"""Single-aircraft ROS 2 publishers. ROS and Newton are optional runtime dependencies."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import argparse

import torch
import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import TransformStamped, Vector3
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image, Imu
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

from . import Environment, ImuConfig, Subscription
from .transport import step_nanoseconds


def _stamp(ns):
    sec, nanosec = divmod(ns, 1_000_000_000)
    return Time(sec=sec, nanosec=nanosec)


def _mount(child, x, xyzw):
    tf = TransformStamped()
    tf.header.frame_id = "base_link"
    tf.child_frame_id = child
    tf.transform.translation.x = x
    q = tf.transform.rotation
    q.x, q.y, q.z, q.w = xyzw
    return tf


class SingleAgentBridge:
    """One producer calls step(). A bounded worker drains GPU snapshots in tick order."""

    depth = 3

    def __init__(
        self,
        node,
        *,
        control_dt=0.01,
        imu=ImuConfig(),
        scene="room",
        camera_every=None,
        camera_size=(256, 192),
    ):
        if camera_every is not None and (
            type(camera_every) is not int or camera_every < 1
        ):
            raise ValueError("camera_every must be a positive tick interval")
        self.tick_ns = step_nanoseconds(control_dt)
        self.every = camera_every
        specs = [
            Subscription("imu", destination="host", required=True, depth=self.depth)
        ]
        if camera_every:
            specs.append(
                Subscription(
                    "pose",
                    required=True,
                    depth=self.depth,
                    period_ns=camera_every * self.tick_ns,
                )
            )
        self.env = Environment(
            1,
            1,
            control_dt=control_dt,
            imu=imu,
            scene=scene,
            auto_reset=False,
            subscriptions=specs,
        )
        self.env.reset_state[2].fill_(2)
        self.env.reset()
        self.env.capture(random=False)
        self.clock = node.create_publisher(Clock, "/clock", 10)
        self.imu = node.create_publisher(Imu, "imu/data_raw", qos_profile_sensor_data)
        w, x, y, z = imu.mount_wxyz
        mounts = [_mount("imu_link", 0.0, (x, y, z, w))]
        self.stream = torch.cuda.Stream(device=self.env.device)
        if camera_every:
            from .camera import TiledCamera

            with torch.cuda.stream(self.stream):
                self.image_valid = torch.empty(1, dtype=torch.bool, pin_memory=True)
                self.camera = TiledCamera(
                    1, *camera_size, device=self.env.device, scene=self.env.scene
                )
                self.pixels = torch.empty(
                    (self.camera.height, self.camera.width, 3),
                    dtype=torch.uint8,
                    pin_memory=True,
                )
            self.image = node.create_publisher(
                Image, "camera/image_raw", qos_profile_sensor_data
            )
            self.info = node.create_publisher(
                CameraInfo, "camera/camera_info", qos_profile_sensor_data
            )
            mounts.append(
                _mount(
                    "camera_optical_frame",
                    self.camera.forward_offset_m,
                    (-0.5, 0.5, -0.5, 0.5),
                )
            )
        self.tf = StaticTransformBroadcaster(node)
        self.tf.sendTransform(mounts)
        self.tick = 0
        self.pending = deque()
        self.worker = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="flightlib-ros"
        )
        self.closed = False

    def step(self):
        if self.closed:
            raise RuntimeError("bridge is closed")
        # Retain at most depth exposures. A slow publisher applies backpressure
        # before we touch the simulator, so a completed step is never retried.
        while self.pending and (
            len(self.pending) >= self.depth or self.pending[0].done()
        ):
            self.pending[0].result()
            self.pending.popleft()
        out = self.env.step(graph=True, random=False)
        self.tick += 1
        self.pending.append(self.worker.submit(self._publish, self.tick))
        return out

    def drain(self):
        while self.pending:
            self.pending[0].result()
            self.pending.popleft()

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.drain()
        finally:
            self.worker.shutdown(wait=True)
            self.env.close_subscriptions()

    def _publish(self, tick):
        ns = tick * self.tick_ns
        channel = self.env.subscriptions[0]
        with channel.acquire(wait=True) as frame:
            if frame.acquired_ns != ns or frame.sequence != tick:
                raise RuntimeError("unexpected IMU delivery tick")
            self.clock.publish(Clock(clock=_stamp(ns)))
            for i in range(frame.count):
                if not frame.valid[i]:
                    continue
                msg = Imu()
                msg.header.stamp = _stamp(int(frame.packet_acquired_ns[i]))
                msg.header.frame_id = "imu_link"
                msg.orientation_covariance[0] = -1.0
                gx, gy, gz, ax, ay, az = frame.values[i].tolist()
                msg.angular_velocity = Vector3(x=gx, y=gy, z=gz)
                msg.linear_acceleration = Vector3(x=ax, y=ay, z=az)
                self.imu.publish(msg)
        if self.every and tick % self.every == 0:
            with torch.cuda.stream(self.stream):
                exposure = self.camera.consume(
                    self.env.subscriptions[1], tick // self.every, ns
                )
                self.pixels.copy_(exposure.rgb[0], non_blocking=True)
                self.image_valid.copy_(exposure.valid, non_blocking=True)
                self.stream.synchronize()
                if self.image_valid.item():
                    self._image(exposure.acquired_ns)

    def _image(self, ns):
        msg = Image()
        msg.header.frame_id = "camera_optical_frame"
        msg.header.stamp = _stamp(ns)
        msg.height, msg.width = self.pixels.shape[:2]
        msg.encoding, msg.step = "rgb8", msg.width * 3
        msg.data.frombytes(self.pixels.numpy().tobytes())
        info = CameraInfo()
        info.header = msg.header
        info.width, info.height = msg.width, msg.height
        f = self.camera.focal_px
        cx, cy = (msg.width - 1) * 0.5, (msg.height - 1) * 0.5
        info.distortion_model, info.d = "plumb_bob", [0.0] * 5
        info.k = [f, 0.0, cx, 0.0, f, cy, 0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [f, 0.0, cx, 0.0, 0.0, f, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.info.publish(info)
        self.image.publish(msg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steps", type=int, help="stop after this many simulation ticks"
    )
    parser.add_argument(
        "--camera-every", type=int, help="publish one image every N ticks"
    )
    parser.add_argument("--scene", default="room")
    from rclpy.utilities import remove_ros_args

    args = parser.parse_args(remove_ros_args()[1:])
    from rclpy.signals import SignalHandlerOptions

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    try:
        node = rclpy.create_node("flightmare_bridge")
        bridge = SingleAgentBridge(
            node, scene=args.scene, camera_every=args.camera_every
        )
        try:
            while rclpy.ok() and (args.steps is None or bridge.tick < args.steps):
                bridge.step()
        except KeyboardInterrupt:
            pass
        finally:
            bridge.close()
            node.destroy_node()
    finally:
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
