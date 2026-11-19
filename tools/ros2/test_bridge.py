"""Live DDS subscriber checks. Run inside the ROS 2 Compose service."""

import json
import math
import time

import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image, Imu
from tf2_msgs.msg import TFMessage

from flightlib_cuda import ImuConfig
from flightlib_cuda.ros2 import SingleAgentBridge


def ns(stamp):
    return stamp.sec * 10**9 + stamp.nanosec


def until(node, ready):
    deadline = time.monotonic() + 10
    while not ready():
        assert time.monotonic() < deadline, "DDS discovery or delivery timeout"
        rclpy.spin_once(node, timeout_sec=0.02)


@pytest.mark.parametrize("camera", [False, True])
def test_wire_timing_and_frames(tmp_path, camera):
    scene = tmp_path / "target.flightcourse"
    scene.write_text(
        json.dumps(
            dict(
                schema="flightmare.boxes.v1",
                assets=[
                    dict(
                        name="target",
                        boxes=[dict(size=[0.5, 0.5, 0.5], center=[0, 0, 0])],
                    )
                ],
                instances=[dict(asset="target", yaw_deg=0, position=[4, -1, 1.5])],
            )
        )
    )
    rclpy.init()
    sender = rclpy.create_node("flightmare_test_source")
    reader = rclpy.create_node(
        "flightmare_test_reader",
        parameter_overrides=[Parameter("use_sim_time", value=True)],
    )
    h = math.sqrt(0.5)
    bridge = SingleAgentBridge(
        sender,
        scene=scene,
        camera_every=2 if camera else None,
        camera_size=(64, 48),
        imu=ImuConfig(
            rate_hz=200,
            latency_s=0.003,
            mount_wxyz=(h, h, 0.0, 0.0),
            gyro_bias=(1.0, 2.0, 3.0),
        ),
    )
    try:
        clocks, imus, images, infos, mounts = [], [], [], [], []
        reliable = QoSProfile(depth=500)
        sensor = QoSProfile(depth=500, reliability=ReliabilityPolicy.BEST_EFFORT)
        tf = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        watch = [
            (Clock, "/clock", clocks, reliable),
            (Imu, "imu/data_raw", imus, sensor),
            (TFMessage, "/tf_static", mounts, tf),
        ]
        if camera:
            watch.extend(
                [
                    (Image, "camera/image_raw", images, sensor),
                    (CameraInfo, "camera/camera_info", infos, sensor),
                ]
            )
        else:
            assert len(bridge.env.subscriptions) == 1
            assert sender.count_publishers("camera/image_raw") == 0
        subs = [
            reader.create_subscription(cls, topic, buf.append, qos)
            for cls, topic, buf, qos in watch
        ]
        until(
            reader,
            lambda: (
                all(sub.get_publisher_count() for sub in subs)
                and bridge.imu.get_subscription_count()
                and mounts
                and (
                    not camera
                    or bridge.image.get_subscription_count()
                    and bridge.info.get_subscription_count()
                )
            ),
        )
        steps = 8 if camera else 104
        for _ in range(steps):
            bridge.step()
        bridge.drain()
        end = steps * 10_000_000
        until(
            reader,
            lambda: (
                len(clocks) == steps
                and len(imus) == steps * 2 - 1
                and reader.get_clock().now().nanoseconds == end
                and (not camera or len(images) == len(infos) == steps // 2)
            ),
        )
        assert [ns(m.clock) for m in clocks] == list(
            range(10_000_000, end + 1, 10_000_000)
        )
        assert [ns(m.header.stamp) for m in imus] == list(
            range(5_000_000, end, 5_000_000)
        )
        for imu in imus:
            assert (
                imu.header.frame_id == "imu_link"
                and imu.orientation_covariance[0] == -1
            )
            np.testing.assert_allclose(
                [
                    imu.angular_velocity.x,
                    imu.angular_velocity.y,
                    imu.angular_velocity.z,
                ],
                [1, 2, 3],
                atol=1e-4,
            )
        # Positive body-Z thrust projects to sensor +Y under the 90-degree X mount.
        acc = imus[-1].linear_acceleration
        assert acc.y > 1 and abs(acc.z) < 1e-4 and abs(acc.x) < 1e-4
        transforms = {t.child_frame_id: t for t in mounts[-1].transforms}
        assert all(t.header.frame_id == "base_link" for t in transforms.values())
        rot = transforms["imu_link"].transform.rotation
        np.testing.assert_allclose([rot.w, rot.x, rot.y, rot.z], [h, h, 0, 0])
        if camera:
            t = transforms["camera_optical_frame"].transform
            assert t.translation.x == 0.16
            q = t.rotation
            # Optical +X right, +Y down, +Z forward in body FLU.
            v = np.array([q.x, q.y, q.z])
            axes = np.eye(3)
            rotated = axes + 2 * np.cross(v, np.cross(v, axes) + q.w * axes)
            np.testing.assert_allclose(rotated, [[0, -1, 0], [0, 0, -1], [1, 0, 0]])
            for i, (image, info) in enumerate(zip(images, infos), 1):
                assert ns(image.header.stamp) == i * 20_000_000
                assert (
                    info.header == image.header
                    and image.header.frame_id == "camera_optical_frame"
                )
                assert (image.width, image.height, image.step, image.encoding) == (
                    64,
                    48,
                    192,
                    "rgb8",
                )
                np.testing.assert_allclose(info.k, [40, 0, 31.5, 0, 40, 23.5, 0, 0, 1])
                rgb = np.frombuffer(image.data, np.uint8).reshape(48, 64, 3)
                # Warm-colored target is to the right and below the optical axis.
                ys, xs = np.where(
                    rgb[:, :, 0].astype(int) > rgb[:, :, 2].astype(int) + 5
                )
                assert len(xs) and xs.mean() > 31.5 and ys.mean() > 23.5
        assert all(c.stats["dropped"] == 0 for c in bridge.env.subscriptions)
    finally:
        bridge.close()
        reader.destroy_node()
        sender.destroy_node()
        rclpy.shutdown()
