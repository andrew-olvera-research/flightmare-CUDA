# Single-aircraft ROS 2 bridge

The Jazzy bridge runs one Flightlib environment with one aircraft, collision queries and an IMU. The default control period is 10 ms and the ideal IMU samples at 200 Hz. The camera is opt-in. Simulation time advances as fast as the producer and its bounded publisher queue allow.

## Run

Build the CUDA Python environment once in the dev service. The ROS service uses the same checkout and build volume, the official Jazzy ROS base image and its system Python 3.12 ROS bindings.

```sh
docker compose up -d dev
docker compose exec -T dev bash scripts/setup-cuda-python.sh
docker compose --profile ros2 up ros2
```

For RGB, install the [Newton camera dependencies](../newton/README.md#run-in-the-gpu-dev-container) in the shared build volume, then run:

```sh
docker compose run --rm ros2 /build/flightlib-cuda-python/bin/python -m flightlib_cuda.ros2 --scene split_s --camera-every 2 --steps 500
```

Run one clock publisher per ROS domain. Set `ROS_DOMAIN_ID` on the Compose invocation and use the same domain on subscribers. Set `use_sim_time:=true` on consumers. For the default running service, inspect messages with `docker compose exec ros2 /ros_entrypoint.sh ros2 topic echo --once /imu/data_raw`. The CLI is an open-loop sensor source initialized at (0, 0, 2) m. For caller-supplied actions, use the Python interface.

```python
from flightlib_cuda.ros2 import SingleAgentBridge

bridge = SingleAgentBridge(node, scene="room", camera_every=2)
try:
    for action in actions:
        bridge.env.actions[0, 0].copy_(action)
        transition = bridge.step()
finally:
    bridge.close()
```

Use a single producer on the simulation stream. Only `bridge.step()` advances this environment. Returned transition tensors are borrowed until the next step. Autoreset is disabled. For an intentional reset, drain the bridge, update reset state and call `bridge.env.reset()`. The simulation clock continues across resets. The caller owns the ROS node and its lifecycle.

## Wire contract

| Topic | Messages and timing |
| --- | --- |
| `/clock` | One `rosgraph_msgs/Clock` per completed simulation tick. Integer nanoseconds start one control period after zero. Reliable, depth 10. |
| `imu/data_raw` | One `sensor_msgs/Imu` per valid delivered sample. Header uses the **acquisition** time, including when several samples arrive in a later tick. Gyro XYZ in rad/s and specific force XYZ in m/s² are expressed in `imu_link`. Orientation covariance starts with -1 to indicate an unavailable orientation estimate. Zero gyro/acceleration covariance means unknown. |
| `camera/image_raw`, `camera/camera_info` | One matching `rgb8` image and ideal pinhole calibration per valid requested exposure, every N ticks. Both headers carry the pose acquisition stamp and `camera_optical_frame`. No distortion. |
| `/tf_static` | Sensor mounts relative to `base_link` (forward, left, up). The IMU is at the body origin with its configured sensor-to-body rotation. The FPV camera is 0.16 m forward; optical X points right, Y down, Z forward. |

Sensor publishers use the ROS sensor-data QoS profile (best effort). Time stamps allow matching across topics regardless of DDS arrival order. A local Z-up simulation frame has no implicit geographic heading. The bridge publishes sensor mounts; vehicle localization remains the consumer's responsibility.

## Graph boundary and validation

The producer replays the physics/sensor graph, then Flightlib exports immutable snapshots through required subscriptions. The IMU uses its bounded asynchronous host transport. The optional camera receives a **device** pose lease, renders on a separate CUDA stream and copies the result into reusable pinned memory. A single publisher worker serializes the snapshots and sends DDS messages. The producer can run up to three outstanding ticks. At capacity it waits for the oldest publication before advancing. `drain()` waits for local publication and propagates worker failures; remote best-effort delivery depends on DDS.

Run the live subscriber tests in an isolated domain:

```sh
docker compose run --rm -e ROS_DOMAIN_ID=42 ros2 /build/flightlib-cuda-python/bin/python -m pytest -q tools/ros2/test_bridge.py
```

The tests check a ROS-time consumer through the one-second boundary, two IMU samples per tick with 3 ms delivery latency, gyro signs, mounted specific force, both static transforms, image/calibration timing and the projected position of an asymmetric target. On the RTX 5070 Ti, inspecting graph nodes during instantiation showed four kernel nodes and zero memcpy nodes with the camera both enabled and disabled. Host transfers and DDS serialization occur after replay.
