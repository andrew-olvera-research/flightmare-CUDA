# Single-aircraft PX4 SITL qualification

Flightlib advances one aircraft at 250 Hz. Its ROS 2 bridge publishes the clock and sensors, and the simulator sends the **same host IMU sample** to PX4 over its MAVLink simulator TCP interface. PX4 v1.17.0 runs commander, EKF2, position/attitude/rate control and control allocation. The optional 50 Hz FPV camera publishes RGB and calibration to ROS consumers. EKF2 uses inertial, magnetic, pressure and GNSS measurements in this qualification.

## Reproduce

Prepare the [ROS 2 bridge](../ros2/README.md) and its optional Newton dependencies for camera runs. Install the optional MAVLink codec and ULog reader in the shared build volume:

```sh
docker compose exec -T dev /build/flightlib-cuda-python/bin/python -m pip install --target /build/flightmare-mavlink --no-deps pymavlink==2.4.50 pyulog==1.3.0 lxml fastcrc
```

Use a lockstep-enabled Ubuntu 24.04 x86_64 PX4 `px4_sitl_default` build, including its `etc` directory and `px4-commander` command. The reference revision is `d6f12ad1c4f70ad3230afd7d86e971421e02fef4` (v1.17.0). The helper stages the binaries, startup files and linked runtime libraries from an existing build. Run it inside the build environment with the Flightmare checkout and build volume mounted:

```sh
bash /workspace/tools/px4/stage_sitl.sh /path/to/PX4-Autopilot/build/px4_sitl_default /build/px4-flightmare-v117
```

To build from upstream sources, the equivalent upstream build steps in an Ubuntu 24.04 build environment are:

```sh
git clone --branch v1.17.0 --depth 1 --recursive https://github.com/PX4/PX4-Autopilot.git
cd PX4-Autopilot
bash Tools/setup/ubuntu.sh --no-nuttx --no-sim-tools
make px4_sitl_default
```

The staged directory is needed only once. Use a fresh output directory for each run. The runner starts a dedicated instance in the ROS service container and closes it on exit. Its simulator socket is loopback TCP 4560, instance 0. See the [PX4 simulation interface and lockstep description](https://docs.px4.io/v1.17/en/simulation/).

```sh
docker compose run --rm ros2 /build/flightlib-cuda-python/bin/python tools/px4/bench_sitl.py --build /build/px4-flightmare-v117 --output /build/px4-test-1
docker compose run --rm ros2 /build/flightlib-cuda-python/bin/python tools/px4/bench_sitl.py --build /build/px4-flightmare-v117 --output /build/px4-camera-1 --camera-every 5
docker compose run --rm -e ROS_DOMAIN_ID=42 ros2 /build/flightlib-cuda-python/bin/python -m pytest -q tools/px4/test_link.py tools/ros2/test_bridge.py
```

## Timing, frames and control

- The adapter feeds a stationary sensor bootstrap until PX4 first returns lockstep actuator output. That PX4 timestamp defines the epoch corresponding to Flightlib time zero. After startup, only the response for the exact sensor timestamp is accepted. Timeout, stale/future feedback or a missing lockstep flag fails the run.
- For each tick, apply the previous actuator frame, integrate for 4 ms, then read the valid acquired IMU packet through the ROS bridge's bounded host transport. The simulation also copies 13 state floats to host for synthetic aiding and independent truth logging. MAVLink, device/host transfers and ROS publication are outside the stepping graph.
- The room's local X/Y/Z axes are north/west/up. The body is forward/left/up. The PX4 measurements use local NED and body FRD. Simulated body rates and specific force come from the aligned Flightlib IMU. A fixed geographic origin, a local magnetic field and a standard-atmosphere pressure model provide 10 Hz GNSS and 50 Hz magnetometer/barometer signals. Magnetic and pressure noise is seeded (0.0005 gauss and 0.012 hPa standard deviation). GNSS is ideal apart from wire quantization.
- The `none_iris` rotor functions on outputs 1–4 are front-right, back-left, front-left, back-right. The adapter remaps those outputs to Flightlib rotor order and uses a linear thrust curve with 50% hover for the default 1 kg model. The benchmark configures the allocation geometry and moment coefficients to match. A simulated level launch cradle holds the reset pose and commands hover support before liftoff. Once armed collective thrust exceeds weight by 1%, all four PX4 motor commands drive the unconstrained plant. This fixture covers takeoff and hover; it has no landing contact solver.
- To embed the adapter, construct a fresh `SingleAgentBridge` with a 4 ms control period and an aligned 250 Hz IMU, accept the PX4 simulator TCP connection, and construct `flightlib_cuda.px4_sitl.SingleAgentSITL(bridge, socket)`. Call its `step()` from the single producer. The caller owns the socket, firmware process and ROS node.

## Acceptance and results

At 14 simulated seconds, the runner requests a normal commander takeoff to 2.5 m above home. It requires a sustained armed climb. It aligns the onboard ULog attitude and local position/velocity estimates to plant truth by **sample time** over the entire released flight, translates the EKF local reference origin and checks validity and sample coverage. Qualification limits are 0.20 m 3-D position RMS, 0.20 m/s 3-D velocity RMS and 3 degrees quaternion angular RMS. No ground-truth pose message is sent to the estimator.

Outputs include the firmware log, PX4 ULog, `report.json`, and `truth.npy`. Truth rows hold PX4 sensor time in microseconds, position XYZ, body-to-world WXYZ, world velocity XYZ, body rates XYZ, armed flag, cradle-release flag, four applied Flightlib actions and the source actuator timestamp. The runner verifies a 4,000 microsecond difference between those two timestamps on every row.

Two runs per configuration on the RTX 5070 Ti and Ryzen 7 9800X3D used the local v1.17.0 SITL build, 8,000 ticks per run, ROS publication and ULog recording. Timing includes local sensor publication, EKF/control execution, the actuator barrier and logging during the run. It excludes firmware boot, camera initialization and offline scoring.

| ROS camera | Wall time for 32 simulated seconds | Real-time factor | Wall tick p95 |
| --- | ---: | ---: | ---: |
| Off | 8.51–8.57 s | 3.73–3.76× | 1.21–1.22 ms |
| 256×192 RGB at 50 Hz | 11.40–11.45 s | 2.79–2.81× | 1.62–1.66 ms |

All four flights climbed 2.49 m. Airborne 3-D position RMS ranged from 0.0090 to 0.0094 m, velocity RMS from 0.0054 to 0.0068 m/s and attitude RMS from 0.17 to 0.29 degrees. These results describe this short, aided vertical flight and local ROS publication. They do not measure estimator robustness to sensor outages or a downstream visual estimator.
