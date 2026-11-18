# Sim-to-real integration plan

Proposed scope for the next release series. Flightlib CUDA owns physics, collision results, episode identity and simulation time. Renderers consume scene poses. Flight software consumes measurements and returns actuator commands. Each adapter can be installed and tested independently.

## Execution modes and ownership

| Mode | Who advances time | Critical dependency |
| --- | --- | --- |
| Headless RL | CUDA rollout driver | Policy actions and any scheduled GPU estimator update |
| FPV RL | CUDA rollout driver | The camera observation required by that policy, subject to the configured sensor delay |
| PX4 software-in-the-loop (SITL) | Dedicated validation runner coordinating Flightlib and the pinned PX4 simulator protocol | The flight stack's actuator output at its negotiated control boundary |

Give a validation vehicle or interacting validation swarm its own Simulation instance. Independent training batches can advance while the validation instance waits. Vehicles that can collide or see each other share one world and clock. Assign exactly one actuator writer per vehicle. ROS offboard commands enter PX4 through its application interface and PX4 retains motor authority. A read-only viewer can attach to any mode.

## CUDA and sensor work

CUDA-13 implements the resident sensor timeline, IMU and local odometry. Renderer
and flight-software adapters can consume the same delivery timestamps, episode IDs
and validity masks. See the [sensor API and qualification](cuda13-sensors.md).

1. CUDA-11 now checks the spherical body against static gate and room colliders over every integration substep. A shared BVH serves all lanes, and the task consumes per-lane hit and validity before reward, terminal observation and autoreset. CUDA-12 adds opt-in continuous sphere pairs with environment-local candidate capacity and overflow reporting. [Moving contact contract](cuda12-swarm-collision.md). [Frozen collision and asset contract](cuda11-static-collision.md).
2. Add an integer simulation timebase and independent sensor periods, sample phases and delivery deadlines alongside the existing integration schedule. Preserve existing floating-point integration semantics. Sample at an integration boundary or specify how intermediate truth is reconstructed. Capture terminal samples before reset. Use an episode generation to invalidate pending old samples. Keep the external flight-software clock monotonic across a reset, or restart the entire session explicitly.
3. Provide a batched sensor interface with setup-time mount, calibration, rate and seed, per-lane persistent model state, and caller-owned output storage. Each packet carries sensor and vehicle IDs, episode, sequence, acquisition and availability timestamps, validity and units/frame from the schema. Begin with ideal output, then add selectable calibration, bias drift, white noise with explicit density units, delay, saturation and dropout. Users can replace the physical measurement function or the error model. Size delay queues from configured rate and maximum latency.
4. Ship gyro and accelerometer at the body origin first. The ideal gyro reads body rate; an aligned accelerometer reads body-frame specific force, gravity removed from inertial acceleration before rotation. Document acceleration during world-box corrections or resets; discontinuities must never masquerade as a normal IMU sample. Support rotated mounts, and either model rotational acceleration at an offset mount or reject that configuration initially. Add delayed local pose/odometry for racing and barometer, magnetic field and GNSS for the selected outdoor PX4 airframe. Specify pressure reference, magnetic reference and geodetic origin. One published indoor profile can instead configure external vision aiding explicitly. Aircraft-specific error distributions remain configurable.
5. Compute cheap batched measurements on GPU. If a high-rate IMU requires intermediate derivatives, add an opt-in sampling point inside the dynamics loop and write only due measurements. Run lower-rate/noisy transforms as a batched sensor stage. Compare fused and separate versions using completed step time, registers, memory and numerical parity. A disabled sensor must incur no extra launch or buffer. In training, schedule a user estimator against delivered measurements; in SITL, run the actual PX4 estimator and expose its health and innovations for validation. Keep truth available for scoring and supervision through a separate interface.

## Device and host transport

CUDA-14 supplies the installed producer API, pre-reset pose capture, device leases
and pinned-host transport. The Isaac and Unity adapters can build on this boundary.
See the [implemented contract and measurements](cuda14-transport.md).

Use the runtime's stream ordering and stable bindings. Allocate subscriptions, device staging buffers, pinned host buffers and queue capacity at setup. A bridge worker owns protocol serialization and socket or middleware I/O. [Runtime API](../include/flightlib_cuda/runtime.hpp).

| Consumer | Data movement and ownership |
| --- | --- |
| CUDA estimator or policy | Borrow typed device views on an ordered stream. Retain a slot until consumption finishes. |
| Isaac FPV policy | Gather transforms for visible actors in selected environments and request only subscribed cameras. First qualify a same-process Isaac runner hosting the simulation extension, renderer and policy on the same GPU. Verify the supported pose update path, image layout, GPU buffer ownership and event ordering on the pinned build. Count any host staging and device copies. Use a bounded host-staged compatibility path if required. |
| PX4 and conventional ROS 2 messages | Pack due small records for selected vehicles into device staging on the simulation stream. Signal a CUDA event. On a transfer stream wait for that event, copy the packed block asynchronously into a pinned slot and signal completion. A worker consumes only completed slots. Reverse the path for commands and order consumption by the target instance. |
| Unity compatibility | Reuse the legacy ZeroMQ pose/frame protocol for selected vehicles. Budget host pose serialization and returned image bytes. Preserve original scene IDs and camera conventions; match replies by frame ID and validate payload sizes. |

Use independent bounded queues for control/sensors, camera export and best-effort viewing. Give device and pinned slots explicit producer, transfer and consumer lifetimes; reuse requires completion on **both** sides. Increment sequence/overrun counters. Under pressure a live viewer may drop old frames; a validation runner must pause or fail explicitly. A training policy consumes the last sample available under its declared sensor model, or waits for the required frame. Copy an image to host only for a subscribed CPU consumer. GPU pointers and CUDA events cannot simply be passed as ordinary pointers between processes. Qualify a separate interprocess transport only if process isolation is needed.

## External integrations

**PX4.** Implement one simulator endpoint using the upstream generated MAVLink library and PX4's Simulator MAVLink API. Start with a single pinned multirotor airframe. Negotiate startup, send timestamped HIL_SENSOR with correct field-update masks and HIL_GPS when due, and map HIL_ACTUATOR_CONTROLS through the airframe's rotor order, normalization, thrust curve and existing motor lag. Bypass the internal rate controller when PX4 supplies motor commands. Validate boot, disarmed output, sensor cadence, actuator hold, arm, timeout and reconnect. Use the pinned firmware's actual timing handshake; an IMU tick need not correspond to a new motor command. [PX4 simulator API](https://docs.px4.io/main/en/simulation/) and [MAVLink message definitions](https://mavlink.io/en/messages/common.html#HIL_SENSOR).

**ROS 2.** Run PX4's existing uXRCE-DDS client and the upstream agent for flight-stack telemetry and offboard commands. Pin compatible firmware, agent, ROS distribution and px4_msgs definitions in the release manifest. Add only simulator-owned topics such as simulation clock, ground truth, camera calibration, images and collision events. Specify QoS and frame IDs per topic and namespaces per vehicle. Use one clock publisher, configure simulated time and PX4 time synchronization coherently, and test ENU/FLU to NED/FRD conversions including quaternions and covariance. Reuse upstream frame transforms where applicable. [PX4 ROS 2 guide](https://docs.px4.io/main/en/ros2/user_guide) and [DDS integration](https://docs.px4.io/main/en/middleware/uxrce_dds).

**Isaac and Unity.** Use Isaac as the primary visual backend. Load a versioned scene and camera calibration once, drive actor poses from CUDA, and render on camera acquisition ticks. Start with RGB and explicitly defined metric depth. Return frame and episode identity, capture time, calibration ID and buffer ownership. Exercise a delayed FPV policy and a dropped viewer frame independently. Evaluate tiled cameras and CUDA annotator outputs with the actual trainer; measure pose submission through policy-ready tensor, including renderer memory and contention. Keep Unity as an optional legacy-protocol adapter with RGB/depth compatibility fixtures and a separately supplied compatible renderer. Expose Unity collision feedback as a diagnostic for scene alignment. [Isaac camera API](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/sensors/isaacsim_sensors_camera.html) and [legacy bridge](../../flightlib/src/bridges/unity_bridge.cpp).

## Delivery and release tests

| Order | Deliverable | Acceptance evidence |
| --- | --- | --- |
| 1 | Headless scene and sensor timeline | Gate opening versus post, coincident isolated worlds, terminal versus reset pose, stationary versus free-fall IMU, deterministic seed replay, rate mismatch and delayed sample across reset. Compare existing disabled-mode benchmarks and measure enabled cost. |
| 2 | Packed transport and one PX4 vehicle | First run a loopback peer to force delays and queue wrap. Then boot actual pinned PX4 SITL, initialize its estimator, arm, hover and land from raw measurements. Record PX4 ULog, truth, command timestamps, mapping and estimator errors. Repeat with a delayed or dropped aiding sensor. |
| 3 | Out-of-box ROS 2 validation | A single documented launch starts the sim, SITL, DDS agent and sample ROS node with pinned configs. A clean install executes a ROS-commanded trajectory and captures ULog plus rosbag with common time and frames. Include timeout, restart and actuator ownership tests. Label this software-in-the-loop; hardware-in-the-loop needs separate hardware qualification. |
| 4 | Isaac visual training | Run the same gate scene through a state-based controller and a camera policy. Check known target pixels and metric depth against geometry, episode/frame pairing, and no truth leakage. Publish resolution, cameras, FPS, end-to-end observation age, GPU memory, copies and dropped frames. |
| 5 | Unity compatibility and combined example | Repeat pose/image fixtures against a documented Unity build. Then run one Isaac camera, ROS companion and PX4 estimator/controller together with a headless baseline. Measure whether camera export impacts control deadlines. |

Milestones 2 and 4 can proceed independently once scene, time and buffer contracts are fixed. Profile D2H/H2D bytes per tick by subscriber, transfer completion latency, queue high-water marks, sensor age, control deadline misses and completed training throughput. Test slow consumers, renderer disconnect, PX4 pause and in-flight shutdown. Keep headless dependencies small; publish renderer and SITL/ROS launch environments separately. Freeze exact dependency revisions after the first working vertical slices and retain those fixtures for upgrade testing.
