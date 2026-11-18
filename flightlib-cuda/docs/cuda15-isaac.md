# CUDA-15 · Isaac Sim cameras

`flightmare-isaac` is an optional visual consumer. Flightlib integrates the
aircraft, resolves contact, timestamps the pose, and owns episode resets. Kit
loads the same Split-S geometry used to build the collision fixture and renders
subscribed cameras at the supplied poses with its timeline paused. A training
run without a pose subscription incurs no renderer work or Kit dependencies.

## Run the installed packages

The Compose profile pins NVIDIA Isaac Sim 6.2.0 for Linux x86-64. Use a Linux
RTX host with NVIDIA Container Toolkit and working Vulkan graphics. Review the
[NVIDIA installation requirements](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/requirements.html)
and [container license and setup](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/install_container.html).
Starting the profile keeps the container available. Kit starts when the
qualification command runs. The profile shares the installed Flightlib build
read-only and keeps the XDG and CUDA compute caches in a writable data volume.

```sh
docker compose up -d --build dev
docker compose exec -T dev bash scripts/setup-cuda-python.sh
docker compose exec -T dev python3 scripts/local_ci.py verify
docker compose exec -T dev bash scripts/setup-isaac-renderer.sh
# Set this for the command after accepting the NVIDIA Omniverse license.
ACCEPT_EULA=Y docker compose -f compose.yaml -f compose.isaac.yaml --profile isaac up -d isaac
docker compose -f compose.yaml -f compose.isaac.yaml exec -T isaac bash -c \
  'export PYTHONPATH=/build/flightmare-isaac LD_LIBRARY_PATH=/build/flightmare-isaac/deps:${LD_LIBRARY_PATH:-}; ./python.sh /workspace/tools/isaac/qualify.py --second-camera --preview /tmp/split-s.png'
```

The default qualification hosts Flightlib, RTX and the PyTorch image policy in
the same Kit Python process on GPU 0. On Windows you can instead run RTX in
the native workstation package and Flightlib in the prepared Linux dev
container. From the repository root in PowerShell, with the matching NVIDIA
workstation distribution unpacked at `C:\isaacsim`:

```powershell
python -m pip install --no-deps --upgrade --target .local-ci/isaac-render ./flightmare-isaac
$env:PYTHONPATH=(Resolve-Path .local-ci/isaac-render).Path
C:\isaacsim\python.bat tools/isaac/qualify.py --docker-producer --second-camera --course flightlib-cuda/python/flightlib_cuda/assets/split_s.usda --preview .local-ci/split-s.png --/app/vulkan=false
```

The Windows fixture uses stdio to exchange selected poses and four action
values. Both image tensors and image inference stay on the workstation GPU.
Use `--trace PATH` to write a Torch/CUPTI trace of a warm step. Saving a preview
explicitly copies one image to the CPU outside the measured render interval.
For a 6.2 Windows CUDA trace, append
`C:\isaacsim\extsDeprecated\omni.isaac.ml_archive\pip_prebundle` to
`PYTHONPATH` before launching. This lets the qualification start Torch's
profiler before Kit claims the CUPTI subscriber.

On this Windows/Docker Desktop host the 6.2.0 Linux image detected the RTX 5070 Ti
and its bundled Python 3.12 / Torch 2.11.0+cu128 ran the installed Flightlib
extension with both collision queries enabled. Vulkan returned
`ERROR_INCOMPATIBLE_DRIVER` during Kit startup. The native Windows renderer
provided the measured end-to-end FPV path. Same-process RTX on a native Linux
graphics host still needs a hardware run. The tested Linux amd64 image digest
is `sha256:bcb96e0b87c28ad446a2a6a2f0fdb7d82b9867f18b96a94c796912c61f07337f`.

## Connect a custom policy or viewer

Start a dedicated Kit process with `flightmare_isaac.launch_visual_app()`.
When embedding in an existing Kit application, set
`--/rtx/hydra/supportMultiTickRate=false` at launch. The adapter requests
captures at a fixed Kit timeline time. Isaac 6.2's independent sensor clock
can otherwise leave the CUDA annotators empty. Create the renderer on the Kit
thread. The FPV benchmark gives each environment one aircraft and one camera.
With `world=None, slots=1`, camera rigs share the immutable metre-scale Z-up USD
course. Each rig follows its own lane. The stage contains no aircraft meshes
that could appear in another environment's image. This layout applies to
environments with identical static course geometry and lighting. Set
`fixture_lighting=False` for a course with its own lights.

```python
from importlib.resources import files
from flightlib_cuda import Environment, Subscription
from flightmare_isaac import Camera, IsaacRenderer

env = Environment(16, 1, control_dt=.02, scene="split_s", terminate_on_collision=True,
                  subscriptions=(Subscription("pose", actors=((7, 0), (12, 0)),
                                 destination="host", required=True),))
renderer = IsaacRenderer(files("flightlib_cuda").joinpath("assets", "split_s.usda"),
                         world=None, slots=1, actors=(7, 12),
                         cameras=(Camera(7, latency_ns=2_000_000), Camera(12)))
```

On each due policy tick, submit the Flightlib step and call
`renderer.consume(env.subscriptions[0], expected_ns=tick_ns)`. The result is a
tuple ordered by camera. An invalid observer yields `None`. Match the observer's episode, requested acquisition and delivery
deadline with `frame.observation(episode=episode, acquired_ns=capture_ns,
time_ns=now_ns)`. It returns CUDA RGB (H×W×3, uint8) and axial depth in metres
(H×W, float32). Set `Camera(..., depth=False)` to create only the RGB
annotator and receive `None` for depth. The centered pinhole uses square pixels, focal length in
pixels, clipping distances in metres, a body-FLU mount offset and WXYZ
orientation. Camera latency is expressed in simulation nanoseconds. An image
from the terminal tick belongs to the ending episode. Clear delayed policy
history at reset.

For a viewer, configure a best-effort pose subscription and poll `consume(channel)`
on the Kit thread. An empty queue returns `None`. Running the stepper on its
own thread lets it continue when the viewer is busy. A strict FPV policy pays
the render latency at the tick where it needs the observation. Close the
renderer on the Kit thread, close the subscriptions, then shut down Kit.
See the [bounded queue and capture semantics](cuda14-transport.md).

## Data movement and qualification

Each due pose copies 48 bytes per selected actor into a pinned host record for
USD transform authoring. The entire simulated population only enters this
copy if subscribed. RTX allocates RGB and axial-depth annotators per camera.
Warp exposes each annotator buffer to Torch on CUDA. The adapter copies out
RGB (stripping alpha) and depth so a caller may retain either tensor after
subsequent Kit captures. It waits for the consumer stream to finish those
copies before allowing Kit to recycle the buffers. At 256×192, an owned RGB
image occupies 147,456 bytes and depth occupies 196,608 bytes. Retaining more
frames in a trainer increases VRAM accordingly.

The [qualification](../../tools/isaac/qualify.py) drives eight real CUDA steps,
checks left/right/top gate bars and the room wall at known image coordinates,
invalidates a second environment, forces a terminal pose and verifies the first
image after reset. A ray through the gate opening passes through the position
occupied by the other environment's aircraft and reaches the wall. A
deterministic policy computes five actions solely from delayed RGB and depth.
The optional second environment's camera has its own resolution, mount and
depth checks. The [installed-package tests](../../flightmare-isaac/tests/test_contract.py)
exercise the host pose subscription, capture identity, dropped viewer frames
and latency without loading Isaac.

On an RTX 5070 Ti (16 GB, Windows driver 610.88), the producer's Nsight trace
recorded one 96-byte device-to-host copy for two pose records. The 6.2 CUDA
renderer trace recorded 196,608-byte and 49,152-byte device-to-device depth
copies for 256×192 and 128×96 images and GPU kernels for texture extraction and
RGB packing. The Windows harness also transfers the four policy action values
back to the producer. CUDA tracing covers CUDA-visible transfers. The three
gate-bar probes read 2.94 m and the wall through the opening read 16.27 m.
Board-wide VRAM was 2,117 MiB after producer startup and 2,490 MiB after
the first pair of images. These snapshots include other GPU users and
driver allocations.

## Vision-in-the-loop PPO benchmark

[bench_fpv.py](../../tools/isaac/bench_fpv.py) measures independent Split-S
environments with exactly one aircraft each at a 20 ms control period. Every
aircraft receives its own 256×192 RGB exposure on each tick.
The policy pools RGB to 16×16 on the GPU and passes all
768 color features through two 112-unit ReLU layers and a shared actor/value
output. Four learned log standard deviations parameterize the sampled,
tanh-squashed actions. BF16 autocast covers the MLP. GAE, Gaussian likelihoods
and optimizer state use FP32. The benchmark allocates no IMU, odometry or
depth channels. Its public entry point uses the installed renderer and core.

The default benchmark warms eight ticks and collects three 32-step rollouts
per camera count. Each rollout is followed by four PPO epochs with four
shuffled minibatches, a 0.2 ratio clip, 0.99 discount, 0.95 GAE lambda and
fused Adam at 0.0003. Swept collisions with the room and gates terminate episodes.
A preflight step crosses a gate post in the first environment and confirms
that the other environments remain collision-free. On autoreset, a black
image initializes the new episode until its first capture. Collection rates
count transitions from camera-controlled aircraft. Training rates include GAE
and optimization. The real-time factor compares completed world ticks to 50 Hz.

```powershell
C:\isaacsim\python.bat tools/isaac/bench_fpv.py --docker-producer --course flightlib-cuda/python/flightlib_cuda/assets/split_s.usda --env-counts 1 4 8 16 32 64 100 --warmup 24 --/app/vulkan=false
```

In the Linux container, use the qualification command above with
`/workspace/tools/isaac/bench_fpv.py --env-counts 1 4 8 16 32 64 100 --warmup 24`
as the script and arguments. Render times include pose submission, RTX and
the owned CUDA RGB copy. The Windows collection timer also includes stdio,
action transfer and the return of rewards and terminal/valid flags.

### Measured on RTX 5070 Ti

The 256×192 sweep used 24 warm-up ticks and 96 measured ticks per batch size.
Every environment had one camera, one aircraft, a 99,353-parameter BF16
actor-critic, and collision termination. The board VRAM column is a snapshot
after the update. Collection includes rendering, inference, action feedback
and rollout storage. The PPO column also includes GAE and all optimizer steps.
[Per-rollout timings and collision counts](cuda15-fpv-benchmark.json)

| Environments | World ticks/s | Collection transitions/s | With PPO updates, transitions/s | Median render ms/tick | Board VRAM MiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 38.83 | 38.8 | 34.8 | 21.2 | 2,408 |
| 4 | 30.17 | 120.7 | 115.6 | 29.8 | 2,749 |
| 8 | 15.03 | 120.2 | 117.8 | 51.3 | 3,180 |
| 16 | 10.41 | 166.6 | 164.3 | 88.9 | 3,485 |
| 32 | 4.54 | 145.2 | 144.3 | 200.7 | 4,658 |
| 64 | 3.19 | 204.4 | 203.6 | 303.9 | 7,002 |
| 100 | 1.22 | 121.6 | 120.8 | 812.0 | 9,520 |

The highest measured aggregate rate was at 64 environments. At 100, the
renderer took a median 812 ms per batch and the timed rollouts recorded
133 course collisions. All configurations maintained a 20 ms simulation tick.
None of these 256×192 configurations achieved 50 ticks per wall-clock second.
Points through 64 were measured consecutively in one Kit process. Rebuilding
100 render products in that process triggered an RTX driver crash. The
100-environment measurement completed in a fresh process. The released sweep
launches a fresh Kit process for each point and saves a completed point before
starting the next.

At 128×96, one environment completed 40.09 ticks/s and four environments
completed 23.90 ticks/s (95.6 aggregate transitions/s). Smaller images alone
did not bring this Windows harness to real-time 50 Hz.

The [matched CPU and CUDA PPO comparison](cuda15-ppo-comparison.md) measures
fresh-process camera throughput with course queries on and off, checks
collision parity and times 100 million headless training transitions.
