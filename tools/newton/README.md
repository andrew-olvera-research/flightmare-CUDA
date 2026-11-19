# Newton tiled FPV prototype

The optional renderer uses one static Split-S scene, one aircraft and camera per environment, and one shared pinhole ray table. Flightlib owns dynamics, collision queries and episode resets. Newton 1.6.1 renders the gate and room boxes with a directional light and shadows. The full-resolution packed color and optional forward depth stay on the device. This fixture tests timing and geometry. Photometric equivalence to an FPV camera or Isaac remains unqualified.

## Run in the GPU dev container

Build the Flightlib CUDA Python extension from this checkout. In the dev container with the native Python build available:

```sh
PY=/build/flightlib-cuda-python/bin/python
$PY -m pip install --target /build/newton-renderer --no-deps newton==1.6.1 warp-lang==1.19.0
export PYTHONPATH=/build/newton-renderer:/build/flightlib-cuda-python-native/python
$PY -m pytest -q tools/newton/test_tiled.py
for n in 1 32 128 256; do
  $PY tools/newton/bench_newton.py --envs "$n" --mode graph
  $PY tools/newton/bench_newton.py --envs "$n" --mode subscription
done
```

## Capture and replay

Construct the environment without subscriptions and create the camera on the same caller-owned CUDA stream. Warm the Warp renderer before recording. The graph input is the bound action tensor. A downstream collector can also record its policy and rollout writes in this graph.

```python
with torch.cuda.stream(stream):
    camera = TiledCamera(env.environments, device=env.device)
    camera.draw(env.state[:7].T)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        obs, reward, done, valid = camera.record_tick(env)
    env.actions.copy_(actions)
    graph.replay()
    # Read obs and camera.rgb on this stream before the next replay.
```

Each replay advances dynamics, computes collisions and reward, reads the pre-reset state, normalizes the body orientation, renders all lanes, pools RGB to a 3×16×16 policy observation, then applies due resets. The terminal RGB is available even though the benchmark zeros terminal policy inputs. Inputs, outputs, camera buffers and optional count accumulator must retain their addresses for the graph lifetime. Results are borrowed until the next replay. Order consumers on the replay stream or provide explicit dependencies. The subscription path remains available for independently paced cameras with owned exposure batches and episode metadata.

## Baseline qualification on an RTX 5070 Ti

The baseline below used pixel-priority traversal (commit `83bb4af`). Pass `--pixel-priority` to reproduce its render order.

The tests check CPU and GPU forward depth against independent ray/box intersections, same-frame RGB permutation, a delayed cross-stream reader, and captured versus subscription RGB, depth, policy observations, validity, reward, and state across a swept gate collision and partial reset.

The benchmark collects a fresh 256×192 RGB frame every simulated 20 ms. Both paths use the shared BF16 99,353-parameter PPO workload, 24 warmup steps and four 32-step rollouts with four update epochs. The table is one run per configuration, including the first optimizer update. The subscription benchmark waits on reader backpressure when its bounded queue fills. The graph benchmark includes the device action copy, policy, and PPO work outside the captured simulation/render tick.

| Environments | Subscription collection/s | Graph collection/s | Graph full PPO transitions/s |
| ---: | ---: | ---: | ---: |
| 1 | 1,093 | 1,026 | 219 |
| 32 | 21,936 | 27,417 | 9,216 |
| 128 | 35,096 | 40,840 | 23,946 |
| 256 | 37,086 | 42,303 | 27,187 |

To isolate the simulation/render graph from policy and optimizer work:

```sh
nsys profile --sample=none --cpuctxsw=none --trace=cuda --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi --capture-range-end=stop -o render-256 \
  $PY tools/newton/bench_newton.py --envs 256 --profile-ticks 64
nsys stats --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum render-256.nsys-rep
```

The 64 warmed fixed-action replays took 233 ms including a single final stream drain. Host submission took 0.70 ms total. The trace recorded 64 graph launches, 64 capture-status checks, and zero CUDA memcpy or per-tick stream wait calls. Per batch the Newton render megakernel averaged 3.53 ms (97.2% of kernel time), collision-enabled dynamics 0.042 ms, and policy-image pooling 0.037 ms. Render cost depends on the views and scene. This trace measures the fixed-action workload independently of the PPO trajectory.

## Kernel traversal comparison

Newton's default pixel-priority order assigns neighboring threads to the same pixel in different cameras. With camera-major images those threads write one image stride apart. Multi-camera batches now assign contiguous groups of threads to 16×8 tiles inside each view. This changes the ray processing order while preserving the camera model, lighting and output layout. A single camera retains the original traversal.

Run `$PY tools/newton/bench_kernel.py --envs 256 --repeats 5 --ticks 32` to compare traversal, tile size and thread block size with CUDA events. It renders identical fixed poses across configurations, checks exact RGB equality for the shadowed variants, and alternates timing order. The unshadowed variant isolates the cost of shadow rays. To compare whole PPO runs, alternate `bench_newton.py --envs 256 --pixel-priority` with `bench_newton.py --envs 256`.

On the RTX 5070 Ti, two runs per configuration in baseline/tiled/tiled/baseline order gave these **warmed** rates. Each rate uses the last three of four 32-step rollouts, excluding the first optimizer update and its setup costs. Values are the mean of the two run rates.

| Environments | Pixel-priority collection/s | Tiled collection/s | Pixel-priority full PPO/s | Tiled full PPO/s |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 34,706 | 49,978 | 17,823 | 21,066 |
| 256 | 44,616 | 83,037 | 38,586 | 64,631 |

At one camera the warmed collector measured 3,469–3,474 transitions/s with the original traversal and 3,240–3,362 with tiles. These results apply to this GPU, Split-S geometry and 256×192 shadowed RGB workload.

Across five timing rounds at 256 varied fixed poses, the shadowed renderer took 5.104 ms per batch with pixel-priority traversal and 2.693 ms with 16�8 tiles at 64 threads/block. All 20 shadowed configurations produced byte-identical RGB. Reducing the tiled block to 32 threads gave 2.662 ms, a 1% difference on this workload. Disabling shadows for diagnosis gave 2.343 ms at 64 threads. The default retains shadows and Newton's 64-thread block. A separate Nsight trace on the fixed-action simulation workload measured 2.78 ms per render kernel (96.6% of kernel time) and one graph launch per tick.
