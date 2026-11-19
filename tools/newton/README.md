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

## Qualification on an RTX 5070 Ti

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
