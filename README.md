# Flightmare CUDA — v0.1.2

The [hardening report](flightlib-cuda/docs/v0.1.2-hardening.md) records invalid
training rejection, setup/capture guards and current validation limits.

A CUDA-accelerated implementation of Flightmare's headless Flightlib simulation
and quadrotor task, with persistent GPU state, PyTorch tensor APIs and resident RL.
It retains the Flightmare dynamics and model preparation; it does not use PhysX.

## Implemented capabilities

- Batched physical state, commands, motors, model configuration/randomization,
  integration, world bounds and separately callable core mathematical helpers.
- Environment observations, rewards, terminal decisions, device random resets,
  partial reset, scalar stepping and vector autoreset.
- Persistent storage, explicit streams/ownership, CUDA graph execution, grouped
  environment/agent views and explicit host-copy adapters.
- Resident policy inference, rollout history, fused episode statistics and GAE,
  PPO training, save/load/evaluation and a reproducible parallel CPU comparison.
- Local validation, frozen CPU oracle, numerical qualification and Nsight gates
  for transfers, allocations and launch counts. No cloud CI is required.

This is the completed **scoped headless rewrite**, not a replacement for Unity
rendering, sensors, scene collision geometry, ROS/PX4 or every legacy RL framework.
Those upstream components remain in the repository but are not GPU-accelerated by
this package. CPU/GPU random sequences and TensorFlow checkpoints are not identical.
See the [capability mapping](flightlib-cuda/docs/cuda9-usage-contract.md),
[core API](flightlib-cuda/docs/core-api.md) and
[numerical qualification](flightlib-cuda/docs/cuda9-validation.md).

## Build and validate

The supported development path is the existing Linux NVIDIA Docker Compose
container (also tested through Windows Docker Desktop/WSL2). Configure
`FLIGHTMARE_CUDA_ARCHITECTURES` for your GPU; the checked-in default targets the
RTX 5070 Ti. For a build targeting the GPU visible in your container, put
`FLIGHTMARE_CUDA_ARCHITECTURES=native` in a repository-root `.env` before
starting Compose. See [local development](docs/local-development.md) for toolchain,
reports, CPU oracle integrity and known sanitizer/multi-GPU coverage limits.

```sh
docker compose up -d --build
docker compose exec -T dev bash scripts/setup-cuda-python.sh
docker compose exec -T dev python3 scripts/local_ci.py verify
```

Verify builds native/Python packages and tests a fresh install outside the source
checkout, including train/save/reload/evaluate. Core is installed with CMake;
`flightrl-cuda` is a separate Python distribution with a small CUDA library.
For a persistent installation inside the prepared container:

```sh
docker compose exec -T dev cmake --install /build/flightlib-cuda-python-native --prefix /build/flightmare-install
docker compose exec -T dev /build/flightlib-cuda-python/bin/pip install --no-deps --no-build-isolation ./flightrl-cuda
```

Set `PYTHONPATH=/build/flightmare-install` when using the installed core. Optional
`flightrl-cuda[spaces]` supplies Gymnasium Box spaces; the host adapter is not a
Gymnasium five-return Env or Stable Baselines VecEnv subclass.

## Tensor simulation

```python
import torch
from flightrl_cuda import Config, TensorEnv

env = TensorEnv(Config(num_envs=100, slots=100))  # 10,000 independent agents
env.reset()
env.capture()
env.actions.zero_()                            # float32 CUDA [10_000, 4]
for _ in range(100):
    result = env.step()                        # borrowed GPU outputs
    observation = result.observation           # [10_000, 12]
    grouped = env.grouped.observation          # zero-copy [100, 100, 12]
env.close()
```

N = environments × agents. A CUDA thread advances one agent's fused dynamics;
threads for agents inside one environment run in parallel just like agents in
separate environments. Grouping is an indexing/ownership convention, not a
shared swarm reward, inter-agent collision or communication model. Increasing N
exposes more GPU parallelism until compute, occupancy or memory limits dominate.
The scalar integration dependency within one agent remains sequential.

Construct and submit on one chosen Torch stream. Bound storage/shapes stay fixed;
mutate values in place. Results are borrowed and overwritten on the next step;
copy into resident history if retaining them. Setup/configuration changes require
recapture. `.cpu()`, NumPy adapters, logging and checkpoints are explicit transfer
boundaries. Configuration accepts packaged defaults and legacy YAML with explicit
overrides; `max_t` remains metadata rather than an added episode timeout.

## RL example and performance

```sh
docker compose exec -T -e PYTHONPATH=/build/flightmare-install dev /build/flightlib-cuda-python/bin/python -m flightrl_cuda.train --envs 10 --slots 100 --steps 64 --updates 20 --checkpoint /build/policy.pt
docker compose exec -T -e PYTHONPATH=/build/flightmare-install dev /build/flightlib-cuda-python/bin/python -m flightrl_cuda.train --envs 10 --slots 100 --steps 64 --evaluate --checkpoint /build/policy.pt
```

On the tested RTX 5070 Ti / Ryzen 9800X3D, matched 32-step rollouts plus GAE and
two PPO epochs measured about **3.4×** speedup at 1,024 agents and **4.1–4.3×**
at 16,384 against the best tested parallel CPU configuration, including CPU
physics with a GPU learner. CPU wins at very small batches. These are controlled
iteration benchmarks, not convergence or sim-to-real certification. Resident
collection traces show zero host/device transfers and zero allocations; PPO
updates allocate temporary storage and are measured separately.

See [RL evidence and reproduction](flightlib-cuda/docs/cuda9-validation.md) and
[grouped benchmark reproduction](flightlib-cuda/docs/v0.1.1-scaling.md#november-18-reproduction).
Large batches consume rollout and learner memory as well as simulation state;
there is no claim of unbounded batch size. Further optimization should follow
measured occupancy/register pressure, launch overhead and learner cost.

## Upstream attribution

This project builds on [UZH RPG Flightmare](https://github.com/uzh-rpg/flightmare).
The original source and frozen CPU behavior oracle are retained. The upstream
Unity features are described in the [original documentation](https://flightmare.readthedocs.io/).
Please retain the MIT [license](LICENSE) and cite the original work:

```bibtex
@inproceedings{song2020flightmare,
  title={Flightmare: A Flexible Quadrotor Simulator},
  author={Song, Yunlong and Naji, Selim and Kaufmann, Elia and Loquercio, Antonio and Scaramuzza, Davide},
  booktitle={Conference on Robot Learning},
  year={2020}
}
```
