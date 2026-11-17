# Flightmare - 左青龙

![License](https://img.shields.io/badge/License-MIT-blue.svg) ![website]( https://img.shields.io/website-up-down-green-red/https/naereen.github.io.svg)

**Flightmare** is a flexible modular quadrotor simulator.
Flightmare is composed of two main components: a configurable rendering engine built on Unity and a flexible physics engine for dynamics simulation.
Those two components are totally decoupled and can run independently from each other. 
Flightmare comes with several desirable features: (i) a large multi-modal sensor suite, including an interface to extract the 3D point-cloud of the scene; (ii) an API for reinforcement learning which can simulate hundreds of quadrotors in parallel; and (iii) an integration with a virtual-reality headset for interaction with the simulated environment.
Flightmare can be used for various applications, including path-planning, reinforcement learning, visual-inertial odometry, deep learning, human-robot interaction, etc.

**[Website](https://uzh-rpg.github.io/flightmare/)** & 
**[Documentation](https://flightmare.readthedocs.io/)** 

[![IMAGE ALT TEXT HERE](./docs/flightmare_main.png)](https://youtu.be/m9Mx1BCNGFU)

## Installation
Installation instructions can be found in our [Wiki](https://github.com/uzh-rpg/flightmare/wiki).

## CUDA development

The rewrite baseline is **v0.1.0**. Use `./scripts/check.ps1 fast` during
development and `./scripts/check.ps1 verify` before checkpoints. See
[local validation and versioning](docs/local-development.md) for test selection,
saved reports, oracle protection, and local release tags. No cloud CI is used.

The new [flightlib-cuda package](flightlib-cuda/README.md) starts with frozen
[v1 tensor contracts and architecture](flightlib-cuda/docs/architecture.md).
It preserves Flightmare dynamics and uses no PhysX backend. The dev check also
compiles and exercises the contracts with both C++ and CUDA.

The separate [Python systems profiler](flightmare-profile/README.md) measures
GPU calls, submission/completion timing, persistent memory, and Nsight traces.
See the [pre-CUDA-4 assessment](flightlib-cuda/docs/systems-profile.md) for the
initial measurements and implications for kernel fusion and launch overhead.
CUDA-4 now provides real fused stepping; see its
[parity and scaling results](flightlib-cuda/docs/cuda4-performance.md), including
the optimized CPU comparison and measured limits.
CUDA-5 adds native environment stepping and device episode resets; its
[systems evidence](flightlib-cuda/docs/cuda5-performance.md) compares task/reset
fusion and explains the optional matched wall-time regression gate.

The dev container uses NVIDIA's CUDA **13.3.1** development image on Ubuntu 24.04,
pinned by digest. It includes CMake, Ninja, GCC, Python 3, NumPy, pytest, and
the native dependencies for the CPU `flightlib` and `flightgym` builds. On
Windows, use Docker Desktop's WSL 2 backend with an NVIDIA driver that supports
CUDA 13.3 (R610 or later).

From the repository root:

```sh
docker compose up -d --build
docker compose exec dev bash docker/check.sh
docker compose exec dev bash
```

The check runs CUDA contract tests, the CPU parity suite, and a Python reset/step
test against the freshly built `flightgym`. The upstream suite and installation
smoke are available through the separate `legacy` and `doctor` profiles. The
local architecture defaults to `120-real` for the RTX 5070 Ti. For a different
GPU, set `FLIGHTMARE_CUDA_ARCHITECTURES` before starting Compose. The check uses
four build jobs by default. Set `CMAKE_BUILD_PARALLEL_LEVEL` in the container
to change that limit.

The checkout is mounted at `/workspace`. The named `/build` volume keeps the
CPU and CUDA build trees and compiler cache between container starts.
`PYTHONPATH` points to the CMake-built extension in `/build/flightlib`.
The CPU oracle uses installed dependencies and `ENABLE_FAST=OFF` to preserve
its NaN-based command validation. `docker compose stop` stops the container
and keeps its cache.

### CPU parity tests

`flightmare-parity-tests` links the frozen C++ simulator. The full dev check
runs it automatically. For a faster CPU-only loop, run the suite directly or
filter to a behavior group with CTest's `-R` option.

```sh
docker compose exec dev bash docker/check-parity.sh
docker compose exec dev bash docker/check-parity.sh -R WorldBoxContract
docker compose exec -T dev bash -c '/build/flightmare-parity-tests/flightmare_oracle_trace > /build/flightmare-oracle.json'
```

The versioned JSON contains deterministic reset inputs, commands, and
per-step C++ outputs. Physics traces include all 25 state fields, simulation
time, and four rotor speeds and thrusts. The scalar environment trace records
observation order, reward, and the terminal flag and reward adjustment. This
trace steps the scalar environment continuously. The vector environment tests
exercise auto-reset separately, checking the returned reset observation and
terminal reward across lanes using the CPU's random resets.

## Updates
 *  17.11.2020 [Spotlight](https://youtu.be/8JyrjPLt8wo) Talk at CoRL 2020 
 *  04.09.2020 Release Flightmare

## Publication

If you use this code in a publication, please cite the following paper **[PDF](http://rpg.ifi.uzh.ch/docs/CoRL20_Yunlong.pdf)**

```
@inproceedings{song2020flightmare,
    title={Flightmare: A Flexible Quadrotor Simulator},
    author={Song, Yunlong and Naji, Selim and Kaufmann, Elia and Loquercio, Antonio and Scaramuzza, Davide},
    booktitle={Conference on Robot Learning},
    year={2020}
}
```

## License
This project is released under the MIT License. Please review the [License file](LICENSE) for more details.
