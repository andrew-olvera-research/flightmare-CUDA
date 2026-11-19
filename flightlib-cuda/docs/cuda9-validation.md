# CUDA-9: resident RL implementation and validation

This checkpoint implements the headless usage/RL layer at 0.1.0, following the
[plan](history/cuda9-usage-plan.md) and [implemented contracts](cuda9-usage-contract.md).
Public README/migration presentation and clean release qualification are deferred
to the next turn. No tag, push or publication is part of this checkpoint.

## Matched RL result

Hardware: Ryzen 7 9800X3D, 8 cores / 16 logical CPUs available without a cgroup
CPU quota; RTX 5070 Ti, driver 610.88; Docker Linux under WSL/WDDM. Python 3.12,
Torch 2.9.1+cu130, CUDA 13.0. TF32 disabled. Original CPU Flightlib is optimized
with the existing strict-FP build, not the deliberately unoptimized oracle.

Both paths execute the same float32 12-128-128 policy, 32-step closed-loop
rollout, GAE and two full-batch PPO epochs. This is actual inference, sampling,
simulation, history collection, gradient computation and Adam updates. Each
timed iteration starts from matched weights, noise, optimizer state and initial
state; repeats are a controlled workload, not a learning-convergence experiment.
An excluded 20-step takeoff puts the batch airborne. Setup, explicit reset,
capture, optimizer initialization, telemetry, logging and save are excluded.
Completed collection, GAE and update time are measured separately and summed.

The broad sweep tests original CPU physics with OpenMP and persistent processes,
1/2/4/8/16 workers, CPU learners with 1/8 Torch threads and GPU learners with
8 host threads. Processes share action/output arrays and send only commands
over pipes; worker affinity prefers physical cores before SMT. All-CPU and
CPU-physics/GPU-learner setups compete for the best tested result. Thread counts
for physics and learner are independent. CPU constructors read the exact frozen
YAML copied to the Linux build volume to avoid Windows mount setup overhead.

Five measured samples follow two warmups. GPU measurements bracket each CPU
sweep; the GPU statistic pools those ten samples. CPU medians determine the
best tested configuration; best collection and best total are selected
independently. Both happen to select the same configurations in these runs.

| Lanes | Broad sweep: GPU collection | Collection speedup | GPU complete iteration | Complete iteration speedup | GPU transitions/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 33 | 5.242 ms | 0.71x | 8.609 ms | 0.94x | 0.123 million |
| 1,024 | 5.453 ms | 4.84x | 8.897 ms | 3.38x | 3.68 million |
| 16,384 | 7.984 ms | 14.55x | 35.021 ms | 4.09x | 14.97 million |

Broad report: `.local-ci/rl-20261117T050438.906645Z/report.json`, 90 CPU cases,
six GPU cases, all numerical gates passed. At 33 lanes the winning CPU setup is
one physics/learner thread. At 1,024 and 16,384 lanes it is respectively 8 and
16 physics threads **with a GPU learner**. These are not sequential-CPU speedups.

After completing lane-zero/host compatibility, a refresh tested 1/8/16 physics
threads and both learner devices/thread settings, including all previous winners:
`.local-ci/rl-20261117T052219.670391Z/report.json`. It passed numerical, trace and
the new symmetric 1.15x before/after timing-stability gates. Its complete-iteration
speedups are **0.94x / 3.41x / 4.34x**; collection speedups **0.71x / 4.90x /
16.17x**. At 16,384 lanes it measures 34.935 ms total, 7.633 ms collection and
15.01 million transitions/s. The refresh is not a second full process sweep.
Normal host/clock variation explains why these are measured ranges, not latency
guarantees; no specific cause is claimed for individual outliers.

The primary kernel result holds: the architecture benefits large parallel
rollouts while small batches remain CPU-favored. At the largest batch, roughly
27 ms is the PPO update; physics-only ratios would substantially overstate
end-to-end RL gains. Further small-batch work should target submission overhead,
and large-batch training work should measure the learner before changing physics.

## Numerical acceptance and the gimbal observation

Existing native/Python core parity tests, frozen oracle and tolerances are
unchanged. The RL benchmark compares every timestep's observation, reward,
done/valid masks, sampled action, value, log probability, advantage and return,
then compares PPO losses. Discrete masks are exact and every sample must be valid.
Normalized error is `abs(a-b)/max(1,abs(b))`: observation/action/value/logp 5e-4,
reward 2e-4, advantages/returns 1e-3, PPO losses 2e-3.

The initial strict 16,384-lane, 32-step run failed at lane 6315, step 30, near
pitch -pi/2. Raw Euler discrepancy reached 0.001312 rad although independently
reconstructed rotation matrices differ by at most 2.364e-6. Replaying the exact
same GPU actions on the CPU still exhibited the coordinate sensitivity. The
failure is retained; no physics equation was changed to hide it.

Following the user's direction to preserve downstream accuracy rather than raw
coordinate identity, the benchmark has a **separate conditioned gate**. It may
accept an observation failure only in Euler columns, only when both
`abs(cos(pitch)) < .01`, with every raw angle difference <= .005 rad and float64
ZYX reconstructed matrix max error <= 5e-5. It cannot waive reward, terminal,
policy or learning-output checks. `--euler-gate strict` retains the original
benchmark gate; the comparator also defaults to strict for callers/tests.

Across the broad run, worst normalized reward error is about 1.384e-5, action
9.761e-7, value 1.325e-4, advantage 1.327e-4 and return 9.925e-6. Maximum PPO
loss error is 3.884e-7. There are no terminal-mask mismatches. The actual failing
Euler fixture is a regression test: strict rejects it, conditioned accepts it,
and deliberate orientation/reward corruptions still fail. This measured envelope
does not imply universal trajectory, trained-policy or sim-to-real equivalence.

A separate reset-heavy strict run,
`.local-ci/rl-20261117T051437.122909Z/report.json`, passed at all three sizes with
every transition terminal/reset (up to 524,288 terminal transitions per sample).
It uses 1/8 thread/process workers with a GPU learner and is a stress qualification,
not the primary best-CPU comparison. Random resets, selected resets, no-autoreset,
non-warp tails and multi-slot grouping have independent acceptance tests.

## Systems evidence

Nsight captures the real warmed collector at 33 and 16,384 lanes. Both the broad
and final refresh traces pass: 65 policy/environment graph launches for T=32,
96 simulation kernel nodes, 32 episode-statistics launches, no host/device copies,
no CUDA allocation/free/host registration, no stream/device synchronization and
one explicit completion-event wait at the measurement boundary. Torch allocator
counters also show zero allocations/frees during measured collection.

There are 322 device-to-device copies: 133,716 bytes at 33 lanes and 66,387,968
bytes at 16,384 lanes, exactly `(126*T+20)*N`, within the explicit history/action
budget. Total observed kernels are 887 and 755 respectively, reflecting policy
GEMM selection as well as simulation. These are not three-kernel whole-RL steps.
Trace scope uses fixed noise and matched explicit resets; resampling and random
reset functionality are exercised separately by training/tests.

At 16,384 lanes: simulation arena 9,112,832 bytes, rollout-owned tensors
62,455,808 bytes, peak Torch allocated memory across the iteration approximately
1.211 GB (excludes the native arena). PPO is not claimed allocation-free. The
two RL kernels compile with zero local/stack storage or spills: GAE 28 registers,
episode statistics 19. Their equations/ownership are separate from frozen state.

The unchanged native Environment paths passed the existing 1.15x wall-time
regression gate for eight eager/graph cases, 1 through 65,536 lanes and 50%
initial reset density: `.local-ci/profile-20261117T051805.400370Z/report.json`,
ratios 0.971–1.074 against CUDA-8's `profile-20261117T034545.362284Z` baseline.

## Tests, installation and provenance

Acceptance was established before implementation (missing package failed first).
Lane-zero/host compatibility tests likewise failed before those APIs were added.
Tests cover real transitions, episode accounting, independent GAE arithmetic,
invalid masks/horizon bootstrap, storage lifetime/stream rejection, configuration
ownership, reseeding, graph invalidation, checkpoint replay, real parameter updates,
CPU thread/process rollouts and corruption controls. Performance timing drift has
a separate rejection test. No new implementation-only getter suite was added.

Local `verify` now builds/installs the RL library, runs its acceptance suite and
trains/saves/reloads/evaluates from a fresh install outside the checkout, alongside
the original core/native/oracle checks. The final verify report path is recorded
in the implementation commit's Validation trailer. A continuous additional run
of 20 updates, 1,024 environments and 64-step horizons completed with valid samples
and finite losses; this is a stability smoke test, not learning convergence.

Durable [machine-readable result summary](cuda9-rl-results.json) retains run
identities, source/binary hashes, comparison configuration, medians and trace
counts. Full logs/SQLite traces remain in the ignored local report directories.
Reports validate their recorded source fingerprints; later documentation does
not change the measured binaries.

Earlier unsuccessful results remain retained: the first installed-package path
failed and was fixed; initial warmup-only cuBLAS allocation was incorrectly
classified as steady-state and the gate was corrected to measured samples;
inherited learner thread settings were found and fixed before the broad sweep.
Strict gimbal failure: `rl-20261117T044424.235273Z`. A later refresh
`rl-20261117T051915.585447Z` passed correctness but had 2.643x GPU total-time drift
at 1,024 lanes; its speedups are excluded even though that older harness wrote
status=passed. This finding motivated the hard bracketing gate. A subsequent
missing CPU CMake target (`rl-20261117T051854.888858Z`) revealed dependence on
previous profiler configuration; the RL harness now explicitly configures its
CPU targets before building, and local verify explicitly builds the shared CPU
adapter used by the RL differential tests. No timing threshold or downstream tolerance was
relaxed in response to these failures.

WSL/WDDM Compute Sanitizer initialization, hardware-counter coverage and multi-GPU
coverage remain unavailable/unqualified, as in earlier layers. Rendering,
sensors and scene collision are outside scope. No old TensorFlow checkpoint,
Stable Baselines subclass or five-return Gymnasium compatibility is claimed.

## Reproduce locally

Run `docker compose exec -T dev bash scripts/setup-cuda-python.sh`, then
`docker compose exec -T dev python3 scripts/local_ci.py verify` to build and test
the exact native/core/RL packages. The setup includes the optional pinned Box
spaces dependency for acceptance tests. After verify, run the broad sweep:

```sh
docker compose exec -T \
  -e PYTHONPATH=/workspace/flightrl-cuda:/workspace/flightmare-profile:/build/flightlib-cuda-python-native/python \
  -e OMP_PROC_BIND=false -e OMP_WAIT_POLICY=PASSIVE -e OPENBLAS_NUM_THREADS=1 \
  dev /build/flightlib-cuda-python/bin/python -m flightmare_profile.rl_benchmark \
  --lanes 33,1024,16384 --workers 1,2,4,8,16 --steps 32 --samples 5 --warmup 2 \
  --epochs 2 --cpu-policy-devices cpu,cuda:0 --cpu-learner-threads 1,8 --trace
```

The final refresh substitutes `--workers 1,8,16 --modes thread`. Reset stress
uses `--spin 0 --euler-gate strict --workers 1,8` with the default GPU learner.
Keep native source/builds stable while profiling; the harness locks shared build
access and rejects source fingerprint changes. Do not run another GPU workload
concurrently with timing. These commands are Linux shell syntax; PowerShell can
put the Docker command on one line.
