# Swept contacts between agents

Enable `agent_collisions=True` on `Environment` or `TensorEnv` to give every
active slot a spherical body of radius `collision_radius` metres. Each group
of slots forms one world. The option works independently or alongside a static
course. Existing collectors default to independent agents.

```python
import torch
from flightlib_cuda import Environment

env = Environment(1, 1000, scene="split_s", agent_collisions=True,
                  collision_radius=.125, candidate_capacity=8000,
                  terminate_on_collision=True)
i = torch.arange(1000, device=env.device)
env.reset_state[0] = i % 10 - 5
env.reset_state[1] = i // 10 % 10 - 5
env.reset_state[2] = 2 + .4 * (i // 100)
env.reset()
env.capture(random=False)
out = env.step(graph=True, random=False)
peer_contact = env.agent_collision_hit
any_contact = env.collision_hit
valid_for_training = out["valid"]
```

An agent-to-agent hit marks both owners on the same transition. The optional
termination setting applies to peer and course hits. The transition observation
is the end-of-step pose. Autoreset preserves the flags until the next step.
Explicit reset clears the selected lanes' flags. Candidate tensors always
describe the most recent query pass. All outputs are borrowed CUDA views.
`TensorEnv` exposes them through `env.engine`.

## Candidate capacity and motion

`candidate_pairs` has shape [worlds, capacity, 2]. For world `e`, only entries
before `candidate_count[e]` are populated. Each pair contains two ascending
flattened lane IDs from that world. Pair order within the list is unspecified.
Candidates indicate overlapping swept bounding boxes. Use `agent_collision_hit`
for confirmed contact and check `collision_valid` before using collision flags.

The default capacity per world is the smaller of eight times the slot count and
the number of possible distinct pairs, with a minimum allocation of one. On
overflow the stored count saturates at capacity, `candidate_overflow[e]` becomes
one, and every transition in that world is invalid for training. Other worlds
advance independently. Active lanes with failed dynamics or numerically
unresolved moving-pair queries also invalidate their world. Inactive slots
never generate candidates. Provision capacity for the training distribution,
including resets. Starting every agent at the same location generates a dense
world immediately. A subsequent complete step recomputes validity and overflow.

The dynamics kernel records actual RK positions and accelerations for each
substep and builds a conservative swept AABB. CUB sorts owners in X-minimum
order in runs of up to 1,024. Within a world, the broad phase visits each
later run, pruning by X and testing the other axes before reserving a bounded
candidate slot. Dense or poorly separated bounds can require quadratic work.
There is no dense pair matrix. A separate kernel tests the relative quadratic
trajectory at matching times. It minimizes squared sphere separation over the
substep, including both ends and interior minima. Tangency counts. The two
committed positions after world-bound correction are also tested. Teleports
and episode resets do not introduce swept segments.

Moving agents share one step clock. A partial reset changes its selected
starting poses. Legacy first-lane-only stepping cannot advance this interacting
world and is rejected before submission. The model provides event detection.
Velocities and positions continue to follow the dynamics until task handling.

## Qualification

The tests cover pass-throughs, clear paths crossing at different times, a
curved interior graze, tangency and a one-float-step miss. They compare random
relative curves with an independent CPU polynomial solver and two 1,031-agent
worlds with exhaustive CPU pair checks. They also cover mixed static and moving
contacts, inactive slots, environment isolation, exact capacity, overflow,
invalid commands, partial resets, world-bound jumps and recovery in graph mode.
The installed RL check exercises a gate strike and a pair collision in the same
100-agent rollout, followed by GAE and a PPO update.

| Agents in one world | Disabled (µs) | Static (µs) | Peers (µs) | Static + peers (µs) | Candidates / capacity | Native arena, disabled → peers (bytes) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 16.88 | 55.48 | 34.50 | 73.25 | 25 / 800 | 63,588 → 84,740 |
| 1,000 | 17.93 | 64.12 | 140.36 | 189.04 | 250 / 8,000 | 558,056 → 762,372 |

These RTX 5070 Ti medians cover completed graph steps at 10 ms simulated time,
nine samples of 50 steps, identical moving formations and query-only contacts.
Arena figures exclude PyTorch bindings and the shared 2,680-byte Split-S BVH.
A 20-step Nsight trace of the combined 1,000-agent case recorded six kernels
per graph replay and zero allocation calls or memory copies during stepping.
Candidate generation took about 60% of kernel time in this formation.

```sh
docker compose exec -T dev python3 scripts/local_ci.py verify
docker compose exec -T -e PYTHONPATH=/build/flightlib-cuda-python-native/python dev /build/flightlib-cuda-python/bin/python tools/course_assets/bench_swarm.py
docker compose exec -T -e PYTHONPATH=/workspace/flightrl-cuda:/build/flightlib-cuda-python-native/python dev /build/flightlib-cuda-python/bin/python -m flightrl_cuda.course_rollout --agent-collisions
```
