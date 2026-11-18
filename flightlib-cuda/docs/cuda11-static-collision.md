# Static course collisions

The installed Python package includes `room` and `split_s`. A course can serve
every agent and environment in a batch. At the end of each control step, CUDA
tests a sphere centered at the vehicle position against the static course.
Custom collectors can use the two resident masks directly.

```python
from flightlib_cuda import Environment

env = Environment(1, 1000, scene="split_s", collision_radius=0.125,
                  terminate_on_collision=True)
env.reset_state[2].fill_(3.6)
env.reset()
env.capture(random=False)
transition = env.step(graph=True, random=False)
hits = env.collision_hit.view(1, 1000)
valid = env.collision_valid.view(1, 1000)
```

`flightrl_cuda.TensorEnv` accepts the same scene options. Its masks are on
`env.engine`. The default radius is 0.15 m and termination is opt-in. On a
terminal step, the flags and transition observation describe the pre-reset pose.
The returned policy observation describes the reset pose. Inactive or failed
dynamics lanes have zero hit and validity. Explicit reset clears both flags for
selected lanes. As with other borrowed outputs, copy into rollout storage to
retain a history and use the environment's stream ordering for other consumers.

## Course assets and import

The [asset directory](../python/flightlib_cuda/assets/) contains the portable
course files and their matching visual USD stages. Coordinates are in metres,
right-handed and Z-up. Box dimensions are full extents. Each instance places a
named prototype at a translation and yaw. The room has a 32 × 24 × 8 m interior
and six solid slabs. Split-S adds seven four-piece gate frames. Gate local X is
normal to the opening. The opening is 1.5 m square, the frame is 0.1 m wide and
the depth is 0.12 m.

The seven positions and quaternion orientations are from RPG's published track.
The room and frame dimensions are project fixtures. The upstream scale is a
multiplier for its original visual asset. For comparisons requiring the precise
original collider or visual appearance, qualify that asset separately. The
pinned YAML and its MIT license ship alongside the fixtures. [Upstream Split-S
track](https://github.com/uzh-rpg/agile_flight/blob/03dc1179c011f414aba1264bb6db92679be66b4f/envsim/parameters/tracks/SplitS.yaml)

- **CUDA** accepts either built-in name or a path to a `.flightcourse` file.
  An asset contains axis-aligned boxes in prototype coordinates. Its instances
  can be translated and rotated about Z.
- **Isaac Sim** can open `room.usda` or `split_s.usda`. Visible cubes carry
  USD Physics colliders and gate placements reference one prototype. The offline
  importer also reads composed box colliders, including instance proxies, and
  converts stage units. Its supported authoring subset is static Z-up stages
  with upright unfiltered boxes. Meshes, rigid bodies, animated and tilted
  colliders need a different representation. Render-only meshes can coexist
  with separately authored box proxies.
- **Unity** uses the two [Editor scripts](../../tools/course_assets/Unity/Editor/).
  Copy them to `Assets/Editor/` and the two `.flightcourse` files to
  `Assets/Courses/`. Drag an imported course into a scene. Every piece has a
  visible cube and BoxCollider. The importer maps world XYZ to Unity
  (−Y, Z, X). The batch check counts rendered and collision shapes and probes
  the floor, room interior, gate opening and post.

```sh
# Optional authoring environment with PyYAML, usd-core and pytest installed
python tools/course_assets/build_courses.py
python tools/course_assets/usd_course.py export flightlib-cuda/python/flightlib_cuda/assets/split_s.flightcourse split_s.usda
python tools/course_assets/usd_course.py import split_s.usda custom.flightcourse
python -m pytest -q tools/course_assets/test_usd_course.py
Unity -batchmode -nographics -quit -projectPath PROJECT -executeMethod FlightmareCourseValidation.ValidateFixtures
```

## Runtime and validation

At construction the loader expands the instances once and builds a balanced
bounding volume hierarchy. Leaves store exact oriented boxes. Internal nodes
store conservative axis-aligned bounds and the index following their subtree.
A missed node skips that subtree. Traversal needs no per-agent stack. The
Split-S table is 67 nodes × 40 bytes = **2,680 bytes on device**, shared by
live environments on the same GPU with identical course bytes. Radius and
termination settings are per environment object. The flags reuse the runtime's
two existing bytes per lane. Environment groups within one batch use one
layout. Separate objects may select different layouts.

The task kernel tests squared sphere-to-box distance at the integrated,
world-constrained position before calculating termination. Tangency counts as
a hit. Sampling happens once per control step. A fast vehicle can pass completely
through a post between samples. The tests exercise this case explicitly.
Contact response, swept queries and vehicle-to-vehicle collisions are outside
this implementation. The stock height terminal still applies independently.

The device tests compare graph and ordinary execution to an exhaustive NumPy
oracle on 1,000 positions, including rotated frames, openings, wall and floor
contacts, plus invalid lanes, resets and terminal observation ordering. The
installed RL example collects 800 transitions and runs GAE and one PPO update.
It exercises the training pipeline with the stock reward. Course completion and
racing policy quality require a separate task and evaluation.

```sh
docker compose exec -T dev python3 scripts/local_ci.py verify
docker compose exec -T -e PYTHONPATH=/workspace/flightrl-cuda:/build/flightlib-cuda-python-native/python dev /build/flightlib-cuda-python/bin/python -m flightrl_cuda.course_rollout
docker compose exec -T -e PYTHONPATH=/build/flightlib-cuda-python-native/python dev /build/flightlib-cuda-python/bin/python tools/course_assets/bench_collision.py --steps 50 --repeats 9
```

On the RTX 5070 Ti, the graph-step benchmark gave the following medians in
microseconds. The paired runs use the same initial states near gate one, 10 ms
control steps and query-only collisions. Setup, policy inference and rendering
are excluded. Spatially dispersed agents can traverse different branches.

| Agents in one environment | Headless | With Split-S |
| ---: | ---: | ---: |
| 1 | 11.75 | 17.72 |
| 1,000 | 18.38 | 22.35 |
| 65,536 | 65.36 | 75.82 |

A 20-step Nsight Systems capture with 1,000 agents showed 20 action preparation,
20 dynamics and 20 fused task kernels, 20 graph launches and zero recorded
memory copies or allocation calls in the measured interval.
