# CUDA-7 collision-first geometry qualification and design

**Deferred historical proposal.** This document is retained for the later
scene/fidelity extension and is not the current CUDA-7 scope. The CPU Flightlib
bridge receives Unity collision results; it supplies no equivalent geometry
oracle to reproduce here. See [the current scope](flightlib-scope.md). References
below to CUDA-7/8 describe the superseded proposal, not current issue assignments.

User clarification: current scope is collision queries and state. Rendering and
sensing are separate. Rays are optional line-intersection queries, not the primary
CUDA-7 deliverable. Revise the existing CUDA-7/8 chronology accordingly.

Recommendation: retain the frozen state contracts and qualify a native CUDA BVH
backend against body-overlap workloads before production integration. Start with
cuBQL as the native candidate and Warp as the higher-level alternative. Defer
OptiX/ray specialization unless a future measured workload requires it. This is
a design recommendation, not a measured backend selection; no candidate has
passed this repository's collision fixtures yet.

## Architecture

Separate shared immutable mesh assets from scene templates and moving instances.
Build object-space triangle acceleration structures once per unique asset. For
rigid motion, transform query proxy centers/endpoints into object coordinates;
update instance transforms and world AABBs, not mesh vertices or mesh BVHs.
Rigid-only transforms preserve sphere radii and distances; bake scale at asset
preparation as the frozen contract requires. Optional rays can use the same
object-space representation later.

Identical static scene layouts should share their acceleration structure across
environments, with a mapping from template-local hits to stable instance IDs.
Each query selects its environment explicitly. Coincident coordinates in two
environments must never cause cross-environment collisions. Do not use artificial world
offsets as the sole isolation mechanism. Distinct layouts get distinct templates;
do not assume every environment has the same obstacles.

Keep dynamic instances in a separate structure. Benchmark a bounded per-environment
instance scan first for small populations; larger populations can use a batched
instance BVH/refit. Avoid one host launch per environment. Refit only affected
dynamic bounds; moving a rigid instance does not require a triangle-BVH rebuild.
Topology or capacity changes are explicit setup/recapture boundaries. Refit can
degrade traversal quality after large rearrangements, so benchmark both steady
motion and reset teleports before defining an explicit rebuild policy.

The stepping schedule is preparation -> D -> G -> collision queries -> fused T/R. G snapshots
post-dynamics/pre-reset poses. Query outputs refer to that snapshot, including
terminal lanes; a query of reset poses requires an explicit refreshed G pass.
Do not implement this by cloning the entire physical-state tensor. Query-disabled
stepping retains the current three-kernel path without geometry update overhead.

## Reuse options

| Candidate | Why qualify it | What must be demonstrated locally |
| --- | --- | --- |
| cuBQL | Mostly header-only CUDA/C++; generic BVH builders and traversal templates fit the native runtime | Exact proxy/triangle overlap, bounded traversal storage, explicit streams, allocator behavior, refit/update approach and capture safety |
| Warp | Mesh/BVH queries and refit plus array interoperability and graph facilities reduce implementation work | Proxy/containment semantics, no extra ownership/stream copies, pinned-version API stability and native-package integration costs |

Defer OptiX: its instance acceleration structures and stream-based launch/update
APIs are relevant to a future ray-heavy workload, but do not directly solve the
current confirmed body-overlap problem.

cuBQL supplies traversal infrastructure rather than a complete scene/collision
system. Warp's convenience does not remove the need to adapt query semantics.
OptiX is a query backend, not a physics engine; using it would not require PhysX.
Do not introduce an Omniverse/Isaac Sim application dependency merely for scene
bookkeeping. Prefer a small backend boundary owning setup resources and enqueueing
G/Q against existing device views and the supplied runtime stream.

Primary sources reviewed 2026-11-16:
- [NVIDIA cuBQL](https://github.com/NVIDIA/cuBQL): builders, traversal templates and stream parameter; pin an audited revision before adoption.
- [Warp runtime](https://nvidia.github.io/warp/v1.19/user_guide/runtime.html) and [interoperability](https://nvidia.github.io/warp/v1.19/user_guide/interoperability.html): mesh/BVH facilities, arrays and graph model.
- [OptiX acceleration structures](https://raytracing-docs.nvidia.com/optix9/api/group__optix__host__api__acceleration__structures.html) and [launch API](https://raytracing-docs.nvidia.com/optix9/api/group__optix__host__api__launches.html): build/update and explicit CUDA streams. API availability does not establish our capture or performance guarantees.

## Qualification before commitment

First freeze supported body proxies, obstacle representations and overlap semantics.
Decide whether triangle obstacles represent surfaces or closed solids: sphere-to-
triangle distance detects surface intersection but misses a body wholly contained
inside a solid. Solid containment needs a separate supported test; do not label
surface-only queries as general solid overlap. An initial sphere/compound-sphere
body with analytic box/sphere obstacles is a useful exact-proxy baseline; mesh
obstacles should be added with explicit containment semantics. Conservative body
proxies may report contacts earlier than the rendered drone, by design.

Also define geometry handling of invalid/nonunit physical quaternions. Geometry
must never silently modify legacy physical state to repair a rigid transform.
Use explicit geometry validity and a documented derived-transform policy.
Then run an independent brute-force CPU overlap reference against candidate adapters:
separation, touching, penetration, rotations, exclusions, inactive/invalid IDs,
coincident environments, dense clusters, overflow and reset teleports. Confirm
overlap after broad-phase AABB selection; an AABB intersection alone is not a
body collision. No contact response or new task termination is implied.

If the optional ray interface is implemented later, cover double-sided triangles, winding normals, t=0 and t=max,
parallel/grazing rays, equal-distance hits, transformed instances, exclusions,
inactive inputs, invalid IDs and coincident environments. Closest hit means the
contract's distance/instance/primitive ordering, not whichever traversal visits
first. A stock closest-hit API without documented tie semantics is insufficient.
Do not fix this by silently changing frozen tolerances or hit ordering.

Then measure one static obstacle asset and many rigid body instances, followed by
representative scene layouts. Sweep E, bodies and obstacles per environment,
triangle count, moving fraction, separation/contact density and candidate capacity.
Include intermediate
shapes: CUDA-6 showed that backend algorithms can change between the extremes.

Record cold asset preparation/build separately from warmed transform, update,
query, and combined step wall time. Break memory into shared asset/BVH bytes,
instance/template metadata, query input/output, and persistent scratch. Desired
scaling is shared asset bytes plus instance/query/scratch capacity, never mesh
vertices multiplied by environment count. Account for library-internal allocation
and memory operations independently of our runtime counters.

Require zero hot-loop host state readbacks, stable addresses, bounded scratch,
explicit errors, graph/native agreement and nondefault-stream lifetime tests.
Benchmark refit and reset bursts. A documented device launch path can be accepted
if graph capture is unsupported, but its full wall cost must be visible; never
fall back to CPU queries. Choose a backend using total stepping cost and memory,
not standalone rays/second or SDK marketing figures.

## Collision boundary and next implementation turns

Revise CUDA-7 to own shared scene/instance state, collision-proxy semantics and a
small validated overlap vertical slice. CUDA-8 then scales candidate generation,
narrow phase, overflow handling and integration. Pull the semantics gate forward
from CUDA-8 so the chosen backend is evaluated on the actual need. Sphere or
capsule proxies are candidates, not an implicit definition of drone geometry.
Ray hits cannot stand in for confirmed body overlap. Keep optional ray support
separately tracked; do not spend this sprint optimizing sensing or rendering.

Reuse the shared static acceleration structures for obstacle candidates and
environment-local dynamic structures for drone pairs where beneficial. Keep
canonical pair IDs, owner filtering, bounded K, device count/overflow and invalid
result propagation. Avoid a global all-drone pair matrix. No contact impulses,
new task termination or swept-collision claims are part of this work.

Implement in three reviewable turns within CUDA-7: (1) collision semantics and
backend/overlap microbenchmark qualification; (2) shared assets and moving G with
capacity accounting; (3) a minimal overlap path, pre-reset ordering fixtures and
the collision-inclusive profiler/graph path. Do not postpone ordering integration to
CUDA-9. Keep the native dynamics implementation intact unless combined profiling
provides evidence for changing it.
