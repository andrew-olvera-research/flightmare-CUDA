#include <math_constants.h>

#include <algorithm>
#include <cub/block/block_radix_sort.cuh>

#include "swept_collision.cuh"

namespace flightlib_cuda::v1::detail {
namespace {
constexpr Index threads = 128;
constexpr Index overflowBit = 1, invalidMotionBit = 2;

// Sort independent runs within each world. All temporary sort storage is shared
// memory. The last run can be partial and inactive owners sort after live ones.
template<int Items>
__global__ void sortRuns(RuntimeViews v, Index runs) {
  using Sort = cub::BlockRadixSort<float, threads, Items, Index>;
  __shared__ typename Sort::TempStorage temp;
  Index env = blockIdx.x / runs, run = blockIdx.x % runs,
        slots = v.topology.batch.slots_per_environment;
  Offset first = (Offset{run} * threads + threadIdx.x) * Items;
  float key[Items];
  Index owner[Items];
  for (int j = 0; j < Items; ++j) {
    key[j] = CUDART_INF_F;
    owner[j] = kInvalidIndex;
    if (first + j < slots) {
      Index lane = env * slots + first + j;
      if (v.topology.active(lane)) {
        if (v.diagnostics.status(lane) == LaneStatus::Ok) {
          key[j] = v.swarm.bounds(0, lane);
          owner[j] = lane;
        } else
          atomicOr(&v.swarm.issue(env), invalidMotionBit);
      }
    }
  }
  Sort(temp).Sort(key, owner);
  for (int j = 0; j < Items && first + j < slots; ++j)
    v.swarm.order(env * slots + first + j) = owner[j];
}

// Saturate the stored count at K, including in dense worlds. The next
// candidate makes that world's entire collision result unusable for training.
__device__ Index reserve(CandidatePairs pairs, Index env, Index capacity) {
  Index* count = &pairs.count(env);
  Index seen = atomicAdd(count, 0u);
  while (seen < capacity) {
    Index previous = atomicCAS(count, seen, seen + 1);
    if (previous == seen) return seen;
    seen = previous;
  }
  return capacity;
}

__global__ void candidates(RuntimeViews v, Index runSize) {
  Offset rank = Offset{blockIdx.x} * blockDim.x + threadIdx.x;
  if (rank >= v.topology.batch.lanes) return;
  Index a = v.swarm.order(rank);
  if (a == kInvalidIndex) return;
  Index slots = v.topology.batch.slots_per_environment, env = rank / slots;
  Offset base = Offset{env} * slots, end = base + slots;
  Index capacity = v.candidates.instance_pair.shape[1];
  // Only visit later ranks and later runs: one unordered pair per unique
  // owner. Within each run X minima are monotone.
  for (Offset run = base + ((rank - base) / runSize) * runSize; run < end;
       run += runSize) {
    Offset first = rank + 1 > run ? rank + 1 : run,
           last = run + runSize < end ? run + runSize : end;
    for (Offset j = first; j < last; ++j) {
      Index b = v.swarm.order(j);
      if (b == kInvalidIndex || v.swarm.bounds(0, b) > v.swarm.bounds(1, a))
        break;
      if (v.swarm.bounds(1, b) < v.swarm.bounds(0, a)) continue;
      if (v.swarm.bounds(2, b) > v.swarm.bounds(3, a) ||
          v.swarm.bounds(3, b) < v.swarm.bounds(2, a) ||
          v.swarm.bounds(4, b) > v.swarm.bounds(5, a) ||
          v.swarm.bounds(5, b) < v.swarm.bounds(4, a))
        continue;
      Index index = reserve(v.candidates, env, capacity);
      if (index == capacity) {
        atomicOr(&v.swarm.issue(env), overflowBit);
        return;
      }
      v.candidates.instance_pair(env, index, 0) = min(a, b);
      v.candidates.instance_pair(env, index, 1) = max(a, b);
    }
  }
}

__device__ bool movingContact(RuntimeViews v, Index a, Index b, Mask& valid) {
  double radius = 2. * v.swarm.radius;
  double committed2 = 0;
  for (int c = 0; c < 3; ++c) {
    double d = double(v.state.x(c, a)) - v.state.x(c, b);
    committed2 = fma(d, d, committed2);
  }
  if (committed2 <= radius * radius) return true;
  const float point[3] = {};
  auto path = v.swarm.trajectory;
  for (Offset s = 0; s < v.schedule.substep_dt.size(); ++s) {
    Arc relative[3];
    double dt = v.schedule.substep_dt(s), scale = radius;
    for (int c = 0; c < 3; ++c) {
      auto& p = relative[c];
      p.start = double(path(2 * s, c, a)) - path(2 * s, c, b);
      p.end = double(path(2 * s + 2, c, a)) - path(2 * s + 2, c, b);
      p.bend =
        .5 * dt * dt * (double(path(2 * s + 1, c, a)) - path(2 * s + 1, c, b));
      scale = fmax(scale, fmax(fabs(p.start), fmax(fabs(p.end), fabs(p.bend))));
    }
    double allowance = 64 * DBL_EPSILON * scale;
    if (allowance > fmax(1e-9, 1e-7 * radius)) {
      valid = 0;
      return false;
    }
    double threshold = (radius + allowance) * (radius + allowance);
    if (distance2(relative, point, 0) <= threshold ||
        intervalContact<true>(relative, point, 0, 1, threshold))
      return true;
  }
  // A zero-duration world-bound correction also has an occupied source.
  if (!v.schedule.substep_dt.size()) {
    double start2 = 0;
    for (int c = 0; c < 3; ++c) {
      double d = double(path(0, c, a)) - path(0, c, b);
      start2 = fma(d, d, start2);
    }
    return start2 <= radius * radius;
  }
  return false;
}

__global__ void contacts(RuntimeViews v, Index tiles) {
  Index env = blockIdx.x / tiles;
  Offset slot = Offset{blockIdx.x % tiles} * blockDim.x + threadIdx.x;
  for (; slot < v.candidates.count(env); slot += Offset{tiles} * blockDim.x) {
    if (atomicAdd(&v.swarm.issue(env), 0u)) return;
    Index a = v.candidates.instance_pair(env, slot, 0),
          b = v.candidates.instance_pair(env, slot, 1);
    Mask valid = 1;
    if (movingContact(v, a, b, valid)) {
      atomicOr(&v.swarm.hit(a), 1u);
      atomicOr(&v.swarm.hit(b), 1u);
    }
    if (!valid) atomicOr(&v.swarm.issue(env), invalidMotionBit);
  }
}
}  // namespace

void launchMovingPairs(const RuntimeViews& v, cudaStream_t stream) {
  auto batch = v.topology.batch;
  Index items = batch.slots_per_environment <= threads ? 1 : 8,
        runSize = threads * items;
  Index runs = (Offset{batch.slots_per_environment} + runSize - 1) / runSize;
  if (items == 1)
    sortRuns<1><<<batch.environments * runs, threads, 0, stream>>>(v, runs);
  else
    sortRuns<8><<<batch.environments * runs, threads, 0, stream>>>(v, runs);
  candidates<<<(Offset{batch.lanes} + 127) / 128, 128, 0, stream>>>(v, runSize);
  Index tiles =
    std::min<Offset>((Offset{batch.slots_per_environment} + 127) / 128,
                     (v.candidates.instance_pair.shape[1] + 127) / 128);
  contacts<<<batch.environments * tiles, 128, 0, stream>>>(v, tiles);
}
}  // namespace flightlib_cuda::v1::detail
