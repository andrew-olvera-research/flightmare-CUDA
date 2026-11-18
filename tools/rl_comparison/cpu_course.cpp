#include "flightlib/envs/quadrotor_env/quadrotor_env.hpp"
namespace flightlib_cuda::v1::detail {
inline double rsqrt(double x) { return 1.0 / std::sqrt(x); }
}  // namespace flightlib_cuda::v1::detail
#include "swept_collision.cuh"
#include <algorithm>
#include <omp.h>

namespace {
using namespace flightlib_cuda::v1;
namespace query = flightlib_cuda::v1::detail;

struct Probe {
  query::StaticScene scene;
  float origin[3], last[3];
  double clearance;
  bool hit;
  Mask valid;
  void begin(const float* p) {
    std::copy_n(p, 3, origin);
    valid = 1;
    auto initial = query::startClearance(scene, p, valid);
    hit = initial.hit;
    clearance = initial.travel;
  }
  void arc(const float* p, const float* next, const float* a, float dt) {
    std::copy_n(next, 3, last);
    if (hit || !valid) return;
    query::Arc path[3];
    for (int k = 0; k < 3; ++k)
      path[k] = {p[k], next[k], .5 * double(dt) * dt * a[k]};
    if (!query::insideClearance(path, origin, clearance))
      hit = query::sweptOverlap(scene, path, valid);
  }
  void boundary(const float* p) {
    if (!std::equal(last, last + 3, p)) {
      const float a[3]{};
      arc(p, p, a, 0);
    }
  }
};
thread_local Probe* probe = nullptr;
thread_local const flightlib::QuadState* reset_state = nullptr;
thread_local float* pose_out = nullptr;
}  // namespace
#include "course_instrumented.inc"

namespace {
struct Batch {
  std::vector<std::unique_ptr<flightlib::QuadrotorEnv>> lanes;
  std::vector<flightlib::QuadState> starts;
  std::vector<float> nodes;
  float options[2] = {.125f, 1.f};
  int workers;
};
}  // namespace

extern "C" void* course_create(int n, int workers, const char* config,
                               const float* nodes, int rows) {
  try {
    auto batch = std::make_unique<Batch>();
    batch->workers = workers;
    batch->nodes.assign(nodes, nodes + rows * 10);
    for (int i = 0; i < n; ++i) {
      auto env = std::make_unique<flightlib::QuadrotorEnv>(config);
      flightlib::QuadState state;
      state.setZero();
      state.p << -.9f + (i % 5 - 2) * .1f, -4.27f - (i % 3) * .2f, 3.48f;
      state.x[3] = state.x[6] = std::sqrt(.5f);
      batch->starts.push_back(state);
      if (i == 0) {
        state.p << -.1f, -1.60f, 3.48f;
        state.v[1] = 70;
      }
      flightlib::Vector<12> obs;
      reset_state = &state;
      env->reset(obs, false);
      reset_state = nullptr;
      batch->lanes.push_back(std::move(env));
    }
    return batch.release();
  } catch (...) {
    reset_state = nullptr;
    return nullptr;
  }
}

extern "C" void course_destroy(Batch* batch) { delete batch; }

extern "C" int course_step(Batch* b, const float* actions, bool queries,
                           bool terminate, float* obs, float* poses,
                           float* rewards, unsigned char* done,
                           unsigned char* hits) {
  int failure = 0;
#pragma omp parallel for schedule(static) num_threads(b->workers) \
  reduction(| : failure)
  for (size_t i = 0; i < b->lanes.size(); ++i) {
    Probe query_state{};
    query_state.scene = {b->nodes.data(),
                         static_cast<Index>(b->nodes.size() / 10), b->options};
    probe = queries ? &query_state : nullptr;
    pose_out = poses ? poses + 7 * i : nullptr;
    auto& env = *b->lanes[i];
    Eigen::Map<flightlib::Vector<12>> observation(obs + 12 * i);
    flightlib::Vector<4> action =
      Eigen::Map<const flightlib::Vector<4>>(actions + 4 * i);
    rewards[i] = env.step(action, observation);
    float extra = 0;
    hits[i] = queries && query_state.hit;
    done[i] = env.isTerminalState(extra) || (terminate && hits[i]);
    probe = nullptr;
    pose_out = nullptr;
    if (done[i]) {
      rewards[i] -= .02f;
      reset_state = &b->starts[i];
      env.reset(observation, false);
      reset_state = nullptr;
    }
    failure |= (queries && !query_state.valid) || !observation.allFinite() ||
               !std::isfinite(rewards[i]);
  }
  return failure;
}
