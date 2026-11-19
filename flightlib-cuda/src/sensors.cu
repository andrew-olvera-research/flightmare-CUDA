#include <curand_kernel.h>

#include "dynamics_launch.hpp"

namespace flightlib_cuda::v1::detail {
namespace {
struct Quaternion {
  float w, x, y, z;
};
__device__ Quaternion product(Quaternion a, Quaternion b) {
  return {a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
          a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
          a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
          a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w};
}
__device__ Quaternion inverse(Quaternion q) { return {q.w, -q.x, -q.y, -q.z}; }
__device__ float3 rotate(Quaternion q, float3 v) {
  Quaternion r = product(product(q, {0, v.x, v.y, v.z}), inverse(q));
  return make_float3(r.x, r.y, r.z);
}

// A separate counter stream for cumulative readout error. Reusing the
// preceding endpoint gives adjacent differences their negative covariance.
__device__ float readout(std::uint64_t key, std::uint64_t seq) {
  auto x = key + seq * 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return float((x ^ (x >> 31)) >> 40) * (1.f / 16777216.f);
}
template<int Channels>
__device__ bool measure(SensorBuffers b, const float* x, float3 gravity,
                        Index lane, std::uint64_t episode, std::uint64_t seq,
                        float* out) {
  const auto& cfg = b.settings;
  curandStatePhilox4_32_10_t rng;
  if (cfg.stochastic) {
    auto key = cfg.seed ^ (episode * 0x9e3779b97f4a7c15ULL) ^
               (Channels == 6 ? 0x6a09e667ULL : Channels == 1 ? 0x3c6ef372ULL : 0xbb67ae85ULL);
    curand_init(key, lane, seq * 64, &rng);
  }
  Quaternion body{x[3], x[4], x[5], x[6]},
    mount{cfg.mount[0], cfg.mount[1], cfg.mount[2], cfg.mount[3]};
  double norm = sqrt(double(body.w) * body.w + double(body.x) * body.x +
                     double(body.y) * body.y + double(body.z) * body.z);
  if (!(norm > 0) || !isfinite(norm)) return false;
  body = {float(body.w / norm), float(body.x / norm), float(body.y / norm),
          float(body.z / norm)};
  Quaternion sensor = product(body, mount);
  float3 omega = make_float3(x[10], x[11], x[12]);
  if constexpr (Channels == 6) {
    auto w = rotate(inverse(mount), omega);
    auto f = rotate(
      inverse(sensor),
      make_float3(x[13] - gravity.x, x[14] - gravity.y, x[15] - gravity.z));
    out[0] = w.x;
    out[1] = w.y;
    out[2] = w.z;
    out[3] = f.x;
    out[4] = f.y;
    out[5] = f.z;
    float root_dt = sqrtf(float(cfg.period_ns * 1e-9));
    for (int i = 0; i < 6; ++i) {
      if (cfg.walk[i])
        b.drift(i, lane) += cfg.walk[i] * root_dt * curand_normal(&rng);
      out[i] += cfg.bias[i] + b.drift(i, lane);
      if (cfg.noise[i]) out[i] += cfg.noise[i] / root_dt * curand_normal(&rng);
      if (cfg.increment_width[i]) {
        auto key = cfg.seed ^ (episode * 0x94d049bb133111ebULL) ^
                   (lane * 0xbf58476d1ce4e5b9ULL) ^
                   (std::uint64_t(i + 1) * 0xd2b74407b1ce6e93ULL);
        out[i] += float(cfg.increment_width[i] *
                        (readout(key, seq) - readout(key, seq - 1)) /
                        (cfg.period_ns * 1e-9));
      }
      if (cfg.limit[i] > 0)
        out[i] = fminf(cfg.limit[i], fmaxf(-cfg.limit[i], out[i]));
    }
  } else if constexpr (Channels == 1) {
    auto offset = rotate(body, make_float3(cfg.offset[0], cfg.offset[1], cfg.offset[2]));
    double altitude = double(cfg.origin_altitude_m) + x[2] + offset.z;
    if (altitude > 11000.) return false; // Tropospheric lapse-rate model.
    out[0] = cfg.sea_level_pa * pow(1. - altitude / 44330.77, 5.25588) + cfg.bias[0];
    if (cfg.noise[0]) out[0] += cfg.noise[0] * curand_normal(&rng);
    if (!(out[0] > 0)) return false;
  } else {
    auto r = make_float3(cfg.offset[0], cfg.offset[1], cfg.offset[2]);
    auto p = rotate(body, r);
    auto v = rotate(body, make_float3(omega.y * r.z - omega.z * r.y,
                                      omega.z * r.x - omega.x * r.z,
                                      omega.x * r.y - omega.y * r.x));
    out[0] = x[0] + p.x;
    out[1] = x[1] + p.y;
    out[2] = x[2] + p.z;
    out[7] = x[7] + v.x;
    out[8] = x[8] + v.y;
    out[9] = x[9] + v.z;
    float angle[3] = {};
    for (int i = 0; i < 3; ++i) {
      if (cfg.noise[i]) out[i] += cfg.noise[i] * curand_normal(&rng);
      if (cfg.noise[3 + i])
        out[7 + i] += cfg.noise[3 + i] * curand_normal(&rng);
      if (cfg.attitude_std[i])
        angle[i] = cfg.attitude_std[i] * curand_normal(&rng);
    }
    float a = norm3df(angle[0], angle[1], angle[2]);
    if (a > 0) {
      float k = sinf(.5f * a) / a;
      sensor = product(
        sensor, {cosf(.5f * a), k * angle[0], k * angle[1], k * angle[2]});
    }
    out[3] = sensor.w;
    out[4] = sensor.x;
    out[5] = sensor.y;
    out[6] = sensor.z;
  }
  for (int i = 0; i < Channels; ++i)
    if (!isfinite(out[i])) return false;
  return cfg.dropout == 0 || curand_uniform(&rng) > cfg.dropout;
}

template<int Channels>
__global__ void sensorStep(RuntimeViews v, SensorBuffers b, Index lanes) {
  Index n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= lanes) return;
  constexpr int channels = Channels;
  const auto& cfg = b.settings;
  auto begin = b.state(0, n), end = begin + v.sensor_step_ns,
       episode = v.episodes.episode(n);
  bool active = v.topology.active(n);
  if (b.state(1, n) != episode) {
    b.state(1, n) = episode;
    b.state(2, n) = begin;
    b.state(3, n) = b.state(4, n) = 0;
    if constexpr (Channels == 6)
      for (int c = 0; c < 6; ++c) b.drift(c, n) = 0;
  }
  b.state(0, n) = end;
  auto epoch = b.state(2, n), produced = b.state(3, n),
       consumed = b.state(4, n);
  if (!active) {
    // Preserve sequence uniqueness when the same episode is later reactivated.
    if (end >= epoch + cfg.phase_ns)
      produced = 1 + (end - epoch - cfg.phase_ns) / cfg.period_ns;
    consumed = produced;
  }
  auto deadline = epoch + cfg.phase_ns + produced * cfg.period_ns;
  double elapsed = 0.;
  auto steps = v.schedule.substep_dt.size();
  bool good = v.diagnostics.status(n) == LaneStatus::Ok;
  for (Offset s = 0; active && deadline <= end && s < steps; ++s) {
    elapsed += v.schedule.substep_dt(s);
    auto time = s + 1 == steps
                  ? end
                  : begin + static_cast<std::uint64_t>(llround(elapsed * 1e9));
    if (time > end) time = end;
    if (deadline > time) continue;
    float x[16] = {};
    if (good)
      for (int c = 0; c < 16; ++c) x[c] = v.sensor_truth.samples(s, c, n);
    Index model = v.topology.model_id(n);
    float3 gravity = make_float3(0, 0, 0);
    if constexpr (Channels == 6)
      if (good)
        gravity =
          make_float3(v.models.gravity(0, model), v.models.gravity(1, model),
                      v.models.gravity(2, model));
    do {
      auto index = produced % b.queue.shape[0];
      float value[channels] = {};
      bool valid = good && !(time == end && v.sensor_truth.corrected(n)) &&
                   measure<Channels>(b, x, gravity, n, episode, produced + 1, value);
      b.acquired(index, n) = time;
      b.queue_valid(index, n) = valid;
      for (int c = 0; c < channels; ++c)
        b.queue(index, n, c) = valid ? value[c] : 0;
      ++produced;
      deadline += cfg.period_ns;
    } while (deadline <= time);
  }
  for (Offset slot = 0; slot < b.values.shape[0]; ++slot) {
    auto index = consumed % b.queue.shape[0];
    bool due =
      consumed < produced && b.acquired(index, n) + cfg.latency_ns <= end;
    auto acquired = due ? b.acquired(index, n) : 0;
    b.metadata(0, slot, n) = due ? consumed + 1 : 0;
    b.metadata(1, slot, n) = due ? episode : 0;
    b.metadata(2, slot, n) =
      due ? epoch + cfg.phase_ns + consumed * cfg.period_ns : 0;
    b.metadata(3, slot, n) = acquired;
    b.metadata(4, slot, n) = due ? acquired + cfg.latency_ns : 0;
    b.metadata(5, slot, n) = due ? end : 0;
    b.valid(slot, n) = due && b.queue_valid(index, n);
    for (int c = 0; c < channels; ++c)
      b.values(slot, n, c) = due ? b.queue(index, n, c) : 0;
    if (due) ++consumed;
  }
  b.state(3, n) = produced;
  b.state(4, n) = consumed;
}
}  // namespace
void launchSensors(const RuntimeViews& v, cudaStream_t stream, bool first) {
  Index lanes = first ? 1 : v.topology.batch.lanes;
  auto blocks = (Offset{lanes} + 127) / 128;
  if (v.imu.values.data)
    sensorStep<6><<<blocks, 128, 0, stream>>>(v, v.imu, lanes);
  if (v.barometer.values.data)
    sensorStep<1><<<blocks, 128, 0, stream>>>(v, v.barometer, lanes);
  if (v.odometry.values.data)
    sensorStep<10><<<blocks, 128, 0, stream>>>(v, v.odometry, lanes);
}
}  // namespace flightlib_cuda::v1::detail
