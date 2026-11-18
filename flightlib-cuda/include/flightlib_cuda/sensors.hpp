#pragma once
#include "flightlib_cuda/contracts.hpp"

namespace flightlib_cuda::v1 {
struct SensorSettings {
  std::uint64_t period_ns{}, phase_ns{}, latency_ns{}, seed{};
  float mount[4]{1, 0, 0, 0}, offset[3]{}, dropout{};
  // IMU: gyro XYZ then accelerometer XYZ, Kalibr continuous-time units.
  // Odometry: position XYZ then world velocity XYZ, per-sample standard
  // deviations.
  float noise[6]{}, bias[6]{}, walk[6]{}, limit[6]{}, increment_width[6]{},
    attitude_std[3]{};
  bool stochastic{};
};
struct SensorBuffers {
  Matrix<std::uint64_t> state;  // clock, episode, epoch, produced, consumed
  Matrix<Scalar> drift;
  TensorView<Scalar, 3> queue, values;  // [packet, lane, channel]
  Matrix<std::uint64_t> acquired;
  Matrix<Mask> queue_valid, valid;
  TensorView<std::uint64_t, 3> metadata;  // [field,packet,lane]
  SensorSettings settings;
};
struct SensorTruth {
  TensorView<Scalar, 3> samples;  // [substep,16,lane], state x[0:16]
  Vector<Mask> corrected;
};
}  // namespace flightlib_cuda::v1
