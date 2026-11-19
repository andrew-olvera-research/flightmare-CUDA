#pragma once
#include "flightlib_cuda/runtime.hpp"

namespace flightlib_cuda::v1 {
enum class ExportKind { Pose, Imu, Odometry, Barometer };
// Packed rows are 8-byte aligned. Pose: uint64 episode, uint32 lane/valid,
// float32 xyz/wxyz, padding. Sensor: six uint64 packet metadata fields in the
// sensor API order, uint32 lane/valid, then float32 measurement channels.
Bindings bindExport(Simulation&, ExportKind, BindingRequest lanes,
                    BindingRequest rows, BindingRequest count);
void packExportFrom(Simulation&, const Bindings&, ExportKind, cudaStream_t,
                    cudaEvent_t ready=nullptr);
}
