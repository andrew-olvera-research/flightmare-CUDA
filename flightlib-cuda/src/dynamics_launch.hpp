#pragma once
#include "flightlib_cuda/runtime.hpp"
namespace flightlib_cuda::v1::detail {
// Internal ordered stage; caller owns the runtime submission boundary.
void launchDynamics(const RuntimeViews&,cudaStream_t,const Mask* enabled=nullptr,bool require_command_ok=false);
// Legacy VecEnv::testStep: launch only lane zero, preserving all original strides.
void launchDynamicsFirst(const RuntimeViews&,cudaStream_t,const Mask* enabled);
}
