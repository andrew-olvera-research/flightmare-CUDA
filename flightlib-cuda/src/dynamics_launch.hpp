#pragma once
#include "flightlib_cuda/runtime.hpp"
namespace flightlib_cuda::v1::detail {
struct StaticScene {const float* nodes; Index count; const float* options;};
void launchSceneDynamics(const RuntimeViews&,cudaStream_t,const Mask* enabled,StaticScene,bool first);
void launchSwarmDynamics(const RuntimeViews&,cudaStream_t,const Mask* enabled,StaticScene);
void launchMovingPairs(const RuntimeViews&,cudaStream_t);
void launchSensors(const RuntimeViews&,cudaStream_t,bool first);
// Internal ordered stage; caller owns the runtime submission boundary.
void launchDynamics(const RuntimeViews&,cudaStream_t,const Mask* enabled=nullptr,bool require_command_ok=false);
// Legacy VecEnv::testStep: launch only lane zero, preserving all original strides.
void launchDynamicsFirst(const RuntimeViews&,cudaStream_t,const Mask* enabled);
}
