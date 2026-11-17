#pragma once
#include "flightlib_cuda/lifecycle.hpp"
namespace flightlib_cuda::v1 {
// One asynchronous fused launch using the configured schedule and accepted commands.
// Invalid inputs and numerical failure retain prior physical telemetry per lane.
void stepDynamics(Simulation&);
}
