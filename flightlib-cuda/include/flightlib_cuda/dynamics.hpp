#pragma once
#include "flightlib_cuda/lifecycle.hpp"
namespace flightlib_cuda::v1 {
// One asynchronous fused launch using the configured schedule and accepted commands.
// Invalid inputs and numerical failure retain prior physical telemetry per lane.
void stepDynamics(Simulation&);
void stepDynamicsFrom(Simulation&,cudaStream_t producer);
Graph captureDynamics(Simulation&);
// Accept selected commands and step in one submission. Rejected selected lanes
// do not advance (legacy run(command, dt)); unselected lanes use accepted commands.
void runDynamics(Simulation&,const Bindings& commands);
void runDynamicsFrom(Simulation&,const Bindings& commands,cudaStream_t producer);
Graph captureRun(Simulation&,const Bindings& commands);
}
