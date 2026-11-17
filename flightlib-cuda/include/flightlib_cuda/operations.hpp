#pragma once
#include "flightlib_cuda/lifecycle.hpp"
namespace flightlib_cuda::v1 {
enum class CoreOperation { Derivative, Controller, Motors, OmegaToThrust, ThrustToOmega,
                           ClampThrust, ClampMotorOmega, ClampBodyRates, ConstrainWorld };
// Input/output are disjoint caller-owned component-major float tensors. Binding
// validates shapes and retains owners; no scratch is added to the frozen arena.
Bindings bindOperation(Simulation&,CoreOperation,BindingRequest input,BindingRequest output);
void evaluateOperation(Simulation&,CoreOperation,const Bindings&,Scalar dt=0);
void evaluateOperationFrom(Simulation&,CoreOperation,const Bindings&,cudaStream_t,Scalar dt=0);
}
