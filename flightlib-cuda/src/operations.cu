#include "flightlib_cuda/operations.hpp"
#include "dynamics_math.cuh"
#include <utility>
namespace flightlib_cuda::v1 {
namespace {
std::pair<Index,Index> rows(CoreOperation op){switch(op){
  case CoreOperation::Derivative:case CoreOperation::ConstrainWorld:return {25,25};case CoreOperation::Controller:return {3,4};
  case CoreOperation::Motors:return {4,8};case CoreOperation::ClampBodyRates:return {3,3};
  case CoreOperation::OmegaToThrust:case CoreOperation::ThrustToOmega:
  case CoreOperation::ClampThrust:case CoreOperation::ClampMotorOmega:return {4,4};
}throw std::invalid_argument("unknown core operation");}
TensorSpec spec(Index r,Index n){TensorSpec s{DType::Float32,2,{}};s.shape[0]=r;s.shape[1]=n;return s;}
struct Context {CoreOperation op;float dt;};
template<CoreOperation op>
__global__ void operationKernel(RuntimeViews v,float dt,const float* input,float* output){
  constexpr Index nr=op==CoreOperation::Derivative||op==CoreOperation::ConstrainWorld?25:
    op==CoreOperation::Controller||op==CoreOperation::ClampBodyRates?3:4;
  constexpr Index no=op==CoreOperation::Derivative||op==CoreOperation::ConstrainWorld?25:
    op==CoreOperation::Motors?8:op==CoreOperation::ClampBodyRates?3:4;
  Index n=blockIdx.x*blockDim.x+threadIdx.x,N=v.topology.batch.lanes;if(n>=N)return;
  if(!v.topology.active(n)){v.diagnostics.status(n)=LaneStatus::Inactive;return;}
  Index p=v.topology.model_id(n);float x[25],out[25]={},j[3],ji[3];bool valid=true;
  #pragma unroll
  for(Index c=0;c<nr;++c){x[c]=input[Offset{c}*N+n];valid &= isfinite(x[c]);}
  if(!valid){v.diagnostics.status(n)=LaneStatus::InvalidState;return;}
  for(int c=0;c<3;++c){j[c]=v.models.inertia(c,p);ji[c]=v.models.inverse_inertia(c,p);}
  float a=v.models.thrust_polynomial(0,p),b=v.models.thrust_polynomial(1,p),c=v.models.thrust_polynomial(2,p);
  if(op==CoreOperation::ConstrainWorld){
    valid=isfinite(v.state.time(n));for(int i=0;i<25;++i){out[i]=v.state.x(i,n);valid &= isfinite(out[i]);}
    if(!valid){v.diagnostics.status(n)=LaneStatus::InvalidState;return;}
    detail::constrain(out,x,v.topology,n);
  }else if(op==CoreOperation::Derivative)detail::derivative(x,x+13,x+16,j,ji,out);
  else if(op==CoreOperation::Controller){
    float rates[3],command[4];valid=v.commands.mode(n)==CommandMode::CollectiveBodyRates&&isfinite(v.commands.time(n));
    for(int i=0;i<3;++i){rates[i]=v.state.x(10+i,n);valid &= isfinite(rates[i]);}
    for(int i=0;i<4;++i){command[i]=v.commands.payload(i,n);valid &= isfinite(command[i]);}
    if(!valid){v.diagnostics.status(n)=LaneStatus::InvalidCommand;return;}
    detail::controller(v.models,p,j,rates,x,command,out);
  }else if(op==CoreOperation::Motors){
    for(int i=0;i<4;++i)out[i]=v.state.motor_omega(i,n);
    if(!detail::motors(v.models,p,dt,x,out,out+4)){v.diagnostics.status(n)=LaneStatus::NumericalFailure;return;}
  }else for(Index i=0;i<nr;++i){
    if(op==CoreOperation::OmegaToThrust)out[i]=detail::motorForce(x[i],a,b,c);
    if(op==CoreOperation::ThrustToOmega)out[i]=detail::motorTarget(x[i],a,b,c);
    if(op==CoreOperation::ClampThrust)out[i]=detail::clamp(x[i],v.models.thrust_bounds(0,p),v.models.thrust_bounds(1,p));
    if(op==CoreOperation::ClampMotorOmega)out[i]=detail::clamp(x[i],v.models.motor_speed_bounds(0,p),v.models.motor_speed_bounds(1,p));
    if(op==CoreOperation::ClampBodyRates)out[i]=detail::clamp(x[i],-v.models.body_rate_limit(i,p),v.models.body_rate_limit(i,p));
  }
  #pragma unroll
  for(Index i=0;i<no;++i)if(!isfinite(out[i])){v.diagnostics.status(n)=LaneStatus::NumericalFailure;return;}
  #pragma unroll
  for(Index i=0;i<no;++i)output[Offset{i}*N+n]=out[i];
  if(op==CoreOperation::Motors)for(int i=0;i<4;++i){v.state.motor_omega(i,n)=out[i];v.state.motor_thrust(i,n)=out[4+i];}
  if(op==CoreOperation::ConstrainWorld)for(int i=0;i<25;++i)v.state.x(i,n)=out[i];
  v.diagnostics.status(n)=LaneStatus::Ok;
}
void launch(const RuntimeViews& v,cudaStream_t stream,const std::vector<BindingRequest>& b,void* ptr){
  auto ctx=*static_cast<Context*>(ptr);
  auto input=static_cast<const float*>(b[0].tensor.data);auto output=static_cast<float*>(b[1].tensor.data);
  // Specialize only when explicitly invoked; no generic per-thread dispatch or
  // dynamically indexed local arrays in the downstream helper path.
  #define FLC_OPERATION(kind) case CoreOperation::kind: operationKernel<CoreOperation::kind><<<(Offset{v.topology.batch.lanes}+127)/128,128,0,stream>>>(v,ctx.dt,input,output);break
  switch(ctx.op){
    FLC_OPERATION(Derivative);FLC_OPERATION(Controller);FLC_OPERATION(Motors);
    FLC_OPERATION(OmegaToThrust);FLC_OPERATION(ThrustToOmega);FLC_OPERATION(ClampThrust);
    FLC_OPERATION(ClampMotorOmega);FLC_OPERATION(ClampBodyRates);FLC_OPERATION(ConstrainWorld);
  }
  #undef FLC_OPERATION
}
Context validate(const Simulation& s,CoreOperation op,const Bindings& b,float dt){
  auto [nr,no]=rows(op);if(!std::isfinite(dt)||dt<0)throw std::invalid_argument("invalid operation timestep");
  if(b.inputs().size()!=2)throw std::invalid_argument("wrong operation bindings");
  int i=0;for(auto r:{nr,no}){auto expected=spec(r,s.views().topology.batch.lanes),actual=b.inputs()[i++].tensor.spec;
    if(actual.dtype!=expected.dtype||actual.rank!=expected.rank||actual.shape!=expected.shape)throw std::invalid_argument("wrong operation tensor schema");}
  return {op,dt};
}
}
Bindings bindOperation(Simulation& s,CoreOperation op,BindingRequest input,BindingRequest output){
  auto [nr,no]=rows(op);input.expected=spec(nr,s.views().topology.batch.lanes);output.expected=spec(no,s.views().topology.batch.lanes);
  return s.bind({input,output});
}
void evaluateOperation(Simulation& s,CoreOperation op,const Bindings& b,float dt){auto ctx=validate(s,op,b,dt);s.enqueue(launch,&ctx,&b);}
void evaluateOperationFrom(Simulation& s,CoreOperation op,const Bindings& b,cudaStream_t stream,float dt){auto ctx=validate(s,op,b,dt);s.enqueueFrom(stream,launch,&ctx,&b);}
}
