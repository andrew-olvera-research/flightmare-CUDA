#include "flightlib_cuda/lifecycle.hpp"
#include "flightlib_cuda/dynamics.hpp"
#include "dynamics_launch.hpp"
#include <cuda_runtime.h>
#include <algorithm>
namespace flightlib_cuda::v1 {
namespace {
__global__ void stateKernel(RuntimeViews v,const Mask* selected,const Scalar* x,const Scalar* time,bool reset) {
  const Index n=blockIdx.x*blockDim.x+threadIdx.x;
  const Offset N=v.topology.batch.lanes;
  if(n>=N)return;
  if(!v.topology.active(n)) {v.diagnostics.status(n)=LaneStatus::Inactive;return;}
  v.diagnostics.status(n)=LaneStatus::Ok;
  if(!selected[n])return;
  bool valid=selected[n]==1&&isfinite(time[n]);
  for(Index c=0;c<25;++c) valid=valid&&isfinite(x[c*N+n]);
  if(!valid){v.diagnostics.status(n)=reset?LaneStatus::InvalidReset:LaneStatus::InvalidState;return;}
  for(Index c=0;c<25;++c)v.state.x(c,n)=x[c*N+n];
  v.state.time(n)=time[n];
  if(reset)for(Index c=0;c<4;++c){v.state.motor_omega(c,n)=0;v.state.motor_thrust(c,n)=0;}
}
__global__ void commandKernel(RuntimeViews v,const Mask* selected,const CommandMode* mode,const Scalar* time,const Scalar* payload) {
  const Index n=blockIdx.x*blockDim.x+threadIdx.x;
  const Offset N=v.topology.batch.lanes;
  if(n>=N)return;
  if(!v.topology.active(n)){v.diagnostics.status(n)=LaneStatus::Inactive;return;}
  v.diagnostics.status(n)=LaneStatus::Ok;
  if(!selected[n])return;
  bool valid=selected[n]==1&&isfinite(time[n])&&(mode[n]==CommandMode::RotorThrust||mode[n]==CommandMode::CollectiveBodyRates);
  for(Index c=0;c<4;++c)valid=valid&&isfinite(payload[c*N+n]);
  const Index p=v.topology.model_id(n);
  if(!valid||p>=v.models.mass.size()||!(v.models.mass(p)>0)){
    v.diagnostics.status(n)=LaneStatus::InvalidCommand;return;
  }
  v.commands.mode(n)=mode[n];v.commands.time(n)=time[n];
  for(Index c=0;c<4;++c){
    const bool rate=mode[n]==CommandMode::CollectiveBodyRates&&c>0;
    const Scalar low=rate?-v.models.body_rate_limit(c-1,p):v.models.thrust_bounds(0,p);
    const Scalar high=rate?v.models.body_rate_limit(c-1,p):v.models.thrust_bounds(1,p);
    v.commands.payload(c,n)=fminf(high,fmaxf(low,payload[c*N+n]));
  }
}
TensorSpec spec(DType type,std::initializer_list<Offset> shape){
  TensorSpec s{type,static_cast<Index>(shape.size()),{}};
  std::copy(shape.begin(),shape.end(),s.shape.begin());return s;
}
void validate(const Bindings& b,std::initializer_list<TensorSpec> expected){
  if(b.inputs().size()!=expected.size())throw std::invalid_argument("wrong lifecycle input count");
  for(size_t i=0;i<expected.size();++i){const auto& s=b.inputs()[i].tensor.spec;
    const auto& e=expected.begin()[i];
    if(s.dtype!=e.dtype||s.rank!=e.rank||s.shape!=e.shape)
      throw std::invalid_argument("wrong lifecycle input schema");}
}
void stateLaunch(const RuntimeViews& v,cudaStream_t stream,const std::vector<BindingRequest>& b,void* reset){
  stateKernel<<<(Offset{v.topology.batch.lanes}+127)/128,128,0,stream>>>(v,
    static_cast<const Mask*>(b[0].tensor.data),static_cast<const Scalar*>(b[1].tensor.data),
    static_cast<const Scalar*>(b[2].tensor.data),reset!=nullptr);
}
void commandLaunch(const RuntimeViews& v,cudaStream_t stream,const std::vector<BindingRequest>& b,void*){
  commandKernel<<<(Offset{v.topology.batch.lanes}+127)/128,128,0,stream>>>(v,
    static_cast<const Mask*>(b[0].tensor.data),static_cast<const CommandMode*>(b[1].tensor.data),
    static_cast<const Scalar*>(b[2].tensor.data),static_cast<const Scalar*>(b[3].tensor.data));
}
void runLaunch(const RuntimeViews& v,cudaStream_t stream,const std::vector<BindingRequest>& b,void*){
  commandLaunch(v,stream,b,nullptr);detail::launchDynamics(v,stream,nullptr,true);
}
void validateCommands(const Simulation& s,const Bindings& b){const Offset n=s.views().topology.batch.lanes;
  validate(b,{spec(DType::UInt8,{n}),spec(DType::UInt8,{n}),spec(DType::Float32,{n}),spec(DType::Float32,{4,n})});}
void validateState(const Simulation& s,const Bindings& b){const Offset n=s.views().topology.batch.lanes;
  validate(b,{spec(DType::UInt8,{n}),spec(DType::Float32,{25,n}),spec(DType::Float32,{n})});}
}
Bindings bindStateInput(Simulation& s,BindingRequest selected,BindingRequest x,BindingRequest time){
  const Offset n=s.views().topology.batch.lanes;
  selected.expected=spec(DType::UInt8,{n});x.expected=spec(DType::Float32,{25,n});time.expected=spec(DType::Float32,{n});
  return s.bind({selected,x,time});
}
Bindings bindCommandInput(Simulation& s,BindingRequest selected,BindingRequest mode,BindingRequest time,BindingRequest payload){
  const Offset n=s.views().topology.batch.lanes;
  selected.expected=spec(DType::UInt8,{n});mode.expected=spec(DType::UInt8,{n});
  time.expected=spec(DType::Float32,{n});payload.expected=spec(DType::Float32,{4,n});
  return s.bind({selected,mode,time,payload});
}
void resetState(Simulation& s,const Bindings& b){const Offset n=s.views().topology.batch.lanes;
  validate(b,{spec(DType::UInt8,{n}),spec(DType::Float32,{25,n}),spec(DType::Float32,{n})});
  s.enqueue(stateLaunch,reinterpret_cast<void*>(1),&b);}
void setState(Simulation& s,const Bindings& b){const Offset n=s.views().topology.batch.lanes;
  validate(b,{spec(DType::UInt8,{n}),spec(DType::Float32,{25,n}),spec(DType::Float32,{n})});
  s.enqueue(stateLaunch,nullptr,&b);}
void setCommand(Simulation& s,const Bindings& b){validateCommands(s,b);s.enqueue(commandLaunch,nullptr,&b);}
void resetStateFrom(Simulation& s,const Bindings& b,cudaStream_t stream){validateState(s,b);s.enqueueFrom(stream,stateLaunch,reinterpret_cast<void*>(1),&b);}
void setStateFrom(Simulation& s,const Bindings& b,cudaStream_t stream){validateState(s,b);s.enqueueFrom(stream,stateLaunch,nullptr,&b);}
void setCommandFrom(Simulation& s,const Bindings& b,cudaStream_t stream){validateCommands(s,b);s.enqueueFrom(stream,commandLaunch,nullptr,&b);}
void runDynamics(Simulation& s,const Bindings& b){validateCommands(s,b);s.enqueue(runLaunch,nullptr,&b);}
void runDynamicsFrom(Simulation& s,const Bindings& b,cudaStream_t stream){validateCommands(s,b);s.enqueueFrom(stream,runLaunch,nullptr,&b);}
Graph captureRun(Simulation& s,const Bindings& b){validateCommands(s,b);return s.capture(runLaunch,nullptr,&b);}
} // namespace flightlib_cuda::v1
