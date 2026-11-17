#include "flightlib_cuda/dynamics.hpp"
#include "dynamics_launch.hpp"
#include "dynamics_math.cuh"
namespace flightlib_cuda::v1 {
namespace {
using detail::derivative;
template<bool CheckCommand>
__global__ void dynamicsKernel(QuadState state,CommandStorage cmd,QuadTopologyView topo,
                              QuadModelView m,StepSchedule schedule,Vector<LaneStatus> status,const Mask* enabled){
  Index n=blockIdx.x*blockDim.x+threadIdx.x;
  if(n>=topo.batch.lanes)return;
  if(enabled&&!enabled[n])return;
  if constexpr(CheckCommand)if(status(n)!=LaneStatus::Ok)return;
  if(!topo.active(n)){status(n)=LaneStatus::Inactive;return;}
  float x[25],omega[4],thrust[4],command[4];
  bool valid=isfinite(state.time(n));
  #pragma unroll
  for(int c=0;c<25;++c){x[c]=state.x(c,n);valid &= isfinite(x[c]);}
  if(!valid){status(n)=LaneStatus::InvalidState;return;}
  auto mode=cmd.mode(n);Index p=topo.model_id(n);
  valid=isfinite(cmd.time(n))&&(mode==CommandMode::RotorThrust||mode==CommandMode::CollectiveBodyRates)&&p<m.mass.size();
  #pragma unroll
  for(int c=0;c<4;++c){command[c]=cmd.payload(c,n);valid &= isfinite(command[c]);omega[c]=state.motor_omega(c,n);thrust[c]=state.motor_thrust(c,n);}
  if(!valid){status(n)=LaneStatus::InvalidCommand;return;}
  float old[3]={x[0],x[1],x[2]},j[3],ji[3];
  #pragma unroll
  for(int c=0;c<3;++c){j[c]=m.inertia(c,p);ji[c]=m.inverse_inertia(c,p);}
  for(Offset s=0;s<schedule.substep_dt.size();++s){
    float dt=schedule.substep_dt(s),desired[4];
    if(mode==CommandMode::RotorThrust){
      #pragma unroll
      for(int c=0;c<4;++c)desired[c]=command[c];
    }else detail::controller(m,p,j,x+10,x+10,command,desired);
    if(!detail::motors(m,p,dt,desired,omega,thrust)){status(n)=LaneStatus::NumericalFailure;return;}
    float ft[4];
    #pragma unroll
    // Eigen's fixed-size column-major packet product starts with a multiply,
    // then uses explicit pmadd intrinsics even with host contraction disabled.
    // Match that ordering: cancellation sign matters to legacy Euler rewards.
    for(int r=0;r<4;++r){ft[r]=m.allocation(r*4,p)*thrust[0];
      #pragma unroll
      for(int c=1;c<4;++c)ft[r]=fmaf(m.allocation(r*4+c,p),thrust[c],ft[r]);}
    // Eigen quaternion-vector transform also supports non-unit input without normalization.
    float tx=2*x[5]*ft[0],ty=-2*x[4]*ft[0];
    x[13]=(x[3]*tx-x[6]*ty)/m.mass(p)+m.gravity(0,p);
    x[14]=(x[3]*ty+x[6]*tx)/m.mass(p)+m.gravity(1,p);
    x[15]=(ft[0]+x[4]*ty-x[5]*tx)/m.mass(p)+m.gravity(2,p);
    #pragma unroll
    for(int c=0;c<3;++c)x[16+c]=ft[c+1];
    float slope[13],sum[13],tmp[13];
    derivative(x,x+13,x+16,j,ji,slope);
    #pragma unroll
    for(int c=0;c<13;++c){sum[c]=dt*slope[c]*(1.0f/6);tmp[c]=x[c]+.5f*dt*slope[c];}
    derivative(tmp,x+13,x+16,j,ji,slope);
    #pragma unroll
    for(int c=0;c<13;++c){sum[c]+=dt*slope[c]*(2.0f/6);tmp[c]=x[c]+.5f*dt*slope[c];}
    derivative(tmp,x+13,x+16,j,ji,slope);
    #pragma unroll
    for(int c=0;c<13;++c){sum[c]+=dt*slope[c]*(2.0f/6);tmp[c]=x[c]+dt*slope[c];}
    derivative(tmp,x+13,x+16,j,ji,slope);
    #pragma unroll
    for(int c=0;c<13;++c)x[c]+=sum[c]+dt*slope[c]*(1.0f/6);
    // CPU normalizes the old quaternion then overwrites it with next_state.
  }
  // Detect failure before world-box clamps can conceal nonfinite intermediates.
  #pragma unroll
  for(int c=0;c<25;++c)if(!isfinite(x[c])){status(n)=LaneStatus::NumericalFailure;return;}
  detail::constrain(x,old,topo,n);
  float time=state.time(n)+schedule.control_dt;valid=isfinite(time);
  #pragma unroll
  for(int c=0;c<25;++c)valid &= isfinite(x[c]);
  #pragma unroll
  for(int c=0;c<4;++c)valid &= isfinite(omega[c])&&isfinite(thrust[c]);
  if(!valid){status(n)=LaneStatus::NumericalFailure;return;}
  #pragma unroll
  for(int c=0;c<25;++c)state.x(c,n)=x[c];
  #pragma unroll
  for(int c=0;c<4;++c){state.motor_omega(c,n)=omega[c];state.motor_thrust(c,n)=thrust[c];}
  state.time(n)=time;status(n)=LaneStatus::Ok;
}
}
void detail::launchDynamics(const RuntimeViews& v,cudaStream_t stream,const Mask* enabled,bool require_command_ok){
  if(require_command_ok)dynamicsKernel<true><<<(Offset{v.topology.batch.lanes}+127)/128,128,0,stream>>>(v.state,v.commands,v.topology,v.models,v.schedule,v.diagnostics.status,enabled);
  else dynamicsKernel<false><<<(Offset{v.topology.batch.lanes}+127)/128,128,0,stream>>>(v.state,v.commands,v.topology,v.models,v.schedule,v.diagnostics.status,enabled);
}
void detail::launchDynamicsFirst(const RuntimeViews& v,cudaStream_t stream,const Mask* enabled){
  dynamicsKernel<false><<<1,1,0,stream>>>(v.state,v.commands,v.topology,v.models,v.schedule,v.diagnostics.status,enabled);
}
namespace {
void dynamicsLaunch(const RuntimeViews& v,cudaStream_t stream,const std::vector<BindingRequest>&,void*){
  detail::launchDynamics(v,stream);
}
}
void stepDynamics(Simulation& s){s.enqueue(dynamicsLaunch);}
void stepDynamicsFrom(Simulation& s,cudaStream_t stream){s.enqueueFrom(stream,dynamicsLaunch);}
Graph captureDynamics(Simulation& s){return s.capture(dynamicsLaunch);}
}
