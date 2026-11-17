#include "flightlib_cuda/dynamics.hpp"
#include "dynamics_launch.hpp"
namespace flightlib_cuda::v1 {
namespace {
__device__ float clamp(float x,float a,float b){return fminf(b,fmaxf(a,x));}
// Only the first 13 fields have nonzero derivatives. Acceleration and torque
// are held fixed throughout each RK step, exactly as in the CPU implementation.
__device__ void derivative(const float* x,const float* acc,const float* torque,
                           const float* j,const float* ji,float* d){
  #pragma unroll
  for(int c=0;c<3;++c){
    d[c]=x[7+c];d[7+c]=acc[c];
    int a=(c+1)%3,b=(c+2)%3;
    d[10+c]=ji[c]*(torque[c]-(x[10+a]*j[b]*x[10+b]-x[10+b]*j[a]*x[10+a]));
  }
  float w=x[3],qx=x[4],qy=x[5],qz=x[6],a=x[10],b=x[11],c=x[12];
  d[3]=0.5f*(-a*qx-b*qy-c*qz);
  d[4]=0.5f*(a*w+c*qy-b*qz);
  d[5]=0.5f*(b*w-c*qx+a*qz);
  d[6]=0.5f*(c*w+b*qx-a*qy);
}
__global__ void dynamicsKernel(QuadState state,CommandStorage cmd,QuadTopologyView topo,
                              QuadModelView m,StepSchedule schedule,Vector<LaneStatus> status,const Mask* enabled){
  Index n=blockIdx.x*blockDim.x+threadIdx.x;
  if(n>=topo.batch.lanes)return;
  if(enabled&&!enabled[n])return;
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
    }else{
      float ft[4];ft[0]=m.mass(p)*command[0];
      #pragma unroll
      for(int c=0;c<3;++c){int a=(c+1)%3,b=(c+2)%3;
        ft[c+1]=j[c]*m.rate_gain(c,p)*(command[c+1]-x[10+c])+
          (x[10+a]*j[b]*x[10+b]-x[10+b]*j[a]*x[10+a]);}
      #pragma unroll
      // Same packet-product ordering as the forward allocation below.
      for(int r=0;r<4;++r){float v=m.inverse_allocation(r*4,p)*ft[0];
        #pragma unroll
        for(int c=1;c<4;++c)v=fmaf(m.inverse_allocation(r*4+c,p),ft[c],v);
        desired[r]=clamp(v,m.thrust_bounds(0,p),m.thrust_bounds(1,p));}
    }
    float a=m.thrust_polynomial(0,p),b=m.thrust_polynomial(1,p),c=m.thrust_polynomial(2,p);
    float decay=expf(-dt*m.motor_tau_inverse(p));
    #pragma unroll
    for(int r=0;r<4;++r){
      float scale=1.0f/(2.0f*a);
      float target=-b*scale+scale*sqrtf(b*b-4.0f*a*(c-desired[r]));
      if(!isfinite(target)){status(n)=LaneStatus::NumericalFailure;return;}
      target=clamp(target,m.motor_speed_bounds(0,p),m.motor_speed_bounds(1,p));
      omega[r]=decay*omega[r]+(1.0f-decay)*target;
      // Eigen's 4x3 polynomial product uses a packet multiply then pmadd.
      thrust[r]=clamp(fmaf(omega[r],b,omega[r]*omega[r]*a)+c,m.thrust_bounds(0,p),m.thrust_bounds(1,p));
    }
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
  #pragma unroll
  for(int c=0;c<2;++c)if(x[c]<topo.world_bounds(2*c,n)||x[c]>topo.world_bounds(2*c+1,n)){x[c]=old[c];x[7+c]=0;}
  if(x[2]<=topo.world_bounds(4,n)||x[2]>topo.world_bounds(5,n)){
    x[2]=topo.world_bounds(4,n);x[7]=x[8]=0;
    #pragma unroll
    for(int c=0;c<3;++c){x[10+c]=0;x[13+c]=0;}
  }
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
void detail::launchDynamics(const RuntimeViews& v,cudaStream_t stream,const Mask* enabled){
  dynamicsKernel<<<(Offset{v.topology.batch.lanes}+127)/128,128,0,stream>>>(v.state,v.commands,v.topology,v.models,v.schedule,v.diagnostics.status,enabled);
}
void stepDynamics(Simulation& s){s.enqueue([](const RuntimeViews& v,cudaStream_t stream,const std::vector<BindingRequest>&,void*){
  detail::launchDynamics(v,stream);
});}
}
