#pragma once
#include "flightlib_cuda/contracts.hpp"
#include <cuda_runtime.h>
namespace flightlib_cuda::v1::detail {
__device__ __forceinline__ float clamp(float x,float a,float b){return fminf(b,fmaxf(a,x));}
__device__ __forceinline__ void constrain(float* x,const float* old,QuadTopologyView topo,Index n){
  #pragma unroll
  for(int c=0;c<2;++c)if(x[c]<topo.world_bounds(2*c,n)||x[c]>topo.world_bounds(2*c+1,n)){x[c]=old[c];x[7+c]=0;}
  if(x[2]<=topo.world_bounds(4,n)||x[2]>topo.world_bounds(5,n)){
    x[2]=topo.world_bounds(4,n);x[7]=x[8]=0;
    #pragma unroll
    for(int c=0;c<3;++c){x[10+c]=0;x[13+c]=0;}
  }
}
__device__ __forceinline__ void derivative(const float* x,const float* acc,const float* torque,
                                         const float* j,const float* ji,float* d){
  #pragma unroll
  for(int c=0;c<3;++c){d[c]=x[7+c];d[7+c]=acc[c];int a=(c+1)%3,b=(c+2)%3;
    d[10+c]=ji[c]*(torque[c]-(x[10+a]*j[b]*x[10+b]-x[10+b]*j[a]*x[10+a]));}
  float w=x[3],qx=x[4],qy=x[5],qz=x[6],a=x[10],b=x[11],c=x[12];
  d[3]=0.5f*(-a*qx-b*qy-c*qz);d[4]=0.5f*(a*w+c*qy-b*qz);
  d[5]=0.5f*(b*w-c*qx+a*qz);d[6]=0.5f*(c*w+b*qx-a*qy);
}
__device__ __forceinline__ float motorTarget(float force,float a,float b,float c){
  float scale=1.0f/(2.0f*a);
  return -b*scale+scale*sqrtf(b*b-4.0f*a*(c-force));
}
__device__ __forceinline__ float motorForce(float omega,float a,float b,float c){
  // Eigen's packet polynomial product: multiply, pmadd, then constant addition.
  return fmaf(omega,b,omega*omega*a)+c;
}
__device__ __forceinline__ void controller(QuadModelView m,Index p,const float* j,
                                         const float* state_rates,const float* measured_rates,
                                         const float* command,float* desired){
  float ft[4];ft[0]=m.mass(p)*command[0];
  #pragma unroll
  for(int c=0;c<3;++c){int a=(c+1)%3,b=(c+2)%3;
    ft[c+1]=j[c]*m.rate_gain(c,p)*(command[c+1]-measured_rates[c])+
      (state_rates[a]*j[b]*state_rates[b]-state_rates[b]*j[a]*state_rates[a]);}
  #pragma unroll
  for(int r=0;r<4;++r){float v=m.inverse_allocation(r*4,p)*ft[0];
    #pragma unroll
    for(int c=1;c<4;++c)v=fmaf(m.inverse_allocation(r*4+c,p),ft[c],v);
    desired[r]=clamp(v,m.thrust_bounds(0,p),m.thrust_bounds(1,p));}
}
__device__ __forceinline__ bool motors(QuadModelView m,Index p,float dt,const float* desired,float* omega,float* thrust){
  float a=m.thrust_polynomial(0,p),b=m.thrust_polynomial(1,p),c=m.thrust_polynomial(2,p);
  float decay=expf(-dt*m.motor_tau_inverse(p));
  #pragma unroll
  for(int r=0;r<4;++r){float target=motorTarget(desired[r],a,b,c);
    if(!isfinite(target))return false;
    target=clamp(target,m.motor_speed_bounds(0,p),m.motor_speed_bounds(1,p));
    omega[r]=decay*omega[r]+(1.0f-decay)*target;
    thrust[r]=clamp(motorForce(omega[r],a,b,c),m.thrust_bounds(0,p),m.thrust_bounds(1,p));}
  return true;
}
}
