#include <cuda_runtime.h>
#include <cstdint>

__global__ void stats(const float* reward,const uint8_t* done,const uint8_t* valid,
                      float* running,int64_t* length,float* returns,int64_t* lengths,int n,bool reset){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
  returns[i]=0;lengths[i]=0;
  if(!valid[i])return;
  float r=running[i]+reward[i];int64_t l=length[i]+1;
  if(done[i]){returns[i]=r;lengths[i]=l;if(reset){r=0;l=0;}}
  running[i]=r;length[i]=l;
}
__global__ void gae(const float* reward,const float* value,const uint8_t* done,const uint8_t* valid,
                    const float* last,float* advantage,float* returns,int n,int steps,float gamma,float lambda){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
  float next=last[i],a=0;bool next_valid=true;
  for(int t=steps-1;t>=0;--t){
    uint64_t k=uint64_t(t)*n+i;
    bool ok=valid[k],cont=ok&&!done[k]&&next_valid;
    float delta=ok?(reward[k]+(cont?gamma*next:0)-value[k]):0;
    a=delta+(cont?gamma*lambda*a:0);
    advantage[k]=a;returns[k]=a+value[k];
    next=value[k];next_valid=ok;
  }
}
extern "C" {
int rl_kernel_abi(){return 1;}
int rl_stats(const float* reward,const uint8_t* done,const uint8_t* valid,float* running,int64_t* length,
              float* returns,int64_t* lengths,int n,bool reset,cudaStream_t stream){
  if(n<1)return int(cudaErrorInvalidValue);
  stats<<<(uint64_t(n)+127)/128,128,0,stream>>>(reward,done,valid,running,length,returns,lengths,n,reset);
  return int(cudaGetLastError());
}
int rl_gae(const float* reward,const float* value,const uint8_t* done,const uint8_t* valid,const float* last,
            float* advantage,float* returns,int n,int steps,float gamma,float lambda,cudaStream_t stream){
  if(n<1||steps<1)return int(cudaErrorInvalidValue);
  gae<<<(uint64_t(n)+127)/128,128,0,stream>>>(reward,value,done,valid,last,advantage,returns,n,steps,gamma,lambda);
  return int(cudaGetLastError());
}
}
