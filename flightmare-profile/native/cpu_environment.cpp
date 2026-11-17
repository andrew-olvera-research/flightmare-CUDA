// Benchmark adapter only. Physics/task implementation is unchanged Flightlib.
#include "flightlib/envs/quadrotor_env/quadrotor_env.hpp"
#include <memory>
#include <vector>
#include <stdexcept>
#include <omp.h>
struct Batch {
  std::vector<std::unique_ptr<flightlib::QuadrotorEnv>> env;
  std::vector<flightlib::Vector<12>> obs,transition;
  std::vector<float> reward;
  std::vector<unsigned char> done;
  flightlib::Vector<4> action;
  unsigned long long resets=0;
  Batch(int n,const char* path,float a):obs(n),transition(n),reward(n),done(n){
    action << a+.1f,a-.1f,a+.05f,a-.05f;
    for(int i=0;i<n;++i){env.emplace_back(std::make_unique<flightlib::QuadrotorEnv>(path));env.back()->reset(obs[i],false);}
  }
};
extern "C" {
void batch_threads(int n){omp_set_dynamic(0);omp_set_num_threads(n);}
void* batch_create(int n,const char* path,float action){try{return new Batch(n,path,action);}catch(...){return nullptr;}}
void batch_destroy(void* p){delete static_cast<Batch*>(p);}
void batch_reset(void* p,float a){auto& b=*static_cast<Batch*>(p);b.action<<a+.1f,a-.1f,a+.05f,a-.05f;b.resets=0;
  for(size_t i=0;i<b.env.size();++i)b.env[i]->reset(b.obs[i],false);}
int batch_step(void* p,int steps){try{auto& b=*static_cast<Batch*>(p);
  unsigned long long resets=0;
  #pragma omp parallel reduction(+:resets)
  { for(int k=0;k<steps;++k){
  #pragma omp for schedule(static)
  for(size_t i=0;i<b.env.size();++i){
    float extra=0;b.reward[i]=b.env[i]->step(b.action,b.obs[i]);b.transition[i]=b.obs[i];
    b.done[i]=b.env[i]->isTerminalState(extra);b.reward[i]+=extra;
    if(b.done[i]){b.env[i]->reset(b.obs[i],false);++resets;}
  } } } b.resets+=resets;return 0;}catch(...){return -1;}}
// Readback is outside timing. All lanes deliberately have identical inputs.
int batch_snapshot(void* p,float* out,unsigned long long* resets){auto& b=*static_cast<Batch*>(p);
  for(size_t i=0;i<b.env.size();++i){if(!b.obs[i].allFinite()||!b.transition[i].allFinite()||!std::isfinite(b.reward[i]))return -1;
    for(int c=0;c<12;++c)if(b.obs[i][c]!=b.obs[0][c]||b.transition[i][c]!=b.transition[0][c])return -1;
    if(b.reward[i]!=b.reward[0]||b.done[i]!=b.done[0])return -1;}
  for(int c=0;c<12;++c){out[c]=b.obs[0][c];out[c+12]=b.transition[0][c];}
  out[24]=b.reward[0];out[25]=b.done[0];*resets=b.resets;return 0;}
}
