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
  int rl_threads=1;
  Batch(int n,const char* path,float a):obs(n),transition(n),reward(n),done(n){
    action << a+.1f,a-.1f,a+.05f,a-.05f;
    for(int i=0;i<n;++i){env.emplace_back(std::make_unique<flightlib::QuadrotorEnv>(path));env.back()->reset(obs[i],false);}
  }
};
extern "C" {
void batch_threads(int n){omp_set_dynamic(0);omp_set_num_threads(n);}
void* batch_create(int n,const char* path,float action){try{return new Batch(n,path,action);}catch(...){return nullptr;}}
void batch_destroy(void* p){delete static_cast<Batch*>(p);}
void batch_rl_threads(void* p,int threads){static_cast<Batch*>(p)->rl_threads=threads;}
// RL adapter: dynamic per-lane actions and complete row-major outputs. The
// environment equations remain the original QuadrotorEnv::step implementation.
int batch_rl_reset(void* p,int spin,float* observation){try{auto& b=*static_cast<Batch*>(p);
  flightlib::Vector<4> launch;launch<<.8f,.82f,.79f,.81f;
  #pragma omp parallel for schedule(static) num_threads(b.rl_threads)
  for(size_t i=0;i<b.env.size();++i){b.env[i]->reset(b.obs[i],false);
    // Deliberate setup-only takeoff without vector autoreset; matched on GPU.
    for(int t=0;t<spin;++t)b.env[i]->step(launch,b.obs[i]);
    for(int c=0;c<12;++c)observation[12*i+c]=b.obs[i][c];
  }return 0;}catch(...){return -1;}}
int batch_rl_step(void* p,const float* actions,float* observation,float* transition,float* reward,unsigned char* done){try{auto& b=*static_cast<Batch*>(p);
  int failure=0;
  #pragma omp parallel for schedule(static) num_threads(b.rl_threads) reduction(|:failure)
  for(size_t i=0;i<b.env.size();++i){
    flightlib::Vector<4> a;for(int c=0;c<4;++c)a[c]=actions[4*i+c];
    float extra=0;reward[i]=b.env[i]->step(a,b.obs[i]);done[i]=b.env[i]->isTerminalState(extra);reward[i]+=extra;
    for(int c=0;c<12;++c)transition[12*i+c]=b.obs[i][c];
    if(done[i])b.env[i]->reset(b.obs[i],false);
    for(int c=0;c<12;++c)observation[12*i+c]=b.obs[i][c];
    failure|=!b.obs[i].allFinite()||!std::isfinite(reward[i]);
  }return failure?-1:0;}catch(...){return -1;}}
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
