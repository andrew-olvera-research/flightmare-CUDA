#include "flightlib_cuda/environment.hpp"
#include "flightlib/objects/quadrotor.hpp"
#include <iostream>
#include <memory>
#include <algorithm>
using namespace flightlib_cuda::v1;
#define CHECK(x) do { if(!(x)) throw std::runtime_error("line "+std::to_string(__LINE__)+": " #x); } while(false)
#define CUDA(x) CHECK((x)==cudaSuccess)
template<class F> void rejects(F f){bool rejected=false;try{f();}catch(const std::exception&){rejected=true;}CHECK(rejected);}
template<class T> std::vector<T> download(const T* p,size_t n){std::vector<T> out(n);CUDA(cudaMemcpy(out.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return out;}
template<class T> void upload(void* p,const std::vector<T>& v){CUDA(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
template<class T> BindingRequest input(const std::vector<T>& data,DType type,std::initializer_list<Offset> shape){
  void* p=nullptr;CUDA(cudaMalloc(&p,data.size()*sizeof(T)));upload(p,data);
  auto owner=std::shared_ptr<void>(p,[](void* ptr){cudaFree(ptr);});
  TensorSpec spec{type,static_cast<Index>(shape.size()),{}};std::copy(shape.begin(),shape.end(),spec.shape.begin());
  std::array<Offset,3> strides{};Offset stride=1;for(Index i=spec.rank;i-->0;){strides[i]=stride;stride*=spec.shape[i];}
  return {{p,spec,strides},spec,owner};
}


#include "flightlib/envs/quadrotor_env/quadrotor_env.hpp"
std::string context;
void near(float a,float b,float tol=5e-4f){if(!std::isfinite(a)||std::abs(a-b)>tol*std::max(1.f,std::abs(b)))throw std::runtime_error(context+" mismatch gpu="+std::to_string(a)+" cpu="+std::to_string(b));}
std::vector<float> box(int N){std::vector<float>b(6*N);for(int n=0;n<N;++n){b[n]=b[2*N+n]=-20;b[N+n]=b[3*N+n]=b[5*N+n]=20;b[4*N+n]=0;}return b;}
struct Fixture {
  Simulation sim;Index N;std::vector<float> state,time,action;std::vector<Mask> selected;
  BindingRequest ai,si,xi,ti;Bindings inputs;
  Fixture(Index E,Index A,float dt=0.02f,std::uint64_t seed=42):sim([&]{RuntimeConfig c;c.environments=E;c.slots=A;c.substeps=prepareSchedule(dt).size();c.seed=seed;return c;}()),N(E*A),state(25*N),time(N),action(4*N),selected(N,1){
    configure(sim,{ModelConfig{}},std::vector<Index>(N,0),selected,box(N),dt);configureTask(sim,{TaskConfig{}});
    for(Index n=0;n<N;++n)state[3*N+n]=1;
    ai=input(action,DType::Float32,{E,A,4});si=input(selected,DType::UInt8,{N});xi=input(state,DType::Float32,{25,N});ti=input(time,DType::Float32,{N});
    inputs=bindTaskInputs(sim,ai,si,xi,ti);
  }
  void syncInputs(){sim.synchronize();upload(ai.tensor.data,action);upload(si.tensor.data,selected);upload(xi.tensor.data,state);upload(ti.tensor.data,time);}
};
void scalarEnv(bool saturated=false){
  auto path=std::string(getenv("FLIGHTMARE_PATH"))+"/flightlib/configs/quadrotor_env.yaml";
  flightlib::QuadrotorEnv cpu(path);auto yaml=YAML::LoadFile(path);yaml["quadrotor_env"]["max_t"]=.001f;CHECK(cpu.loadParam(yaml));
  float dt=cpu.getSimTimeStep();Fixture f(1,1,dt);auto model=loadModel(path);
  configure(f.sim,{model},{0},{1},box(1),dt);configureTask(f.sim,{legacyTaskConfig(model.mass)});
  flightlib::Vector<12> obs;cpu.reset(obs,false);resetEnvironment(f.sim,f.inputs);
  int episodes=1,steps=0;float clock=0;
  for(int i=0;i<120;++i){flightlib::Vector<4> act;act<<1.f,1.f,1.f,1.f;if(i>40)act<<-.8f,-.2f,.5f,.1f;if(i==10)act<<3.f,-2.f,1.5f,.1f;
    // Saturation + exact symmetry exposes cancellation-sign changes in the
    // allocation product: equivalent Euler orientations can change the reward.
    if(saturated)act.setConstant(3.f);
    for(int c=0;c<4;++c)f.action[c]=act[c];f.syncInputs();
    float reward=cpu.step(act,obs),extra=0;bool done=cpu.isTerminalState(extra);auto transition=obs.eval();
    reward+=extra;++steps;clock+=dt;if(done){cpu.reset(obs,false);++episodes;steps=0;clock=0;}
    stepEnvironment(f.sim,f.inputs,ResetSource::Explicit,i%2?TaskExecution::SeparateReset:TaskExecution::FusedReset);f.sim.synchronize();
    auto actual=download(f.sim.views().task_output.observation.data,12),tr=download(f.sim.views().task_output.transition_observation.data,12);
    for(int c=0;c<12;++c){context="scalar step="+std::to_string(i)+" field="+std::to_string(c);near(actual[c],obs[c]);near(tr[c],transition[c]);}
    near(download(f.sim.views().task_output.reward.data,1)[0],reward,2e-4f);
    CHECK(download(f.sim.views().task_output.done.data,1)[0]==done);CHECK(download(f.sim.views().task_output.valid.data,1)[0]==1);
    CHECK(download(f.sim.views().episodes.episode.data,1)[0]==episodes);CHECK(download(f.sim.views().episodes.step.data,1)[0]==steps);
    CHECK(download(f.sim.views().episodes.command_time.data,1)[0]==clock);
  }
}
void mixedLanes(){
  Fixture f(3,11,.006125f);const int N=33;std::vector<Mask> active(N,1);active[32]=0;
  configure(f.sim,{ModelConfig{}},std::vector<Index>(N,0),active,box(N),.006125f);
  std::vector<TaskConfig> params(N);
  for(int n=0;n<N;++n){params[n].goal[0]=(n%4)*.1f;params[n].reward_coeff[0]*=1+n%3;
    for(int k=0;k<4;++k){params[n].action_mean[k]+=.01f*n;params[n].action_scale[k]*=1+.001f*n;}}
  configureTask(f.sim,params);
  std::vector<std::unique_ptr<flightlib::Quadrotor>> cpu;
  std::vector<std::uint64_t> episodes(N,1),steps(N,0);std::vector<float> clocks(N,0);episodes[32]=0;
  for(int n=0;n<N;++n){cpu.emplace_back(std::make_unique<flightlib::Quadrotor>(flightlib::QuadrotorDynamics(1.f,.25f)));flightlib::Matrix<3,2> b;b<<-20,20,-20,20,0,20;cpu[n]->setWorldBox(b);
    flightlib::QuadState x;x.setZero();x.p.z()=n%3?5:0;x.t=7+n*.01f;
    Eigen::Quaternionf q(Eigen::AngleAxisf((n-16)*.19f,Eigen::Vector3f::UnitZ())*Eigen::AngleAxisf((n%5-2)*.7f,Eigen::Vector3f::UnitY()));
    x.q(q);x.x[19]=.123f;cpu[n]->reset(x);f.time[n]=x.t;for(int c=0;c<25;++c)f.state[c*N+n]=x.x[c];}
  f.syncInputs();resetEnvironment(f.sim,f.inputs);f.sim.synchronize();
  for(int i=0;i<80;++i){
    for(int n=0;n<N;++n)for(int c=0;c<4;++c)f.action[4*n+c]=n%3==0?-1.f:float((n+c+i)%9-4)*.12f;
    bool invalid=i==5;if(invalid)f.action[8]=NAN;f.syncInputs();
    stepEnvironment(f.sim,f.inputs,ResetSource::Explicit,i%2?TaskExecution::SeparateReset:TaskExecution::FusedReset);f.sim.synchronize();
    auto obs=download(f.sim.views().task_output.observation.data,12*N),tr=download(f.sim.views().task_output.transition_observation.data,12*N);
    auto rewards=download(f.sim.views().task_output.reward.data,N),state=download(f.sim.views().state.x.data,25*N);
    auto valid=download(f.sim.views().task_output.valid.data,N),done=download(f.sim.views().task_output.done.data,N);
    auto times=download(f.sim.views().state.time.data,N),cmdtimes=download(f.sim.views().episodes.command_time.data,N);
    for(int n=0;n<N;++n){
      if(!active[n]||(invalid&&n==2)){CHECK(!valid[n]&&!done[n]&&rewards[n]==0);for(int c=0;c<12;++c)CHECK(obs[12*n+c]==0&&tr[12*n+c]==0);continue;}
      flightlib::Vector<4> raw,force;for(int c=0;c<4;++c){raw[c]=f.action[n*4+c];force[c]=raw[c]*params[n].action_scale[c]+params[n].action_mean[c];}
      clocks[n]+=.006125f;CHECK(cpu[n]->run(flightlib::Command(clocks[n],force),.006125f));++steps[n];
      flightlib::QuadState x;CHECK(cpu[n]->getState(&x));flightlib::Vector<12> expected;expected<<x.p,x.q().toRotationMatrix().eulerAngles(2,1,0),x.v,x.w;
      auto goal=expected.eval();for(int k=0;k<12;++k)goal[k]-=params[n].goal[k];
      float r=.1f;for(int group=0;group<4;++group)r+=params[n].reward_coeff[group]*goal.segment<3>(group*3).squaredNorm();r+=params[n].reward_coeff[4]*raw.norm();
      bool terminal=x.p.z()<=.02f;CHECK(done[n]==terminal&&valid[n]);if(terminal)r-=.02f;
      for(int c=0;c<12;++c){context="mixed step="+std::to_string(i)+" lane="+std::to_string(n)+" field="+std::to_string(c);near(tr[12*n+c],expected[c]);}near(rewards[n],r);
      if(terminal){for(int c=0;c<25;++c)x.x[c]=f.state[c*N+n];x.t=f.time[n];cpu[n]->reset(x);expected<<x.p,x.q().toRotationMatrix().eulerAngles(2,1,0),x.v,x.w;++episodes[n];steps[n]=0;clocks[n]=0;}
      for(int c=0;c<12;++c){context="returned step="+std::to_string(i)+" lane="+std::to_string(n)+" field="+std::to_string(c);near(obs[12*n+c],expected[c]);}for(int c=0;c<25;++c){context="state step="+std::to_string(i)+" lane="+std::to_string(n)+" field="+std::to_string(c);near(state[c*N+n],x.x[c]);}
      CHECK(times[n]==x.t&&cmdtimes[n]==clocks[n]);
    }
    CHECK(download(f.sim.views().episodes.episode.data,N)==episodes);CHECK(download(f.sim.views().episodes.step.data,N)==steps);
  }
  // Failed automatic reset retains the terminal physical state, zeroes outputs,
  // and does not count a successful reset. Neighbors continue normally.
  f.state[0]=NAN;f.syncInputs();auto before=download(f.sim.views().episodes.episode.data,N);
  stepEnvironment(f.sim,f.inputs,ResetSource::Explicit);f.sim.synchronize();
  CHECK(download(f.sim.views().diagnostics.status.data,N)[0]==LaneStatus::InvalidReset);
  CHECK(download(f.sim.views().episodes.episode.data,N)[0]==before[0]);CHECK(!download(f.sim.views().task_output.valid.data,N)[0]);
  // Invalid schema/owner/config reject before launching or changing task parameters.
  Fixture other(3,11);rejects([&]{stepEnvironment(other.sim,f.inputs);});
  auto bad=TaskConfig{};bad.goal[0]=NAN;rejects([&]{configureTask(f.sim,{bad});});
}
void graphParity(){
  Fixture a(3,11,.006125f,123),b(3,11,.006125f,123);const int N=33;
  for(int n=0;n<N;++n){a.state[2*N+n]=b.state[2*N+n]=n%2?5:0;}
  a.syncInputs();b.syncInputs();resetEnvironment(a.sim,a.inputs);resetEnvironment(b.sim,b.inputs);
  b.sim.synchronize();auto before=download(b.sim.views().state.x.data,25*N);
  Graph graph=captureEnvironment(b.sim,b.inputs,ResetSource::Random);
  CHECK(download(b.sim.views().state.x.data,25*N)==before);
  for(int i=0;i<30;++i){
    for(int n=0;n<N*4;++n)a.action[n]=b.action[n]=float((n+i)%9-4)*.1f;
    if(i==3)a.action[0]=b.action[0]=NAN;
    a.syncInputs();b.syncInputs();
    if(i==8){resetEnvironment(a.sim,a.inputs);resetEnvironment(b.sim,b.inputs);}
    if(i==10){configureTask(a.sim,{TaskConfig{}});configureTask(b.sim,{TaskConfig{}});
      rejects([&]{b.sim.replay(graph);});b.sim.synchronize();graph=captureEnvironment(b.sim,b.inputs,ResetSource::Random);}
    stepEnvironment(a.sim,a.inputs,ResetSource::Random);b.sim.replay(graph);a.sim.synchronize();b.sim.synchronize();
    for(const char* name:{"state.x","state.time","state.motor_omega","state.motor_thrust","commands.payload",
                         "task_output.observation","task_output.transition_observation","task_output.reward"}){
      auto it=std::find_if(a.sim.allocationPlan().buffers.begin(),a.sim.allocationPlan().buffers.end(),[&](auto& x){return x.name==name;});
      CHECK(download(static_cast<float*>(a.sim.buffer(name)),it->bytes/4)==download(static_cast<float*>(b.sim.buffer(name)),it->bytes/4));}
    CHECK(download(a.sim.views().task_output.done.data,N)==download(b.sim.views().task_output.done.data,N));
    CHECK(download(a.sim.views().task_output.valid.data,N)==download(b.sim.views().task_output.valid.data,N));
    CHECK(download(a.sim.views().episodes.episode.data,N)==download(b.sim.views().episodes.episode.data,N));
    CHECK(download(a.sim.views().episodes.random_counter.data,N)==download(b.sim.views().episodes.random_counter.data,N));
  }
  rejects([&]{a.sim.replay(graph);});
}
void failureIsolation(){
  Fixture f(1,3);for(int n=0;n<3;++n)f.state[2*3+n]=5;f.state[10*3+2]=1e30f;
  f.syncInputs();resetEnvironment(f.sim,f.inputs);f.sim.synchronize();
  f.action[0]=NAN;upload(f.ai.tensor.data,f.action);
  auto before=download(f.sim.views().state.x.data,25*3);before[1]=NAN;upload(f.sim.buffer("state.x"),before);
  stepEnvironment(f.sim,f.inputs,ResetSource::Random);f.sim.synchronize();
  auto status=download(f.sim.views().diagnostics.status.data,3);
  CHECK(status[0]==LaneStatus::InvalidCommand&&status[1]==LaneStatus::InvalidState&&status[2]==LaneStatus::NumericalFailure);
  CHECK(download(f.sim.views().task_output.valid.data,3)==std::vector<Mask>(3,0));
  CHECK(download(f.sim.views().task_output.done.data,3)==std::vector<Mask>(3,0));
  CHECK(download(f.sim.views().task_output.reward.data,3)==std::vector<float>(3,0));
  CHECK(download(f.sim.views().episodes.episode.data,3)==std::vector<std::uint64_t>(3,1));
  CHECK(download(f.sim.views().episodes.random_counter.data,3)==std::vector<std::uint64_t>(3,0));
  auto after=download(f.sim.views().state.x.data,25*3);for(int k=0;k<25*3;++k)CHECK((std::isnan(before[k])&&std::isnan(after[k]))||before[k]==after[k]);
}
void randomObservationParity(){
  Fixture f(1,1025,0,293);resetEnvironment(f.sim,f.inputs,ResetSource::Random);f.sim.synchronize();
  auto x=download(f.sim.views().state.x.data,25*1025),obs=download(f.sim.views().task_output.observation.data,12*1025);
  for(int n=0;n<1025;++n){Eigen::Quaternionf q(x[3*1025+n],x[4*1025+n],x[5*1025+n],x[6*1025+n]);
    auto e=q.toRotationMatrix().eulerAngles(2,1,0);for(int k=0;k<3;++k){context="random Euler lane="+std::to_string(n);near(obs[12*n+3+k],e[k],2e-6f);}}
}
void rngAndBranches(){
  Fixture a(3,11,0,123),b(1,33,0,123),c(3,11,0,124);const int N=33;
  resetEnvironment(a.sim,a.inputs,ResetSource::Random);resetEnvironment(c.sim,c.inputs,ResetSource::Random);
  // Same logical lanes, partitioned selection and different E/A grouping.
  for(int n=0;n<N;++n)b.selected[n]=n%2;b.syncInputs();resetEnvironment(b.sim,b.inputs,ResetSource::Random);b.sim.synchronize();
  for(int n=0;n<N;++n)b.selected[n]=1-n%2;b.syncInputs();resetEnvironment(b.sim,b.inputs,ResetSource::Random);
  a.sim.synchronize();b.sim.synchronize();c.sim.synchronize();
  CHECK(download(a.sim.views().state.x.data,25*N)==download(b.sim.views().state.x.data,25*N));
  CHECK(download(a.sim.views().state.x.data,25*N)!=download(c.sim.views().state.x.data,25*N));
  CHECK(download(a.sim.views().episodes.random_counter.data,N)==std::vector<std::uint64_t>(N,10));
  auto x=download(a.sim.views().state.x.data,25*N);for(int n=0;n<N;++n){CHECK(x[2*N+n]>=4&&x[2*N+n]<=6);float norm=0;for(int k=3;k<7;++k)norm+=x[k*N+n]*x[k*N+n];near(norm,1,1e-6f);}
  // Explicit reset cannot rewind RNG. Auto reset consumes only terminal lanes.
  for(int n=0;n<N;++n){a.state[2*N+n]=n%2?5:0;b.state[2*N+n]=n%2?5:0;b.selected[n]=1;}
  a.syncInputs();b.syncInputs();resetEnvironment(a.sim,a.inputs);resetEnvironment(b.sim,b.inputs);
  stepEnvironment(a.sim,a.inputs,ResetSource::Random);stepEnvironment(b.sim,b.inputs,ResetSource::Random,TaskExecution::SeparateReset);
  a.sim.synchronize();b.sim.synchronize();CHECK(download(a.sim.views().state.x.data,25*N)==download(b.sim.views().state.x.data,25*N));
  auto counters=download(a.sim.views().episodes.random_counter.data,N);for(int n=0;n<N;++n)CHECK(counters[n]==(n%2?10:20));
  // Euler branch boundaries checked directly against Eigen, including gimbal lock.
  for(int n=0;n<N;++n){float yaw=n%3==0?-1e-6f:n%3==1?0:3.1415926f;
    Eigen::Quaternionf q(Eigen::AngleAxisf(yaw,Eigen::Vector3f::UnitZ())*Eigen::AngleAxisf((n%5-2)*1.57079632679f,Eigen::Vector3f::UnitY()));
    a.state[3*N+n]=q.w();a.state[4*N+n]=q.x();a.state[5*N+n]=q.y();a.state[6*N+n]=q.z();}
  a.syncInputs();resetEnvironment(a.sim,a.inputs);a.sim.synchronize();auto obs=download(a.sim.views().task_output.observation.data,12*N);
  for(int n=0;n<N;++n){Eigen::Quaternionf q(a.state[3*N+n],a.state[4*N+n],a.state[5*N+n],a.state[6*N+n]);auto e=q.toRotationMatrix().eulerAngles(2,1,0);for(int k=0;k<3;++k)near(obs[12*n+3+k],e[k],2e-6f);}
}
int main(){try{scalarEnv();scalarEnv(true);mixedLanes();rngAndBranches();randomObservationParity();failureIsolation();graphParity();std::cout<<"Environment parity and lifecycle passed\n";return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
