#include "flightlib_cuda/lifecycle.hpp"
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
int main(){try{
  const auto path=std::string(getenv("FLIGHTMARE_PATH"))+"/flightlib/configs/quadrotor_env.yaml";
  auto configured=loadModel(path);ModelConfig normal;
  auto table=prepareModels({normal,configured});
  flightlib::QuadrotorDynamics d0(normal.mass,normal.arm_length),d1;
  CHECK(d1.updateParams(YAML::LoadFile(path)));
  std::vector<flightlib::QuadrotorDynamics*> models{&d0,&d1};
  auto near=[](float a,float b){CHECK(std::abs(a-b)<=2e-6f*std::max(1.0f,std::abs(b)));};
  for(int p=0;p<2;++p){auto& d=*models[p];
    CHECK(table["models.mass"][p]==d.getMass());
    CHECK(table["models.thrust_bounds"][2+p]==d.clampThrust(1000));
    near(table["models.motor_tau_inverse"][p],d.getMotorTauInv());
    for(int k=0;k<3;++k){near(table["models.inertia"][k*2+p],d.getJ()(k,k));near(table["models.inverse_inertia"][k*2+p],d.getJInv()(k,k));}
    auto allocation=d.getAllocationMatrix();flightlib::Matrix<4,4> inverse=allocation.inverse();
    for(int r=0;r<4;++r)for(int c=0;c<4;++c){near(table["models.allocation"][(r*4+c)*2+p],allocation(r,c));near(table["models.inverse_allocation"][(r*4+c)*2+p],inverse(r,c));}
  }
  auto schedule=prepareSchedule(0.006125f);CHECK(schedule.size()==3);
  CHECK(schedule[0]==0.0025f&&schedule[1]==0.0025f&&schedule[2]==(0.006125f-0.0025f)-0.0025f);
  CHECK(prepareSchedule(0).empty());rejects([]{prepareSchedule(-1);});rejects([]{prepareSchedule(1e10f);});
  rejects([]{prepareSchedule(1,0.001f,2);});
  auto bad=normal;bad.arm_length=0;rejects([&]{prepareModels({bad});});
  constexpr Index N=33;
  RuntimeConfig config;config.environments=3;config.slots=11;config.models=2;config.substeps=3;
  Simulation sim(config);std::vector<Index> ids(N);std::vector<Mask> active(N,1),selected(N,1);
  active[3]=0;selected[2]=0;for(Index n=0;n<N;++n)ids[n]=n%2;
  std::vector<float> bounds(6*N);for(Index c=0;c<6;++c)for(Index n=0;n<N;++n)bounds[c*N+n]=c%2?100:-100;
  configure(sim,{normal,configured},ids,active,bounds,0.006125f);
  CHECK(sim.views().schedule.control_dt==0.006125f);
  CHECK(download(sim.views().schedule.substep_dt.data,3)==schedule);
  for(const auto& entry:table)CHECK(download(static_cast<const float*>(sim.buffer(entry.first)),entry.second.size())==entry.second);
  auto wrong_ids=ids;wrong_ids[0]=2;rejects([&]{configure(sim,{normal,configured},wrong_ids,active,bounds,0.006125f);});
  rejects([&]{configure(sim,{normal,configured},ids,active,bounds,0.0025f);});
  CHECK(download(sim.views().topology.model_id.data,N)==ids);
  std::vector<std::unique_ptr<flightlib::Quadrotor>> cpu;
  std::vector<float> old(25*N),times(N),omega(4*N),thrust(4*N),candidate(25*N),new_times(N,9);
  for(Index n=0;n<N;++n){cpu.emplace_back(std::make_unique<flightlib::Quadrotor>(*models[n%2]));
    flightlib::QuadState state;state.setZero();state.p.z()=5;
    CHECK(cpu.back()->reset(state));CHECK(cpu.back()->run(flightlib::Command(0,flightlib::Vector<4>::Constant(3)),0.01f));
    CHECK(cpu.back()->getState(&state));times[n]=state.t;
    flightlib::Vector<4> w,t;cpu.back()->getMotorOmega(w);cpu.back()->getMotorThrusts(t);
    for(Index c=0;c<25;++c){old[c*N+n]=state.x[c];candidate[c*N+n]=0.1f*(c+1)+n;}
    for(Index c=0;c<4;++c){omega[c*N+n]=w[c];thrust[c*N+n]=t[c];}
  }
  upload(sim.buffer("state.x"),old);upload(sim.buffer("state.time"),times);
  upload(sim.buffer("state.motor_omega"),omega);upload(sim.buffer("state.motor_thrust"),thrust);
  candidate[1]=NAN;candidate[2]=NAN; // invalid selected lane and ignored lane
  auto selection=input(selected,DType::UInt8,{N});
  auto xs=input(candidate,DType::Float32,{25,N}),ts=input(new_times,DType::Float32,{N});
  auto state_inputs=bindStateInput(sim,selection,xs,ts);
  setState(sim,state_inputs);sim.synchronize();
  auto actual=download(sim.views().state.x.data,25*N);
  auto actual_time=download(sim.views().state.time.data,N);
  for(Index n=0;n<N;++n){flightlib::QuadState state;for(Index c=0;c<25;++c)state.x[c]=candidate[c*N+n];state.t=9;
    if(active[n]&&selected[n]) CHECK(cpu[n]->setState(state)==(n!=1));
    CHECK(cpu[n]->getState(&state));CHECK(actual_time[n]==state.t);for(Index c=0;c<25;++c)CHECK(actual[c*N+n]==state.x[c]);}
  CHECK(download(sim.views().state.motor_omega.data,4*N)==omega);
  resetState(sim,state_inputs);sim.synchronize();
  auto speeds=download(sim.views().state.motor_omega.data,4*N);
  auto forces=download(sim.views().state.motor_thrust.data,4*N);
  for(Index n=0;n<N;++n){flightlib::QuadState state;for(Index c=0;c<25;++c)state.x[c]=candidate[c*N+n];state.t=9;
    if(active[n]&&selected[n])CHECK(cpu[n]->reset(state)==(n!=1));
    flightlib::Vector<4> w,t;cpu[n]->getMotorOmega(w);cpu[n]->getMotorThrusts(t);
    for(Index c=0;c<4;++c){CHECK(speeds[c*N+n]==w[c]);CHECK(forces[c*N+n]==t[c]);}}
  auto statuses=download(sim.views().diagnostics.status.data,N);
  CHECK(statuses[1]==LaneStatus::InvalidReset&&statuses[3]==LaneStatus::Inactive);
  CHECK(download(sim.views().episodes.episode.data,N)==std::vector<std::uint64_t>(N,0));
  std::vector<CommandMode> modes(N);std::vector<float> ct(N),payload(4*N);
  for(Index n=0;n<N;++n){LegacyCommand c;c.time=3;
    flightlib::Command oracle;
    if(n%2){c.collective=1000;c.rates={30,-40,50};oracle=flightlib::Command(3,1000,flightlib::Vector<3>(30,-40,50));}
    else {c.thrusts={-1,2,100,4};oracle=flightlib::Command(3,flightlib::Vector<4>(-1,2,100,4));}
    if(n==1){c.time=NAN;oracle.t=NAN;}if(n==4){c.collective=1;c.rates={1,1,1};oracle.collective_thrust=1;oracle.omega.setOnes();}
    auto command=adaptCommand(c);CHECK((command.mode!=CommandMode::Invalid)==oracle.valid());
    modes[n]=command.mode;ct[n]=command.time;for(Index k=0;k<4;++k)payload[k*N+n]=command.payload[k];
    if(active[n]&&selected[n])CHECK(cpu[n]->setCommand(oracle)==oracle.valid());
  }
  auto commands=bindCommandInput(sim,selection,input(modes,DType::UInt8,{N}),input(ct,DType::Float32,{N}),input(payload,DType::Float32,{4,N}));
  setCommand(sim,commands);sim.synchronize();
  auto accepted=download(sim.views().commands.payload.data,4*N);auto accepted_modes=download(sim.views().commands.mode.data,N);
  for(Index n=0;n<N;++n){if(!active[n]||!selected[n]||modes[n]==CommandMode::Invalid){CHECK(accepted_modes[n]==CommandMode::Invalid);continue;}
    CHECK(accepted_modes[n]==modes[n]);auto& d=*models[n%2];
    if(n%2){CHECK(accepted[n]==d.clampThrust(1000));auto rates=d.clampBodyrates(flightlib::Vector<3>(30,-40,50));for(Index k=1;k<4;++k)CHECK(accepted[k*N+n]==rates[k-1]);}
    else {auto t=d.clampThrust(flightlib::Vector<4>(-1,2,100,4));for(Index k=0;k<4;++k)CHECK(accepted[k*N+n]==t[k]);}}
  resetState(sim,state_inputs);sim.synchronize();CHECK(download(sim.views().commands.payload.data,4*N)==accepted);
  // A later invalid command must retain an already accepted command, not merely
  // leave a zero-initialized slot untouched.
  std::vector<CommandMode> invalid_modes(N,CommandMode::Invalid);
  auto invalid_commands=bindCommandInput(sim,selection,input(invalid_modes,DType::UInt8,{N}),
    input(ct,DType::Float32,{N}),input(payload,DType::Float32,{4,N}));
  setCommand(sim,invalid_commands);sim.synchronize();
  CHECK(download(sim.views().commands.payload.data,4*N)==accepted);
  CHECK(download(sim.views().commands.mode.data,N)==accepted_modes);
  rejects([&]{setCommand(sim,state_inputs);});
  auto before_update=download(sim.views().state.x.data,25*N);
  auto motors_before_update=download(sim.views().state.motor_omega.data,4*N);
  normal.mass=1.2f;
  configure(sim,{normal,configured},ids,active,bounds,0.006125f);
  CHECK(download(sim.views().models.mass.data,2)[0]==1.2f);
  CHECK(download(sim.views().state.x.data,25*N)==before_update);
  CHECK(download(sim.views().state.motor_omega.data,4*N)==motors_before_update);
  CHECK(download(sim.views().commands.payload.data,4*N)==accepted);
  Simulation zero(RuntimeConfig{});
  configure(zero,{normal},{0},{1},{-1,1,-1,1,-1,1},0);
  CHECK(zero.views().schedule.control_dt==0&&zero.views().schedule.substep_dt.size()==0);
  std::cout<<"CPU/GPU model, lifecycle, command and schedule parity passed (33 lanes)\n";
  return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
