#include "flightlib_cuda/dynamics.hpp"
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

#include <chrono>
int main(int argc,char** argv){try{
  if(argc==5){
    int N=std::stoi(argv[1]),steps=std::stoi(argv[2]),warmup=std::stoi(argv[3]),samples=std::stoi(argv[4]);
    std::vector<std::unique_ptr<flightlib::Quadrotor>> cpu;
    for(int n=0;n<N;++n){cpu.emplace_back(std::make_unique<flightlib::Quadrotor>(flightlib::QuadrotorDynamics{}));
      flightlib::QuadState x;x.setZero();x.p.z()=5;x.t=2;cpu.back()->reset(x);
      cpu.back()->setCommand(flightlib::Command(0,flightlib::Vector<4>::Constant(3)));}
    for(int i=0;i<warmup;++i)for(auto& q:cpu)q->run(.01f);
    std::cout<<"{\"samples_us\":[";
    for(int sample=0;sample<samples;++sample){
    auto start=std::chrono::steady_clock::now();
    for(int i=0;i<steps;++i)for(auto& q:cpu)CHECK(q->run(.01f));
    double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    if(sample)std::cout<<',';std::cout<<ms*1000/steps;
    }std::cout<<"]}\n";return 0;
  }
  constexpr int N=33;
  for(float dt:{0.f,.0025f,.006125f,.01f}){
    RuntimeConfig cfg;cfg.environments=3;cfg.slots=11;cfg.models=2;cfg.substeps=prepareSchedule(dt).size();Simulation sim(cfg);
    ModelConfig other;other.mass=1.4f;other.arm_length=.3f;
    std::vector<Index> ids(N);std::vector<Mask> active(N,1);active[32]=0;
    std::vector<float> bounds(6*N),xs(25*N),times(N),payload(4*N);
    std::vector<CommandMode> modes(N);std::vector<std::unique_ptr<flightlib::Quadrotor>> cpu;
    for(int n=0;n<N;++n){ids[n]=n%2;for(int c=0;c<6;++c)bounds[c*N+n]=c%2?10:-10;
      cpu.emplace_back(std::make_unique<flightlib::Quadrotor>(flightlib::QuadrotorDynamics(n%2?1.4f:1.f,n%2?.3f:.25f)));
      flightlib::Matrix<3,2> box;box<<-10,10,-10,10,-10,10;cpu[n]->setWorldBox(box);
      flightlib::QuadState x;x.setZero();x.p.z()=5;x.v<<.1f,-.2f,.3f;x.w<<.1f,.2f,-.1f;
      x.qx[0]=1.01f; x.x[19]=.123f;
      if(n<6)x.p[n/2]=n%2?10.01f:-10.01f;
      cpu[n]->reset(x);for(int c=0;c<25;++c)xs[c*N+n]=x.x[c];
    }
    configure(sim,{ModelConfig{},other},ids,active,bounds,dt);
    auto selection=input(std::vector<Mask>(N,1),DType::UInt8,{N});
    resetState(sim,bindStateInput(sim,selection,input(xs,DType::Float32,{25,N}),input(times,DType::Float32,{N})));
    sim.synchronize();upload(sim.buffer("state.x"),xs);
    auto modeinput=input(std::vector<CommandMode>(N),DType::UInt8,{N});auto pay=input(payload,DType::Float32,{4,N});
    auto commands=bindCommandInput(sim,selection,modeinput,input(times,DType::Float32,{N}),pay);
    for(int step=0;step<300;++step){
      for(int n=0;n<N;++n){bool rates=n%2; modes[n]=rates?CommandMode::CollectiveBodyRates:CommandMode::RotorThrust;
        float sign=step%60<30?1:-1;
        flightlib::Command command=rates?flightlib::Command(0,9.81f,flightlib::Vector<3>(.3f*sign,-.2f*sign,.1f)):
          flightlib::Command(0,flightlib::Vector<4>(3+sign*.1f,3,3,3));
        for(int c=0;c<4;++c)payload[c*N+n]=rates?(c?command.omega[c-1]:command.collective_thrust):command.thrusts[c];
        if(active[n])CHECK(cpu[n]->run(command,dt));
      }
      sim.synchronize();upload(modeinput.tensor.data,modes);upload(pay.tensor.data,payload);setCommand(sim,commands);stepDynamics(sim);sim.synchronize();
      auto actual=download(sim.views().state.x.data,25*N),t=download(sim.views().state.time.data,N);
      auto omega=download(sim.views().state.motor_omega.data,4*N),thrust=download(sim.views().state.motor_thrust.data,4*N);
      auto status=download(sim.views().diagnostics.status.data,N);
      for(int n=0;n<N;++n){flightlib::QuadState x;CHECK(cpu[n]->getState(&x));CHECK(t[n]==x.t);
        CHECK(status[n]==(active[n]?LaneStatus::Ok:LaneStatus::Inactive));
        for(int c=0;c<25;++c){float tol=(c<13?3e-4f:1e-4f)*std::max(1.f,std::abs(x.x[c]));
          if(std::abs(actual[c*N+n]-x.x[c])>tol)throw std::runtime_error("trajectory mismatch dt="+std::to_string(dt)+" step="+std::to_string(step)+" lane="+std::to_string(n)+" field="+std::to_string(c)+" gpu="+std::to_string(actual[c*N+n])+" cpu="+std::to_string(x.x[c]));}
        flightlib::Vector<4> w,f;cpu[n]->getMotorOmega(w);cpu[n]->getMotorThrusts(f);
        for(int c=0;c<4;++c){CHECK(std::abs(omega[c*N+n]-w[c])<.02f);CHECK(std::abs(thrust[c*N+n]-f[c])<1e-4f);}
      }
    }
    // Invalid input rejection is transactional and isolated to the affected lane.
    auto before=download(sim.views().state.x.data,25*N);modes[0]=CommandMode::Invalid;
    upload(sim.buffer("commands.mode"),modes);stepDynamics(sim);sim.synchronize();
    auto after=download(sim.views().state.x.data,25*N);auto status=download(sim.views().diagnostics.status.data,N);
    CHECK(status[0]==LaneStatus::InvalidCommand);for(int c=0;c<25;++c)CHECK(after[c*N]==before[c*N]);
    modes[0]=CommandMode::RotorThrust;upload(sim.buffer("commands.mode"),modes);
    before=after;before[1]=NAN;before[10*N+2]=1e30f;upload(sim.buffer("state.x"),before);
    stepDynamics(sim);sim.synchronize();status=download(sim.views().diagnostics.status.data,N);
    CHECK(status[1]==LaneStatus::InvalidState);
    if(dt>0){CHECK(status[2]==LaneStatus::NumericalFailure);after=download(sim.views().state.x.data,25*N);
      for(int c=0;c<25;++c)CHECK(after[c*N+2]==before[c*N+2]);}
  }
  std::cout<<"Dynamics trajectory parity passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
