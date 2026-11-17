#include <flightlib_cuda/dynamics.hpp>
#include <iostream>
using namespace flightlib_cuda::v1;
int main(){
  RuntimeConfig config;config.environments=2;config.slots=3;config.substeps=prepareSchedule(.01f).size();
  Simulation sim(config);
  std::vector<float> bounds;for(float v:{-100.f,100.f,-100.f,100.f,-100.f,100.f})for(int n=0;n<6;++n)bounds.push_back(v);
  configure(sim,{ModelConfig{}},std::vector<Index>(6,0),std::vector<Mask>(6,1),bounds,.01f);
  // No accepted command yet: the installed core must reject, not advance time.
  auto graph=captureDynamics(sim);sim.replay(graph);sim.synchronize();
  LaneStatus status;float time;
  if(cudaMemcpy(&status,sim.views().diagnostics.status.data,sizeof(status),cudaMemcpyDeviceToHost)!=cudaSuccess ||
     cudaMemcpy(&time,sim.views().state.time.data,sizeof(time),cudaMemcpyDeviceToHost)!=cudaSuccess ||
     status!=LaneStatus::InvalidCommand || time!=0)return 1;
  std::cout<<"Installed native core consumer passed\n";
}
