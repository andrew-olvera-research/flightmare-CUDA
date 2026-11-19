#include "flightlib_cuda/environment.hpp"
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <map>
#include <string>

using namespace flightlib_cuda::v1;
using Clock=std::chrono::steady_clock;
static bool watching=false;
enum Counter {Alloc,Free,Copy,Sync,Launch,GraphLaunch,Wait,Record,GetDevice,SetDevice,StreamDevice,Flags,Capture,LastError,Count};
static unsigned long long calls[Count]{};
const char* names[]={"allocations","frees","copies","host_syncs","kernel_launches","graph_launches","event_waits","event_records","get_device","set_device","stream_device","stream_flags","capture_queries","last_error"};
#define WRAP(name,args,values,counter) extern "C" cudaError_t __real_##name args; \
 extern "C" cudaError_t __wrap_##name args {if(watching)++calls[counter];return __real_##name values;}
WRAP(cudaMalloc,(void** p,size_t n),(p,n),Alloc)
WRAP(cudaMallocAsync,(void** p,size_t n,cudaStream_t s),(p,n,s),Alloc)
WRAP(cudaFree,(void* p),(p),Free)
WRAP(cudaFreeAsync,(void* p,cudaStream_t s),(p,s),Free)
WRAP(cudaMemcpy,(void* d,const void* s,size_t n,cudaMemcpyKind k),(d,s,n,k),Copy)
WRAP(cudaMemcpyAsync,(void* d,const void* s,size_t n,cudaMemcpyKind k,cudaStream_t t),(d,s,n,k,t),Copy)
WRAP(cudaDeviceSynchronize,(),(),Sync)
WRAP(cudaStreamSynchronize,(cudaStream_t s),(s),Sync)
WRAP(cudaEventSynchronize,(cudaEvent_t e),(e),Sync)
WRAP(cudaLaunchKernel,(const void* f,dim3 g,dim3 b,void** a,size_t m,cudaStream_t s),(f,g,b,a,m,s),Launch)
WRAP(__cudaLaunchKernel,(cudaKernel_t f,dim3 g,dim3 b,void** a,size_t m,cudaStream_t s),(f,g,b,a,m,s),Launch)
WRAP(cudaGraphLaunch,(cudaGraphExec_t g,cudaStream_t s),(g,s),GraphLaunch)
WRAP(cudaStreamWaitEvent,(cudaStream_t s,cudaEvent_t e,unsigned f),(s,e,f),Wait)
WRAP(cudaEventRecord,(cudaEvent_t e,cudaStream_t s),(e,s),Record)
WRAP(cudaGetDevice,(int* p),(p),GetDevice)
WRAP(cudaSetDevice,(int d),(d),SetDevice)
WRAP(cudaStreamGetDevice,(cudaStream_t s,int* d),(s,d),StreamDevice)
WRAP(cudaStreamGetFlags,(cudaStream_t s,unsigned* f),(s,f),Flags)
WRAP(cudaStreamIsCapturing,(cudaStream_t s,cudaStreamCaptureStatus* p),(s,p),Capture)
WRAP(cudaGetLastError,(),(),LastError)
#undef WRAP
void check(cudaError_t r){if(r!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(r));}
double ms(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::milli>(b-a).count();}
template<class T> BindingRequest input(const std::vector<T>& v,DType type,std::initializer_list<Offset> shape){
 void* p=nullptr;check(cudaMalloc(&p,v.size()*sizeof(T)));check(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));
 auto owner=std::shared_ptr<void>(p,[](void* x){cudaFree(x);});TensorSpec spec{type,static_cast<Index>(shape.size()),{}};
 std::copy(shape.begin(),shape.end(),spec.shape.begin());std::array<Offset,3> strides{};Offset stride=1;
 for(Index i=spec.rank;i-->0;){strides[i]=stride;stride*=spec.shape[i];}return {{p,spec,strides},spec,owner};
}
__global__ void touch(Scalar* state,Index n){Index i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)state[i]+=1;}
void touchLaunch(const RuntimeViews& v,cudaStream_t s,const std::vector<BindingRequest>&,void*){
 touch<<<(v.topology.batch.lanes+127)/128,128,0,s>>>(v.state.x.data,v.topology.batch.lanes);
}
void marker(const RuntimeViews&,cudaStream_t s,const std::vector<BindingRequest>&,void* p){check(cudaEventRecord(*static_cast<cudaEvent_t*>(p),s));}
int main(int argc,char** argv){try{
 if(argc!=11)throw std::invalid_argument("E A iterations samples warmup workload active_percent trace control_dt reset_percent");
 Index E=std::stoul(argv[1]),A=std::stoul(argv[2]);int iterations=std::stoi(argv[3]),samples=std::stoi(argv[4]),warmup=std::stoi(argv[5]);
 std::string workload=argv[6];int percent=std::stoi(argv[7]);bool trace=std::stoi(argv[8]);float control_dt=std::stof(argv[9]);int reset_percent=std::stoi(argv[10]);bool environment=workload.rfind("environment",0)==0;
 if(iterations<1||samples<1||warmup<0||percent<0||percent>100||reset_percent<0||reset_percent>100)throw std::invalid_argument("invalid measurement dimensions");
 if(workload!="direct"&&workload!="runtime"&&workload!="dispatch"&&workload!="state"&&workload!="command"&&workload!="mixed"&&workload!="dynamics"&&workload!="environment"&&workload!="environment-split"&&workload!="environment-random"&&workload!="environment-graph")throw std::invalid_argument("unknown workload");
 auto setup_begin=Clock::now();RuntimeConfig config;config.environments=E;config.slots=A;config.substeps=(workload=="dynamics"||environment)?prepareSchedule(control_dt).size():0;
 Simulation sim(config);Index N=sim.views().topology.batch.lanes;
 std::vector<Mask> active(N),selected(N,1);for(Index i=0;i<N;++i)active[i]=(i%100)<static_cast<Index>(percent);
 std::vector<Scalar> bounds(6ULL*N);for(Index c=0;c<6;++c)for(Index i=0;i<N;++i)bounds[c*N+i]=(c%2)?100:-100;
 if(environment)for(Index i=0;i<N;++i)bounds[4ULL*N+i]=0;
 configure(sim,{ModelConfig{}},std::vector<Index>(N,0),active,bounds,(workload=="dynamics"||environment)?control_dt:0);
 std::vector<Scalar> state(25ULL*N,0),time(N,2),payload(4ULL*N,3);
 for(Index i=0;i<N;++i){state[i]=workload=="dynamics"?0:7;state[2ULL*N+i]=workload=="dynamics"?5:0;state[3ULL*N+i]=1;}
 if(environment)for(Index i=0;i<N;++i){state[i]=0;state[2ULL*N+i]=(i%100)<Index(reset_percent)?0:50;time[i]=0;}
 auto selection=input(selected,DType::UInt8,{N});
 auto states=bindStateInput(sim,selection,input(state,DType::Float32,{25,N}),input(time,DType::Float32,{N}));
 auto commands=bindCommandInput(sim,selection,input(std::vector<CommandMode>(N,CommandMode::RotorThrust),DType::UInt8,{N}),input(time,DType::Float32,{N}),input(payload,DType::Float32,{4,N}));
 if(workload=="dynamics"){resetState(sim,states);setCommand(sim,commands);}
 Bindings task;ResetSource source=workload=="environment-random"?ResetSource::Random:ResetSource::Explicit;
 TaskExecution execution=workload=="environment-split"?TaskExecution::SeparateReset:TaskExecution::FusedReset;
 if(environment){
   configureTask(sim,{TaskConfig{}});std::vector<float> actions(4ULL*N);
   for(Index i=0;i<N;++i)for(int c=0;c<4;++c)actions[4ULL*i+c]=(i%100)<Index(reset_percent)?-1:0;
   task=bindTaskInputs(sim,input(actions,DType::Float32,{E,A,4}),selection,states.inputs()[1],states.inputs()[2]);
   resetEnvironment(sim,task);
 }
 Graph graph;double capture_ms=0;
 if(workload=="environment-graph"){auto begin=Clock::now();graph=captureEnvironment(sim,task,source,execution);capture_ms=ms(begin,Clock::now());}
 cudaStream_t direct;check(cudaStreamCreateWithFlags(&direct,cudaStreamNonBlocking));
 cudaEvent_t start,end;check(cudaEventCreate(&start));check(cudaEventCreate(&end));sim.synchronize();
 const double setup_ms=ms(setup_begin,Clock::now());
 auto step=[&]{
   if(workload=="environment-graph")sim.replay(graph);
   else if(environment)stepEnvironment(sim,task,source,execution);
   else if(workload=="dynamics")stepDynamics(sim);
   else if(workload=="direct")touch<<<(N+127)/128,128,0,direct>>>(sim.views().state.x.data,N);
   else if(workload=="runtime")sim.enqueue(touchLaunch);
   else if(workload=="dispatch")sim.enqueue([](const RuntimeViews&,cudaStream_t,const std::vector<BindingRequest>&,void*){});
   else if(workload=="state")setState(sim,states);
   else if(workload=="command")setCommand(sim,commands);
   else{setState(sim,states);setCommand(sim,commands);resetState(sim,states);}
 };
 for(int i=0;i<warmup;++i)step();check(cudaStreamSynchronize(direct));sim.synchronize();
 std::cout<<"{\"schema_version\":1,\"active_lanes\":"<<std::count(active.begin(),active.end(),Mask{1})<<",\"setup_ms\":"<<setup_ms<<",\"capture_ms\":"<<capture_ms<<",\"arena_bytes\":"<<sim.allocationPlan().arena_bytes
   <<",\"payload_bytes\":"<<sim.allocationPlan().payload_bytes<<",\"external_input_bytes\":"<<(environment?142ULL:126ULL)*N
   <<",\"samples\":[";
 for(int sample=0;sample<samples;++sample){
   if(trace)check(cudaProfilerStart());
   auto begin=Clock::now();
   if(workload=="direct")check(cudaEventRecord(start,direct));else sim.enqueue(marker,&start);
   auto submit_begin=Clock::now();watching=true;
   for(int i=0;i<iterations;++i)step();
   watching=false;auto submitted=Clock::now();
   if(workload=="direct")check(cudaEventRecord(end,direct));else sim.enqueue(marker,&end);
   check(cudaEventSynchronize(end));auto complete=Clock::now();
   if(trace)check(cudaProfilerStop());
   float elapsed=0;check(cudaEventElapsedTime(&elapsed,start,end));
   if(sample)std::cout<<',';
   std::cout<<"{\"submit_ms\":"<<ms(submit_begin,submitted)<<",\"wall_ms\":"<<ms(begin,complete)<<",\"gpu_timeline_ms\":"<<elapsed<<'}';
 }
 check(cudaGetLastError());sim.synchronize();check(cudaStreamSynchronize(direct));
 std::vector<Scalar> actual(N);check(cudaMemcpy(actual.data(),sim.views().state.x.data,N*4,cudaMemcpyDeviceToHost));
 if(workload=="direct"||workload=="runtime")for(auto value:actual)if(value!=static_cast<float>(warmup+iterations*samples))throw std::runtime_error("touch result mismatch");
 if(workload=="state"||workload=="mixed")for(Index i=0;i<N;++i)if(actual[i]!=(active[i]?7:0))throw std::runtime_error("state result mismatch");
 if(workload=="command"||workload=="mixed"){
   check(cudaMemcpy(actual.data(),sim.views().commands.payload.data,N*4,cudaMemcpyDeviceToHost));
   for(Index i=0;i<N;++i)if(actual[i]!=(active[i]?3:0))throw std::runtime_error("command result mismatch");
 }
 if(workload=="dynamics"){
   auto statuses=std::vector<LaneStatus>(N);check(cudaMemcpy(statuses.data(),sim.views().diagnostics.status.data,N*sizeof(LaneStatus),cudaMemcpyDeviceToHost));
   check(cudaMemcpy(actual.data(),sim.views().state.time.data,N*4,cudaMemcpyDeviceToHost));
   float expected=2;for(int i=0;i<warmup+iterations*samples;++i)expected+=control_dt;
   for(Index i=0;i<N;++i)if(statuses[i]!=(active[i]?LaneStatus::Ok:LaneStatus::Inactive)||actual[i]!=(active[i]?expected:0))throw std::runtime_error("dynamics completion mismatch");
 }
 std::uint64_t resets=0,draws=0;unsigned final_done=0;
 if(environment){
   std::vector<Mask> valid(N),done(N);std::vector<LaneStatus> status(N);std::vector<std::uint64_t> episodes(N),rng(N);
   std::vector<float> obs(12ULL*N),reward(N);
   check(cudaMemcpy(valid.data(),sim.views().task_output.valid.data,N,cudaMemcpyDeviceToHost));
   check(cudaMemcpy(done.data(),sim.views().task_output.done.data,N,cudaMemcpyDeviceToHost));
   check(cudaMemcpy(status.data(),sim.views().diagnostics.status.data,N*sizeof(LaneStatus),cudaMemcpyDeviceToHost));
   check(cudaMemcpy(episodes.data(),sim.views().episodes.episode.data,N*8,cudaMemcpyDeviceToHost));
   check(cudaMemcpy(rng.data(),sim.views().episodes.random_counter.data,N*8,cudaMemcpyDeviceToHost));
   check(cudaMemcpy(obs.data(),sim.views().task_output.observation.data,obs.size()*4,cudaMemcpyDeviceToHost));
   check(cudaMemcpy(reward.data(),sim.views().task_output.reward.data,N*4,cudaMemcpyDeviceToHost));
   for(Index n=0;n<N;++n){
     if(valid[n]!=active[n]||status[n]!=(active[n]?LaneStatus::Ok:LaneStatus::Inactive)||!std::isfinite(reward[n])||episodes[n]<active[n]||episodes[n]>std::uint64_t(warmup+iterations*samples+1)||(!active[n]&&(done[n]||reward[n]!=0)))throw std::runtime_error("environment completion mismatch");
     for(int c=0;c<12;++c)if(!std::isfinite(obs[12ULL*n+c])||(!active[n]&&obs[12ULL*n+c]!=0))throw std::runtime_error("invalid environment observation");
     resets+=episodes[n]-active[n];draws+=rng[n];final_done+=done[n];
   }
 }
 std::cout<<"],\"auto_resets_including_warmup\":"<<resets<<",\"random_draws\":"<<draws<<",\"final_done_lanes\":"<<final_done<<",\"calls\":{";for(int i=0;i<Count;++i){if(i)std::cout<<',';std::cout<<'"'<<names[i]<<"\":"<<calls[i];}
 std::cout<<"},\"arena_categories\":{";std::map<std::string,Offset> cats;for(auto& b:sim.allocationPlan().buffers)cats[b.category]+=b.bytes;
 bool first=true;for(auto& c:cats){if(!first)std::cout<<',';first=false;std::cout<<'"'<<c.first<<"\":"<<c.second;}
 std::cout<<"},\"validated\":true}\n";
 check(cudaEventDestroy(start));check(cudaEventDestroy(end));check(cudaStreamDestroy(direct));return 0;
 }catch(const std::exception& e){watching=false;std::cerr<<e.what()<<'\n';return 1;}}
