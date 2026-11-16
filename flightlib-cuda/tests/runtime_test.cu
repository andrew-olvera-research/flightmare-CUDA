#include "flightlib_cuda/runtime.hpp"
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <iostream>
#include <map>
#include <mutex>
#include <thread>

using namespace flightlib_cuda::v1;
#define REQUIRE(x) do { if (!(x)) throw std::runtime_error("line " + std::to_string(__LINE__) + ": " #x); } while (false)
#define CUDA(x) REQUIRE((x) == cudaSuccess)

// Link-time wrappers observe application runtime calls, not driver-internal
// allocations. Enable only during the warmed, repeated launch section.
static bool watching = false;
static bool profile_requested = false;
static int allocations = 0, frees = 0, transfers = 0, synchronizations = 0;
#ifdef FLC_TRACE_RUNTIME_CALLS
#define WRAP(name, args, call, counter) \
  extern "C" cudaError_t __real_##name args; \
  extern "C" cudaError_t __wrap_##name args { if (watching) ++counter; return __real_##name call; }
WRAP(cudaMalloc, (void** p, size_t n), (p,n), allocations)
WRAP(cudaFree, (void* p), (p), frees)
WRAP(cudaMallocAsync, (void** p, size_t n, cudaStream_t s), (p,n,s), allocations)
WRAP(cudaFreeAsync, (void* p, cudaStream_t s), (p,s), frees)
WRAP(cudaMemcpy, (void* d, const void* s, size_t n, cudaMemcpyKind k), (d,s,n,k), transfers)
WRAP(cudaMemcpyAsync, (void* d, const void* s, size_t n, cudaMemcpyKind k, cudaStream_t t), (d,s,n,k,t), transfers)
WRAP(cudaDeviceSynchronize, (), (), synchronizations)
WRAP(cudaStreamSynchronize, (cudaStream_t s), (s), synchronizations)
WRAP(cudaEventSynchronize, (cudaEvent_t e), (e), synchronizations)
#undef WRAP
#endif

template <typename F> void rejects(F f) {
  bool failed = false;
  try { f(); } catch (const std::exception&) { failed = true; }
  REQUIRE(failed);
}
template <typename T, Index R>
std::vector<std::remove_const_t<T>> read(TensorView<T,R> view) {
  std::vector<std::remove_const_t<T>> result(view.size());
  if (!result.empty()) CUDA(cudaMemcpy(result.data(), view.data,
      result.size()*sizeof(T), cudaMemcpyDeviceToHost));
  return result;
}
__global__ void increment(RuntimeViews views) {
  const Index n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < views.topology.batch.lanes && views.topology.active(n)) views.state.x(0,n) += 1;
}
void incrementLaunch(const RuntimeViews& v, cudaStream_t stream,
                     const std::vector<BindingRequest>&, void*) {
  increment<<<(v.topology.batch.lanes+63)/64,64,0,stream>>>(v);
}
void activate(const RuntimeViews& v, cudaStream_t stream,
              const std::vector<BindingRequest>&, void*) {
  CUDA(cudaMemsetAsync(const_cast<Mask*>(v.topology.active.data), 1,
                      v.topology.batch.lanes-1, stream));
}
__global__ void externalWork(RuntimeViews v, Scalar* input, int operation) {
  const Index n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= v.topology.batch.lanes) return;
  if (operation == 0) input[n] = Scalar(n+2);
  if (operation == 1) v.state.x(0,n) = input[n] * 2;
  if (operation == 2) input[n] = v.state.x(0,n) + 1;
}
void externalLaunch(const RuntimeViews& v, cudaStream_t stream,
                    const std::vector<BindingRequest>& inputs, void* context) {
  externalWork<<<(v.topology.batch.lanes+63)/64,64,0,stream>>>(v,
      static_cast<Scalar*>(inputs.at(0).tensor.data), *static_cast<int*>(context));
}
BindingRequest request(void* data, Offset n, std::shared_ptr<void> owner = {}) {
  TensorSpec spec{DType::Float32, 1, {n,0,0}};
  return {{data, spec, {1,0,0}}, spec, std::move(owner)};
}

void testStorageAndLaunch() {
  RuntimeConfig config;
  config.environments=3; config.slots=11; config.models=2; config.substeps=3;
  config.instances=4; config.rays=5; config.candidates_per_environment=7;
  Simulation sim(config);
  sim.synchronize();
  const auto& v=sim.views();
  REQUIRE(v.topology.batch.lanes == 33);
  const auto state=read(v.state.x);
  for (Index c=0;c<25;++c) for (Index n=0;n<33;++n)
    REQUIRE(state[c*33+n] == (c==3 ? 1 : 0));
  for (auto x:read(v.topology.active)) REQUIRE(x==0);
  for (auto x:read(v.commands.mode)) REQUIRE(x==CommandMode::Invalid);
  for (auto x:read(v.diagnostics.status)) REQUIRE(x==LaneStatus::Inactive);
  for (auto x:read(v.task_output.valid)) REQUIRE(x==0);
  for (auto x:read(v.ray_output.distance)) REQUIRE(x==std::numeric_limits<float>::infinity());
  for (auto x:read(v.ray_output.instance)) REQUIRE(x==kInvalidIndex);
  for (auto x:read(v.instances.owner_lane)) REQUIRE(x==kInvalidIndex);
  std::map<std::string,Offset> categories;
  Offset previous=0, payload=0;
  for (const auto& buffer:sim.allocationPlan().buffers) {
    categories[buffer.category]+=buffer.bytes;
    payload+=buffer.bytes;
    if (buffer.bytes) {
      REQUIRE(buffer.offset%256==0 && buffer.offset>=previous);
      REQUIRE(reinterpret_cast<std::uintptr_t>(sim.buffer(buffer.name))%256==0);
      previous=buffer.offset+buffer.bytes;
    }
  }
  REQUIRE(categories["state"]==33*136);
  REQUIRE(payload==sim.allocationPlan().payload_bytes);
  REQUIRE(previous==sim.allocationPlan().arena_bytes);
  for (const auto& entry:categories) std::cout<<entry.first<<" bytes="<<entry.second<<'\n';
  std::cout<<"arena bytes="<<sim.allocationPlan().arena_bytes
           <<" padding="<<sim.allocationPlan().arena_bytes-payload<<'\n';
  sim.enqueue(incrementLaunch); // All inactive; no state change.
  sim.synchronize();
  REQUIRE(read(v.state.x)==state);
  sim.enqueue(activate);
  sim.enqueue(incrementLaunch); // Warmup.
  sim.synchronize();
  auto* stable=v.state.x.data;
  if(profile_requested) CUDA(cudaProfilerStart());
  watching=true;
  for(int i=0;i<128;++i) sim.enqueue(incrementLaunch);
  watching=false;
  sim.synchronize();
  if(profile_requested) CUDA(cudaProfilerStop());
  REQUIRE(v.state.x.data==stable);
  auto after=read(v.state.x);
  for(Index n=0;n<33;++n) REQUIRE(after[n]==(n==32 ? 0 : 129));
#ifdef FLC_TRACE_RUNTIME_CALLS
  REQUIRE(allocations==0 && frees==0 && transfers==0 && synchronizations==0);
  std::cout<<"128 warmed submissions: allocations=0 frees=0 copies=0 host_syncs=0\n";
#endif
  REQUIRE(sim.submissions()==131);
  auto bad=config; bad.memory_budget_bytes=sim.allocationPlan().arena_bytes-1;
  rejects([&]{Simulation failed(bad);});
  bad=config; bad.environments=kInvalidIndex-1; bad.slots=1;
  bad.candidates_per_environment=kInvalidIndex-1;
  rejects([&]{Simulation::plan(bad);}); // Product/byte overflow before device allocation.
  bad=config; bad.environments=kInvalidIndex-1; bad.slots=1; bad.candidates_per_environment=0;
  rejects([&]{Simulation failed(bad);}); // Valid plan, cannot fit on this GPU.
  int device_count=0; CUDA(cudaGetDeviceCount(&device_count));
  bad=config; bad.device=device_count;
  rejects([&]{Simulation failed(bad);});
  REQUIRE(read(v.state.x)==after); // Failed creation cannot mutate an existing owner.
  Simulation empty(RuntimeConfig{});
  REQUIRE(empty.views().rays.origin.data==nullptr);
  REQUIRE(empty.views().geometry.pose.data==nullptr);
  REQUIRE(empty.views().schedule.substep_dt.data==nullptr);
}

void testBindingsAndStreams() {
  RuntimeConfig config; config.environments=3; config.slots=11;
  Simulation sim(config);
  Scalar* external=nullptr; CUDA(cudaMalloc(&external,33*sizeof(Scalar)));
  auto owner=std::shared_ptr<void>(external,[](void* p){cudaFree(p);});
  std::weak_ptr<void> lifetime=owner;
  auto good=request(external,33,owner);
  auto inputs=sim.bind({good});
  auto bad=good; bad.tensor.spec.dtype=DType::UInt32;
  rejects([&]{sim.bind({bad});});
  bad=good; bad.tensor.spec.shape[0]=32;
  rejects([&]{sim.bind({bad});});
  bad=good; bad.tensor.strides[0]=2;
  rejects([&]{sim.bind({bad});});
  bad=good; bad.tensor.data=nullptr;
  rejects([&]{sim.bind({bad});});
  rejects([&]{sim.bind({request(external,34)});});
  rejects([&]{sim.bind({request(external+32,2)});});
  rejects([&]{sim.bind({request(reinterpret_cast<char*>(external)+1,1)});});
  rejects([&]{sim.bind({request(sim.views().state.x.data,33)});});
  rejects([&]{sim.bind({good,request(external+1,32)});});
  Scalar host[33]{};
  rejects([&]{sim.bind({request(host,33)});});
  Scalar* managed=nullptr; CUDA(cudaMallocManaged(&managed,33*sizeof(Scalar)));
  rejects([&]{sim.bind({request(managed,33)});}); CUDA(cudaFree(managed));
  cudaStream_t stream; CUDA(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
  CUDA(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
  rejects([&]{sim.enqueueOn(stream,incrementLaunch);});
  cudaGraph_t graph=nullptr; CUDA(cudaStreamEndCapture(stream,&graph));
  CUDA(cudaGraphDestroy(graph));
  cudaStream_t blocking; CUDA(cudaStreamCreate(&blocking));
  rejects([&]{sim.enqueueOn(blocking,incrementLaunch);});
  rejects([&]{sim.enqueueOn(nullptr,incrementLaunch);});
  CUDA(cudaStreamDestroy(blocking));
  Simulation other(config);
  rejects([&]{other.enqueue(externalLaunch,nullptr,&inputs);});
  int op=0; sim.enqueueOn(stream,externalLaunch,&op,&inputs);
  op=1; sim.enqueue(externalLaunch,&op,&inputs);
  op=2; sim.enqueueOn(stream,externalLaunch,&op,&inputs);
  op=1; sim.enqueue(externalLaunch,&op,&inputs);
  // Registration retains the external owner even after temporary tokens die.
  inputs=Bindings{}; good.lifetime.reset(); bad.lifetime.reset(); owner.reset();
  REQUIRE(!lifetime.expired());
  sim.synchronize();
  const auto state=read(sim.views().state.x);
  for(Index n=0;n<33;++n) REQUIRE(state[n]==Scalar((n+2)*4+2));
  CUDA(cudaStreamDestroy(stream));
  int count=0; CUDA(cudaGetDeviceCount(&count));
  if(count>1) {
    CUDA(cudaSetDevice(1)); Scalar* foreign=nullptr; CUDA(cudaMalloc(&foreign,132));
    rejects([&]{sim.bind({request(foreign,33)});});
    int current=-1; CUDA(cudaGetDevice(&current)); REQUIRE(current==1);
    CUDA(cudaFree(foreign)); CUDA(cudaSetDevice(0));
  } else std::cout<<"Cross-GPU pointer rejection not exercised: one GPU available\n";
}

struct Gate { std::mutex mutex; std::condition_variable cv; bool entered=false, release=false; };
void gateLaunch(const RuntimeViews&,cudaStream_t,const std::vector<BindingRequest>&,void* arg) {
  auto& gate=*static_cast<Gate*>(arg);
  std::unique_lock<std::mutex> lock(gate.mutex);
  gate.entered=true; gate.cv.notify_all();
  gate.cv.wait(lock,[&]{return gate.release;});
}
void throwingLaunch(const RuntimeViews& v,cudaStream_t stream,
                    const std::vector<BindingRequest>& inputs,void* context) {
  externalLaunch(v,stream,inputs,context);
  throw std::runtime_error("test callback failure after launch");
}
void testOwnershipAndFailures() {
  RuntimeConfig config; config.slots=33;
  auto sim=std::make_unique<Simulation>(config);
  auto retained=*sim;
  auto* pointer=sim->views().state.x.data;
  sim.reset();
  REQUIRE(retained.views().state.x.data==pointer);
  retained.enqueue(incrementLaunch);
  Gate gate;
  std::exception_ptr error;
  std::thread worker([&]{try {retained.enqueue(gateLaunch,&gate);} catch(...){error=std::current_exception();}});
  {std::unique_lock<std::mutex> lock(gate.mutex);gate.cv.wait(lock,[&]{return gate.entered;});}
  bool rejected=false;
  try {retained.enqueue(incrementLaunch);} catch(const std::logic_error&){rejected=true;}
  {std::lock_guard<std::mutex> lock(gate.mutex);gate.release=true;gate.cv.notify_all();}
  worker.join(); if(error)std::rethrow_exception(error);
  REQUIRE(rejected);
  retained.synchronize();
  Scalar* external=nullptr; CUDA(cudaMalloc(&external,132));
  cudaStream_t stream; CUDA(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
  {
    Simulation fail(config);
    auto inputs=fail.bind({request(external,33)});
    int op=0;
    rejects([&]{fail.enqueueOn(stream,throwingLaunch,&op,&inputs);});
    rejects([&]{fail.enqueue(incrementLaunch);}); // Partial launch poisons the owner.
  } // Last owner drains external work even on the exception path.
  Vector<Scalar> result{external,{33}};
  auto values=read(result);
  for(Index n=0;n<33;++n) REQUIRE(values[n]==Scalar(n+2));
  CUDA(cudaStreamDestroy(stream)); CUDA(cudaFree(external));
}

int main(int argc, char** argv) {
  try {
    profile_requested = argc==2 && std::string(argv[1])=="--profile";
    testStorageAndLaunch();
    if(!profile_requested) {
      testBindingsAndStreams();
      testOwnershipAndFailures();
    }
    std::cout<<"Runtime storage, validation, ordering, and lifetime checks passed\n";
    return 0;
  } catch(const std::exception& e) { watching=false;std::cerr<<e.what()<<'\n';return 1; }
}
