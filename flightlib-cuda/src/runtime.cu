#include "flightlib_cuda/runtime.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <mutex>

namespace flightlib_cuda::v1 {
namespace {
void check(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    // Report this failure once without leaving a recoverable API error in the
    // calling thread's last-error slot to mislabel a later kernel launch.
    cudaGetLastError();
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
  }
}
struct DeviceGuard {
  int prior{};bool changed{};
  explicit DeviceGuard(int device) {
    check(cudaGetDevice(&prior), "get device");
    if(prior!=device)check(cudaSetDevice(device), "select device");
    changed=prior!=device;
  }
  ~DeviceGuard() { if(changed)cudaSetDevice(prior); }
};
Offset add(Offset a, Offset b) {
  if (b > std::numeric_limits<Offset>::max() - a)
    throw std::overflow_error("runtime byte sum overflow");
  return a + b;
}
Offset multiply(Offset a, Offset b) {
  if (b && a > std::numeric_limits<Offset>::max() / b)
    throw std::overflow_error("runtime extent overflow");
  return a * b;
}
Offset elementBytes(DType type) {
  switch (type) {
    case DType::Float32: case DType::UInt32: return 4;
    case DType::UInt8: return 1;
    case DType::UInt64: return 8;
  }
  throw std::invalid_argument("unknown dtype");
}
Offset bytes(const TensorSpec& spec) {
  if (!spec.rank || spec.rank > 3) throw std::invalid_argument("invalid tensor rank");
  Offset count = 1;
  for (Index d = 0; d < spec.rank; ++d) count = multiply(count, spec.shape[d]);
  return checkedBytes<std::uint8_t>(multiply(count, elementBytes(spec.dtype)));
}
template <typename T> constexpr DType dtype() {
  if constexpr (std::is_same_v<T, Scalar>) return DType::Float32;
  else if constexpr (sizeof(T) == 1) return DType::UInt8;
  else if constexpr (sizeof(T) == 4) return DType::UInt32;
  else { static_assert(sizeof(T) == 8); return DType::UInt64; }
}
template <typename T>
void append(AllocationPlan& plan, const char* name, const char* category,
            std::initializer_list<Offset> dimensions) {
  TensorSpec spec{dtype<T>(), static_cast<Index>(dimensions.size()), {}};
  std::copy(dimensions.begin(), dimensions.end(), spec.shape.begin());
  const auto count = bytes(spec);
  const auto offset = count ? (add(plan.arena_bytes, 255) & ~Offset{255}) : 0;
  plan.buffers.push_back({name, category, spec, offset, count});
  plan.payload_bytes = add(plan.payload_bytes, count);
  if (count) plan.arena_bytes = add(offset, count);
}
template <typename T, Index Rank>
void assign(TensorView<T, Rank>& view, void* arena, const BufferInfo& buffer) {
  view.data = buffer.bytes ? reinterpret_cast<T*>(static_cast<char*>(arena) + buffer.offset) : nullptr;
  for (Index d = 0; d < Rank; ++d) view.shape[d] = buffer.spec.shape[d];
}
__device__ std::uint64_t seedKey(std::uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}
__global__ void initialize(RuntimeViews v, std::uint64_t seed) {
  const Offset n = Offset{blockIdx.x} * blockDim.x + threadIdx.x;
  if (n < v.topology.batch.lanes) {
    v.state.x(3, n) = 1;
    const_cast<Scalar*>(v.reset.x.data)[3 * v.topology.batch.lanes + n] = 1;
    v.diagnostics.status(n) = LaneStatus::Inactive;
    v.episodes.random_key(n) = seedKey(seed + n);
  }
  if (n < v.instances.asset.size()) {
    const_cast<Index*>(v.instances.asset.data)[n] = kInvalidIndex;
    const_cast<Index*>(v.instances.owner_lane.data)[n] = kInvalidIndex;
    const_cast<Scalar*>(v.instances.local_pose.data)[3 * v.instances.asset.size() + n] = 1;
    v.geometry.pose(3, n) = 1;
  }
  if (n < v.ray_output.status.size()) {
    const_cast<Index*>(v.rays.excluded_owner.data)[n] = kInvalidIndex;
    v.ray_output.distance(n) = CUDART_INF_F;
    v.ray_output.instance(n) = kInvalidIndex;
    v.ray_output.primitive(n) = kInvalidIndex;
  }
}
} // namespace

namespace detail {
struct RuntimeStorage {
  int device{};
  AllocationPlan plan;
  RuntimeViews views{};
  void* arena{};
  cudaStream_t stream{};
  cudaEvent_t tail{},ready{};
  cudaStream_t last{};
  std::uint64_t generation{0};
  mutable std::mutex mutex;
  bool poisoned{false};
  std::atomic<std::uint64_t> submissions{0};
  std::vector<std::shared_ptr<void>> external_owners;
  ~RuntimeStorage() {
    int prior = 0;
    cudaGetDevice(&prior);
    cudaSetDevice(device);
    // All registered external streams are chained through tail. Teardown is
    // deliberately a completion boundary; ordinary enqueue never synchronizes.
    auto result = tail ? cudaEventSynchronize(tail) : cudaSuccess;
    if (result != cudaSuccess)
      std::fprintf(stderr, "flightlib-cuda teardown: %s\n", cudaGetErrorString(result));
    if (stream) cudaStreamSynchronize(stream); // also covers partial initialization
    if (arena) cudaFree(arena);
    if (tail) cudaEventDestroy(tail);
    if (ready) cudaEventDestroy(ready);
    if (stream) cudaStreamDestroy(stream);
    external_owners.clear(); // Release registered allocations only after completion.
    cudaSetDevice(prior);
  }
};
struct GraphStorage {
  std::shared_ptr<RuntimeStorage> owner;
  cudaGraph_t graph{};cudaGraphExec_t executable{};std::uint64_t generation{};
  ~GraphStorage(){
    if(!owner)return;
    std::lock_guard<std::mutex> lock(owner->mutex);
    int prior=0;cudaGetDevice(&prior);cudaSetDevice(owner->device);
    cudaEventSynchronize(owner->tail);
    if(executable)cudaGraphExecDestroy(executable);
    if(graph)cudaGraphDestroy(graph);
    cudaSetDevice(prior);
  }
};
} // namespace detail

AllocationPlan Simulation::plan(const RuntimeConfig& config) {
  const auto batch = makeBatchShape(config.environments, config.slots);
  if (!config.models || config.models == kInvalidIndex || config.instances == kInvalidIndex ||
      config.rays == kInvalidIndex || config.substeps == kInvalidIndex ||
      config.candidates_per_environment == kInvalidIndex)
    throw std::invalid_argument("invalid runtime capacity");
  const Offset E = batch.environments, A = batch.slots_per_environment, N = batch.lanes;
  const Offset P = config.models, S = config.substeps, I = config.instances;
  const Offset Q = config.rays, K = config.candidates_per_environment;
  AllocationPlan result;
#define FLC_BUFFER(member, type, category, ...) append<type>(result, #member, category, {__VA_ARGS__});
#include "buffers.def"
#undef FLC_BUFFER
  if (config.memory_budget_bytes && result.arena_bytes > config.memory_budget_bytes)
    throw std::length_error("runtime arena exceeds configured memory budget");
  return result;
}

Simulation::Simulation(const RuntimeConfig& config) {
  auto layout = plan(config); // Validate every extent before allocating anything.
  int device_count = 0;
  check(cudaGetDeviceCount(&device_count), "count devices");
  if (config.device < 0 || config.device >= device_count)
    throw std::invalid_argument("runtime device ordinal is out of range");
  DeviceGuard guard(config.device);
  std::size_t free_bytes = 0, total_bytes = 0;
  check(cudaMemGetInfo(&free_bytes, &total_bytes), "query device memory");
  if (layout.arena_bytes > free_bytes) throw std::bad_alloc();
  auto s = std::make_shared<detail::RuntimeStorage>();
  s->device = config.device;
  s->plan = std::move(layout);
  check(cudaStreamCreateWithFlags(&s->stream, cudaStreamNonBlocking), "create stream");
  check(cudaEventCreateWithFlags(&s->tail, cudaEventDisableTiming), "create completion event");
  check(cudaEventCreateWithFlags(&s->ready, cudaEventDisableTiming), "create producer event");
  check(cudaMalloc(&s->arena, s->plan.arena_bytes), "allocate runtime arena");
  std::size_t cursor = 0;
#define FLC_BUFFER(member, type, category, ...) assign(s->views.member, s->arena, s->plan.buffers.at(cursor++));
#include "buffers.def"
#undef FLC_BUFFER
  s->views.topology.batch = makeBatchShape(config.environments, config.slots);
  check(cudaMemsetAsync(s->arena, 0, s->plan.arena_bytes, s->stream), "initialize arena");
  const Offset count = std::max({Offset{s->views.topology.batch.lanes},
                                 Offset{config.instances}, Offset{config.rays}});
  initialize<<<static_cast<unsigned>((count + 255) / 256), 256, 0, s->stream>>>(s->views, config.seed);
  check(cudaGetLastError(), "initialize views");
  check(cudaEventRecord(s->tail, s->stream), "record initialization");
  s->last=s->stream;
  storage_ = std::move(s);
}

const RuntimeViews& Simulation::views() const { return storage_->views; }
const AllocationPlan& Simulation::allocationPlan() const { return storage_->plan; }
int Simulation::device() const { return storage_->device; }
std::uint64_t Simulation::submissions() const { return storage_->submissions.load(); }
void* Simulation::buffer(const std::string& name) const {
  for (const auto& b : storage_->plan.buffers)
    if (b.name == name) return b.bytes ? static_cast<char*>(storage_->arena) + b.offset : nullptr;
  throw std::invalid_argument("unknown runtime buffer: " + name);
}

Bindings Simulation::bind(std::vector<BindingRequest> requests) const {
  std::unique_lock<std::mutex> lock(storage_->mutex, std::try_to_lock);
  if (!lock.owns_lock()) throw std::logic_error("concurrent simulation binding");
  DeviceGuard guard(device());
  // Validate all requests before returning any registration. No device writes.
  std::vector<std::pair<Offset, Offset>> ranges;
  const auto arena = reinterpret_cast<Offset>(storage_->arena);
  ranges.emplace_back(arena, add(arena, storage_->plan.arena_bytes));
  for (const auto& request : requests) {
    const auto& t = request.tensor;
    const auto& expected = request.expected;
    const auto length = bytes(expected);
    if (t.spec.dtype != expected.dtype || t.spec.rank != expected.rank ||
        t.spec.shape != expected.shape)
      throw std::invalid_argument("external binding shape/dtype mismatch");
    Offset stride = 1;
    for (Index d = expected.rank; d-- > 0;) {
      if (t.strides[d] != stride) throw std::invalid_argument("external binding is not dense");
      stride = multiply(stride, expected.shape[d]);
    }
    if (!length) {
      if (t.data) throw std::invalid_argument("empty binding must be null");
      continue;
    }
    if (!t.data || reinterpret_cast<Offset>(t.data) % elementBytes(expected.dtype))
      throw std::invalid_argument("null or misaligned external binding");
    cudaPointerAttributes attributes{};
    check(cudaPointerGetAttributes(&attributes, t.data), "inspect external pointer");
    if (attributes.type != cudaMemoryTypeDevice || attributes.device != device())
      throw std::invalid_argument("external binding must be device memory on simulation device");
    CUdeviceptr base = 0;
    std::size_t capacity = 0;
    if (cuMemGetAddressRange(&base, &capacity, reinterpret_cast<CUdeviceptr>(t.data)) != CUDA_SUCCESS)
      throw std::invalid_argument("external allocation extent cannot be verified");
    const auto begin = reinterpret_cast<Offset>(t.data), end = add(begin, length);
    if (begin < base || end > add(base, capacity))
      throw std::invalid_argument("external binding exceeds allocation");
    for (const auto& range : ranges)
      if (begin < range.second && range.first < end)
        throw std::invalid_argument("external bindings alias protected storage");
    ranges.emplace_back(begin, end);
  }
  Bindings result;
  result.owner_ = storage_;
  result.inputs_ = std::move(requests);
  // Allocation owners are retained for the simulation lifetime, so destroying a
  // temporary Bindings token immediately after enqueue cannot free in-flight data.
  auto owners = storage_->external_owners;
  for (const auto& request : result.inputs_)
    if (request.lifetime) owners.push_back(request.lifetime);
  storage_->external_owners.swap(owners);
  return result;
}

void Simulation::enqueue(Launch launch, void* context, const Bindings* inputs) {
  enqueueOn(storage_->stream, launch, context, inputs);
}
void Simulation::enqueueOn(cudaStream_t stream,Launch launch,void* context,const Bindings* inputs){
  submit(stream,launch,context,inputs,false,false);
}
void Simulation::enqueueFrom(cudaStream_t producer,Launch launch,void* context,const Bindings* inputs){
  submit(producer,launch,context,inputs,true,false);
}
void Simulation::enqueueSetup(Launch launch,void* context){submit(storage_->stream,launch,context,nullptr,false,true);}
void Simulation::submit(cudaStream_t stream,Launch launch,void* context,const Bindings* inputs,bool bridge,bool setup,const Graph* graph){
  auto& s=*storage_;std::unique_lock<std::mutex> lock(s.mutex,std::try_to_lock);
  if(!lock.owns_lock())throw std::logic_error("concurrent simulation submission");
  if(s.poisoned)throw std::logic_error("simulation launch failed; recreate runtime");
  if(!launch||(!bridge&&!stream)||(inputs&&inputs->owner_!=storage_))throw std::invalid_argument("invalid launch, stream, or foreign bindings");
  if(graph&&(!graph->storage_||graph->storage_->owner!=storage_||graph->storage_->generation!=s.generation))
    throw std::invalid_argument("foreign or stale graph; recapture after configuration changes");
  DeviceGuard guard(s.device);
  if(stream!=s.stream){
    int device=-1;check(cudaStreamGetDevice(stream,&device),"inspect stream device");
    unsigned flags=0;check(cudaStreamGetFlags(stream,&flags),"inspect stream flags");
    if(device!=s.device||(!bridge&&!(flags&cudaStreamNonBlocking)))throw std::invalid_argument("invalid external stream");
  }
  cudaStreamCaptureStatus capture=cudaStreamCaptureStatusNone;
  check(cudaStreamIsCapturing(stream,&capture),"inspect stream capture");
  if(capture!=cudaStreamCaptureStatusNone)throw std::invalid_argument("external capture unsupported; use explicit capture API");
  static const std::vector<BindingRequest> empty;
  cudaStream_t target=bridge?s.stream:stream;
  if(setup)++s.generation;
  try{
    if(bridge){check(cudaEventRecord(s.ready,stream),"record producer");check(cudaStreamWaitEvent(target,s.ready,0),"wait producer");}
    // Elide only on the private stream: external handles can be destroyed/reused.
    if(target!=s.stream||s.last!=s.stream)check(cudaStreamWaitEvent(target,s.tail,0),"wait previous use");
    launch(s.views,target,inputs?inputs->inputs_:empty,context);
    check(cudaGetLastError(),"simulation launch");
    check(cudaEventRecord(s.tail,target),"record completion");s.last=target;
    if(bridge)check(cudaStreamWaitEvent(stream,s.tail,0),"wait simulation output");
    ++s.submissions;
  }catch(...){s.poisoned=true;if(cudaEventRecord(s.tail,target)!=cudaSuccess)cudaStreamSynchronize(target);s.last=target;throw;}
}
Graph Simulation::capture(Launch launch,void* context,const Bindings* inputs){
  auto& s=*storage_;std::unique_lock<std::mutex> lock(s.mutex,std::try_to_lock);
  if(!lock.owns_lock())throw std::logic_error("concurrent graph capture");
  if(s.poisoned)throw std::logic_error("poisoned simulation");
  if(!launch||(inputs&&inputs->owner_!=storage_))throw std::invalid_argument("invalid capture inputs");
  DeviceGuard guard(s.device);
  const auto drained=cudaEventSynchronize(s.tail);
  if(drained!=cudaSuccess)s.poisoned=true;
  check(drained,"capture setup drain");
  // Allocate host ownership before creating CUDA resources, so allocation failure
  // cannot leak an instantiated graph.
  auto graph_storage=std::make_shared<detail::GraphStorage>();
  cudaGraph_t captured=nullptr;cudaGraphExec_t executable=nullptr;
  check(cudaStreamBeginCapture(s.stream,cudaStreamCaptureModeThreadLocal),"begin capture");
  static const std::vector<BindingRequest> empty;
  try{launch(s.views,s.stream,inputs?inputs->inputs_:empty,context);check(cudaGetLastError(),"capture launch");}
  catch(...){cudaStreamEndCapture(s.stream,&captured);if(captured)cudaGraphDestroy(captured);throw;}
  check(cudaStreamEndCapture(s.stream,&captured),"end capture");
  auto result=cudaGraphInstantiate(&executable,captured,0);
  if(result!=cudaSuccess){cudaGraphDestroy(captured);check(result,"instantiate graph");}
  Graph out;out.storage_=std::move(graph_storage);
  out.storage_->owner=storage_;out.storage_->graph=captured;out.storage_->executable=executable;out.storage_->generation=s.generation;
  // No kernel executes during capture. First replay upload costs belong to warmup.
  return out;
}
namespace {
void graphLaunch(const RuntimeViews&,cudaStream_t stream,const std::vector<BindingRequest>&,void* context){
  check(cudaGraphLaunch(static_cast<detail::GraphStorage*>(context)->executable,stream),"graph replay");
}
void noop(const RuntimeViews&,cudaStream_t,const std::vector<BindingRequest>&,void*){}
}
void Simulation::replay(const Graph& graph){submit(storage_->stream,graphLaunch,graph.storage_.get(),nullptr,false,false,&graph);}
void Simulation::replayFrom(const Graph& graph,cudaStream_t stream){submit(stream,graphLaunch,graph.storage_.get(),nullptr,true,false,&graph);}
void Simulation::waitOn(cudaStream_t stream) const{
  auto& s=*storage_;std::unique_lock<std::mutex> lock(s.mutex,std::try_to_lock);
  if(!lock.owns_lock())throw std::logic_error("concurrent consumer wait");
  if(s.poisoned)throw std::logic_error("poisoned simulation");
  DeviceGuard guard(s.device);int device=-1;check(cudaStreamGetDevice(stream,&device),"inspect consumer");
  cudaStreamCaptureStatus status;check(cudaStreamIsCapturing(stream,&status),"inspect consumer capture");
  if(device!=s.device||status!=cudaStreamCaptureStatusNone)throw std::invalid_argument("invalid consumer stream");
  check(cudaStreamWaitEvent(stream,s.tail,0),"consumer wait");
}
void Simulation::recordUsage(cudaStream_t stream){enqueueFrom(stream,noop);}
void Simulation::synchronize() const {
  auto& s = *storage_;
  std::unique_lock<std::mutex> lock(s.mutex, std::try_to_lock);
  if (!lock.owns_lock()) throw std::logic_error("concurrent simulation synchronization");
  DeviceGuard guard(s.device);
  const auto result = cudaEventSynchronize(s.tail);
  if (result != cudaSuccess) s.poisoned = true;
  check(result, "simulation completion");
  if (s.poisoned) throw std::logic_error("simulation is poisoned by a failed launch");
}

} // namespace flightlib_cuda::v1
