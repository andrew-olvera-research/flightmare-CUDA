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
  int prior{};
  explicit DeviceGuard(int device) {
    check(cudaGetDevice(&prior), "get device");
    check(cudaSetDevice(device), "select device");
  }
  ~DeviceGuard() { cudaSetDevice(prior); }
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
  cudaEvent_t tail{};
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
    if (stream) cudaStreamDestroy(stream);
    external_owners.clear(); // Release registered allocations only after completion.
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
void Simulation::enqueueOn(cudaStream_t stream, Launch launch, void* context, const Bindings* inputs) {
  auto& s = *storage_;
  std::unique_lock<std::mutex> lock(s.mutex, std::try_to_lock);
  if (!lock.owns_lock()) throw std::logic_error("concurrent simulation submission");
  if (s.poisoned) throw std::logic_error("simulation launch failed; destroy and recreate runtime");
  if (!launch || !stream || (inputs && inputs->owner_ != storage_))
    throw std::invalid_argument("invalid launch, stream, or foreign bindings");
  DeviceGuard guard(s.device);
  int stream_device = -1;
  check(cudaStreamGetDevice(stream, &stream_device), "inspect stream device");
  unsigned flags = 0;
  check(cudaStreamGetFlags(stream, &flags), "inspect stream flags");
  if (stream_device != s.device || !(flags & cudaStreamNonBlocking))
    throw std::invalid_argument("stream must be explicit, nonblocking, and on simulation device");
  cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
  check(cudaStreamIsCapturing(stream, &capture), "inspect stream capture");
  if (capture != cudaStreamCaptureStatusNone)
    throw std::invalid_argument("graph capture is not supported by the CUDA-2 runtime");
  static const std::vector<BindingRequest> empty;
  // Reusing the event is safe: each wait snapshots the preceding record, before
  // the next record. Host serialization prevents record/wait races.
  check(cudaStreamWaitEvent(stream, s.tail, 0), "wait for previous simulation use");
  try {
    launch(s.views, stream, inputs ? inputs->inputs_ : empty, context);
    check(cudaGetLastError(), "simulation launch");
    check(cudaEventRecord(s.tail, stream), "record simulation completion");
    ++s.submissions;
  } catch (...) {
    s.poisoned = true; // Arbitrary callbacks cannot provide transactional rollback.
    if (cudaEventRecord(s.tail, stream) != cudaSuccess) cudaStreamSynchronize(stream);
    throw;
  }
}
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
