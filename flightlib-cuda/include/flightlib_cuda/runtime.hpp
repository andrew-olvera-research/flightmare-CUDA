#pragma once

#include "flightlib_cuda/contracts.hpp"
#include <cuda_runtime_api.h>
#include <array>
#include <memory>
#include <string>
#include <vector>

namespace flightlib_cuda::v1 {

enum class DType { Float32, UInt8, UInt32, UInt64 };
struct TensorSpec {
  DType dtype{};
  Index rank{};
  std::array<Offset, 3> shape{};
};
struct TensorArgument {
  void* data{};
  TensorSpec spec;
  std::array<Offset, 3> strides{}; // Elements, dense row-major only.
};
struct BindingRequest {
  TensorArgument tensor;
  TensorSpec expected;
  // Retains an owner until simulation teardown if supplied. Otherwise the caller must
  // keep device memory alive through completion of every submitted use.
  std::shared_ptr<void> lifetime;
};
struct BufferInfo {
  std::string name;
  std::string category;
  TensorSpec spec;
  Offset offset{};
  Offset bytes{};
};
struct AllocationPlan {
  std::vector<BufferInfo> buffers;
  Offset payload_bytes{};
  Offset arena_bytes{}; // Includes 256-byte field alignment.
};
struct RuntimeConfig {
  Index environments{1};
  Index slots{1};
  Index models{1};
  Index substeps{0};
  Index instances{0};
  Index rays{0};
  Index candidates_per_environment{0};
  int device{0};
  std::uint64_t seed{0};
  Offset memory_budget_bytes{0}; // 0 = no user cap; free memory still checked.
};

struct RuntimeViews {
  QuadState state;
  CommandStorage commands;
  QuadModelView models;
  QuadTopologyView topology;
  StepSchedule schedule;
  ResetInput reset;
  LaneDiagnostics diagnostics;
  TaskParameters task_parameters;
  TaskInput task_input;
  TaskOutput task_output;
  EpisodeState episodes;
  GeometryInstances instances;
  GeometryWorldState geometry;
  RayQueryInput rays;
  RayQueryOutput ray_output;
  CandidatePairs candidates;
  CollisionFlags collisions;
};

namespace detail { struct RuntimeStorage; struct GraphStorage; }
class Graph {
 public:
  Graph() = default;
 private:
  friend class Simulation;
  std::shared_ptr<detail::GraphStorage> storage_;
};

class Bindings {
 public:
  const std::vector<BindingRequest>& inputs() const { return inputs_; }
 private:
  friend class Simulation;
  std::shared_ptr<detail::RuntimeStorage> owner_;
  std::vector<BindingRequest> inputs_;
};

// Copies share ownership of ONE simulation; they do not clone its state.
// Raw RuntimeViews do not retain storage. Keep a Simulation/Bindings handle
// alive until borrowed descriptors and all external accesses are finished.
class Simulation {
 public:
  using Launch = void (*)(const RuntimeViews&, cudaStream_t,
                          const std::vector<BindingRequest>&, void*);
  static AllocationPlan plan(const RuntimeConfig& config); // CPU only.
  explicit Simulation(const RuntimeConfig& config);

  const RuntimeViews& views() const;
  const AllocationPlan& allocationPlan() const;
  int device() const;
  // Owning arena base for explicit setup uploads and diagnostics, not stepping.
  void* buffer(const std::string& name) const;

  // Register external CUDA allocations once, before the hot path. Rejects
  // host/managed/wrong-device storage, bad extents/strides, and all aliasing
  // with the arena or other inputs. External allocations must be cudaMalloc
  // (or driver-address-range-queryable) allocations, not opaque pooled slices.
  Bindings bind(std::vector<BindingRequest> requests) const;

  // Trusted launch callbacks must only enqueue onto the supplied stream. No
  // recursive submission, synchronization, allocations, or raw-stream escape.
  // These methods enqueue and return; they do not wait for GPU completion.
  void enqueue(Launch launch, void* context = nullptr, const Bindings* inputs = nullptr);
  void enqueueOn(cudaStream_t stream, Launch launch, void* context = nullptr,
                 const Bindings* inputs = nullptr);
  // Framework bridge: persistent events order producer -> private stream ->
  // producer. Default streams are allowed here, never captured implicitly.
  void enqueueFrom(cudaStream_t producer, Launch, void* context=nullptr, const Bindings* inputs=nullptr);
  void enqueueSetup(Launch, void* context=nullptr); // invalidates captured schedules
  Graph capture(Launch, void* context=nullptr, const Bindings* inputs=nullptr);
  void replay(const Graph&);
  void replayFrom(const Graph&,cudaStream_t producer);
  void waitOn(cudaStream_t consumer) const;
  void recordUsage(cudaStream_t consumer);
  // Explicit completion boundary for snapshots/errors. Not part of step.
  void synchronize() const;
  std::uint64_t submissions() const;
 private:
  void submit(cudaStream_t,Launch,void*,const Bindings*,bool bridge,bool setup,const Graph* graph=nullptr);
  std::shared_ptr<detail::RuntimeStorage> storage_;
};

} // namespace flightlib_cuda::v1
