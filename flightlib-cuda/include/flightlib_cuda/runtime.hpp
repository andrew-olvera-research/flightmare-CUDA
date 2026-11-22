#pragma once

#include "flightlib_cuda/contracts.hpp"
#include "flightlib_cuda/sensors.hpp"
#include "flightlib_cuda/shared_channel.hpp"
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
// Transition probabilities per simulation-clock period. A zero period disables outages.
struct OutageProfile { Index period_ns{}; Scalar enter{}, recover{}; };

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
  // Diagnostic layout only: reserve gaps around every field for test canaries.
  // No checks or extra launches on the production path. Shapes are unchanged.
  bool debug_guards{false};
  Scalar agent_collision_radius{0}; // Positive enables one moving sphere per active slot.
  bool terminate_on_agent_collision{false};
  std::uint64_t sensor_step_ns{};
  SensorSettings imu{}, odometry{}, barometer{};
  packet::SharedProfile shared_profile{};
  Index network_tick_ns{}, network_fanin{};
  std::vector<Index> network_offsets{0};
  Index carrier_group_size{}, hidden_per_edge{};
  bool network_edge_outage_rates{}, network_sample_loss{}, network_record_outcomes{}, network_dynamic_subscriptions{};
  std::vector<Scalar> network_range_edges;
  OutageProfile outage{};
  Index network_loss_history_bits{};
};

struct SwarmCollision {
  TensorView<Scalar,3> trajectory; // [2*S+1,3,N]: alternate position and acceleration
  Matrix<Scalar> bounds;           // [6,N] swept AABB including the committed endpoint
  Vector<Index> order, hit, issue; // sorted lanes, atomic hit [N], error bits [E]
  Scalar radius{};
  bool terminate{};
};

struct NetworkBuffers {
  Matrix<Mask> media, offers, senders, results;
  Matrix<Index> sources;
  TensorView<std::uint64_t,3> inbox;
  Vector<std::uint64_t> reset_ns;
  Vector<Index> offsets;
  TensorView<Index,3> hidden_sources;
  Matrix<std::uint64_t> hidden_end;
  Index group_size{};
  Vector<Scalar> range_edges;
  TensorView<Scalar,3> loss_after_rx, loss_after_miss;
  Matrix<Mask> receiver_outcomes;
  Matrix<std::uint64_t> reset_cancellation;
  Matrix<Index> bound_source;
  Matrix<std::uint64_t> bound_since_ns;
  Matrix<std::uint64_t> history;
  TensorView<Scalar,3> context_loss; // One history tree per submission offset.
  Matrix<std::uint64_t> context_last_ns;
  Index context_mask{};
  TensorView<Scalar,3> outage_rates;
  Matrix<std::uint64_t> outage_tag, outage_last_ns;
  OutageProfile outage{};
  packet::SharedProfile profile{};
  Index tick_ns{};
};

struct RuntimeViews {
  NetworkBuffers network;
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
  SwarmCollision swarm;
  SensorTruth sensor_truth;
  SensorBuffers imu, odometry, barometer;
  std::uint64_t sensor_step_ns{};
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
  // Append kernels to an active caller capture. Caller retains all storage and
  // orders graph replays with waitOn/recordUsage outside the capture.
  void recordInto(cudaStream_t stream,Launch launch,void* context,const Bindings* inputs);
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
  void submit(cudaStream_t,Launch,void*,const Bindings*,bool bridge,bool setup,const Graph* graph=nullptr,bool capture_only=false);
  std::shared_ptr<detail::RuntimeStorage> storage_;
};

} // namespace flightlib_cuda::v1
