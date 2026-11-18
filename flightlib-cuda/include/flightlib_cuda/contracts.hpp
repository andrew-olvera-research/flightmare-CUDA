#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

#if defined(__CUDACC__)
#define FLC_HD __host__ __device__
#else
#define FLC_HD
#endif

namespace flightlib_cuda::v1 {

// Frozen semantic schema, not a serialized ABI. See docs/architecture.md.
inline constexpr std::uint32_t kContractVersion = 1;
using Scalar = float;
using Index = std::uint32_t;
using Offset = std::uint64_t;
using Mask = std::uint8_t;  // Exactly 0 or 1; never vector<bool>.
inline constexpr Index kInvalidIndex = std::numeric_limits<Index>::max();
inline constexpr Index kStateSize = 25;
inline constexpr Index kActionSize = 4;
inline constexpr Index kObservationSize = 12;

enum class StateField : Index {
  Position = 0, QuaternionWxyz = 3, LinearVelocity = 7,
  BodyAngularVelocity = 10, LinearAcceleration = 13, BodyTorque = 16,
  AngularBias = 19, AccelerationBias = 22
};
enum class CommandMode : std::uint8_t {
  Invalid = 0, RotorThrust = 1, CollectiveBodyRates = 2
};
enum class LaneStatus : std::uint32_t {
  Ok = 0, Inactive = 1, InvalidCommand = 2, InvalidState = 3,
  InvalidReset = 4, NumericalFailure = 5
};
enum class QueryStatus : std::uint32_t {
  Inactive = 0, Miss = 1, Hit = 2, InvalidInput = 3, Overflow = 4
};

// Non-owning dense row-major descriptor. In production data points to device
// memory; host allocation is permitted for contract tests and oracle adapters.
// No hidden padding, pitch, allocation, conversion, bounds check, or transfer.
template <typename T, Index Rank>
struct TensorView {
  static_assert(Rank > 0);
  T* data{};
  Offset shape[Rank]{};

  FLC_HD constexpr Offset size() const {
    Offset result = 1;
    for (Index i = 0; i < Rank; ++i) result *= shape[i];
    return result;
  }
  FLC_HD constexpr Offset stride(Index axis) const {
    Offset result = 1;
    for (Index i = axis + 1; i < Rank; ++i) result *= shape[i];
    return result;
  }
  template <typename... I>
  FLC_HD constexpr Offset offset(I... indices) const {
    static_assert(sizeof...(I) == Rank);
    const Offset coords[Rank]{static_cast<Offset>(indices)...};
    Offset result = 0;
    for (Index i = 0; i < Rank; ++i) result = result * shape[i] + coords[i];
    return result;
  }
  template <typename... I>
  FLC_HD T& operator()(I... indices) const { return data[offset(indices...)]; }
};
template <typename T> using Vector = TensorView<T, 1>;
template <typename T> using Matrix = TensorView<T, 2>;

// Construct through makeBatchShape at the host API boundary. N < invalid index.
struct BatchShape {
  Index environments{};       // E
  Index slots_per_environment{};  // A, fixed for this quadrotor pool
  Index lanes{};              // N = E*A
  FLC_HD constexpr Index lane(Index env, Index slot) const {
    return env * slots_per_environment + slot;
  }
  FLC_HD constexpr Index environment(Index lane_id) const {
    return lane_id / slots_per_environment;
  }
};

inline BatchShape makeBatchShape(Index environments, Index slots) {
  const Offset lanes = Offset{environments} * slots;
  if (environments == 0 || slots == 0 || lanes >= kInvalidIndex)
    throw std::invalid_argument("flightlib-cuda: invalid batch capacity");
  return {environments, slots, static_cast<Index>(lanes)};
}

template <typename T>
inline Offset checkedBytes(Offset elements) {
  if (elements > std::numeric_limits<std::size_t>::max() / sizeof(T))
    throw std::overflow_error("flightlib-cuda: allocation size overflow");
  return elements * sizeof(T);
}

// Use T=Scalar for writers, T=const Scalar for readers. x is one SoA allocation;
// its contiguous row groups are position, attitude, velocity, etc.
template <typename T>
struct QuadStateView {
  Matrix<T> x;             // [25,N], exact legacy field order
  Vector<T> time;          // [N], seconds, independent of command time
  Matrix<T> motor_omega;   // [4,N], rad/s
  Matrix<T> motor_thrust;  // [4,N], N
};
using QuadState = QuadStateView<Scalar>;
using ConstQuadState = QuadStateView<const Scalar>;

struct CommandInput {
  Vector<const CommandMode> mode;  // [N]
  Vector<const Scalar> time;       // [N]
  Matrix<const Scalar> payload;    // [4,N]: rotor forces OR collective,wx,wy,wz
};
struct CommandStorage {
  Vector<CommandMode> mode;  // [N], persistent last accepted command
  Vector<Scalar> time;      // [N]
  Matrix<Scalar> payload;   // [4,N], clamped accepted values
};

struct QuadModelView {
  Vector<const Scalar> mass;                  // [P]
  Matrix<const Scalar> inertia;               // [3,P], diagonal J
  Matrix<const Scalar> inverse_inertia;       // [3,P]
  Matrix<const Scalar> allocation;            // [16,P], row*4+column
  Matrix<const Scalar> inverse_allocation;    // [16,P]
  Matrix<const Scalar> thrust_polynomial;     // [3,P], quadratic,linear,constant
  Vector<const Scalar> motor_tau_inverse;     // [P]
  Matrix<const Scalar> motor_speed_bounds;    // [2,P], min,max
  Matrix<const Scalar> thrust_bounds;         // [2,P], min,max (legacy cached cap)
  Matrix<const Scalar> body_rate_limit;       // [3,P]
  Matrix<const Scalar> rate_gain;             // [3,P]
  Matrix<const Scalar> gravity;               // [3,P], world acceleration
};

struct QuadTopologyView {
  BatchShape batch;
  Vector<const Index> model_id;      // [N], in [0,P)
  Vector<const Mask> active;         // [N]
  Matrix<const Scalar> world_bounds; // [6,N]: xmin,xmax,ymin,ymax,zmin,zmax
};

// Host-owned immutable schedule; durations are device-resident. Build by the
// oracle's float remaining-time subtraction, not ceil(dt/dt_max).
struct StepSchedule {
  Scalar control_dt{};
  Vector<const Scalar> substep_dt;  // [S], positive, exact legacy sequence
};

struct ResetInput {
  Vector<const Mask> selected;     // [N]
  Matrix<const Scalar> x;          // [25,N]
  Vector<const Scalar> time;       // [N]
};
struct LaneDiagnostics {
  Vector<LaneStatus> status;       // [N], overwritten on each operation
};

// Legacy task is defined only for A=1. For A>1 these are per-platform outputs;
// no shared reward, done reduction, or group reset policy is implied.
struct TaskParameters {
  Vector<const Mask> ctbr;            // [N], one flag per lane
  Matrix<const Scalar> action_mean;   // [4,N]
  Matrix<const Scalar> action_scale;  // [4,N]
  Matrix<const Scalar> goal;          // [12,N]
  Matrix<const Scalar> reward_coeff;  // [5,N]: pos,orientation,velocity,rates,action
};
struct TaskInput {
  TensorView<const Scalar, 3> action; // [E,A,4], raw policy actions
};
struct TaskOutput {
  TensorView<Scalar, 3> observation;          // [E,A,12], returned/reset obs
  TensorView<Scalar, 3> transition_observation; // [E,A,12], before auto reset
  Matrix<Scalar> reward;                      // [E,A], includes terminal adjustment
  Matrix<Mask> done;                          // [E,A], pre-reset terminal flag
  Matrix<Mask> valid;                         // [E,A], failed step must not be trained on
};
struct EpisodeState {
  Vector<std::uint64_t> episode;     // [N], increments on successful task reset
  Vector<std::uint64_t> step;        // [N], zeroed on successful task reset
  Vector<std::uint64_t> random_key;  // [N], independent lane streams
  Vector<std::uint64_t> random_counter; // [N], reset sampling advances this
  Vector<Scalar> command_time;       // [N], task clock, reset to zero
};

// Geometry uses environment-local world coordinates, as does dynamics.
// Geometry/backend asset storage is opaque and deliberately not frozen here.
struct GeometryInstances {
  Vector<const Index> environment;  // [I], in [0,E)
  Vector<const Index> asset;        // [I], opaque stable asset ID
  Vector<const Index> owner_lane;   // [I], invalid = static world instance
  Matrix<const Scalar> local_pose;  // [7,I], xyz,wxyz relative to owner or world
  Vector<const Mask> active;        // [I]
};
struct GeometryWorldState {
  Matrix<Scalar> pose;        // [7,I], xyz,wxyz in environment world
  Matrix<Scalar> aabb;        // [6,I], xmin,xmax,ymin,ymax,zmin,zmax
};
struct RayQueryInput {
  Vector<const Index> environment;    // [Q]
  Vector<const Index> excluded_owner; // [Q], invalid = exclude none
  Matrix<const Scalar> origin;        // [3,Q]
  Matrix<const Scalar> direction;     // [3,Q], unit length
  Vector<const Scalar> max_distance;  // [Q], finite, >0, meters
  Vector<const Mask> active;          // [Q]
};
struct RayQueryOutput {
  Vector<QueryStatus> status;   // [Q]
  Vector<Scalar> distance;      // [Q], +inf for miss/inactive/error
  Matrix<Scalar> normal;        // [3,Q], world geometric normal; zero unless hit
  Vector<Index> instance;       // [Q], invalid unless hit
  Vector<Index> primitive;      // [Q], stable triangle ID; invalid unless hit
};
// Broad-phase candidates are NOT confirmed contacts. K is per-env capacity.
struct CandidatePairs {
  TensorView<Index, 3> instance_pair; // [E,K,2], canonical (min ID,max ID)
  Vector<Index> count;               // [E], stored count <= K
  Vector<Mask> overflow;             // [E], invalidates that env's query result
};
struct CollisionFlags {
  Vector<Mask> hit;           // [N], confirmed overlap, not broad-phase candidate
  Vector<Mask> valid;         // [N], 0 on overflow/unsupported geometry/error
};

static_assert(sizeof(Scalar) == 4 && std::numeric_limits<Scalar>::is_iec559);
static_assert(sizeof(Index) == 4 && sizeof(Offset) == 8 && sizeof(Mask) == 1);
static_assert(std::is_trivially_copyable_v<QuadState>);
static_assert(std::is_trivially_copyable_v<QuadModelView>);
static_assert(std::is_trivially_copyable_v<TaskOutput>);
static_assert(std::is_standard_layout_v<TensorView<Scalar, 2>>);

}  // namespace flightlib_cuda::v1

#undef FLC_HD
