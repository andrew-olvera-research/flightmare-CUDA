#pragma once
#include "flightlib_cuda/runtime.hpp"
#include <map>
#include <cmath>

namespace flightlib_cuda::v1 {
struct ModelConfig {
  Scalar mass{1}, arm_length{0.25f}, motor_min{150}, motor_max{2000};
  Scalar motor_tau{0.05f}, kappa{0.016f};
  std::array<Scalar,3> thrust_map{1.3298253500372892e-06f,0.0038360810526746033f,-1.7689986848125325f};
  std::array<Scalar,3> rate_limit{6,6,6};
  std::array<Scalar,3> rate_gain{16.6f,16.6f,5.f};
};
// Host setup only. Keys are runtime buffer names, values are component-major.
using ModelTable = std::map<std::string,std::vector<Scalar>>;
ModelConfig loadModel(const std::string& yaml_path);
ModelTable prepareModels(const std::vector<ModelConfig>& models);
std::vector<Scalar> prepareSchedule(Scalar control_dt, Scalar max_dt=0.0025f,
                                    Index capacity=1000000);
void validateSensorPeriods(const std::vector<Scalar>& schedule,
                           const SensorSettings& imu, const SensorSettings& odometry);
// Explicit synchronous setup/update boundary. Validates everything before upload;
// preserves state, motors, accepted commands, and task counters.
void configure(Simulation& simulation, const std::vector<ModelConfig>& models,
               const std::vector<Index>& model_ids, const std::vector<Mask>& active,
               const std::vector<Scalar>& world_bounds, Scalar control_dt,
               Scalar max_dt=0.0025f);

struct LegacyCommand {
  Scalar time{std::numeric_limits<Scalar>::quiet_NaN()};
  Scalar collective{std::numeric_limits<Scalar>::quiet_NaN()};
  std::array<Scalar,3> rates{NAN,NAN,NAN};
  std::array<Scalar,4> thrusts{NAN,NAN,NAN,NAN};
};
struct TypedCommand { CommandMode mode{CommandMode::Invalid}; Scalar time{};
                      std::array<Scalar,4> payload{}; };
TypedCommand adaptCommand(const LegacyCommand& input);

// Validate/register once. Binding owners follow Simulation::bind's rules.
Bindings bindStateInput(Simulation&, BindingRequest selected, BindingRequest x, BindingRequest time);
Bindings bindCommandInput(Simulation&, BindingRequest selected, BindingRequest mode,
                          BindingRequest time, BindingRequest payload);
// Asynchronous, no allocation/readback; input contents are validated per lane.
void resetState(Simulation&, const Bindings&);
void setState(Simulation&, const Bindings&);
void setCommand(Simulation&, const Bindings&);
// Framework bridges order producer -> simulation -> producer without host waits.
void resetStateFrom(Simulation&, const Bindings&, cudaStream_t producer);
void setStateFrom(Simulation&, const Bindings&, cudaStream_t producer);
void setCommandFrom(Simulation&, const Bindings&, cudaStream_t producer);
} // namespace flightlib_cuda::v1
