#include "oracle.hpp"

#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <utility>

#include "flightlib/envs/quadrotor_env/quadrotor_env.hpp"
#include "flightlib/json/json.hpp"

namespace {
using namespace flightlib;
using nlohmann::json;

template<int Size>
json values(const Vector<Size>& vector) {
  return std::vector<Scalar>(vector.data(), vector.data() + vector.size());
}

json commandInput(const parity::Input& input) {
  const auto& command = input.command;
  json result{{"dt", input.dt}, {"command_time", command.t}};
  if (command.isSingleRotorThrusts()) {
    result["mode"] = "rotor_thrust";
    result["thrust"] = values(command.thrusts);
  } else {
    result["mode"] = "collective_thrust_body_rates";
    result["collective_thrust"] = command.collective_thrust;
    result["body_rates"] = values(command.omega);
  }
  return result;
}

json physicsTrace(const parity::Scenario& scenario) {
  json bounds = json::array();
  for (int axis = 0; axis < 3; ++axis) {
    bounds.push_back({scenario.world_box(axis, 0), scenario.world_box(axis, 1)});
  }
  json dynamics{{"mass", 1.0f}, {"arm_length", 0.25f}};
  if (!scenario.dynamics_config.empty()) {
    const auto config = YAML::LoadFile(scenario.dynamics_config);
    dynamics = {{"yaml", YAML::Dump(config["quadrotor_dynamics"])}};
  }
  const auto samples = parity::replay(scenario);
  json steps = json::array();
  for (size_t i = 0; i < samples.size(); ++i) {
    const auto& sample = samples[i];
    steps.push_back({{"input", commandInput(scenario.inputs[i])},
                     {"state", values(sample.state.x)},
                     {"time", sample.state.t},
                     {"motor_omega", values(sample.motor_omega)},
                     {"motor_thrust", values(sample.motor_thrust)}});
  }
  return {{"name", scenario.name},
          {"dynamics", std::move(dynamics)},
          {"world_box", std::move(bounds)},
          {"initial_time", scenario.initial.t},
          {"initial_state", values(scenario.initial.x)},
          {"steps", std::move(steps)}};
}

json environmentTrace() {
  const std::string path = parity::quadrotorConfigPath();
  QuadrotorEnv env(path);
  Vector<12> obs;
  if (!env.reset(obs, false)) {
    throw std::runtime_error("Environment reset failed");
  }
  const json initial_observation = values(obs);
  json steps = json::array();
  for (int i = 0; i < 12; ++i) {
    Vector<4> action = i < 3 ? Vector<4>(1, 1, 1, 1)
                            : Vector<4>(0.75f, -0.25f, 1.1f, 0.35f);
    const Scalar reward = env.step(action, obs);
    Scalar adjustment;
    const bool done = env.isTerminalState(adjustment);
    if (!obs.allFinite() || !std::isfinite(reward)) {
      throw std::runtime_error("Invalid environment output at step " + std::to_string(i));
    }
    steps.push_back({{"action", values(action)},
                     {"observation", values(obs)},
                     {"reward", reward},
                     {"done", done},
                     {"terminal_reward_adjustment", adjustment}});
  }
  return {{"name", "scalar_environment_climb_and_turn"},
          {"config_yaml", YAML::Dump(YAML::LoadFile(path))},
          {"random_reset", false},
          {"initial_observation", initial_observation},
          {"steps", std::move(steps)}};
}
}  // namespace

int main() {
  try {
    json physics = json::array();
    for (const auto& scenario : parity::scenarios()) {
      physics.push_back(physicsTrace(scenario));
    }
    const json trace{
      {"schema_version", 1},
      {"backend", "flightlib_cpu"},
      {"scalar_type", "float32"},
      {"state_fields", {"px", "py", "pz", "qw", "qx", "qy", "qz",
                        "vx", "vy", "vz", "wx", "wy", "wz", "ax", "ay", "az",
                        "tx", "ty", "tz", "bwx", "bwy", "bwz", "bax", "bay", "baz"}},
      {"observation_fields", {"px", "py", "pz", "yaw", "pitch", "roll",
                              "vx", "vy", "vz", "wx", "wy", "wz"}},
      {"physics", std::move(physics)},
      {"environment", environmentTrace()}};
    std::cout << trace.dump(2) << '\n';
    if (!std::cout) throw std::runtime_error("Could not write trace JSON");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
