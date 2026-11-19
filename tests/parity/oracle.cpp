#include "oracle.hpp"

#include <cstdlib>
#include <stdexcept>

namespace parity {
using namespace flightlib;

std::string quadrotorConfigPath() {
  const char* root = std::getenv("FLIGHTMARE_PATH");
  if (!root || !*root) {
    throw std::runtime_error("Set FLIGHTMARE_PATH to the repository root");
  }
  return std::string(root) + "/flightlib/configs/quadrotor_env.yaml";
}

QuadState stateAt(Scalar altitude) {
  QuadState state;
  state.setZero();
  state.p.z() = altitude;
  return state;
}

Sample readSample(const Quadrotor& quad) {
  Sample sample;
  if (!quad.getState(&sample.state) ||
      !quad.getMotorOmega(sample.motor_omega) ||
      !quad.getMotorThrusts(sample.motor_thrust) ||
      !sample.motor_omega.allFinite() || !sample.motor_thrust.allFinite()) {
    throw std::runtime_error("Invalid quadrotor telemetry");
  }
  return sample;
}

std::vector<Scenario> scenarios() {
  Matrix<3, 2> box;
  box << -20.0f, 20.0f, -20.0f, 20.0f, 0.0f, 20.0f;
  const Vector<4> zero = Vector<4>::Zero();
  const Vector<4> asymmetric{1.5f, 3.0f, 2.0f, 1.0f};
  Scenario fall{"free_fall_and_floor", stateAt(0.5f), box, "", {}};
  Scenario motors{"motor_transient", stateAt(5.0f), box, "", {}};
  Scenario rates{"rates_reversal", stateAt(5.0f), box, "", {}};
  Scenario configured{"configured_hover", stateAt(5.0f), box,
                      quadrotorConfigPath(), {}};

  for (int step = 0; step < 32; ++step) {
    fall.inputs.push_back({0.02f, Command(0.02f * step, zero)});
  }

  motors.initial.p.x() = 1.0f;
  motors.initial.v.y() = 0.2f;
  motors.initial.q(Quaternion(Eigen::AngleAxis<Scalar>(0.4f, Vector<3>::UnitZ())));
  const Scalar substep = 0.0025f;
  motors.inputs = {{0.001f, Command(0.0f, asymmetric)},
                   {substep, Command(0.001f, asymmetric)},
                   {0.00625f, Command(0.0035f, asymmetric)},
                   {0.02f, Command(0.00975f, Vector<4>{-1, 30, 1, 10})},
                   {0.02f, Command(0.02975f, zero)}};

  for (int step = 0; step < 80; ++step) {
    const Scalar direction = step < 40 ? 1.0f : -1.0f;
    rates.inputs.push_back(
      {0.01f, Command(0.01f * step, 9.81f,
                      direction * Vector<3>{0.12f, -0.15f, 0.45f})});
  }

  const Quadrotor tuned(configured.dynamics_config);
  QuadrotorDynamics dynamics;
  if (!tuned.getDynamics(&dynamics)) {
    throw std::runtime_error("Invalid configured dynamics");
  }
  const Vector<4> hover =
    Vector<4>::Constant(-Gz * dynamics.getMass() / 4.0f);
  for (int step = 0; step < 250; ++step) {
    configured.inputs.push_back({0.02f, Command(0.02f * step, hover)});
  }
  return {fall, motors, rates, configured};
}

std::vector<Sample> replay(const Scenario& scenario) {
  QuadrotorDynamics dynamics(1.0f, 0.25f);
  if (!scenario.dynamics_config.empty() &&
      !dynamics.updateParams(YAML::LoadFile(scenario.dynamics_config))) {
    throw std::runtime_error(scenario.name + " has invalid dynamics");
  }

  Quadrotor quad(dynamics);
  auto box = scenario.world_box;
  if (!quad.setWorldBox(box) || !quad.reset(scenario.initial)) {
    throw std::runtime_error(scenario.name + " has an invalid initial state or world box");
  }

  std::vector<Sample> samples;
  samples.reserve(scenario.inputs.size());
  for (const auto& input : scenario.inputs) {
    if (!quad.run(input.command, input.dt)) {
      throw std::runtime_error(scenario.name + " rejected step " +
                               std::to_string(samples.size()));
    }
    samples.push_back(readSample(quad));
  }
  return samples;
}

}  // namespace parity
