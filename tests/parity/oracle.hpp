#pragma once

#include <string>
#include <vector>

#include "flightlib/objects/quadrotor.hpp"

namespace parity {

struct Input {
  flightlib::Scalar dt;
  flightlib::Command command;
};

struct Scenario {
  std::string name;
  flightlib::QuadState initial;
  flightlib::Matrix<3, 2> world_box;
  std::string dynamics_config;
  std::vector<Input> inputs;
};

struct Sample {
  flightlib::QuadState state;
  flightlib::Vector<4> motor_omega;
  flightlib::Vector<4> motor_thrust;
};

std::string quadrotorConfigPath();
flightlib::QuadState stateAt(flightlib::Scalar altitude);
Sample readSample(const flightlib::Quadrotor& quad);
std::vector<Scenario> scenarios();
std::vector<Sample> replay(const Scenario& scenario);

}  // namespace parity
