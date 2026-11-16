#include "oracle.hpp"

#include <cmath>
#include <limits>
#include <ostream>

#include <gtest/gtest.h>

namespace parity {
namespace {
using namespace flightlib;

void expectSameSample(const Sample& actual, const Sample& expected) {
  EXPECT_EQ(actual.state.x, expected.state.x);
  EXPECT_FLOAT_EQ(actual.state.t, expected.state.t);
  EXPECT_EQ(actual.motor_omega, expected.motor_omega);
  EXPECT_EQ(actual.motor_thrust, expected.motor_thrust);
}

TEST(QuadrotorContract, ResetClearsMotorsAndRestoresTheWholeSuppliedState) {
  Quadrotor quad;
  auto initial = stateAt(5.0f);
  initial.t = 0.5f;
  initial.v << 1, -2, 3;
  initial.w << 0.2f, 0.3f, -0.4f;
  initial.bw << 0.01f, 0.02f, 0.03f;
  initial.ba << 0.04f, 0.05f, 0.06f;
  ASSERT_TRUE(quad.run(Command(0, Vector<4>::Constant(5)), 0.01f));
  ASSERT_GT(readSample(quad).motor_omega.minCoeff(), 0);
  ASSERT_TRUE(quad.reset(initial));
  const auto reset = readSample(quad);
  EXPECT_EQ(reset.state.x, initial.x);
  EXPECT_FLOAT_EQ(reset.state.t, initial.t);
  EXPECT_TRUE(reset.motor_omega.isZero(0.0f));
  EXPECT_TRUE(reset.motor_thrust.isZero(0.0f));
}

TEST(QuadrotorContract, InvalidCommandAndStateLeaveTelemetryUnchanged) {
  Quadrotor quad;
  ASSERT_TRUE(quad.reset(stateAt(5.0f)));
  ASSERT_TRUE(quad.run(Command(0, Vector<4>::Constant(3)), 0.02f));
  const auto before = readSample(quad);
  auto invalid = before.state;
  invalid.w.x() = std::numeric_limits<Scalar>::infinity();
  EXPECT_FALSE(quad.reset(invalid));
  EXPECT_FALSE(quad.setState(invalid));
  EXPECT_FALSE(quad.run(Command{}, 0.02f));
  expectSameSample(readSample(quad), before);
}

TEST(QuadrotorContract, FreeFallPreservesAttitudeAndMatchesBallisticMotion) {
  Quadrotor quad;
  auto initial = stateAt(5.0f);
  initial.v << 0.4f, -0.3f, 0.2f;
  initial.q(Quaternion(Eigen::AngleAxis<Scalar>(0.4f, Vector<3>::UnitX())));
  ASSERT_TRUE(quad.reset(initial));
  const Command free_fall(0, Vector<4>::Zero());
  for (int i = 1; i <= 20; ++i) {
    SCOPED_TRACE(i);
    ASSERT_TRUE(quad.run(free_fall, 0.02f));
    const auto sample = readSample(quad);
    const Scalar t = 0.02f * i;
    const Vector<3> position = initial.p + t * initial.v + (0.5f * t * t) * GVEC;
    EXPECT_NEAR(sample.state.t, t, 1e-7f);
    EXPECT_NEAR((sample.state.p - position).norm(), 0.0f, 1e-5f);
    EXPECT_NEAR((sample.state.v - initial.v - t * GVEC).norm(), 0.0f, 1e-5f);
    EXPECT_NEAR(sample.state.q().angularDistance(initial.q()), 0.0f, 1e-6f);
    EXPECT_TRUE(sample.state.w.isZero(0.0f));
    EXPECT_TRUE(sample.motor_thrust.isZero(0.0f));
  }
}

TEST(QuadrotorContract, MotorResponseUsesTimeConstantAndStartsFromPriorSpeed) {
  Quadrotor quad;
  ASSERT_TRUE(quad.reset(stateAt(5.0f)));
  const auto& dynamics = quad.getDynamics();
  const Vector<4> thrust{2, 3, 4, 5};
  const auto target = dynamics.motorThrustToOmega(thrust);
  const Scalar dt = 0.00125f;
  const Scalar decay = std::exp(-dt / 0.05f);
  ASSERT_TRUE(quad.run(Command(0, thrust), dt));
  const auto first = readSample(quad);
  EXPECT_NEAR((first.motor_omega - (1 - decay) * target).norm(), 0.0f, 1e-4f);

  ASSERT_TRUE(quad.run(Command(dt, thrust), dt));
  const auto second = readSample(quad);
  const Vector<4> expected_speed = decay * first.motor_omega + (1 - decay) * target;
  EXPECT_NEAR((second.motor_omega - expected_speed).norm(), 0.0f, 1e-4f);
  EXPECT_GT(second.motor_omega.minCoeff(), first.motor_omega.minCoeff());
}

TEST(QuadrotorContract, SubstepSchedulingMatchesExplicitPartition) {
  Quadrotor whole;
  Quadrotor split;
  ASSERT_TRUE(whole.reset(stateAt(5.0f)));
  ASSERT_TRUE(split.reset(stateAt(5.0f)));
  constexpr Scalar substep = 0.0025f;
  constexpr Scalar total = 0.006125f;
  const Command command(0, 9.81f, Vector<3>{0.2f, -0.4f, 0.5f});
  ASSERT_TRUE(whole.run(command, total));
  ASSERT_TRUE(split.run(command, substep));
  ASSERT_TRUE(split.run(command, substep));
  ASSERT_TRUE(split.run(command, (total - substep) - substep));
  const auto expected = readSample(split);
  const auto actual = readSample(whole);
  EXPECT_NEAR((actual.state.x - expected.state.x).norm(), 0.0f, 1e-6f);
  EXPECT_NEAR((actual.motor_omega - expected.motor_omega).norm(), 0.0f, 1e-5f);
  EXPECT_NEAR((actual.motor_thrust - expected.motor_thrust).norm(), 0.0f, 1e-6f);
  EXPECT_FLOAT_EQ(actual.state.t, total);
}

TEST(QuadrotorContract, RatesCommandAppliesTheLegacyCollectiveAndRateClamps) {
  Quadrotor saturated;
  Quadrotor limited;
  ASSERT_TRUE(saturated.reset(stateAt(5.0f)));
  ASSERT_TRUE(limited.reset(stateAt(5.0f)));
  const Scalar legacy_cap = limited.getDynamics().clampThrust(100.0f);
  ASSERT_TRUE(saturated.run(Command(0, 500, Vector<3>{30, -40, 50}), 0.02f));
  ASSERT_TRUE(limited.run(Command(0, legacy_cap, Vector<3>{6, -6, 6}), 0.02f));
  expectSameSample(readSample(saturated), readSample(limited));
}

struct WorldFace {
  const char* name;
  int axis;
  Scalar coordinate;
  bool crosses;
};

void PrintTo(const WorldFace& face, std::ostream* output) {
  *output << face.name;
}

class WorldBoxContract : public testing::TestWithParam<WorldFace> {};

TEST_P(WorldBoxContract, AppliesAxisSpecificConstraint) {
  const auto face = GetParam();
  Matrix<3, 2> box;
  box << -2, 2, -3, 3, 0, 10;
  Quadrotor quad;
  ASSERT_TRUE(quad.setWorldBox(box));
  const auto previous = stateAt(5.0f);
  auto candidate = previous;
  candidate.p[face.axis] = face.coordinate;
  candidate.v << 1, 2, 3;
  candidate.w << 4, 5, 6;
  candidate.a << 7, 8, 9;
  ASSERT_TRUE(quad.setState(candidate));
  ASSERT_TRUE(quad.constrainInWorldBox(previous));

  auto expected = candidate;
  if (face.crosses && face.axis == 2) {
    expected.p.z() = 0;
    expected.v.head<2>().setZero();
    expected.w.setZero();
    expected.a.setZero();
  } else if (face.crosses) {
    expected.p[face.axis] = previous.p[face.axis];
    expected.v[face.axis] = 0;
  }
  EXPECT_EQ(readSample(quad).state.x, expected.x);
}

INSTANTIATE_TEST_SUITE_P(
  Faces, WorldBoxContract,
  testing::Values(WorldFace{"XMin", 0, -2.1f, true},
                  WorldFace{"XMax", 0, 2.1f, true},
                  WorldFace{"YMin", 1, -3.1f, true},
                  WorldFace{"YMax", 1, 3.1f, true},
                  WorldFace{"ZMin", 2, -0.1f, true},
                  WorldFace{"ZMax", 2, 10.1f, true},
                  WorldFace{"AtXMin", 0, -2.0f, false},
                  WorldFace{"AtYMax", 1, 3.0f, false},
                  WorldFace{"AtZMin", 2, 0.0f, true},
                  WorldFace{"AtZMax", 2, 10.0f, false}),
  [](const testing::TestParamInfo<WorldFace>& info) { return info.param.name; });

TEST(ReplayContract, FixedInputsReproduceFullTelemetryAtEveryStep) {
  const auto cases = scenarios();
  ASSERT_EQ(cases.size(), 4U);
  for (const auto& scenario : cases) {
    SCOPED_TRACE(scenario.name);
    const auto first = replay(scenario);
    const auto second = replay(scenario);
    ASSERT_EQ(first.size(), scenario.inputs.size());
    ASSERT_EQ(second.size(), first.size());
    Scalar time = scenario.initial.t;
    for (size_t i = 0; i < first.size(); ++i) {
      SCOPED_TRACE(i);
      time += scenario.inputs[i].dt;
      EXPECT_FLOAT_EQ(first[i].state.t, time);
      expectSameSample(first[i], second[i]);
      EXPECT_NEAR(first[i].state.qx.norm(), 1.0f, 1e-4f);
    }
  }
}

TEST(ReplayContract, TrajectoriesReachTheirBehaviorCheckpoints) {
  for (const auto& scenario : scenarios()) {
    SCOPED_TRACE(scenario.name);
    const auto trace = replay(scenario);
    ASSERT_FALSE(trace.empty());
    if (scenario.name == "free_fall_and_floor") {
      EXPECT_LT(trace.front().state.p.z(), scenario.initial.p.z());
      EXPECT_FLOAT_EQ(trace.back().state.p.z(), 0.0f);
      EXPECT_LT(trace.back().state.v.z(), 0.0f);
    } else if (scenario.name == "motor_transient") {
      EXPECT_GT(trace[2].motor_omega.minCoeff(), trace[0].motor_omega.minCoeff());
      EXPECT_LT(trace.back().motor_omega.maxCoeff(), trace[3].motor_omega.maxCoeff());
    } else if (scenario.name == "rates_reversal") {
      EXPECT_GT(trace[39].state.w.z(), 0.1f);
      EXPECT_LT(trace.back().state.w.z(), -0.1f);
    } else if (scenario.name == "configured_hover") {
      EXPECT_NEAR(trace.back().state.t, 5.0f, 1e-5f);
      EXPECT_NEAR(trace.back().state.p.z(), 5.0f, 0.005f);
    }
  }
}

}  // namespace
}  // namespace parity
