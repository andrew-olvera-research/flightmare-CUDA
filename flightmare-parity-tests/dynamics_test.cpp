#include "oracle.hpp"

#include <cmath>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "flightlib/common/integrator_euler.hpp"
#include "flightlib/common/integrator_rk4.hpp"

namespace parity {
namespace {
using namespace flightlib;

TEST(CommandContract, SelectsExactlyOneFiniteControlMode) {
  const Command rotor(0.0f, Vector<4>{1, 2, 3, 4});
  EXPECT_TRUE(rotor.valid());
  EXPECT_TRUE(rotor.isSingleRotorThrusts());
  EXPECT_FALSE(rotor.isRatesThrust());
  const Command rates(0.0f, 9.81f, Vector<3>{1, 2, 3});
  EXPECT_TRUE(rates.valid());
  EXPECT_FALSE(rates.isSingleRotorThrusts());
  EXPECT_TRUE(rates.isRatesThrust());

  Command both = rotor;
  both.collective_thrust = rates.collective_thrust;
  both.omega = rates.omega;
  EXPECT_FALSE(both.valid());
  EXPECT_FALSE(Command().valid());
}

TEST(CommandContract, RejectsNonfiniteTimestampOrControl) {
  const Scalar infinity = std::numeric_limits<Scalar>::infinity();
  EXPECT_FALSE(Command(infinity, Vector<4>::Ones()).valid());
  EXPECT_FALSE(Command(0.0f, Vector<4>{1, 1, infinity, 1}).valid());
  EXPECT_FALSE(Command(0.0f, infinity, Vector<3>::Zero()).valid());
  EXPECT_FALSE(Command(0.0f, 9.81f, Vector<3>{0, infinity, 0}).valid());
}

TEST(DynamicsContract, MotorMapAndItsInverseMatchAtDistinctSpeeds) {
  const QuadrotorDynamics dynamics;
  const Vector<4> speeds{200, 500, 1000, 2000};
  const Vector<4> thrust = dynamics.motorOmegaToThrust(speeds);
  const Vector<4> expected{-0.94858946f, 0.48149808f, 3.39690709f, 11.22246494f};
  for (int rotor = 0; rotor < 4; ++rotor) {
    EXPECT_NEAR(thrust[rotor], expected[rotor], 2e-6f) << "rotor " << rotor;
    EXPECT_NEAR(dynamics.motorThrustToOmega(thrust)[rotor], speeds[rotor],
                5e-4f) << "rotor " << rotor;
  }
}

TEST(DynamicsContract, AssignsForceAndTorqueByRotorIndex) {
  const Scalar arm = 0.2f / std::sqrt(2.0f);
  Matrix<4, 4> expected;
  expected << 1, 1, 1, 1,
              arm, -arm, -arm, arm,
              -arm, -arm, arm, arm,
              0.016f, -0.016f, 0.016f, -0.016f;
  const auto allocation = QuadrotorDynamics().getAllocationMatrix();
  EXPECT_NEAR((allocation - expected).cwiseAbs().maxCoeff(), 0.0f, 1e-7f);
}

TEST(DynamicsContract, ClampsThrustSpeedAndRatesIndependently) {
  const QuadrotorDynamics dynamics;
  const Vector<4> thrust = dynamics.clampThrust(Vector<4>{-10, 0, 2, 100});
  EXPECT_FLOAT_EQ(thrust[0], 0.0f);
  EXPECT_FLOAT_EQ(thrust[1], 0.0f);
  EXPECT_FLOAT_EQ(thrust[2], 2.0f);
  EXPECT_NEAR(thrust[3], 11.222465f, 2e-6f);
  const Vector<4> expected_speed{150, 150, 600, 2000};
  EXPECT_EQ(dynamics.clampMotorOmega(Vector<4>(-2, 150, 600, 2500)),
            expected_speed);
  const Vector<3> expected_rates{-6, 1, 6};
  EXPECT_EQ(dynamics.clampBodyrates(Vector<3>(-8, 1, 12)), expected_rates);
}

TEST(DynamicsContract, YamlChangesSpeedLimitAndKeepsLegacyThrustCap) {
  QuadrotorDynamics dynamics;
  const Scalar default_cap = dynamics.clampThrust(100.0f);
  ASSERT_TRUE(dynamics.updateParams(YAML::LoadFile(quadrotorConfigPath())));
  EXPECT_FLOAT_EQ(dynamics.getMass(), 0.73f);
  EXPECT_FLOAT_EQ(dynamics.getArmLength(), 0.17f);
  EXPECT_FLOAT_EQ(dynamics.getMotorTauInv(), 10000.0f);
  EXPECT_EQ(dynamics.clampMotorOmega(Vector<4>::Constant(4000)),
            Vector<4>::Constant(3000));
  EXPECT_FLOAT_EQ(dynamics.clampThrust(100.0f), default_cap);
  EXPECT_GT(dynamics.motorOmegaToThrust(Vector<4>::Constant(3000))[0],
            default_cap);
}

TEST(DynamicsContract, DerivativeUsesBodyRatesAndLeavesAuxiliaryFieldsStatic) {
  const QuadrotorDynamics dynamics;
  auto initial = stateAt(5.0f);
  initial.q(Quaternion(Eigen::AngleAxis<Scalar>(0.6f, Vector<3>::UnitY())));
  initial.v << 1.0f, -2.0f, 3.0f;
  initial.w << 0.4f, -0.8f, 1.2f;
  initial.a << -0.5f, 0.25f, -3.0f;
  initial.tau << 0.01f, -0.02f, 0.04f;
  initial.bw.setOnes();
  initial.ba.setConstant(2.0f);

  QuadState derivative;
  ASSERT_TRUE(dynamics.dState(initial, &derivative));
  const Quaternion qdot =
    initial.q() * Quaternion(0, initial.w.x(), initial.w.y(), initial.w.z());
  const Vector<4> expected_qdot{qdot.w(), qdot.x(), qdot.y(), qdot.z()};
  const Vector<3> momentum = dynamics.getJ() * initial.w;
  const Vector<3> expected_rate_change =
    dynamics.getJInv() * (initial.tau - initial.w.cross(momentum));
  EXPECT_EQ(derivative.p, initial.v);
  EXPECT_EQ(derivative.v, initial.a);
  EXPECT_NEAR((derivative.qx - 0.5f * expected_qdot).norm(), 0.0f, 1e-7f);
  EXPECT_NEAR((derivative.w - expected_rate_change).norm(), 0.0f, 1e-6f);
  EXPECT_TRUE(derivative.x.tail<QS::SIZE - QS::ACC>().isZero(0.0f));
}

TEST(IntegratorContract, EulerAndRk4MatchTheirKnownOneStepSolutions) {
  const auto dynamics = [](const Ref<const Vector<>>& x, Ref<Vector<>> dx) {
    dx = x;
    return true;
  };
  IntegratorEuler euler(dynamics, 0.1f);
  IntegratorRK4 rk4(dynamics, 0.1f);
  const Vector<1> initial = Vector<1>::Ones();
  Vector<1> result;
  ASSERT_TRUE(euler.step(initial, 0.1f, result));
  EXPECT_FLOAT_EQ(result[0], 1.1f);
  ASSERT_TRUE(rk4.step(initial, 0.1f, result));
  EXPECT_NEAR(result[0], std::exp(0.1f), 2e-7f);
}

TEST(IntegratorContract, TakesFullSubstepsAndOneRemainder) {
  std::vector<Scalar> evaluation_states;
  const auto dynamics = [&](const Ref<const Vector<>>& x, Ref<Vector<>> dx) {
    evaluation_states.push_back(x[0]);
    dx.setOnes();
    return true;
  };
  IntegratorRK4 rk4(dynamics, 0.125f);
  Vector<1> result;
  ASSERT_TRUE(rk4.integrate(Vector<1>::Zero(), 0.3125f, result));
  ASSERT_EQ(evaluation_states.size(), 12U);
  EXPECT_FLOAT_EQ(evaluation_states[3], 0.125f);
  EXPECT_FLOAT_EQ(evaluation_states[7], 0.25f);
  EXPECT_FLOAT_EQ(evaluation_states[11], 0.3125f);
  EXPECT_FLOAT_EQ(result[0], 0.3125f);
}

TEST(IntegratorContract, RejectsInvalidTimeAndDerivativeFailure) {
  const auto dynamics = [](const Ref<const Vector<>>&, Ref<Vector<>>) {
    return false;
  };
  IntegratorRK4 rk4(dynamics, 0.125f);
  const auto initial = stateAt(5.0f);
  auto result = initial;
  EXPECT_FALSE(rk4.integrate(initial, &result));
  result.t = -1.0f;
  EXPECT_FALSE(rk4.integrate(initial, &result));
  result.t = 0.125f;
  EXPECT_FALSE(rk4.integrate(initial, &result));
}

}  // namespace
}  // namespace parity
