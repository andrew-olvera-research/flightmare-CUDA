#include "oracle.hpp"

#include <algorithm>

#include <gtest/gtest.h>

#include "flightlib/envs/vec_env.hpp"

namespace parity {
namespace {
using namespace flightlib;

Scalar expectedReward(const Vector<12>& obs, const Vector<4>& action) {
  const Vector<3> goal{0, 0, 5};
  return 0.1f - 0.002f * (obs.head<3>() - goal).squaredNorm() -
         0.002f * obs.segment<3>(3).squaredNorm() -
         0.0002f * obs.tail<6>().squaredNorm() - 0.0002f * action.norm();
}

TEST(EnvironmentContract, DeterministicResetInitializesObservationAndActuator) {
  QuadrotorEnv env(quadrotorConfigPath());
  Vector<12> obs;
  ASSERT_EQ(env.getObsDim(), obs.size());
  ASSERT_EQ(env.getActDim(), 4);
  ASSERT_TRUE(env.reset(obs, false));
  EXPECT_TRUE(obs.isZero(0.0f));
  Vector<4> climb = Vector<4>::Ones();
  env.step(climb, obs);
  EXPECT_GT(obs[8], 0.0f);
  ASSERT_TRUE(env.reset(obs, false));
  EXPECT_TRUE(obs.isZero(0.0f));
  Vector<4> action;
  ASSERT_TRUE(env.getAct(action));
  EXPECT_TRUE(action.isZero(0.0f));
  Command command;
  ASSERT_TRUE(env.getAct(&command));
  EXPECT_FLOAT_EQ(command.t, 0.0f);
  EXPECT_TRUE(command.thrusts.isZero(0.0f));
}

TEST(EnvironmentContract, MatchesRawActionScaleObservationOrderAndReward) {
  const std::string config = quadrotorConfigPath();
  QuadrotorEnv env(config);
  Quadrotor quad(config);
  Matrix<3, 2> world;
  world << -20, 20, -20, 20, 0, 20;
  ASSERT_TRUE(quad.setWorldBox(world));
  Vector<12> obs;
  ASSERT_TRUE(env.reset(obs, false));
  const Scalar hover_thrust = 0.73f * 9.81f / 4.0f;
  const Scalar dt = env.getSimTimeStep();
  for (int step = 0; step < 8; ++step) {
    SCOPED_TRACE(step);
    Vector<4> action = step < 3 ? Vector<4>(1, 1, 1, 1)
                               : Vector<4>(0.75f, -0.25f, 1.1f, 0.35f);
    const Vector<4> thrust = (2.0f * action.array() + 1.0f) * hover_thrust;
    ASSERT_TRUE(quad.run(Command((step + 1) * dt, thrust), dt));
    const auto state = readSample(quad).state;
    Vector<12> expected;
    expected << state.p, state.R().eulerAngles(2, 1, 0), state.v, state.w;
    const Scalar reward = env.step(action, obs);
    Vector<4> scaled;
    ASSERT_TRUE(env.getAct(scaled));
    EXPECT_NEAR((scaled - thrust).cwiseAbs().maxCoeff(), 0.0f, 1e-6f);
    EXPECT_NEAR((obs - expected).cwiseAbs().maxCoeff(), 0.0f, 1e-5f);
    EXPECT_NEAR(reward, expectedReward(expected, action), 2e-6f);
  }
  EXPECT_GT(obs.segment<3>(3).norm(), 0.01f);
}

TEST(EnvironmentContract, ActionPenaltyUsesRawL2Norm) {
  const std::string config = quadrotorConfigPath();
  QuadrotorEnv env(config);
  auto params = YAML::LoadFile(config);
  for (const char* key : {"pos_coeff", "ori_coeff", "lin_vel_coeff", "ang_vel_coeff"}) {
    params["rl"][key] = 0.0f;
  }
  params["rl"]["act_coeff"] = -0.25f;
  ASSERT_TRUE(env.loadParam(params));
  Vector<12> obs;
  ASSERT_TRUE(env.reset(obs, false));
  Vector<4> action{-1.0f, 2.0f, -2.0f, 0.0f};
  EXPECT_NEAR(env.step(action, obs), 0.1f - 0.25f * 3.0f, 1e-6f);
}

TEST(EnvironmentContract, TerminalAltitudeAndEpisodeTimeHaveDistinctEffects) {
  const std::string config = quadrotorConfigPath();
  QuadrotorEnv env(config);
  auto params = YAML::LoadFile(config);
  params["quadrotor_env"]["max_t"] = 0.04f;
  ASSERT_TRUE(env.loadParam(params));
  EXPECT_FLOAT_EQ(env.getMaxT(), 0.04f);
  Vector<12> obs;
  ASSERT_TRUE(env.reset(obs, false));
  Scalar adjustment;
  EXPECT_TRUE(env.isTerminalState(adjustment));
  EXPECT_FLOAT_EQ(adjustment, -0.02f);

  Vector<4> climb = Vector<4>::Ones();
  for (int i = 0; i < 2; ++i) {
    env.step(climb, obs);
    EXPECT_GT(obs[2], 0.0f);
    EXPECT_LT(obs[2], 0.02f);
    EXPECT_TRUE(env.isTerminalState(adjustment));
    EXPECT_FLOAT_EQ(adjustment, -0.02f);
  }
  env.step(climb, obs);
  EXPECT_GT(obs[2], 0.02f);
  EXPECT_FALSE(env.isTerminalState(adjustment));
  EXPECT_FLOAT_EQ(adjustment, 0.0f);
}

class VectorEnvContract : public testing::Test {
 protected:
  static constexpr int count = 3;
  VecEnv<QuadrotorEnv> env{
    "{env: {seed: 1, scene_id: 0, num_envs: 3, num_threads: 2, render: false}}",
    false};
  MatrixRowMajor<> obs{count, 12};
  MatrixRowMajor<> acts = MatrixRowMajor<>::Constant(count, 4, -1.0f);
  MatrixRowMajor<> extra{count, 0};
  Vector<> reward{count};
  BoolVector<> done = BoolVector<>::Constant(count, false);
};

TEST_F(VectorEnvContract, ResetRespectsThePublicObservationAndMetadataContract) {
  EXPECT_EQ(env.getNumOfEnvs(), count);
  EXPECT_EQ(env.getObsDim(), obs.cols());
  EXPECT_EQ(env.getActDim(), acts.cols());
  EXPECT_EQ(env.getExtraInfoDim(), 0);
  EXPECT_EQ(env.getEpisodeLength(), 250U);
  ASSERT_TRUE(env.reset(obs));
  EXPECT_TRUE(obs.allFinite());
  for (int lane = 0; lane < count; ++lane) {
    EXPECT_GE(obs(lane, 2), 4.0f);
    EXPECT_LE(obs(lane, 2), 6.0f);
    EXPECT_LE(obs.row(lane).segment<2>(0).cwiseAbs().maxCoeff(), 1.0f);
    EXPECT_LE(obs.row(lane).segment<3>(6).cwiseAbs().maxCoeff(), 1.0f);
    EXPECT_TRUE(obs.row(lane).tail<3>().isZero(0.0f));
  }
  MatrixRowMajor<> readback(count, 12);
  env.getObs(readback);
  EXPECT_EQ(readback, obs);
}

TEST_F(VectorEnvContract, InvalidSizesLeaveTheSimulationAndOutputsUntouched) {
  ASSERT_TRUE(env.reset(obs));
  const MatrixRowMajor<> initial = obs;
  reward.setConstant(7.0f);
  MatrixRowMajor<> bad_action(count, 3);
  EXPECT_FALSE(env.step(bad_action, obs, reward, done, extra));
  MatrixRowMajor<> bad_extra(count, 1);
  EXPECT_FALSE(env.step(acts, obs, reward, done, bad_extra));
  MatrixRowMajor<> bad_reset(count - 1, 12);
  EXPECT_FALSE(env.reset(bad_reset));
  EXPECT_EQ(obs, initial);
  EXPECT_TRUE((reward.array() == 7.0f).all());
  EXPECT_FALSE(done.any());
  env.getObs(obs);
  EXPECT_EQ(obs, initial);
}

TEST_F(VectorEnvContract, RoutesDistinctActionsToTheCorrectLanes) {
  ASSERT_TRUE(env.reset(obs));
  const MatrixRowMajor<> initial = obs;
  acts.row(0).setConstant(-1.0f);
  acts.row(1).setConstant(0.25f);
  acts.row(2).setConstant(1.0f);
  ASSERT_TRUE(env.step(acts, obs, reward, done, extra));
  EXPECT_FALSE(done.any());

  for (int lane = 0; lane < count; ++lane) {
    SCOPED_TRACE(lane);
    const auto angle = initial.row(lane).segment<3>(3);
    const Quaternion attitude =
      Eigen::AngleAxis<Scalar>(angle[0], Vector<3>::UnitZ()) *
      Eigen::AngleAxis<Scalar>(angle[1], Vector<3>::UnitY()) *
      Eigen::AngleAxis<Scalar>(angle[2], Vector<3>::UnitX());
    const Scalar thrust_accel = -Gz * std::max(0.0f, 2.0f * acts(lane, 0) + 1.0f);
    const Vector<3> accel = attitude * Vector<3>(0, 0, thrust_accel) + GVEC;
    const Vector<3> expected_vel =
      initial.row(lane).segment<3>(6).transpose() + 0.02f * accel;
    EXPECT_NEAR((obs.row(lane).segment<3>(6).transpose() - expected_vel).norm(),
                0.0f, 4e-6f);
  }
}

Vector<12> freeFallObservation(const Vector<12>& before, Scalar dt) {
  Vector<12> after = before;
  after.head<3>() += dt * before.segment<3>(6) + (0.5f * dt * dt) * GVEC;
  after.segment<3>(6) += dt * GVEC;
  if (after[2] <= 0.0f) {
    after[2] = 0.0f;
    after.segment<2>(6).setZero();
  }
  return after;
}

TEST_F(VectorEnvContract, TerminalLaneReturnsResetObsAndKeepsItsTerminalReward) {
  ASSERT_TRUE(env.reset(obs));
  const MatrixRowMajor<> initial = obs;
  constexpr Scalar dt = 0.02f;
  Vector<12> terminal = freeFallObservation(obs.row(0).transpose(), dt);
  // Advance lane zero alone to make a mixed terminal/nonterminal batch from
  // the actual random reset. The other lanes still start at their reset state.
  int steps = 0;
  for (; terminal[2] > 0.02f && steps < 80; ++steps) {
    env.testStep(acts, obs, reward, done, extra);
    ASSERT_FALSE(done[0]);
    terminal = freeFallObservation(obs.row(0).transpose(), dt);
  }
  ASSERT_LT(steps, 80);
  ASSERT_GT(obs(0, 2), 0.02f);
  EXPECT_EQ(obs.bottomRows(2), initial.bottomRows(2));

  ASSERT_TRUE(env.step(acts, obs, reward, done, extra));
  EXPECT_TRUE(done[0]);
  EXPECT_GE(obs(0, 2), 4.0f);
  EXPECT_LE(obs(0, 2), 6.0f);
  EXPECT_TRUE(obs.row(0).tail<3>().isZero(0.0f));
  EXPECT_NEAR(reward[0], expectedReward(terminal, acts.row(0).transpose()) - 0.02f,
              3e-5f);
  for (int lane = 1; lane < count; ++lane) {
    SCOPED_TRACE(lane);
    EXPECT_FALSE(done[lane]);
    const Vector<12> expected = freeFallObservation(initial.row(lane).transpose(), dt);
    EXPECT_NEAR((obs.row(lane).transpose() - expected).cwiseAbs().maxCoeff(),
                0.0f, 2e-6f);
    EXPECT_NEAR(reward[lane], expectedReward(expected, acts.row(lane).transpose()),
                2e-6f);
  }
  env.getObs(obs);
  EXPECT_GE(obs(0, 2), 4.0f);
}

}  // namespace
}  // namespace parity
