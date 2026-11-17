#pragma once
#include "flightlib_cuda/dynamics.hpp"
namespace flightlib_cuda::v1 {
struct TaskConfig {
  std::array<Scalar,4> action_mean{2.4525f,2.4525f,2.4525f,2.4525f};
  std::array<Scalar,4> action_scale{4.905f,4.905f,4.905f,4.905f};
  std::array<Scalar,12> goal{0,0,5,0,0,0,0,0,0,0,0,0};
  std::array<Scalar,5> reward_coeff{-.002f,-.002f,-.0002f,-.0002f,-.0002f};
};
TaskConfig legacyTaskConfig(Scalar mass);
// Explicit synchronous setup. One configuration broadcasts; otherwise require N.
void configureTask(Simulation&,const std::vector<TaskConfig>&);
enum class ResetSource { Explicit, Random };
enum class TaskExecution { FusedReset, SeparateReset };
// Bind once: action[E,A,4], selected[N], reset_x[25,N], reset_time[N].
// Explicit automatic reset requires selected=1 for every terminal lane.
Bindings bindTaskInputs(Simulation&,BindingRequest action,BindingRequest selected,
                        BindingRequest reset_x,BindingRequest reset_time);
void resetEnvironment(Simulation&,const Bindings&,ResetSource=ResetSource::Explicit);
// Three launches by default, four for the reference split T/R path. No host decisions.
void stepEnvironment(Simulation&,const Bindings&,ResetSource=ResetSource::Random,
                     TaskExecution=TaskExecution::FusedReset);
}
