"""Generate an instrumented benchmark TU without modifying the frozen CPU oracle."""

from pathlib import Path
import sys


def insert(source, marker, replacement):
    if source.count(marker) != 1:
        raise ValueError(f"CPU benchmark instrumentation needs review at {marker!r}")
    return source.replace(marker, replacement)


root, output = map(Path, sys.argv[1:])
aircraft = (root / "flightlib/src/objects/quadrotor.cpp").read_text()
aircraft = insert(
    aircraft,
    "  QuadState old_state = state_;",
    "  QuadState old_state = state_;\n  if (probe) probe->begin(state_.p.data());",
)
aircraft = insert(
    aircraft,
    "    state_.x = next_state.x;",
    "    if (probe) probe->arc(state_.p.data(), next_state.p.data(), state_.a.data(), sim_dt);\n    state_.x = next_state.x;",
)
aircraft = insert(
    aircraft,
    "  constrainInWorldBox(old_state);",
    "  constrainInWorldBox(old_state);\n  if (probe) probe->boundary(state_.p.data());",
)
env = (root / "flightlib/src/envs/quadrotor_env/quadrotor_env.cpp").read_text()
env = insert(
    env,
    "  quadrotor_ptr_->reset(quad_state_);",
    "  if (reset_state) quad_state_ = *reset_state;\n  quadrotor_ptr_->reset(quad_state_);",
)
env = insert(
    env,
    "  quadrotor_ptr_->getState(&quad_state_);",
    "  quadrotor_ptr_->getState(&quad_state_);\n  if (pose_out) std::copy_n(quad_state_.x.data(), 7, pose_out);",
)
output.write_text(aircraft + "\n" + env)
