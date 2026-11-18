"""Installed Split-S integration check using the stock collector and PPO update."""

import torch
from . import ActorCritic, Config, Rollout, TensorEnv, ppo_update


def main():
    torch.manual_seed(17)
    env = TensorEnv(
        Config(num_envs=1, slots=100),
        random_reset=False,
        scene="split_s",
        collision_radius=0.125,
        terminate_on_collision=True,
    )
    try:
        reset = env.engine.reset_state
        start = torch.tensor([-5.0, -4.5, 3.6], device=env.device)
        reset[:3].copy_(start[:, None])
        # One vehicle starts on the first gate post. After reset it rejoins the others.
        reset[:3, 0] = torch.tensor([-0.1, -1.27, 3.48], device=env.device)
        env.reset()
        reset[:3, 0].copy_(start)
        env.capture()
        policy = ActorCritic().to(env.device)
        rollout = Rollout(8, env, policy)
        rollout.collect()
        assert (
            rollout.valid.all() and rollout.done[0, 0] and not rollout.done[0, 1:].any()
        )
        rollout.finish()
        optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
        assert torch.isfinite(ppo_update(policy, optimizer, rollout, epochs=1)).all()
        print(
            "Validated 800 Split-S transitions, collision autoreset, GAE and PPO update."
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
