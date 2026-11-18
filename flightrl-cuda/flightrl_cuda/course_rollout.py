"""Installed Split-S integration check using the stock collector and PPO update."""

import argparse
import torch
from . import ActorCritic, Config, Rollout, TensorEnv, ppo_update


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-collisions", action="store_true")
    moving = parser.parse_args().agent_collisions
    torch.manual_seed(17)
    env = TensorEnv(
        Config(num_envs=1, slots=100),
        random_reset=False,
        scene="split_s",
        collision_radius=0.125,
        terminate_on_collision=True,
        agent_collisions=moving,
    )
    try:
        reset = env.engine.reset_state
        start = torch.tensor([-5.0, -4.5, 3.6], device=env.device)
        reset[:3].copy_(start[:, None])
        if moving:
            # Distinct reset sites for one interacting world. Slots 1 and 2
            # exchange positions during the first step and both terminate.
            ids = torch.arange(100, device=env.device)
            reset[0] = -12.0 + (ids % 10) * 0.7
            reset[1] = -10.0 + (ids // 10) * 0.7
        starts = reset[:3, :3].clone()
        # Cross the first post with both control endpoints clear, then rejoin the group.
        reset[:3, 0] = torch.tensor([-0.1, -1.60, 3.48], device=env.device)
        reset[8, 0] = 70
        if moving:
            reset[:3, 1] = torch.tensor([-7.0, 2.0, 4.0], device=env.device)
            reset[:3, 2] = torch.tensor([-6.0, 2.0, 4.0], device=env.device)
            reset[7, 1] = 100
            reset[7, 2] = -100
        env.reset()
        reset[:3, :3].copy_(starts)
        reset[8, 0] = 0
        reset[7, 1:3] = 0
        env.capture()
        policy = ActorCritic().to(env.device)
        rollout = Rollout(8, env, policy)
        rollout.collect()
        terminals = 3 if moving else 1
        assert (
            rollout.valid.all()
            and rollout.done[0, :terminals].all()
            and not rollout.done[0, terminals:].any()
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
