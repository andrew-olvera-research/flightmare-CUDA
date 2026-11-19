"""Installed example: collect a GPU batch with a caller-owned policy and history."""
import torch
from torch import nn

from .config import Config
from .env import TensorEnv


@torch.no_grad()
def collect(env, policy, steps):
    """Start from the current observation and retain both sides of each step."""
    obs = env.result.observation
    history = {
        name: tensor.new_empty((steps, *tensor.shape))
        for name, tensor in (
            ("observation", obs), ("action", env.actions),
            ("transition_observation", env.result.transition_observation),
            ("reward", env.result.reward), ("done", env.result.done),
            ("valid", env.result.valid),
        )
    }
    for t in range(steps):
        history["observation"][t].copy_(obs)
        action = policy(obs)
        history["action"][t].copy_(action)
        result = env.step(action)
        # Step outputs borrow storage. Save this tick before the next step overwrites it.
        for name in ("transition_observation", "reward", "done", "valid"):
            history[name][t].copy_(getattr(result, name))
        obs = result.observation  # The next policy input includes any reset.
    return history


def main():
    torch.manual_seed(17)
    device = torch.device("cuda", 0)
    with torch.cuda.stream(torch.cuda.Stream(device=device)):
        env = TensorEnv(Config(num_envs=2, slots=3), random_reset=False)
        try:
            policy = nn.Sequential(nn.Linear(12, 4), nn.Tanh()).to(device)
            # Start lane zero at ground level; all subsequent resets are at 5 m.
            env.engine.reset_state[2].fill_(5)
            env.engine.reset_state[2, 0] = 0
            env.reset()
            env.engine.reset_state[2, 0] = 5
            env.capture()
            history = collect(env, policy, 4)

            # Host checks only after collection. Grouped outputs borrow storage.
            assert env.grouped.observation.shape == (2, 3, 12)
            assert env.grouped.observation.data_ptr() == env.result.observation.data_ptr()
            assert history["valid"].all() and history["done"][0, 0]
            assert history["transition_observation"][0, 0, 2] <= .02
            assert history["observation"][1, 0, 2] == 5
            print("Collected 24 valid transitions, including a same-step reset.")
        finally:
            env.close()


if __name__ == "__main__":
    main()
