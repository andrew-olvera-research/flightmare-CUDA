"""Flightlib side of the FPV throughput benchmark. Also runs inside Kit on Linux."""

import argparse
import json
import sys


class RacingWorld:
    def __init__(self, envs):
        import torch
        from flightlib_cuda import Environment, Subscription

        self.env = env = Environment(
            envs,
            1,
            control_dt=0.02,
            scene="split_s",
            collision_radius=0.125,
            terminate_on_collision=True,
            subscriptions=(Subscription("pose", destination="host", required=True),),
        )
        (self.channel,) = env.subscriptions
        reset = env.reset_state
        lane = torch.arange(envs, device=env.device)
        reset[0] = -0.9 + (lane % 5 - 2) * 0.1
        reset[1] = -4.27 - (lane % 3) * 0.2
        reset[2] = 3.48
        reset[3] = reset[6] = 0.5**0.5
        env.reset()
        # Force a swept gate-post crossing in lane zero before timing.
        env.state[:3, 0] = torch.tensor([-0.1, -1.60, 3.48], device=env.device)
        env.state[8, 0] = 70
        env.capture(random=False)
        out = env.step(graph=True, random=False)
        with self.channel.acquire(wait=True):
            assert env.collision_hit[0] and out["done"][0, 0]
            assert not env.collision_hit[1:].bool().any()
        self.counts = torch.zeros((), device=env.device, dtype=torch.int64)

    def step(self, action):
        import torch

        env = self.env
        env.actions[:, 0].copy_(torch.as_tensor(action, device=env.device))
        out = env.step(graph=True, random=False)
        self.counts.add_(env.collision_hit.sum())
        return torch.stack([out[k].reshape(-1) for k in ("reward", "done", "valid")])

    def close(self):
        self.env.close_subscriptions()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("envs", type=int)
    args = parser.parse_args()
    world = RacingWorld(args.envs)
    try:
        print("CUDA15_PRODUCER_READY", flush=True)
        for line in sys.stdin:
            action = json.loads(line)
            if action is None:
                print(json.dumps(world.counts.tolist()), flush=True)
                continue
            stats = world.step(action)
            with world.channel.acquire(wait=True) as f:
                result = dict(
                    sequence=f.sequence,
                    requested_ns=f.requested_ns,
                    acquired_ns=f.acquired_ns,
                    lanes=f.lane.tolist(),
                    episodes=f.episode.tolist(),
                    valid=f.valid.tolist(),
                    poses=f.values.tolist(),
                )
                print(json.dumps([result, stats.tolist()]), flush=True)
    finally:
        world.close()


if __name__ == "__main__":
    main()
