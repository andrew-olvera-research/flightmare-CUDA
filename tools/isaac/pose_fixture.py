"""CUDA-15 qualification fixture. The optional stdio path crosses Windows/WSL only for testing."""

import json
import math
import sys


class Fixture:
    def __init__(self):
        import torch
        from flightlib_cuda import Environment, Subscription

        self.env = Environment(
            2,
            1,
            scene="split_s",
            subscriptions=(Subscription("pose", destination="host", required=True),),
        )
        (self.channel,) = self.env.subscriptions
        self.env.reset_state[:3] = torch.tensor(
            [[-0.9, -0.55], [-4.27, -2.27], [3.48, 3.48]], device=self.env.device
        )
        self.env.reset_state[3, 0] = self.env.reset_state[6, 0] = math.sqrt(0.5)
        self.env.reset()
        self.env.capture(random=False)
        self.tick = 0

    def advance(self, action, terminal=False, invalidate_second=False):
        import torch

        if terminal:
            self.env.state[2, 0] = 0.01
        if action is not None:
            self.env.actions[0, 0].copy_(
                torch.as_tensor(action, device=self.env.device)
            )
        else:
            self.env.actions[0, 0].zero_()
        self.env.actions[1, 0].fill_(float("nan") if invalidate_second else 0)
        self.env.step(graph=True, random=False)
        self.tick += 1

    def close(self):
        self.env.close_subscriptions()


def main():
    fixture = Fixture()
    try:
        print("CUDA15_PRODUCER_READY", flush=True)
        for line in sys.stdin:
            request = json.loads(line)
            fixture.advance(**request)
            with fixture.channel.acquire(wait=True) as f:
                print(
                    json.dumps(
                        dict(
                            sequence=f.sequence,
                            requested_ns=f.requested_ns,
                            acquired_ns=f.acquired_ns,
                            lanes=f.lane.tolist(),
                            episodes=f.episode.tolist(),
                            valid=f.valid.tolist(),
                            poses=f.values.tolist(),
                        )
                    ),
                    flush=True,
                )
    finally:
        fixture.close()


if __name__ == "__main__":
    main()
