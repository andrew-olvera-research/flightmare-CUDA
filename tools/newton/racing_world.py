"""Split-S rollout fixture for the in-process camera benchmark."""


class RacingWorld:
    def __init__(self, envs, *, pose_destination="host", scene="split_s"):
        import torch
        from flightlib_cuda import Environment, Subscription

        self.env = env = Environment(
            envs,
            1,
            control_dt=0.02,
            scene=scene,
            collision_radius=0.125,
            terminate_on_collision=True,
            subscriptions=(
                Subscription("pose", destination=pose_destination, required=True),
            )
            if pose_destination
            else (),
        )
        self.channel = env.subscriptions[0] if env.subscriptions else None
        reset = env.reset_state
        lane = torch.arange(envs, device=env.device)
        reset[0] = -0.9 + (lane % 5 - 2) * 0.1
        reset[1] = -4.27 - (lane % 3) * 0.2
        reset[2] = 3.48
        reset[3] = reset[6] = 0.5**0.5
        env.reset()
        # Force a swept gate-post crossing in lane zero before timing.
        if scene == "split_s":
            env.state[:3, 0] = torch.tensor([-0.1, -1.60, 3.48], device=env.device)
            env.state[8, 0] = 70
        env.capture(random=False)
        out = env.step(graph=True, random=False)
        if self.channel:
            with self.channel.acquire(wait=True):
                pass
        if scene == "split_s":
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
