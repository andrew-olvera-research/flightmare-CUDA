"""Shared timing loop for state and camera PPO benchmarks."""
import math
import subprocess
import time


def vram_mib():
    value = subprocess.check_output(
        [
            "nvidia-smi",
            "--id=0",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return int(value.strip())


def run_ppo(args, cameras, inputs, hidden, step, counts):
    """Common timed PPO loop for camera and state observations."""
    import torch
    from torch import nn

    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(inputs, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 5),
            )
            self.log_std = nn.Parameter(torch.full((4,), -0.7))

        def forward(self, obs):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = self.net(obs)
            out = out.float()
            return out[..., :4], out[..., 4]

        def logp(self, mu, latent):
            return (
                -0.5
                * (
                    (latent - mu).square() * (-2 * self.log_std).exp()
                    + 2 * self.log_std
                    + math.log(2 * math.pi)
                )
            ).sum(-1)

    torch.manual_seed(17)
    policy = Policy().cuda()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4, fused=True)

    def storage(*tail):
        return torch.empty((args.steps, cameras, *tail), device="cuda")

    obs_buf, latent_buf = storage(inputs), storage(4)
    rewards, dones, valids, values, logps, advantages = (storage() for _ in range(6))
    action = torch.zeros((cameras, 4), device="cuda")
    with torch.no_grad():
        for _ in range(args.warmup):
            obs, *_ = step(action)
            mu, _ = policy(obs)
            action = mu.tanh()
    base_counts = counts()
    collection, updates = [], []
    torch.cuda.synchronize()
    for _ in range(args.rollouts):
        start = time.perf_counter()
        with torch.no_grad():
            for t in range(args.steps):
                obs_buf[t].copy_(obs)
                mu, value = policy(obs)
                latent = mu + policy.log_std.exp() * torch.randn_like(mu)
                latent_buf[t].copy_(latent)
                values[t].copy_(value)
                logps[t].copy_(policy.logp(mu, latent))
                obs, r, d, v = step(latent.tanh())
                rewards[t].copy_(r)
                dones[t].copy_(d)
                valids[t].copy_(v)
            _, bootstrap = policy(obs)
        torch.cuda.synchronize()
        collection.append(time.perf_counter() - start)
        start = time.perf_counter()
        if not valids.all():
            raise RuntimeError("invalid Flightlib transition")
        with torch.no_grad():
            gae = torch.zeros_like(bootstrap)
            for t in reversed(range(args.steps)):
                cont = 1 - dones[t]
                delta = rewards[t] + 0.99 * cont * bootstrap - values[t]
                gae = delta + 0.99 * 0.95 * cont * gae
                advantages[t].copy_(gae)
                bootstrap = values[t]
            targets = (advantages + values).flatten()
            adv = (advantages.flatten() - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )
        for _ in range(args.epochs):
            order = torch.randperm(args.steps * cameras, device="cuda")
            for idx in order.tensor_split(args.minibatches):
                mu, v = policy(obs_buf.flatten(0, 1)[idx])
                ratio = (
                    policy.logp(mu, latent_buf.flatten(0, 1)[idx])
                    - logps.flatten()[idx]
                ).exp()
                gain = torch.minimum(ratio * adv[idx], ratio.clamp(0.8, 1.2) * adv[idx])
                loss = -gain.mean() + 0.5 * (v - targets[idx]).square().mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    policy.parameters(), 0.5, error_if_nonfinite=True
                )
                optimizer.step()
        torch.cuda.synchronize()
        updates.append(time.perf_counter() - start)
    hit = int(counts() - base_counts)
    assert torch.isfinite(loss)
    ticks = args.rollouts * args.steps
    seconds = sum(collection)
    return dict(
        environments=cameras,
        agents_per_environment=1,
        control_hz=50,
        params=sum(p.numel() for p in policy.parameters()),
        gpu=torch.cuda.get_device_name(),
        torch=str(torch.__version__),
        amp="bf16",
        epochs=args.epochs,
        minibatches=args.minibatches,
        steps=args.steps,
        rollouts=args.rollouts,
        warmup=args.warmup,
        collection_s=collection,
        update_s=updates,
        ticks_per_s=ticks / seconds,
        collection_transitions_per_s=ticks * cameras / seconds,
        training_transitions_per_s=ticks * cameras / (seconds + sum(updates)),
        realtime_factor=ticks * 0.02 / seconds,
        course_collision_hits=hit,
        board_mib=vram_mib(),
    )
