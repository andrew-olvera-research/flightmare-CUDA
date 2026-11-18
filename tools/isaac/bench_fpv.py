"""Measure collision-enabled, 50 Hz RGB collection and PPO updates in Isaac Kit."""

import argparse
from contextlib import ExitStack
import json
import math
import os
import sys
import subprocess
import tempfile
from pathlib import Path
import time

from qualify import RemoteFixture, vram_mib


def benchmark(args, cameras):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from flightmare_isaac import Camera, IsaacRenderer
    from rollout_fixture import RacingWorld

    with ExitStack() as clean:
        if args.cpu_workers or args.no_course_queries:
            world = RemoteFixture(
                "../rl_comparison/world.py",
                (
                    str(cameras),
                    "cpu" if args.cpu_workers else "cuda",
                    str(args.cpu_workers),
                    str(int(not args.no_course_queries)),
                ),
            )
        elif args.docker_producer:
            world = RemoteFixture("rollout_fixture.py", (str(cameras),))
        else:
            world = RacingWorld(cameras)
        clean.callback(world.close)
        renderer = IsaacRenderer(
            args.course,
            world=None,
            slots=1,
            actors=range(cameras),
            cameras=tuple(
                Camera(
                    i,
                    width=args.width,
                    height=args.height,
                    fx=args.width * 0.625,
                    depth=False,
                )
                for i in range(cameras)
            ),
        )
        clean.callback(renderer.close)
        tick = 1
        render_ms = []

        def step(action):
            nonlocal tick
            tick += 1
            if args.docker_producer:
                pose, stats = world.exchange(action.tolist())
                pose["poses"] = np.asarray(pose["poses"], dtype=np.float32)
                rendering = time.perf_counter()
                frames = renderer.render(**pose)
            else:
                stats = world.step(action)
                rendering = time.perf_counter()
                frames = renderer.consume(world.channel, expected_ns=tick * 20_000_000)
            render_ms.append((time.perf_counter() - rendering) * 1000)
            if args.docker_producer:
                stats = torch.tensor(stats, dtype=torch.float32, device="cuda")
            if any(
                f is None or f.depth_m is not None or f.acquired_ns != tick * 20_000_000
                for f in frames
            ):
                raise ValueError("missing RGB-only capture at the requested tick")
            rgb = torch.stack([f.rgb for f in frames]).permute(0, 3, 1, 2)
            obs = F.adaptive_avg_pool2d(rgb.float(), (16, 16)).flatten(1).mul_(1 / 255)
            reward, done, valid = stats
            # The terminal image depicts the previous episode. A black initial
            # observation bootstraps the new episode until its first exposure.
            return obs.masked_fill_(done.bool().unsqueeze(1), 0), reward, done, valid

        def counts():
            return np.array(
                world.exchange(None) if args.docker_producer else world.counts.tolist()
            )

        result = run_ppo(args, cameras, 768, 112, step, counts)
        timed_render = render_ms[args.warmup :]
        return dict(
            result,
            cameras=cameras,
            resolution=[args.width, args.height],
            render_s=sum(timed_render) / 1000,
            render_p50_ms=float(np.percentile(timed_render, 50)),
            render_p95_ms=float(np.percentile(timed_render, 95)),
            simulator="cpu" if args.cpu_workers else "cuda",
            cpu_workers=args.cpu_workers,
            collision_queries=not args.no_course_queries,
            producer="docker-stdio" if args.docker_producer else "in-process",
        )


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docker-producer", action="store_true")
    parser.add_argument("--course")
    parser.add_argument(
        "--cpu-workers",
        type=int,
        default=0,
        help="use this many OpenMP simulator workers",
    )
    parser.add_argument("--no-course-queries", action="store_true")
    parser.add_argument(
        "--results",
        type=Path,
        help="write one JSON record per completed camera count",
    )
    parser.add_argument("--env-counts", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--rollouts", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatches", type=int, default=4)
    args, kit_args = parser.parse_known_args()
    if (
        min(args.env_counts) < 1
        or min(args.warmup, args.epochs, args.minibatches, args.rollouts, args.steps)
        < 1
        or args.minibatches > args.steps * min(args.env_counts)
    ):
        parser.error("invalid world, rollout or minibatch dimensions")
    if args.cpu_workers < 0 or (
        (args.cpu_workers or args.no_course_queries) and not args.docker_producer
    ):
        parser.error(
            "producer comparison requires --docker-producer and nonnegative worker count"
        )
    if args.docker_producer and not args.course:
        parser.error("--docker-producer requires a local --course")
    if len(args.env_counts) > 1:
        # A fresh Kit process per point releases RTX descriptor pools and
        # textures. Keep startup outside each collection measurement.
        with ExitStack() as stack:
            directory = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            output = (
                stack.enter_context(args.results.open("w")) if args.results else None
            )
            for count in args.env_counts:
                path = directory / f"{count}.jsonl"
                subprocess.run(
                    [
                        sys.executable,
                        __file__,
                        *sys.argv[1:],
                        "--env-counts",
                        str(count),
                        "--results",
                        str(path),
                    ],
                    check=True,
                )
                record = json.loads(path.read_text())
                if record["environments"] != count:
                    raise ValueError("incorrect environment count in benchmark result")
                if output:
                    print(json.dumps(record), file=output, flush=True)
        return
    sys.argv = [sys.argv[0], *kit_args]
    from flightmare_isaac import launch_visual_app

    app = launch_visual_app()
    try:
        if not args.course:
            from importlib.resources import files

            args.course = files("flightlib_cuda").joinpath("assets", "split_s.usda")
        record = json.dumps(benchmark(args, args.env_counts[0]))
        if args.results:
            args.results.write_text(record + "\n")
        print("CUDA15_FPV " + record, flush=True)
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    app.close()


if __name__ == "__main__":
    main()
