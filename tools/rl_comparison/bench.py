"""Compare full PPO iterations and isolate collision work on identical trajectories."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch
from world import CpuWorld, CudaWorld, prepare

from ppo import run_ppo


def preflight():
    max_pose = max_reward = max_observation = 0.0
    contacts = 0
    rng = np.random.default_rng(53)
    for queries in (False, True):
        c = CpuWorld(64, 8, queries)
        g = CudaWorld(64, queries)
        try:
            for t in range(256):
                actions = rng.uniform(-0.9, 0.9, (64, 4)).astype(np.float32)
                if t == 0:
                    actions.fill(0)
                c.advance(actions)
                g.advance(actions)
                pose = g.packet()
                assert torch.all(g.valid)
                np.testing.assert_array_equal(c.capture_episodes, pose["episodes"])
                np.testing.assert_array_equal(c.done, g.done.cpu().numpy())
                if queries:
                    h = g.hits.cpu().numpy()
                    np.testing.assert_array_equal(c.hits, h)
                    if t == 0:
                        assert h[0] and h.sum() == 1
                    contacts += int(h.sum())
                pose_error = np.abs(c.poses - np.asarray(pose["poses"])).max()
                reward_error = np.abs(c.rew - g.rew.cpu().numpy()).max()
                max_pose = max(max_pose, pose_error)
                max_reward = max(max_reward, reward_error)
                obs_error = np.abs(c.obs - g.obs.cpu().numpy()).max()
                max_observation = max(max_observation, obs_error)
                np.testing.assert_allclose(
                    c.obs, g.obs.cpu().numpy(), atol=5e-4, rtol=1e-5
                )
                assert pose_error < 5e-4 and reward_error < 5e-4, (
                    t,
                    pose_error,
                    reward_error,
                )
        finally:
            g.close()
            c.close()
    return dict(
        mode="parity",
        transitions=2 * 64 * 256,
        contacts=contacts,
        max_pose_abs=float(max_pose),
        max_reward_abs=float(max_reward),
        max_observation_abs=float(max_observation),
    )


def run(args, n, workers, queries):
    cpu = workers > 0
    world = (
        CpuWorld(n, workers, queries, args.mode == "ppo", poses=False)
        if cpu
        else CudaWorld(n, queries, args.mode == "ppo", poses=False)
    )
    try:
        prepare(world, n, queries)
        if args.mode == "raw":
            # With termination on ground only, the on/off trajectories coincide.
            # Hold commands resident on both backends. Time completed step batches.
            actions = np.zeros((n, 4), np.float32)
            if cpu:

                def advance():
                    world.advance(actions)
            else:
                world.env.actions.zero_()

                def advance():
                    world.env.step(graph=True, random=False)

            samples = []
            for index in range(args.rollouts + 1):
                if not cpu:
                    torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(args.raw_steps):
                    advance()
                if not cpu:
                    torch.cuda.synchronize()
                if index:
                    samples.append((time.perf_counter() - start) * 1e6 / args.raw_steps)
            obs = (
                world.obs
                if cpu
                else world.env.outputs["observation"].cpu().numpy().reshape(n, 12)
            )
            row = dict(
                environments=n,
                us_per_tick=samples,
                median_us=statistics.median(samples),
                steps_per_sample=args.raw_steps,
                final_obs_sum=float(obs.astype(np.float64).sum()),
                final_obs_sha256=hashlib.sha256(obs.tobytes()).hexdigest(),
            )
        else:
            valid = torch.ones(n, device="cuda")
            if cpu:
                # Reuse pinned source/destination storage. The next action download
                # waits for the preceding observation upload before the CPU writes it.
                host = [
                    torch.empty(n * size, pin_memory=True, dtype=dtype)
                    for size, dtype in [
                        (12, torch.float32),
                        (1, torch.float32),
                        (1, torch.uint8),
                    ]
                ]
                host[0] = host[0].view(n, 12)
                world.obs, world.rew, world.done = [v.numpy() for v in host]
                device = [v.to("cuda") for v in host]
                act = torch.empty((n, 4), pin_memory=True)

                def step(action):
                    act.copy_(action)
                    world.advance(act.numpy())
                    for target, source in zip(device, host):
                        target.copy_(source, non_blocking=True)
                    return device[0], device[1], device[2], valid
            else:

                def step(action):
                    world.advance(action)
                    return world.obs, world.rew, world.done, world.valid

            row = run_ppo(args, n, 12, 304, step, lambda: int(world.counts))
        return dict(
            row,
            mode=args.mode,
            backend="cpu" if cpu else "cuda",
            workers=workers,
            collision_queries=queries,
        )
    finally:
        world.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("parity", "raw", "ppo"))
    p.add_argument("--envs", nargs="+", type=int, default=[64, 1024, 4096])
    p.add_argument(
        "--workers",
        nargs="+",
        type=int,
        default=[0, 8, 16],
        help="0 uses resident CUDA; positive uses OpenMP",
    )
    p.add_argument("--queries", nargs="+", type=int, choices=(0, 1), default=[0, 1])
    p.add_argument("--raw-steps", type=int, default=96)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--warmup", type=int, default=24)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatches", type=int, default=4)
    p.add_argument("--rollouts", type=int, default=4)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if (
        min(
            args.envs
            + [
                args.steps,
                args.rollouts,
                args.raw_steps,
                args.warmup,
                args.epochs,
                args.minibatches,
            ]
        )
        < 1
        or min(args.workers) < 0
    ):
        p.error("counts must be positive and workers must be nonnegative")
    if args.mode == "ppo" and args.minibatches > args.steps * min(args.envs):
        p.error("more minibatches than transitions")
    with args.output.open("w") as out:
        if args.mode == "parity":
            print(json.dumps(preflight()), file=out, flush=True)
            return
        for n in args.envs:
            for w in args.workers:
                previous_obs = None
                for q in args.queries:
                    case_start = time.perf_counter()
                    row = run(args, n, w, bool(q))
                    row["case_wall_s"] = time.perf_counter() - case_start
                    if args.mode == "raw":
                        if (
                            previous_obs is not None
                            and previous_obs != row["final_obs_sha256"]
                        ):
                            raise RuntimeError(
                                f"raw trajectories diverged at {n} lanes, {w} workers"
                            )
                        previous_obs = row["final_obs_sha256"]
                    print(json.dumps(row), file=out, flush=True)
                    print(
                        args.mode,
                        n,
                        w,
                        q,
                        round(
                            row.get("training_transitions_per_s", row.get("median_us")),
                            2,
                        ),
                        flush=True,
                    )


if __name__ == "__main__":
    main()
