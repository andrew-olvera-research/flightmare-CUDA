"""Collision-enabled single-aircraft FPV PPO, using the in-process Newton prototype."""

import argparse
import json
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F
from flightlib_cuda.transport import Backpressure
from tiled import TiledCamera

sys.path.append(str(Path(__file__).resolve().parents[1] / "isaac"))
from bench_fpv import run_ppo
from rollout_fixture import RacingWorld


def benchmark(count, args):
    world = RacingWorld(
        count,
        scene=args.scene,
        pose_destination="device" if args.mode == "subscription" else None,
    )
    try:
        camera = TiledCamera(count, scene=world.env.scene)
        if args.pixel_priority:
            camera.camera.default_render_config.render_order = (
                camera.camera.RenderOrder.PIXEL_PRIORITY
            )
        tick = 1  # the common racing fixture already consumed its forced-crossing tick

        def step(action):
            nonlocal tick
            try:
                reward, done, valid = world.step(action)
            except Backpressure:
                # Reservation fails before stepping. Wait for retired reader slots.
                camera.stream.synchronize()
                reward, done, valid = world.step(action)
            tick += 1
            frame = camera.consume(world.channel, tick, tick * 20_000_000)
            obs = F.adaptive_avg_pool2d(
                frame.rgb.permute(0, 3, 1, 2).float(), (16, 16)
            ).flatten(1)
            obs.mul_(1 / 255).masked_fill_(done.bool()[:, None], 0)
            return obs, reward, done, valid.bool() & frame.valid

        if args.mode == "graph":
            # Compile both Warp kernels before framework capture.
            camera.draw(world.env.state[:7].T)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=camera.stream):
                outputs = camera.record_tick(world.env, world.counts)

            def step(action):
                world.env.actions[:, 0].copy_(action)
                graph.replay()
                return outputs

        if args.profile_ticks:
            if args.mode != "graph":
                raise ValueError("--profile-ticks requires --mode graph")
            for _ in range(16):
                graph.replay()
            camera.stream.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            start = time.perf_counter()
            for _ in range(args.profile_ticks):
                graph.replay()
            submitted = time.perf_counter()
            camera.stream.synchronize()
            end = time.perf_counter()
            torch.cuda.cudart().cudaProfilerStop()
            return dict(
                envs=count,
                replays=args.profile_ticks,
                submit_ms=(submitted - start) * 1000,
                elapsed_ms=(end - start) * 1000,
            )
        result = run_ppo(args, count, 768, 112, step, lambda: int(world.counts))
        result.update(
            renderer="Newton SensorTiledCamera",
            resolution=[256, 192],
            torch_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
            torch_peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
            mode=args.mode,
            scene=args.scene,
            render_order=camera.camera.default_render_config.render_order.name,
            pose_host_bytes=world.channel.stats["host_bytes"] if world.channel else 0,
        )
        return result
    finally:
        world.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=int, required=True)
    parser.set_defaults(epochs=4, warmup=24)
    parser.add_argument(
        "--scene", default="split_s", help="course alias or authored scene path"
    )
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument(
        "--minibatches",
        type=int,
        default=4,
        help="optimizer minibatches per rollout epoch",
    )
    parser.add_argument("--mode", choices=("graph", "subscription"), default="graph")
    parser.add_argument(
        "--profile-ticks",
        type=int,
        default=0,
        help="profile only warmed graph replays with fixed device actions",
    )
    parser.add_argument(
        "--pixel-priority",
        action="store_true",
        help="use baseline cross-view pixel traversal for A/B comparisons",
    )
    args = parser.parse_args()
    if (
        min(args.envs, args.steps, args.rollouts, args.minibatches) < 1
        or args.minibatches > args.envs * args.steps
    ):
        parser.error("positive sizes required, with at least one sample per minibatch")
    with torch.cuda.stream(torch.cuda.Stream()):
        print(json.dumps(benchmark(args.envs, args)))
