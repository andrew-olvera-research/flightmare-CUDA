"""Run in Kit's Python. Consume real Flightlib steps, render Split-S, and feed image actions back."""

import argparse
import json
import os
from pathlib import Path
from contextlib import nullcontext
import statistics
import subprocess
import sys
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


class RemoteFixture:
    """Test-only Windows harness. Images stay in Kit. Only poses and actions cross stdio."""

    def __init__(self, script="pose_fixture.py", arguments=()):
        self.proc = subprocess.Popen(
            [
                "docker",
                "compose",
                "exec",
                "-T",
                "-e",
                "PYTHONPATH=/build/flightmare-isaac",
                "dev",
                "/build/flightlib-cuda-python/bin/python",
                "-u",
                "tools/isaac/" + script,
                *arguments,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            cwd=Path(__file__).resolve().parents[2],
        )
        if self.proc.stdout.readline().strip() != "CUDA15_PRODUCER_READY":
            self.close()
            raise RuntimeError("Flightlib producer did not initialize")

    def render(self, renderer, action, terminal, invalidate_second):
        import numpy as np

        batch = self.exchange(
            dict(
                action=None if action is None else action.tolist(),
                terminal=terminal,
                invalidate_second=invalidate_second,
            )
        )
        batch["poses"] = np.asarray(batch["poses"], dtype=np.float32)
        return renderer.render(**batch)

    def exchange(self, request):
        self.proc.stdin.write(json.dumps(request) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"Flightlib fixture exited ({self.proc.poll()})")
        return json.loads(line)

    def close(self):
        self.proc.stdin.close()
        if self.proc.wait(timeout=30) != 0:
            raise RuntimeError("Flightlib fixture failed")


def qualify(args):
    from flightmare_isaac import Camera, IsaacRenderer
    import torch
    from pose_fixture import Fixture

    if args.docker_producer:
        if args.course is None:
            raise ValueError("supply the packaged split_s.usda path with --course")
        source = RemoteFixture()
        course = args.course
    else:
        from importlib.resources import files

        course = args.course or files("flightlib_cuda").joinpath(
            "assets", "split_s.usda"
        )
        source = Fixture()

    source_ready_mib = vram_mib()
    camera = Camera(0, offset=(0, 0, 0), latency_ns=2_000_000)
    cameras = (
        (camera, Camera(1, width=128, height=96, fx=80, offset=(0.16, 0, 0)))
        if args.second_camera
        else (camera,)
    )
    renderer = IsaacRenderer(
        course, world=None, slots=1, actors=(0, 1), cameras=cameras
    )
    action = previous = None
    policy_steps = 0
    elapsed, poses = [], []
    try:
        for tick in range(1, 9):
            terminal = tick == 6
            start = time.perf_counter_ns()
            profile = (
                torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                )
                if args.trace and tick == 3
                else None
            )
            with profile if profile else nullcontext():
                if args.docker_producer:
                    images = source.render(renderer, action, terminal, tick == 4)
                else:
                    source.advance(action, terminal, tick == 4)
                    images = renderer.consume(
                        source.channel, expected_ns=tick * 10_000_000
                    )
                image = images[0]
            elapsed.append((time.perf_counter_ns() - start) / 1e6)
            if profile:
                profile.export_chrome_trace(args.trace)
                if not any(
                    event.device_type == torch.autograd.DeviceType.CUDA
                    for event in profile.events()
                ):
                    raise RuntimeError(
                        "CUDA trace is missing device events. Expose Kit's bundled Torch before startup to initialize CUPTI first."
                    )
            expected_episode = 1 if tick <= 6 else 2
            now = tick * 10_000_000
            assert image.episode == expected_episode and image.acquired_ns == now
            rgb, depth = image.rgb, image.depth_m
            assert image.sequence == tick and rgb.is_cuda and depth.is_cuda
            if args.second_camera:
                if tick == 4:
                    assert images[1] is None
                else:
                    assert images[1].depth_m.shape == (96, 128)
                    assert abs(images[1].depth_m[48, 64].item() - 16.39) < 0.1
            # With 2 ms latency, the previous capture is first usable on this
            # 10 ms control tick. Bootstrap again after an episode change.
            action = None
            if previous and previous.episode == image.episode and not terminal:
                pixels, ranges = previous.observation(
                    episode=image.episode, time_ns=now, acquired_ns=now - 10_000_000
                )
                action = (
                    (
                        (
                            ranges[80:112, 110:146].clamp(0, 20).mean() / 20
                            + pixels[..., 0].float().mean() / 255
                        )
                        * 0.5
                        - 0.5
                    )
                    .expand(4)
                    .contiguous()
                )
                assert torch.isfinite(action).all()
                policy_steps += 1
            previous = image
            if tick in (1, 5, 7):
                center, left, right, top, isolation = [
                    depth[v, u].item()
                    for u, v in ((128, 96), (85, 96), (171, 96), (128, 53), (156, 96))
                ]
                poses.append([center, left, right, top, isolation])
                assert center > 8 and all(
                    abs(x - 2.94) < 0.12 for x in (left, right, top)
                ), poses[-1]
                assert isolation > 8, (
                    "aircraft from a different environment appeared in the gate opening"
                )
                assert rgb[96, 171, 0] > rgb[96, 171, 2]
            if tick == 1:
                first_image_mib = vram_mib()
                retained = image
                reference = depth.clone()
                if args.preview:
                    from PIL import Image

                    Image.fromarray(rgb.cpu().numpy()).save(args.preview)
            if tick == 4:
                assert depth[96, 156].item() > 5, (
                    "another environment appeared in this image"
                )
            if tick == 6:
                # Above the horizon the terminal ground-level camera sees past
                # the former gate post. A ray below the horizon hits the floor.
                print(
                    "CUDA15_TERMINAL_DEPTH",
                    depth[72, 171].item(),
                    depth[96, 171].item(),
                    flush=True,
                )
                assert depth[72, 171].item() > 8, (
                    "terminal capture still shows the previous gate"
                )
        assert policy_steps == 5 and torch.equal(reference, retained.depth_m)
        for episode, now in ((2, 12_000_000), (1, 11_999_999)):
            try:
                retained.observation(
                    episode=episode, time_ns=now, acquired_ns=10_000_000
                )
            except ValueError:
                continue
            raise AssertionError("stale or early policy observation was accepted")
        result = dict(
            ticks=8,
            policy_decisions=policy_steps,
            cameras=len(cameras),
            environments=2,
            agents_per_environment=1,
            pixels=poses,
            resolution=[256, 192],
            first_ms=elapsed[0],
            median_warm_loop_ms=statistics.median(elapsed[1:]),
            producer_ready_mib=source_ready_mib,
            first_image_mib=first_image_mib,
            end_gpu_mib=vram_mib(),
            torch=str(torch.__version__),
            producer="docker-stdio" if args.docker_producer else "installed-in-process",
            rgb_bytes=retained.rgb.numel(),
            depth_bytes=retained.depth_m.numel() * 4,
        )
    finally:
        try:
            renderer.close()
        finally:
            source.close()
    print("CUDA15_PASS " + json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docker-producer", action="store_true")
    parser.add_argument("--course")
    parser.add_argument("--preview")
    parser.add_argument(
        "--trace", help="write a Torch/CUPTI trace of one warm pose-to-image tick"
    )
    parser.add_argument("--second-camera", action="store_true")
    args, kit_args = parser.parse_known_args()
    sys.argv = [sys.argv[0], *kit_args]
    print("CUDA15_PRE_KIT_GPU_MIB", vram_mib(), flush=True)
    if args.trace:
        # Kit 6.2 can claim the sole CUPTI subscriber during startup. Give
        # Kineto first use when the bundled Torch is on Python's search path.
        from importlib.util import find_spec

        if find_spec("torch"):
            import torch

            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA]
            ):
                torch.zeros(1, device="cuda").sum().item()
    from flightmare_isaac import launch_visual_app

    app = launch_visual_app()
    try:
        print("CUDA15_READY_GPU_MIB", vram_mib(), flush=True)
        qualify(args)
    except BaseException:
        # Some Kit versions exit the process during close. Preserve failure for CI.
        import traceback

        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    app.close()


if __name__ == "__main__":
    main()
