"""Compare completed headless graph steps with and without the shared Split-S course."""

import argparse
import statistics

import torch
from flightlib_cuda import Environment


def measure(lanes, steps, repeats):
    environments = [
        Environment(1, lanes, control_dt=0.01, scene=scene, collision_radius=0.125)
        for scene in (None, "split_s")
    ]
    timings = [[], []]
    try:
        for env in environments:
            # Hover in the first opening or on its post. Query-only preserves both groups.
            env.reset_state[0, ::2] = -0.9
            env.reset_state[0, 1::2] = -0.1
            env.reset_state[1] = -1.27
            env.reset_state[2] = 3.48
            env.reset()
            env.capture(random=False)
            for _ in range(20):
                env.step(graph=True, random=False)
        for repeat in range(repeats):
            for i in range(2) if repeat % 2 == 0 else reversed(range(2)):
                env = environments[i]
                env.reset()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(steps):
                    env.step(graph=True, random=False)
                end.record()
                end.synchronize()
                timings[i].append(start.elapsed_time(end) * 1000 / steps)
                assert env.outputs["valid"].all()
        return [statistics.median(samples) for samples in timings]
    finally:
        for env in environments:
            env.synchronize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lanes", nargs="+", type=int, default=[1, 1000, 65536])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    print(torch.cuda.get_device_name(), flush=True)
    print("lanes  headless_us  split_s_us  difference_pct", flush=True)
    for lanes in args.lanes:
        base, scene = measure(lanes, args.steps, args.repeats)
        print(
            f"{lanes:5} {base:12.2f} {scene:10.2f} {(scene / base - 1) * 100:+15.2f}",
            flush=True,
        )
