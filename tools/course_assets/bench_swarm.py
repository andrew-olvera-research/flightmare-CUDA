"""Measure CUDA graph environment steps on identical moving formations."""

import statistics
import torch
from flightlib_cuda import Environment


def benchmark(slots, steps=50, repeats=9):
    modes = {
        "disabled": (None, False),
        "static": ("split_s", False),
        "peers": (None, True),
        "combined": ("split_s", True),
    }
    envs, samples = {}, {name: [] for name in modes}
    i = torch.arange(slots, device="cuda")
    cell = i // 2
    for name, (scene, moving) in modes.items():
        env = Environment(
            1, slots, scene=scene, agent_collisions=moving, collision_radius=0.125
        )
        envs[name] = env
        # Quarter as many overlapping pairs as agents. Other paired owners
        # are separated by 0.4 m. Each cell has the same translational velocity.
        env.reset_state[0] = (cell % 10) - 5.0 + (i % 2) * (0.2 + 0.2 * (cell % 2))
        env.reset_state[1] = (cell // 10 % 10) - 5.0
        env.reset_state[2] = 2.0 + (cell // 100) * 0.6
        env.reset_state[7] = 0.2
        env.reset()
        env.capture(random=False)
        for _ in range(20):
            env.step(graph=True, random=False)
    for repeat in range(repeats):
        for name in list(modes) if repeat % 2 == 0 else reversed(modes):
            env = envs[name]
            env.reset()
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            for _ in range(steps):
                env.step(graph=True, random=False)
            end.record()
            end.synchronize()
            samples[name].append(1000 * start.elapsed_time(end) / steps)
            assert env.outputs["valid"].all()
            if modes[name][1]:
                assert (
                    not env.candidate_overflow.item()
                    and env.candidate_count.item() == slots // 4
                )
                assert env.agent_collision_hit.cpu().tolist().count(1) == slots // 2
    for name, env in envs.items():
        count = env.candidate_count.item() if modes[name][1] else 0
        capacity = env.candidate_pairs.shape[1] if modes[name][1] else 0
        print(
            slots,
            name,
            f"{statistics.median(samples[name]):.2f}",
            env._native.arena_bytes,
            count,
            capacity,
            flush=True,
        )


if __name__ == "__main__":
    print(torch.cuda.get_device_name())
    print("agents mode us_per_step native_arena_bytes candidates capacity")
    for slots in (100, 1000):
        benchmark(slots)
