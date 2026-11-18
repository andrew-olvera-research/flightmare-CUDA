"""Completed graph-step time and native arena bytes for resident sensor streams."""

import argparse
import statistics
import torch
from flightlib_cuda import Environment, ImuConfig, OdometryConfig, _native


def environment(lanes, mode):
    imu = (
        ImuConfig(
            rate_hz=200,
            latency_s=0.003,
            gyro_noise_density=0.0002,
            accel_noise_density=0.002,
            gyro_random_walk=0.00002,
            accel_random_walk=0.0002,
        )
        if mode != "disabled"
        else None
    )
    odom = (
        OdometryConfig(rate_hz=50, latency_s=0.01, position_std=0.01)
        if mode == "both"
        else None
    )
    env = Environment(1, lanes, imu=imu, odometry=odom)
    env.reset_state[2] = 5
    env.reset()
    env.capture(random=False)
    for _ in range(20):
        env.step(graph=True, random=False)
    env.synchronize()
    return env


def benchmark(lanes):
    modes = ("disabled", "imu", "both")
    envs = [environment(lanes, mode) for mode in modes]
    times = [[] for _ in modes]
    for rep in range(9):
        for i in range(3) if rep % 2 == 0 else reversed(range(3)):
            env = envs[i]
            env.reset()
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            for _ in range(50):
                env.step(graph=True, random=False)
            end.record()
            end.synchronize()
            times[i].append(start.elapsed_time(end) * 1000 / 50)
            assert env.outputs["valid"].all()
    for mode, env, t in zip(modes, envs, times):
        print(lanes, mode, f"{statistics.median(t):.2f}", env._native.arena_bytes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", choices=["both", "imu", "disabled"])
    args = parser.parse_args()
    if args.trace:
        env = environment(1000, args.trace)
        _native.profile_start()
        for _ in range(20):
            env.step(graph=True, random=False)
        env.synchronize()
        _native.profile_stop()
    else:
        print(torch.cuda.get_device_name())
        print("lanes mode us_per_step native_arena_bytes")
        for n in (100, 1000):
            benchmark(n)
