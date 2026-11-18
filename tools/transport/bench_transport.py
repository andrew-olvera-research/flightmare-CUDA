"""Time installed headless steps and trace exactly 20 steady-state subscription ticks."""

import argparse
import statistics
import time

import torch
from flightlib_cuda import Environment, ImuConfig, Subscription, _native


def run(mode, selected, trace):
    specs = ()
    actors = tuple((0, i) for i in range(selected))
    if mode == "device":
        specs = (Subscription("pose", actors, depth=8, required=True),)
    if mode == "host":
        specs = (
            Subscription("imu", actors, destination="host", depth=2, required=True),
        )
    env = Environment(
        1, 1000, imu=ImuConfig() if mode == "host" else None, subscriptions=specs
    )
    env.reset_state[2] = 5
    env.reset()
    env.capture(random=False)
    producer = torch.cuda.current_stream()

    def batch():
        for _ in range(20):
            env.step(graph=True, random=False)
            for ch in env.subscriptions:
                with ch.acquire(wait=True) as f:
                    if ch.host:
                        assert f.count == selected * 2
        producer.synchronize()

    batch()
    start_bytes = sum(ch.stats["host_bytes"] for ch in env.subscriptions)
    if trace:
        _native.profile_start()
    times = []
    for _ in range(1 if trace else 9):
        start = time.perf_counter_ns()
        batch()
        times.append((time.perf_counter_ns() - start) / 20_000)
    if trace:
        _native.profile_stop()
    host_bytes = sum(ch.stats["host_bytes"] for ch in env.subscriptions) - start_bytes
    steps = 20 * len(times)
    print(
        f"{mode=} actors={selected} {steps=} median_us={statistics.median(times):.2f} "
        f"d2h_bytes_per_step={host_bytes // steps}"
    )
    env.close_subscriptions()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["disabled", "device", "host"])
    parser.add_argument("--selected", type=int, default=4)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    run(args.mode, args.selected, args.trace)
