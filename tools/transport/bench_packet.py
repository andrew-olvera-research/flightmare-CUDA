"""Run matched native plant + synthetic unicast workloads and emit one JSON record per run."""
import argparse
import json
import math
import random
import statistics
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("driver", help="built tools/profile/native profile_driver")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--shared", action="store_true", help="compare shared unicast and broadcast with plant only")
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("rounds must be positive")
    workloads = ["environment"] + [
        "environment-packet-" + kind for kind in ("ideal", "rate", "overload")
    ]
    if args.shared:
        workloads = ["environment", "environment-shared-unicast", "environment-shared-broadcast"]
    cases = [(e, a, mode) for e, a in ((1, 1), (1, 100), (1, 1000), (100, 10))
             for mode in workloads]
    rng = random.Random(22)
    for repetition in range(args.rounds):
        rng.shuffle(cases)
        for environments, slots, mode in cases:
            # Each CUDA event pair times one full step; wait only outside that interval.
            command = [args.driver, str(environments), str(slots), "1", "128", "128",
                       mode, "100", "0", ".01", "0"]
            result = json.loads(subprocess.check_output(command))
            samples = sorted(1000 * s["gpu_timeline_ms"] for s in result.pop("samples"))
            print(json.dumps(dict(result, environments=environments, slots=slots, mode=mode,
                                  repetition=repetition, median_us=statistics.median(samples),
                                  p95_us=samples[math.ceil(.95 * len(samples)) - 1],
                                  sample_us=samples)), flush=True)


if __name__ == "__main__":
    main()
