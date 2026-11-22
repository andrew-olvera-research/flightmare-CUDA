"""Score patched ns-3.44 wifi-bianchi logs from the one-second common-sink fixture."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from statistics import mean, median


def parse(path, senders):
    start, end = 10_000_000_000, 11_000_000_000
    trials = []
    with path.open() as stream:
        for line in stream:
            if line.startswith("Trial "):
                offers, transmitted, waits = {}, set(), defaultdict(list)
                cw_counts, release_times, outstanding = Counter(), Counter(), {}
            elif line.startswith("station-cw "):
                _, t, sender, cw = line.split()
                t, sender = int(t), int(sender)
                if sender and start <= t < end:
                    cw_counts[int(cw)] += 1
                    if sender in outstanding:
                        release_times[t-outstanding.pop(sender)] += 1
            elif line.startswith(("packet-offer ", "packet-air ")):
                kind, t, sender, uid = line.split()
                t, sender = int(t), int(sender)
                key = sender, int(uid)
                if kind == "packet-offer":
                    if key in offers:
                        raise ValueError("duplicate application packet")
                    offers[key] = t
                elif start <= t < end:
                    if key not in offers or key in transmitted or offers[key] > t:
                        raise ValueError("unmatched, repeated or noncausal transmit start")
                    transmitted.add(key)
                    waits[sender].append(t-offers[key])
                    outstanding[sender] = t
            elif line.startswith("Total throughput:"):
                if set(waits) != set(range(1, senders+1)):
                    raise ValueError("incomplete station coverage")
                per_sender = []
                for sender, values in sorted(waits.items()):
                    count = sum(node == sender and start <= t < end for (node,_),t in offers.items())
                    if count != 10000:
                        raise ValueError(f"expected 10000 offers at sender {sender}, saw {count}")
                    values.sort()
                    per_sender.append(dict(sender=sender, offered=count, transmitted=len(values),
                                           wait_ns_p50_p95=[values[math.ceil(q*len(values))-1] for q in (.5,.95)]))
                trial = dict(reported_goodput_mbps=float(line.split()[-2]), per_sender=per_sender)
                if cw_counts:
                    trial.update(cw_counts=dict(cw_counts), data_start_to_cw_update_ns=dict(release_times))
                trials.append(trial)
    if len(trials) != 3:
        raise ValueError("expected three complete trials")
    return trials


def compare_cuda(reference, rows):
    comparison = {}
    for n, trials in reference.items():
        runs = [r for r in rows if r["senders"] == n and r["workload"] == "saturated"]
        if (sorted(r["seed"] for r in runs) != [22,23,24] or
                any(r["seconds"] != 1 or r["offset_interval_us"] != 100 or len(r["per_sender"]) != n or
                    any(s["bytes"] != 1000 or s["offered"] != 10000 for s in r["per_sender"]) for r in runs)):
            raise ValueError("CUDA profile must match the synchronized saturated fixture")
        def tail(results):
            return median(s["wait_ns_p50_p95"][1] for r in results for s in r["per_sender"])/1000
        sim_rate = mean(r["goodput_mbps"] for r in runs)
        ref_rate = mean(r["reported_goodput_mbps"] for r in trials)
        sim_wait, ref_wait = tail(runs), tail(trials)
        passed = abs(sim_rate/ref_rate-1) <= .10 and abs(sim_wait/ref_wait-1) <= .20
        comparison[n] = dict(cuda_mbps=sim_rate, reference_mbps=ref_rate, cuda_p95_us=sim_wait,
                             reference_p95_us=ref_wait, passes=passed)
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, help="contains service-{1,4,8}.log")
    parser.add_argument("--reference", type=Path, help="read previously scored reference trials")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cuda", type=Path, help="compare with a compare_contention.py report")
    parser.add_argument("--require-qualified", action="store_true", help="fail if any matched-load gate misses")
    args = parser.parse_args()
    if bool(args.reference) == bool(args.directory) or (args.require_qualified and not args.cuda):
        parser.error("supply one reference source and a CUDA report when enforcing gates")
    if args.reference:
        data = json.loads(args.reference.read_text())
        results = {n: data[str(n)] for n in (1,4,8)}
        if any(len(v) != 3 for v in results.values()):
            raise ValueError("reference must contain three trials at each load")
    else:
        results = {n: parse(args.directory / f"service-{n}.log", n) for n in (1,4,8)}
    report = dict(reference=results, comparisons=compare_cuda(results, json.loads(args.cuda.read_text()))) if args.cuda else results
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.require_qualified and not all(r["passes"] for r in report["comparisons"].values()):
        raise SystemExit(1)
