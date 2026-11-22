"""Score receiver-generated loss on witnessed Wi-SUN triplets using the public GPU step."""
import argparse
from collections import Counter
import json
from pathlib import Path

import torch
from flightlib_cuda import Environment, NetworkConfig
from validate_wisun import DIGEST, LIMITS, compare, distribution, fortnight, metrics, read_numbered_bursts


def contiguous_runs(days, numbered):
    runs = []
    for day in days:
        previous = None
        for counter, mask in numbered[day]:
            if previous is None or counter != previous + 1:
                runs.append([])
            runs[-1].append(mask)
            previous = counter
    return runs


def fit(runs):
    transitions = Counter()
    for run in runs:
        previous = None
        for mask in run:
            for bit in range(3):
                received = bool(mask & (1 << bit))
                if previous is not None:
                    transitions[previous, received] += 1
                previous = received
    def drop_after(previous):
        return transitions[previous, False] / (transitions[previous, False] + transitions[previous, True])
    return drop_after(True), drop_after(False)


def describe(runs):
    masks = [mask for run in runs for mask in run]
    pairs = Counter((prev == 0, cur == 0) for run in runs for prev, cur in zip(run, run[1:]))
    lengths = Counter()
    for run in runs:
        losses = 0
        for mask in run:
            if mask == 0:
                losses += 1
            elif losses:
                lengths[losses] += 1
                losses = 0
        if losses:
            lengths[losses] += 1
    after_miss = pairs[True, True] + pairs[True, False]
    return dict(repeat_complete_miss_rate=pairs[True, True] / after_miss if after_miss else None,
                longest_complete_miss_run=max(lengths, default=0), periods=len(masks), mask_counts=[masks.count(i) for i in range(8)],
                **metrics(distribution(masks)), complete_misses=masks.count(0),
                consecutive_miss_pairs=pairs[True, True],
                pairs_starting_with_miss=after_miss,
                pairs_starting_with_reception=pairs[False, True] + pairs[False, False],
                miss_after_reception_pairs=pairs[False, True], loss_run_histogram=dict(lengths))


def simulate(lengths, after_rx, after_miss, seed, outage=None, conditions=None, rates=None):
    # One independent two-station environment per uninterrupted counter run.
    # Only run lengths reach the simulator. The three offsets encode ordering;
    # time on this synthetic schedule has no physical delay interpretation.
    config = NetworkConfig(window_slots=1, offsets_ns=(0, 100000, 200000),
                           range_edges_m=(), seed=seed, edge_outage_rates=conditions is not None, **(outage or {}))
    env = Environment(len(lengths), 2, control_dt=.001, auto_reset=False, network=config)
    net = env.network
    env.reset_state[2].fill_(5)
    env.reset()
    with env.access():
        net.sources.zero_()
        net.size[:, :, 0].fill_(32)
        net.broadcast[:, :, 0].fill_(1)
        net.loss_after_rx.fill_(after_rx)
        net.loss_after_miss.fill_(after_miss)
    if conditions is not None:
        if list(map(len, conditions)) != lengths:
            raise ValueError("probe conditions must cover the entire offered schedule")
        classes = torch.nn.utils.rnn.pad_sequence([torch.tensor(row) for row in conditions]).to(env.device)
        table = torch.tensor(rates, device=env.device)
    env.capture(random=False)
    masks = torch.empty((max(lengths), len(lengths)), device=env.device, dtype=torch.uint8)
    invalid = torch.zeros(len(lengths), device=env.device, dtype=torch.bool)
    for tick in range(max(lengths)):
        if conditions is not None:
            with env.access():
                current = table[classes[tick]]
                net.loss_after_rx[:,1,0,0].copy_(current[:,0])
                net.loss_after_miss[:,1,0,0].copy_(current[:,1])
                net.outage_rates[:,1,0].copy_(current[:,2:])
        env.step(graph=True, random=False)
        with env.access():
            events = net.inbox[:, :, 1, 0, 0].to(torch.uint8)
            masks[tick] = events[0] | (events[1] << 1) | (events[2] << 2)
            ev = net.evaluator
            invalid |= ((ev["admission"][:, :, 0] != 1) | (ev["completed"][:, :, 0] != 1) |
                        (ev["expired"][:, :, 0] != 0) | (ev["collided"][:, :, 0] != 0)).any(dim=0)
    if invalid.any().item():
        raise RuntimeError("single-source diagnostic requires every offer admitted and completed without collision or expiry")
    host = masks.T.cpu()
    return [row[:length].tolist() for row, length in zip(host, lengths)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    numbered = read_numbered_bursts(args.csv, set(fortnight(8) + fortnight(9) + fortnight(10)))
    calibration = contiguous_runs(fortnight(8), numbered)
    report = dict(schema=1, sha256=DIGEST, limits=LIMITS,
                  unit="consecutive witnessed packet counters within a day", comparisons={})
    # The split within October is diagnostic. That month's overall result was already known.
    comparisons = (("august_to_september", calibration, fortnight(9)),
                   ("august_to_october", calibration, fortnight(10)),
                   ("october_first_week_to_second", contiguous_runs(fortnight(10)[:7], numbered),
                    fortnight(10)[7:]),
                   ("october_same_sample_capacity_check", contiguous_runs(fortnight(10), numbered),
                    fortnight(10)))
    for name, fit_runs, evaluation_days in comparisons:
        after_rx, after_miss = fit(fit_runs)
        observed = contiguous_runs(evaluation_days, numbered)
        summary = describe(observed)
        results = []
        for seed in (22, 23, 24):
            generated = simulate([len(r) for r in observed], after_rx, after_miss, seed)
            predicted = describe(generated)
            score = compare([n/predicted["periods"] for n in predicted["mask_counts"]],
                            [n/summary["periods"] for n in summary["mask_counts"]])
            ratio = predicted["complete_misses"] / summary["complete_misses"] if summary["complete_misses"] else None
            results.append(dict(seed=seed, simulated=predicted, errors=score["errors"],
                                distribution_passed=score["passed"], miss_count_ratio=ratio,
                                miss_count_passed=(.5 <= ratio <= 2 if summary["complete_misses"] >= 20
                                                   else None)))
        report["comparisons"][name] = dict(loss_after_rx=after_rx, loss_after_miss=after_miss,
                                           measured=summary, runs=len(observed), simulations=results)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not all(s["distribution_passed"] for s in report["comparisons"]["august_to_september"]["simulations"]):
        raise SystemExit("resident September distribution exceeded the established tolerances")
    print(f"Wrote resident receiver diagnostic to {args.output}")


if __name__ == "__main__":
    main()
