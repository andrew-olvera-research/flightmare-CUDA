"""Compare the resident single-outcome and persistent receiver on witnessed packet counters."""
import argparse
from collections import Counter
from datetime import date
from itertools import groupby
import json
import math
from pathlib import Path
import random

from validate_resident_wisun import contiguous_runs, describe, fit, simulate
from validate_wisun import DIGEST, LIMITS, compare, fortnight, read_numbered_bursts

SEEDS = (22, 23, 24)
MIN_DAILY_PERIODS = 720
MIN_TAIL_RUNS = 200


def fit_persistent(runs):
    # Complete period loss is an observable proxy for the bad state at calibration.
    transitions = Counter((a == 0, b == 0) for run in runs for a, b in zip(run, run[1:]))
    enter = transitions[False, True] / (transitions[False, False] + transitions[False, True])
    recover = transitions[True, False] / (transitions[True, True] + transitions[True, False])
    good_runs = [list(group) for run in runs for good, group in groupby(run, bool) if good]
    return (*fit(good_runs), dict(outage_period_ns=1000000, outage_enter=enter, outage_recover=recover))


def complete_runs(runs):
    # Require a witnessed reception on both sides. A gap may hide a longer outage.
    lengths = Counter()
    for run in runs:
        start = None
        for i, mask in enumerate(run):
            if mask and start is not None:
                if start > 0:
                    lengths[i-start] += 1
                start = None
            elif not mask and start is None:
                start = i
    return lengths


def tail(lengths):
    count = sum(lengths.values())
    if not count:
        return None
    rank = math.ceil(.95 * count)
    for length, weight in sorted(lengths.items()):
        rank -= weight
        if rank <= 0:
            return length


def summary(runs):
    result = describe(runs)
    lengths = complete_runs(runs)
    result.update(uncensored_runs=sum(lengths.values()), p95_loss_run=tail(lengths))
    return result


def compare_run(observed, generated):
    predicted = summary(generated)
    score = compare([n/predicted["periods"] for n in predicted["mask_counts"]],
                    [n/observed["periods"] for n in observed["mask_counts"]])
    ratio = (predicted["complete_misses"]/observed["complete_misses"]
             if observed["complete_misses"] >= 20 else None)
    tail_error = None
    if min(observed["uncensored_runs"], predicted["uncensored_runs"]) >= MIN_TAIL_RUNS:
        tail_error = abs(predicted["p95_loss_run"]/observed["p95_loss_run"] - 1)
    repeat_error = None
    if min(observed["pairs_starting_with_miss"], predicted["pairs_starting_with_miss"]) >= 100:
        repeat_error = abs(predicted["repeat_complete_miss_rate"] - observed["repeat_complete_miss_rate"])
    gates = [score["passed"], None if ratio is None else .5 <= ratio <= 2,
             None if tail_error is None else tail_error <= .20,
             None if repeat_error is None else repeat_error <= .10]
    verdict = "rejected" if False in gates else "insufficient_evidence" if None in gates else "qualified"
    return dict(simulated=predicted, distribution_errors=score["errors"], count_ratio=ratio,
                p95_relative_error=tail_error, repeat_probability_error=repeat_error, verdict=verdict)


def day_intervals(day_runs):
    # Resample whole days, preserving gaps and all within-day dependence.
    blocks = [(complete_runs(runs), sum(mask == 0 for run in runs for mask in run),
               sum(map(len, runs))) for runs in day_runs]
    rng = random.Random(24)
    tails, rates = [], []
    for _ in range(1000):
        sample = rng.choices(blocks, k=len(blocks))
        value = tail(sum((b[0] for b in sample), Counter()))
        if value is not None:
            tails.append(value)
        rates.append(sum(b[1] for b in sample)/sum(b[2] for b in sample))
    def interval(values):
        values.sort()
        return [values[int(q*(len(values)-1))] for q in (.025,.975)] if values else None
    return dict(p95_loss_run=interval(tails), complete_loss_fraction=interval(rates))


def evaluate(numbered, fit_days, test_days):
    calibration = contiguous_runs(fit_days, numbered)
    eligible = [d for d in test_days if len(numbered[d]) >= MIN_DAILY_PERIODS]
    result = dict(fit_days=fit_days, daily_coverage={d: len(numbered[d]) for d in test_days})
    if not eligible:
        return dict(**result, verdict="insufficient_evidence")
    if any(len(numbered[d]) < MIN_DAILY_PERIODS for d in fit_days):
        raise ValueError("insufficient calibration coverage")
    days = [contiguous_runs([d], numbered) for d in eligible]
    runs = [r for day in days for r in day]
    measured = summary(runs)
    result.update(measured=measured, measured_day_bootstrap_95=day_intervals(days), models={})
    for name, (after_rx, after_miss, outage) in (
            ("last_outcome", (*fit(calibration), None)), ("persistent", fit_persistent(calibration))):
        seeds = {}
        for seed in SEEDS:
            generated = simulate(list(map(len, runs)), after_rx, after_miss, seed, outage)
            seeds[seed] = compare_run(measured, generated)
        result["models"][name] = dict(after_rx=after_rx, after_miss=after_miss,
                                      outage=outage, seeds=seeds)
    verdicts = [s["verdict"] for s in result["models"]["persistent"]["seeds"].values()]
    result["coverage_sufficient"] = len(eligible) >= 7
    result["verdict"] = ("insufficient_evidence" if not result["coverage_sufficient"] else
                         "rejected" if "rejected" in verdicts else
                         "insufficient_evidence" if "insufficient_evidence" in verdicts else "qualified")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    windows = parser.add_mutually_exclusive_group()
    windows.add_argument("--confirmation", action="store_true", help="score the frozen November window")
    windows.add_argument("--supplemental", action="store_true", help="score the entire unused October 15-31 window")
    parser.add_argument("--require-qualified", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    october = fortnight(10)
    comparisons = {"october_capacity": (october, october),
                   "october_transfer": (october[:7], october[7:])}
    if args.confirmation:
        comparisons = {"november_confirmation": (october, fortnight(11)[:10])}
    if args.supplemental:
        comparisons = {"late_october_confirmation": (october, [date(2019,10,d).isoformat() for d in range(15,32)])}
    requested = {d for fit_days, test in comparisons.values() for d in fit_days + test}
    numbered = read_numbered_bursts(args.csv, requested, min_periods=0)
    results = {k: evaluate(numbered, fit_days, test) for k,(fit_days,test) in comparisons.items()}
    args.output.write_text(json.dumps(dict(schema=1, sha256=DIGEST, limits=LIMITS,
                                           comparisons=results), indent=2) + "\n")
    for name, result in results.items():
        print(name, result["verdict"])
    if args.require_qualified and any(r["verdict"] != "qualified" for r in results.values()):
        raise SystemExit(1)
