"""Qualify within-burst FSK reception on a pinned, stationary Wi-SUN link."""
import argparse
import csv
from collections import Counter, defaultdict
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import random
import subprocess

REVISION = "66c350d88d28fa39516fbc90722bdd3d9930d991"
URL = f"https://raw.githubusercontent.com/wine-uoc/wisun_traces/{REVISION}/data_5563.csv"
DIGEST = "35e5ecc65477a8b6e3f579ec537392b1d4a61f62886ded4c1973c406b8904f76"
# Absolute errors, fixed before examining the September holdout.
LIMITS = {"delivery": .05, "all_lost": .03, "pattern_tv": .10}


def fortnight(month):
    return [(date(2019, month, 1) + timedelta(days=i)).isoformat() for i in range(14)]


def read_bursts(path, days):
    if hashlib.sha256(path.read_bytes()).hexdigest() != DIGEST:
        raise ValueError("unexpected source SHA-256")
    periods = defaultdict(dict)
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            day = row["day"]
            if day not in days:
                continue
            key = day, int(row["pkt_number"])
            rep = int(row["pkt_retry"])
            if rep not in (0, 1, 2) or rep in periods[key]:
                raise ValueError(f"ambiguous repetition in {key}")
            periods[key][rep] = (bool(row["fsk_rssi"]),
                                 bool(row["oqpsk_rssi"] or row["ofdm_rssi"]))
    result = {day: [] for day in days}
    for (day, _), reps in sorted(periods.items()):
        # Independent witness of an active measurement period. Exclude FSK-only
        # periods symmetrically, regardless of their FSK reception pattern.
        if any(witness for _, witness in reps.values()):
            result[day].append(sum(1 << rep for rep, (rx, _) in reps.items() if rx))
    if any(len(values) < 720 for values in result.values()):
        raise ValueError("insufficient coverage: need 720 witnessed periods per day")
    return result


def distribution(values):
    counts = Counter(values)
    return [counts[i] / len(values) for i in range(8)]


def metrics(p):
    return {"delivery": sum(i.bit_count()*x for i, x in enumerate(p))/3,
            "all_lost": p[0]}


def baselines(p):
    marginal = metrics(p)["delivery"]
    iid = [marginal**i.bit_count() * (1-marginal)**(3-i.bit_count()) for i in range(8)]
    initial = sum(p[i] for i in range(8) if i & 1)
    transitions = [[0., 0.], [0., 0.]]
    for mask, prob in enumerate(p):
        for rep in (0, 1):
            transitions[(mask >> rep) & 1][(mask >> (rep+1)) & 1] += prob
    conditional = [row[1]/sum(row) for row in transitions]
    markov = []
    for mask in range(8):
        bits = [(mask >> rep) & 1 for rep in range(3)]
        prob = initial if bits[0] else 1-initial
        for prev, curr in zip(bits, bits[1:]):
            prob *= conditional[prev] if curr else 1-conditional[prev]
        markov.append(prob)
    return {"independent": iid, "two_state": markov, "empirical_triplet": p}


def compare(predicted, observed):
    errors = {k: abs(v - metrics(observed)[k]) for k, v in metrics(predicted).items()}
    errors["pattern_tv"] = sum(abs(a-b) for a, b in zip(predicted, observed))/2
    return {"predicted": metrics(predicted), "errors": errors,
            "passed": all(errors[k] <= limit for k, limit in LIMITS.items())}


def day_intervals(by_day):
    # Bootstrap complete days together to retain within-day dependence.
    daily = [Counter(values) for values in by_day.values()]
    rng = random.Random(22)
    samples = defaultdict(list)
    for _ in range(2000):
        total = Counter()
        for counts in rng.choices(daily, k=len(daily)):
            total.update(counts)
        for k, v in metrics([total[i]/total.total() for i in range(8)]).items():
            samples[k].append(v)
    return {k: [sorted(v)[49], sorted(v)[1949]] for k, v in samples.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("packet_test", help="native CUDA test executable")
    args = parser.parse_args()
    fit, holdout, confirm = fortnight(8), fortnight(9), fortnight(10)
    days = read_bursts(args.csv, set(fit + holdout + confirm))
    train = [mask for day in fit for mask in days[day]]
    test = [mask for day in holdout for mask in days[day]]
    profile, observed = distribution(train), distribution(test)
    def replay_masks(masks):
        replay = subprocess.check_output([args.packet_test, "--replay"],
                                         input="".join(map(str, masks)), text=True)
        if int(replay) != len(masks):
            raise RuntimeError("incomplete CUDA replay")

    replay_masks(test)
    scores = {k: compare(p, observed) for k, p in baselines(profile).items()}
    october = [mask for day in confirm for mask in days[day]]
    replay_masks(october)
    # Frozen after September, before October evaluation. Below 20 events
    # there is insufficient evidence to qualify the rare-event criterion.
    confirmation = {}
    for name, probabilities in baselines(profile).items():
        score = compare(probabilities, distribution(october))
        expected = len(october) * probabilities[0]
        count = october.count(0)
        ratio = expected / count if count else None
        score.update(expected_complete_misses=expected, observed_complete_misses=count,
                     predicted_to_observed=ratio,
                     qualified=score["passed"] and count >= 20 and 0.5 <= ratio <= 2)
        confirmation[name] = score
    print(json.dumps({"schema": 1, "source": URL, "sha256": DIGEST,
                      "node": "5563", "mode": "FSK", "psdu_bytes": 32,
                      "fit_counts_by_day": {d: len(days[d]) for d in fit},
                      "holdout_counts_by_day": {d: len(days[d]) for d in holdout},
                      "profile_mask_probabilities": profile,
                      "holdout_mask_counts": [test.count(mask) for mask in range(8)],
                      "cuda_replayed_attempts": 3*len(test),
                      "observed": metrics(observed),
                      "holdout_day_bootstrap_95": day_intervals({d: days[d] for d in holdout}),
                      "limits": LIMITS, "models": scores,
                      "october_counts_by_day": {d: len(days[d]) for d in confirm},
                      "october_cuda_replayed_attempts": 3*len(october),
                      "october_observed": metrics(distribution(october)),
                      "october_day_bootstrap_95": day_intervals({d: days[d] for d in confirm}),
                      "october_limits": {"min_complete_misses": 20, "miss_ratio": [.5, 2.]},
                      "october_models": confirmation}, indent=2))
    return 0 if scores["empirical_triplet"]["passed"] and confirmation["empirical_triplet"]["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
