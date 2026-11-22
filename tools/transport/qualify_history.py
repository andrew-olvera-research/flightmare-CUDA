"""Score offset-specific packet loss calibrated on pilots and generated in the resident receiver."""
import argparse
from collections import deque
from datetime import date, timedelta
import json
from pathlib import Path

from qualify_outages import MIN_DAILY_PERIODS, SEEDS, compare_run, day_intervals, summary
from qualify_probe import TRANSFER_LINKS, calibration_blocks, fetch_link, read_runs
from validate_resident_wisun import simulate
from validate_wisun import DIGEST, fortnight

HISTORY_BITS = 5  # One preceding three-packet update plus two packets in the current update.
EXTRA_LINKS = {"5632": "54a0828ed8077830e4b8d6ce6b1b0ded85bedea626bd148d1b60cda036ade123",
               "5653": "2da1c0d5bead038c4489272e9fa72a2a4cc9ef48daa96ee5de2f7368c92414f8",
               "630a": "53a1b1938bb475386976c4edd7cb897817f79b84fb17cf9981a252e0a4bfbb33"}


def fit_context(runs, bits, prior=None):
    """Estimate drop probabilities for each offer position and recent packet history."""
    nodes = (2 << bits) - 1
    attempts = [[0]*nodes for _ in range(3)]
    misses = [[0]*nodes for _ in range(3)]
    for run in runs:
        history = depth = 0
        for mask in run:
            for ordinal in range(3):
                lost = not (mask & (1 << ordinal))
                for length in range(depth + 1):
                    start = (1 << length) - 1
                    node = start + (history & start)
                    attempts[ordinal][node] += 1
                    misses[ordinal][node] += lost
                history = ((history << 1) | lost) & ((1 << bits) - 1)
                depth = min(bits, depth + 1)
    probabilities = []
    for ordinal in range(3):
        row = []
        for node, (lost, total) in enumerate(zip(misses[ordinal], attempts[ordinal])):
            depth = (node + 1).bit_length() - 1
            # Back off by dropping the oldest bit, preserving the newest outcomes.
            shorter = (1 << max(0, depth-1)) - 1
            history = node + 1
            backoff = row[shorter + (history & shorter)] if depth else .5
            reference = prior[ordinal][node] if prior is not None else backoff
            prefix = (1 << ordinal) - 1
            if depth > ordinal and (history & prefix) != prefix:
                # A copy already landed. Borrow a local observation conditional on
                # this update's prefix when estimating the remaining copy losses.
                reference = row[prefix + (history & prefix)]
            row.append((lost + reference + backoff) / (total + 2))
        probabilities.append(row)
    return probabilities


def evaluate(days, dates, prior, bits):
    daily, tables = [], []
    for day in dates:
        if sum(map(len, days[day])) < MIN_DAILY_PERIODS:
            continue
        daily.append([])
        pilots = deque(maxlen=8)
        for calibration, suffix in calibration_blocks(days[day], 32, 8):
            pilots.append(calibration)
            if suffix:
                table = fit_context([r for block in pilots for r in block], bits, prior)
                tables.extend([table] * len(suffix))
                daily[-1].extend(suffix)
    if not daily:
        return dict(verdict="insufficient_evidence", eligible_days=0)
    runs = [r for day in daily for r in day]
    measured = summary(runs)
    lengths = list(map(len, runs))
    seeds = {seed: compare_run(measured, simulate(lengths, 0, 0, seed, history_bits=bits,
                                                  context_tables=tables)) for seed in SEEDS}
    outcomes = [s["verdict"] for s in seeds.values()]
    verdict = ("rejected" if "rejected" in outcomes else "insufficient_evidence"
               if len(daily)<7 or "insufficient_evidence" in outcomes else "qualified")
    return dict(eligible_days=len(daily), measured=measured, seeds=seeds, verdict=verdict,
                measured_day_bootstrap_95=day_intervals(daily))


def window(first, last):
    start, end = date.fromisoformat(first), date.fromisoformat(last)
    return [(start+timedelta(days=i)).isoformat() for i in range((end-start).days+1)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download", action="store_true", help="populate missing, pinned measurement files")
    parser.add_argument("--require-case", action="append", default=[])
    parser.add_argument("--require-qualified", action="store_true", help="enforce all measured windows")
    args = parser.parse_args()
    dates = fortnight(10)
    base_path = args.directory / "data_5563.csv"
    if args.download:
        fetch_link(base_path, DIGEST)
    base = read_runs(base_path, set(dates))
    raw = [[mask for mask, _ in run] for d in dates[:7] for run in base[d]]
    prior = fit_context(raw, HISTORY_BITS)
    results = {"5563": dict(sha256=DIGEST, dates=dates[7:], **evaluate(base, dates[7:], prior, HISTORY_BITS))}
    for link, digest in {**TRANSFER_LINKS, **EXTRA_LINKS}.items():
        windows = {link: dates[7:]}
        if link == "55e4" or link in EXTRA_LINKS:
            windows[link+"-late-october"] = window("2019-10-15", "2019-10-28")
        if link == "55e4":
            windows.update({"55e4-august": window("2019-08-01","2019-08-31"),
                            "55e4-september": window("2019-09-01","2019-09-30"),
                            "55e4-final": window("2019-10-29","2019-11-10")})
        path = args.directory / f"data_{link}.csv"
        if args.download:
            fetch_link(path, digest)
        days = read_runs(path, set().union(*windows.values()), digest)
        for name, period in windows.items():
            results[name] = dict(sha256=digest, dates=period, **evaluate(days, period, prior, HISTORY_BITS))
    report = dict(schema=1, fit_link="5563", fit_sha256=DIGEST, fit_dates=dates[:7], history_bits=HISTORY_BITS,
                  block_size=32, pilot_size=8, calibration_blocks=8, comparisons=results)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    for link, case in results.items():
        print(link, case["verdict"])
    required = results if args.require_qualified else args.require_case
    raise SystemExit(int(any(results[name]["verdict"] != "qualified" for name in required)))
