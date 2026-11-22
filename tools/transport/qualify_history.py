"""Score offset-specific packet loss calibrated on pilots and generated in the resident receiver."""
import argparse
from collections import deque
from datetime import date, timedelta
import json
from pathlib import Path

from flightlib_cuda.loss_calibration import TripletLossCalibrator, fit_context
from qualify_outages import MIN_DAILY_PERIODS, MIN_TAIL_RUNS, SEEDS, compare_run, day_intervals, summary
from qualify_probe import CONFIRMATION, LINKS, PILOT_LINKS, TRANSFER_LINKS, calibration_blocks, fetch_link, read_runs
from validate_resident_wisun import simulate
from validate_wisun import DIGEST, fortnight

HISTORY_BITS = 5  # One preceding three-packet update plus two packets in the current update.
EXTRA_LINKS = {"5632": "54a0828ed8077830e4b8d6ce6b1b0ded85bedea626bd148d1b60cda036ade123",
               "5653": "2da1c0d5bead038c4489272e9fa72a2a4cc9ef48daa96ee5de2f7368c92414f8",
               "630a": "53a1b1938bb475386976c4edd7cb897817f79b84fb17cf9981a252e0a4bfbb33"}


def evaluate(days, dates, prior, bits, calibrator=None):
    daily, tables = [], []
    weight_sum = 0.0
    for day in dates:
        if sum(map(len, days[day])) < MIN_DAILY_PERIODS:
            continue
        daily.append([])
        pilots = deque(maxlen=8)
        if calibrator is not None:
            calibrator.reset_session()
        for calibration, suffix in calibration_blocks(days[day], 32, 8):
            if calibrator is not None:
                table = calibrator.observe(calibration)
                weight_sum += calibrator.reference_weight * sum(map(len, suffix))
            else:
                pilots.append(calibration)
            if suffix:
                if calibrator is None:
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
                mean_reference_weight=weight_sum/sum(lengths) if calibrator is not None else None,
                measured_day_bootstrap_95=day_intervals(daily))


def window(first, last):
    start, end = date.fromisoformat(first), date.fromisoformat(last)
    return [(start+timedelta(days=i)).isoformat() for i in range((end-start).days+1)]


def loss_counts_and_repeats_pass(case):
    """Gate distribution, miss count and recurrence; enforce p95 whenever measured support exists."""
    if case["eligible_days"] < 7 or case["verdict"] == "rejected":
        return False
    tail_supported = case["measured"]["uncensored_runs"] >= MIN_TAIL_RUNS
    return all(s["count_ratio"] is not None and s["repeat_probability_error"] is not None
               and (not tail_supported or s["p95_relative_error"] is not None)
               for s in case["seeds"].values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download", action="store_true", help="populate missing, pinned measurement files")
    parser.add_argument("--adaptive", action="store_true", help="estimate borrowing strength from prior pilot blocks including the current one")
    parser.add_argument("--require-case", action="append", default=[])
    parser.add_argument("--require-loss-case", action="append", default=[],
                        help="require supported and passing miss counts and recurrence, plus p95 when measurable")
    parser.add_argument("--require-qualified", action="store_true", help="enforce all measured windows")
    parser.add_argument("--require-no-rejections", action="store_true",
                        help="fail if any measured window violates a supported limit")
    args = parser.parse_args()
    dates = fortnight(10)
    base_path = args.directory / "data_5563.csv"
    if args.download:
        fetch_link(base_path, DIGEST)
    base = read_runs(base_path, set(dates))
    raw = [[mask for mask, _ in run] for d in dates[:7] for run in base[d]]
    prior = fit_context(raw, HISTORY_BITS)
    calibrator = TripletLossCalibrator(raw, HISTORY_BITS) if args.adaptive else None
    results = {"5563": dict(sha256=DIGEST, dates=dates[7:], **evaluate(base, dates[7:], prior, HISTORY_BITS, calibrator))}
    links = {**TRANSFER_LINKS, **EXTRA_LINKS}
    seasonal_links = {**LINKS, "55ad": PILOT_LINKS["55ad"], **CONFIRMATION}
    if args.adaptive:
        links.update(seasonal_links)
    for link, digest in links.items():
        windows = {link: dates[7:]}
        if link == "55e4" or link in EXTRA_LINKS:
            windows[link+"-late-october"] = window("2019-10-15", "2019-10-28")
        if link == "55e4":
            windows.update({"55e4-august": window("2019-08-01","2019-08-31"),
                            "55e4-september": window("2019-09-01","2019-09-30"),
                            "55e4-final": window("2019-10-29","2019-11-10")})
        if link in seasonal_links:
            windows = {link+"-august": window("2019-08-01","2019-08-31"),
                       link+"-september": window("2019-09-01","2019-09-30")}
        path = args.directory / f"data_{link}.csv"
        if args.download:
            fetch_link(path, digest)
        days = read_runs(path, set().union(*windows.values()), digest)
        for name, period in windows.items():
            results[name] = dict(sha256=digest, dates=period, **evaluate(days, period, prior, HISTORY_BITS, calibrator))
            if not args.adaptive and name in ("55e4", "55e4-late-october", "55e4-final"):
                control = evaluate(days, period, None, HISTORY_BITS)
                results[name]["local_pilots_only"] = dict(verdict=control["verdict"], seeds=control.get("seeds", {}))
    report = dict(schema=2, adaptive=args.adaptive, fit_link="5563", fit_sha256=DIGEST, fit_dates=dates[:7], history_bits=HISTORY_BITS,
                  block_size=32, pilot_size=8, calibration_blocks=8, comparisons=results)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    for link, case in results.items():
        print(link, case["verdict"])
    required = results if args.require_qualified else args.require_case
    failed = any(results[name]["verdict"] != "qualified" for name in required)
    if args.require_no_rejections:
        failed |= any(case["verdict"] == "rejected" for case in results.values())
    failed |= any(not loss_counts_and_repeats_pass(results[name]) for name in args.require_loss_case)
    raise SystemExit(int(failed))
