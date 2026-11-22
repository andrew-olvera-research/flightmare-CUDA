"""Compare pooled and probe-conditioned Wi-SUN reception through the public GPU step."""
import argparse
from collections import Counter, defaultdict, deque
import csv
import hashlib
import json
from pathlib import Path
import tempfile
from urllib.request import urlopen

from qualify_outages import MIN_DAILY_PERIODS, SEEDS, compare_run, day_intervals, fit_persistent, summary
from validate_resident_wisun import simulate
from validate_wisun import DIGEST, REVISION, fortnight

LINKS = {"5565": "1b1be5293f45b58d7375e2a79871acf2968c5633062367711e4138179bb07c89",
         "5599": "4f5838811503bdb0b3792ab30194539aa4918de4f40caf2cd46ac110eed8b496"}
PILOT_LINKS = {"55ad": "872483c25b89f2db3b5d8eba317ff6dbf00748b3661e15ae45269be3d1d7867f",
               "55b3": "84bb6b926dc062103503995e6913d13b36936b4dfc8e9bdca7e3cc2419149710"}


def read_runs(path, dates, digest=DIGEST, signal=None):
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError("unexpected Wi-SUN capture")
    periods = defaultdict(dict)
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            day = row["day"]
            if day not in dates:
                continue
            key = day, int(row["pkt_number"])
            rep = int(row["pkt_retry"])
            if not 0 <= rep < 3 or rep in periods[key]:
                raise ValueError(f"ambiguous replica at {key}")
            periods[key][rep] = (float(row["fsk_rssi"]) if row["fsk_rssi"] else None,
                                 bool(row["oqpsk_rssi"]), bool(row["ofdm_rssi"]))

    days = defaultdict(list)
    history = deque(maxlen=12)
    last = None
    for (day, counter), reps in sorted(periods.items()):
        if not any(q or o for _, q, o in reps.values()):
            continue
        if last != (day, counter-1):
            days[day].append([])
            history.clear()
        condition = 0
        if len(history) == history.maxlen:
            fraction = sum(history)/(3*len(history))
            condition = 1 if fraction < .70 else 2 if fraction < .94 else 3
        mask = sum(1 << rep for rep, (fsk, _, _) in reps.items() if fsk is not None)
        if signal is not None:
            signal[day].update(fsk for fsk, _, _ in reps.values() if fsk is not None)
        days[day][-1].append((mask, condition))
        history.append(sum(q for _, q, _ in reps.values()))
        last = day, counter
    return days


def fit_rates(runs):
    rx, miss, outage = fit_persistent([[mask for mask, _ in run] for run in runs])
    pooled = [rx, miss, outage["outage_enter"], outage["outage_recover"]]
    transitions = [Counter() for _ in range(4)]
    packets = [Counter() for _ in range(4)]
    for run in runs:
        prior = None
        for mask, condition in run:
            if prior is not None:
                transitions[condition][bool(prior), bool(mask)] += 1
            if mask:
                previous = bool(prior & 4) if prior else None
                for rep in range(3):
                    received = bool(mask & (1 << rep))
                    if previous is not None:
                        packets[condition][previous, received] += 1
                    previous = received
            prior = mask

    def rate(counts, previous, event, fallback):
        total = counts[previous, True] + counts[previous, False]
        return counts[previous, event] / total if total else fallback

    table = [pooled]
    for state in (1, 2, 3):
        p, t = packets[state], transitions[state]
        enter = rate(t, True, False, pooled[2])
        recover = rate(t, False, True, pooled[3])
        if not 0 < enter + recover <= 1:
            enter, recover = pooled[2:]
        table.append([rate(p, True, False, rx), rate(p, False, False, miss), enter, recover])
    return table, outage


def diagnose(runs):
    cells = []
    for state in range(4):
        values = [m for run in runs for m, c in run if c == state]
        cells.append(dict(periods=len(values), complete_misses=values.count(0),
                          received_replicas=sum(m.bit_count() for m in values)))
    return cells


def signal_median(signal, dates):
    counts = sum((signal[d] for d in dates), Counter())
    halfway = (counts.total()+1)//2
    for value, count in sorted(counts.items()):
        halfway -= count
        if halfway <= 0:
            return value
    return None


def prefix_rates(runs, prior):
    packet, period = Counter(), Counter()
    for run in runs:
        previous = None
        for mask in run:
            if previous is not None:
                period[bool(previous), bool(mask)] += 1
            if mask:
                bit = bool(previous & 4) if previous else None
                for i in range(3):
                    received = bool(mask & (1 << i))
                    if bit is not None:
                        packet[bit, received] += 1
                    bit = received
            previous = mask
    rates = [(counts[src, dst]+p)/(1+counts[src, False]+counts[src, True])
             for p, (counts, src, dst) in zip(prior, ((packet, True, False), (packet, False, False),
                                                      (period, True, False), (period, False, True)))]
    total = sum(rates[2:])
    if total > 1:
        rates[2:] = [r/total for r in rates[2:]]
    return rates


def score_models(daily, table, outage, conditions, candidate):
    runs = [r for day in daily for r in day]
    observed = summary(runs)
    models = {}
    for name in ("pooled", candidate):
        seeds = {}
        for seed in SEEDS:
            generated = simulate(list(map(len, runs)), *table[0][:2], seed, outage,
                                 conditions if name == candidate else None, table)
            seeds[seed] = compare_run(observed, generated)
        models[name] = seeds
    verdicts = [s["verdict"] for s in models[candidate].values()]
    verdict = ("insufficient_evidence" if len(daily) < 7 else "rejected" if "rejected" in verdicts
               else "insufficient_evidence" if "insufficient_evidence" in verdicts else "qualified")
    return dict(measured=observed, models=models, measured_day_bootstrap_95=day_intervals(daily), verdict=verdict)


def calibration_blocks(runs, block_size, pilot_size):
    """Split by witnessed order, retaining counter gaps and excluding pilot labels from scores."""
    offset = 0
    blocks = []
    for run in runs:
        i = 0
        while i < len(run):
            j = offset % block_size
            if j == 0:
                blocks.append(([], []))
            pilot = j < pilot_size
            n = min(len(run)-i, (pilot_size if pilot else block_size)-j)
            blocks[-1][0 if pilot else 1].append([m for m, _ in run[i:i+n]])
            offset += n
            i += n
    return blocks


def evaluate_local(days, dates, prior, outage, distributed):
    table, conditions, daily = [prior], [], []
    coverage = {d: sum(map(len, days[d])) for d in dates}
    prefix_sizes = {}
    for d in dates:
        if coverage[d] < MIN_DAILY_PERIODS:
            continue
        history = deque(maxlen=4)
        blocks = calibration_blocks(days[d], 240 if distributed else coverage[d], 60 if distributed else 240)
        daily.append([])
        prefix_sizes[d] = 0
        for pilot, suffix in blocks:
            prefix_sizes[d] += sum(map(len, pilot))
            history.append(pilot)
            if suffix:
                conditions.extend([[len(table)]*len(run) for run in suffix])
                table.append(prefix_rates([run for block in history for run in block], prior))
                daily[-1].extend(suffix)
    result = dict(coverage=coverage, prefix_sizes=prefix_sizes, parameters=table)
    if not daily:
        return dict(**result, verdict="insufficient_evidence")
    return dict(**result, **score_models(daily, table, outage, conditions, "pilots" if distributed else "local_prefix"))


def evaluate(days, fit_dates, score_dates):
    fit = [run for d in fit_dates for run in days[d]]
    table, outage = fit_rates(fit)
    coverage = {d: sum(map(len, days[d])) for d in score_dates}
    eligible = [d for d, count in coverage.items() if count >= MIN_DAILY_PERIODS]
    result = dict(coverage=coverage, fit_dates=fit_dates, parameters=table, calibration_cells=diagnose(fit))
    if not eligible:
        return dict(**result, verdict="insufficient_evidence")
    daily = [[[m for m, _ in run] for run in days[d]] for d in eligible]
    runs = [run for d in eligible for run in days[d]]
    conditions = [[c for _, c in run] for run in runs]
    return dict(**result, scoring_cells=diagnose(runs), **score_models(daily, table, outage, conditions, "probe"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prefix", action="store_true", help="score calibration prefixes on two independent links")
    modes.add_argument("--pilots", action="store_true", help="score rolling pilot calibration on two further links")
    parser.add_argument("--download", action="store_true", help="fetch missing pinned link captures beside the input")
    args = parser.parse_args()
    october = fortnight(10)
    splits = {"october_diagnostic": (october[:7], october[7:]),
              "august_transfer": (october, [f"2019-08-{d}" for d in range(15,32)]),
              "september_transfer": (october, [f"2019-09-{d}" for d in range(15,31)])}
    dates = {d for fit, score in splits.values() for d in fit+score}
    signal = defaultdict(Counter)
    days = read_runs(args.csv, set(october) if args.prefix or args.pilots else dates, signal=signal)
    if args.prefix or args.pilots:
        prior, outage = fit_rates([run for d in october for run in days[d]])
        report = {}
        for link, digest in (PILOT_LINKS if args.pilots else LINKS).items():
            path = args.csv.with_name(f"data_{link}.csv")
            if args.download and not path.exists():
                with urlopen(f"https://raw.githubusercontent.com/wine-uoc/wisun_traces/{REVISION}/{path.name}",
                             timeout=60) as stream:
                    data = stream.read()
                if hashlib.sha256(data).hexdigest() != digest:
                    raise ValueError("unexpected download")
                with tempfile.TemporaryDirectory(dir=path.parent) as scratch:
                    temporary = Path(scratch)/path.name
                    temporary.write_bytes(data)
                    temporary.replace(path)
            report[link] = dict(sha256=digest, **evaluate_local(read_runs(path, set(october), digest),
                                                                 october, prior[0], outage, args.pilots))
    else:
        report = {name: evaluate(days, *split) for name, split in splits.items()}
        for name, (fit, score) in splits.items():
            eligible = [d for d in score if report[name]["coverage"][d] >= MIN_DAILY_PERIODS]
            report[name]["received_fsk_median_dbm"] = dict(calibration=signal_median(signal, fit),
                                                            evaluation=signal_median(signal, eligible))
    args.output.write_text(json.dumps(dict(schema=1, sha256=DIGEST, comparisons=report), indent=2) + "\n")
    for name, case in report.items():
        print(name, case["verdict"])
