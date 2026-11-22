"""Compare three-opportunity freshness on paired telemetry receiver logs."""
import argparse
import csv
import json
from pathlib import Path


def heartbeat_gaps(runs):
    """Count fully observed three-update windows with no fresh heartbeat."""
    windows = stale = 0
    for run in runs:
        misses = 0
        for index, mask in enumerate(run):
            misses = 0 if mask else misses+1
            if index >= 2:
                windows += 1
                stale += misses >= 3
    return dict(stale=stale, windows=windows)


def recovery_by_age(runs):
    counts = [dict(successors=0, recoveries=0) for _ in range(2)]
    for run in runs:
        age = None
        for mask in run:
            if age:
                cell = counts[min(age, 2)-1]
                cell["successors"] += 1
                cell["recoveries"] += bool(mask)
            if mask:
                age = 0
            elif age is not None:
                age += 1
    return dict(after_one=counts[0], after_two_or_more=counts[1])


def read_capture(path):
    rows = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = row["session"], int(row["opportunity"])
            if not key[0] or key[1] < 0 or row["received"] not in ("0", "1") or key in rows:
                raise ValueError(f"invalid or repeated receiver observation {key}")
            rows[key] = int(row["received"])
    if not rows:
        raise ValueError("empty receiver capture")
    return rows


def compare(measured, simulated, min_windows, tolerance):
    if measured.keys() != simulated.keys() or not measured:
        raise ValueError("receiver logs require the same nonempty eligible schedule")
    if min_windows < 1 or not 0 <= tolerance <= 1:
        raise ValueError("invalid support or tolerance")
    sessions = {}
    previous = None
    for session, index in sorted(measured):
        left, right = sessions.setdefault(session, ([], []))
        if previous != (session, index-1):
            left.append([])
            right.append([])
        previous = session, index
        left[-1].append(measured[previous])
        right[-1].append(simulated[previous])
    result = {}
    for session, pair in sessions.items():
        obs, sim = map(heartbeat_gaps, pair)
        n = obs["windows"]
        error = abs(obs["stale"]-sim["stale"])/n if n else None
        verdict = "insufficient_evidence" if n < min_windows else "qualified" if error <= tolerance else "rejected"
        result[session] = dict(measured=obs, simulated=sim, fraction_error=error, verdict=verdict)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measured", type=Path)
    parser.add_argument("simulated", type=Path)
    parser.add_argument("--min-windows", type=int, required=True)
    parser.add_argument("--tolerance", type=float, required=True)
    args = parser.parse_args()
    report = compare(read_capture(args.measured), read_capture(args.simulated), args.min_windows, args.tolerance)
    print(json.dumps(report, indent=2))
    raise SystemExit(int(any(v["verdict"] != "qualified" for v in report.values())))
