"""Evaluate directional delivery by distance on two complete Bristol driving sessions."""
import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = "https://data.bris.ac.uk/datasets/eupowp7h3jl525yxhm3521f57/rawData"
DAYS = ("11022019", "13022019")
EDGES = [25, 50, 100, 200, 400]


def read(path, day):
    start = int(datetime.strptime(day, "%d%m%Y").replace(tzinfo=timezone.utc).timestamp()*1000)
    tx, rx, audit = [], set(), Counter()
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for raw in f:
            digest.update(raw)
            line = raw.decode("ascii", errors="replace")
            kind = "TX" if "TX-REQ-CAM;" in line else "RX"
            marker = kind + "-REQ-CAM;"
            if marker not in line:
                continue
            fields = line.split(marker, 1)[1].rstrip().split(";")
            row = dict(zip(fields[::2], fields[1::2]))
            try:
                stamp = int(row["Timestamp"])
                key = int(row["StationID"]), int(row["SeqNum"])
                if not start <= stamp < start+86400000:
                    audit["other_day"] += 1
                    continue
                if kind == "TX":
                    # Transmitter GPS at offer time, used for failed and successful packets alike.
                    lat, lon = float(row["GpsLat"]), float(row["GpsLon"])
                    tx.append((stamp, key, lat, lon))
                elif row.get("Validation") == "1":
                    rx.add(key)
            except (ValueError, KeyError):
                audit["malformed"] += 1
    return sorted(tx), rx, dict(audit), digest.hexdigest()


def distance(a, b):
    lat, lon, lat2, lon2 = map(math.radians, (*a, *b))
    h = math.sin((lat2-lat)/2)**2 + math.cos(lat)*math.cos(lat2)*math.sin((lon2-lon)/2)**2
    return 12742000*math.asin(min(1, math.sqrt(h)))


def count(sender, receiver):
    tx, peer_rx = sender[0], receiver[1]
    gps = receiver[0]
    if not tx or not gps:
        raise ValueError("missing TX log")
    keys = Counter(packet[1] for packet in tx)
    stamps = [packet[0] for packet in gps]
    bins = [[0, 0] for _ in range(len(EDGES)+1)]
    excluded = Counter()
    for t, key, lat, lon in tx:
        if keys[key] != 1:
            excluded["ambiguous"] += 1
            continue
        if not stamps[0] <= t <= stamps[-1]:
            excluded["outside_overlap"] += 1
            continue
        i = bisect_left(stamps, t)
        fix = min(gps[max(0,i-1):i+1], key=lambda row: abs(row[0]-t))
        if abs(fix[0]-t) > 250 or not all(50 < x < 53 and -4 < y < -1
                                            for x,y in [(lat,lon),fix[2:]]):
            excluded["unpaired_position"] += 1
            continue
        band = bisect_right(EDGES, distance((lat,lon), fix[2:]))
        bins[band][0] += 1
        bins[band][1] += key in peer_rx
    return {"offered_received": bins, "excluded": dict(excluded)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory", type=Path, help="bristol-DDMMYYYY-{00,01}.log files")
    args = ap.parse_args()
    sessions, sources = [], []
    for day in DAYS:
        pair = []
        for obu in ("00","01"):
            row = read(args.directory / f"bristol-{day}-{obu}.log", day)
            sources.append(dict(url=f"{ROOT}/{day}/OBU-{obu}-{day}-Morning/log_HP.txt",
                                sha256=row[3], audit=row[2]))
            pair.append(row)
        sessions.append([count(*pair), count(*reversed(pair))])
    scores = []
    for direction in range(2):
        for band, (fit, test) in enumerate(zip(*(s[direction]["offered_received"] for s in sessions))):
            if min(fit[0],test[0]) >= 200:
                error = abs(fit[1]/fit[0] - test[1]/test[0])
                scores.append(dict(direction=direction, band=band, absolute_error=error,
                                   passed=error <= .10))
    passed = bool(scores) and all(s["passed"] for s in scores)
    print(json.dumps(dict(sources=sources, distance_edges_m=EDGES, sessions=sessions,
                          scores=scores, qualified=passed), indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
