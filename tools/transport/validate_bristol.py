"""Evaluate directional delivery by distance on two complete Bristol driving sessions."""
import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import random
import shutil
from urllib.request import Request, urlopen
from collections import defaultdict
from pathlib import Path

ROOT = "https://data.bris.ac.uk/datasets/eupowp7h3jl525yxhm3521f57/rawData"
DAYS = ("11022019", "13022019")
EDGES = [25, 50, 100, 200, 400]
SIZES = ((232206813,253880147),(241168035,262880996))


OBU_MAC = ("4c:5e:0c:84:35:f6", "00:0c:42:6f:bc:ad")
HW_MAC = "4c:5e:0c:84:35:fb"
HW_POSITION = (51.4582980, -2.6038980)


DIGESTS = (
    ("55fd6af13e1e687e5d1df5e7df4cb49e4b1490722f5a12207b7cdb17aba4bfc8",
     "5fa0373c3fd68d4327dd2f878eb30d384522f4fc34cf43f9150371c32c365706",
     "915ab3c64a469350436665c714afafbdb3cee71f8ba8060d33eec65ad1068ce1"),
    ("e58d6601da313ea06d3c827cc83d11a2227aee7a391cb652a5d991b2fc022331",
     "00e79ce893ae5c472f607dd4792ed4839dce64ae0d35f970999cadbb6511dd1b",
     "dbc6b6c2f5c478a6948b01889a19b29acdbbcd9572e7101ae1d44ee9db92640c")
)


def fetch(path, url, sliced=False):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    headers={"Range":"bytes=125829120-142606335"} if sliced else {}
    temporary=path.with_suffix(".partial")
    with urlopen(Request(url,headers=headers),timeout=60) as source:
        if sliced and (source.status!=206 or not source.headers["Content-Range"].startswith("bytes 125829120-142606335/")):
            raise ValueError("server returned the wrong source range")
        with temporary.open("wb") as out:
            shutil.copyfileobj(source,out)
    temporary.replace(path)


def read(path, day, wanted_mac):
    start = int(datetime.strptime(day, "%d%m%Y").replace(tzinfo=timezone.utc).timestamp()*1000)
    tx, rx, audit, rsu_rx = [], set(), Counter(), {}
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
                elif row.get("Validation") == "1" and "RxMAC;" in line:
                    mac = line.split("RxMAC;",1)[1].split(";",1)[0].lower()
                    if mac == HW_MAC:
                        if (float(row["GpsLat"]), float(row["GpsLon"])) != HW_POSITION:
                            raise RuntimeError("Hawthorns source MAC and payload position disagree")
                        rsu_rx[key] = min(stamp, rsu_rx.get(key, stamp))
                    if mac == wanted_mac:
                        rx.add(key)
            except (ValueError, KeyError):
                audit["malformed"] += 1
    return sorted(tx), rx, dict(audit), digest.hexdigest(), rsu_rx


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
    blocks = defaultdict(lambda: [[0,0] for _ in bins])
    runs = []
    run = 0
    previous = None
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
        seen = key in peer_rx
        bins[band][1] += seen
        counts = blocks[t//60000][band]
        counts[0] += 1
        counts[1] += seen
        if previous is not None and (key[1]!=previous[1]+1 or t-previous[0]>1000):
            if run: runs.append(run)
            run=0
        if seen:
            if run: runs.append(run)
            run=0
        else:
            run+=1
        previous = t,key[1]
    if run: runs.append(run)
    rng = random.Random(22)
    minutes = list(blocks.values())
    uncertainty = [[] for _ in bins]
    for _ in range(500):
        totals = [[0,0] for _ in bins]
        for block in rng.choices(minutes,k=len(minutes)):
            for total,row in zip(totals,block):
                total[0]+=row[0]
                total[1]+=row[1]
        for samples,(offers,heard) in zip(uncertainty,totals):
            if offers: samples.append(heard/offers)
    ci = [[sorted(xs)[int(.025*len(xs))],sorted(xs)[int(.975*len(xs))]] if xs else None
          for xs in uncertainty]
    return {"offered_received": bins, "excluded": dict(excluded),
            "eligible_minutes": len(minutes), "minute_bootstrap_95": ci,
            "loss_runs": len(runs), "max_loss_run": max(runs,default=0)}


def broadcast_rows(offers, pair, offsets=None, clipped=False):
    """One HW transmission and both active receiver logs, restricted to the same session."""
    times = [[r[0] for r in log[0]] for log in pair]
    unique = Counter(r[1] for r in offers)
    for t, key, lat, lon in (offers[1:-1] if clipped else offers):
        if (lat, lon) != HW_POSITION:
            raise ValueError("unexpected transmitter position in Hawthorns TX log")
        if unique[key] != 1:
            continue
        indexes = [bisect_left(a,t) for a in times]
        if not all(0<i<len(a) and max(t-a[i-1],a[i]-t)<=250 for i,a in zip(indexes,times)):
            continue
        mask = 0
        distances = []
        for peer,(log,i) in enumerate(zip(pair,indexes)):
            if key in log[4]:
                mask |= 1 << peer
                if offsets is not None:
                    offsets[peer].append(log[4][key] - t)
            fix = min(log[0][i-1:i+1], key=lambda row: abs(row[0]-t))
            if 50 < fix[2] < 53 and -4 < fix[3] < -1:
                distances.append(distance((lat,lon),fix[2:]))
        yield t, mask, distances if len(distances) == 2 else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory", type=Path, help="bristol-DDMMYYYY-{00,01}.log files")
    ap.add_argument("--replay-output", type=Path, help="write two-receiver masks for shared_test --broadcast-replay")
    ap.add_argument("--geometry-output", type=Path, help="write eligible roadside offers and receiver distances")
    ap.add_argument("--download",action="store_true",help="fetch missing pinned source files (about 1 GB)")
    args = ap.parse_args()
    trace = bytearray()
    geometry = ["day,time_ms,distance_0,distance_1,mask\n"] if args.geometry_output else None
    sessions, sources, broadcast = [], [], []
    for session, day in enumerate(DAYS):
        pair = []
        for obu in ("00","01"):
            path = args.directory / f"bristol-{day}-{obu}.log"
            url=f"{ROOT}/{day}/OBU-{obu}-{day}-Morning/log_HP.txt"
            if args.download: fetch(path,url)
            if path.stat().st_size != SIZES[session][int(obu)]:
                raise ValueError(f"incomplete or unexpected source: {path}")
            row = read(path, day, OBU_MAC[1-int(obu)])
            if row[3]!=DIGESTS[session][int(obu)]:
                raise ValueError("unexpected OBU source SHA-256")
            sources.append(dict(url=url,
                                sha256=row[3], audit=row[2]))
            pair.append(row)
        sessions.append([count(*pair), count(*reversed(pair))])
        # Same source and byte range on both days. Receiver TX timestamps
        # serve as evidence that each logger is live on both sides of the offer.
        fragment=args.directory/f"bristol-{day}-hw-prefix.log"
        rsu_url=f"{ROOT}/{day}/RSUs-{day}/log_HP_HW.txt"
        if args.download: fetch(fragment,rsu_url,sliced=True)
        if fragment.stat().st_size!=16*1024**2:
            raise ValueError("unexpected HW source range")
        hw=read(fragment,day,"")
        if hw[3]!=DIGESTS[session][2]:
            raise ValueError("unexpected RSU source SHA-256")
        sources.append(dict(url=rsu_url,
                            byte_range=[125829120,142606335],sha256=hw[3]))
        masks = [0]*4
        for t, mask, distances in broadcast_rows(hw[0], pair, clipped=True):
            if geometry is not None and distances is not None:
                geometry.append(f"{day},{t},{distances[0]},{distances[1]},{mask}\n")
            masks[mask] += 1
            trace.append(ord("0")+mask)
        broadcast.append(dict(day=day,receiver_mask_counts=masks,denominator=sum(masks)))
    if geometry is not None:
        args.geometry_output.write_text("".join(geometry))
    if args.replay_output is not None:
        args.replay_output.write_bytes(trace)
    scores = []
    for direction in range(2):
        for band, (fit, test) in enumerate(zip(*(s[direction]["offered_received"] for s in sessions))):
            if min(fit[0],test[0]) >= 200:
                error = abs(fit[1]/fit[0] - test[1]/test[0])
                scores.append(dict(direction=direction, band=band, absolute_error=error,
                                   passed=error <= .10))
    passed = bool(scores) and all(s["passed"] for s in scores)
    print(json.dumps(dict(schema_version=1, replay_sha256=hashlib.sha256(trace).hexdigest(),
                          sources=sources, distance_edges_m=EDGES, sessions=sessions,
                          scores=scores, broadcast=broadcast, qualified=passed), indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
