"""Test February 13 morning-to-afternoon HW broadcast transfer at the same center frequency."""
import argparse
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from qualify_broadcast import evaluate
from validate_bristol import DIGESTS, OBU_MAC, ROOT, broadcast_rows, fetch, read


DAY = "13022019"
SESSIONS = ("Morning", "Afternoon")
# Pinned full logs for the second session and the day-long Hawthorns transmitter.
PM_DIGESTS = ("e026c02a85c4e258c1c1630cd5522bdf4498b43fd562902124f36c953e33d1dc",
              "3dbc51f2d06bc974b1fdbf32960a53fd6292dd7b6edf1f491d08981f30e6b31f")
HW_DIGEST = "04ee84ef13c22042cf5e4c4bfcbafa72bcf2b4192e2a242195c142e9321c2b95"


def load(directory, name, remote, digest, download, peer=""):
    path = directory / f"bristol-{DAY}-{name}.log"
    if download:
        fetch(path, f"{ROOT}/{DAY}/{remote}")
    source = read(path, DAY, peer)
    if source[3] != digest:
        raise ValueError(f"unexpected SHA-256 for {remote}")
    return source


def qualify(directory, download):
    tx = load(directory, "hw-day", f"RSUs-{DAY}/log_HP_HW.txt", HW_DIGEST, download)[0]
    audit = {}
    with TemporaryDirectory() as scratch:
        geometry = Path(scratch) / "offers.csv"
        with geometry.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("day", "time_ms", "distance_0", "distance_1", "mask"))
            for session, digests in zip(SESSIONS, (DIGESTS[1][:2], PM_DIGESTS)):
                pair = []
                for i in range(2):
                    obu = f"{i:02d}"
                    name = obu if session == "Morning" else "pm-" + obu
                    pair.append(load(directory, name, f"OBU-{obu}-{DAY}-{session}/log_HP.txt",
                                     digests[i], download, OBU_MAC[1-i]))
                start = max(log[0][0][0] for log in pair)
                end = min(log[0][-1][0] for log in pair)
                if start >= end or (audit and start <= audit["Morning"]["end_ms"]):
                    raise ValueError("missing or overlapping capture sessions")
                offsets = [[], []]
                counts = [0]*4
                offers = [r for r in tx if start <= r[0] <= end]
                for t, mask, distance in broadcast_rows(offers, pair, offsets):
                    if distance is not None:
                        writer.writerow((session, t, *distance, mask))
                        counts[mask] += 1
                skew = [sorted(map(abs, v))[int(.95*(len(v)-1))] if v else None for v in offsets]
                if any(x is None or x > 250 for x in skew):
                    raise ValueError(f"{session} TX/RX timestamp agreement insufficient for 250 ms GPS coverage: {skew}")
                audit[session] = dict(start_ms=start, end_ms=end, offered_in_overlap=len(offers), geometry_mask_counts=counts,
                                      matched_timestamp_abs_p95_ms=skew)
        result = evaluate(geometry, sessions=SESSIONS)
    if min(result["observed_mask_counts"]) == 0:
        raise ValueError("same-channel holdout must witness all four receiver outcomes")
    return dict(**result, date=DAY, hp_center_mhz=5200, sessions=audit,
                source_sha256=dict(morning=DIGESTS[1][:2], afternoon=PM_DIGESTS, transmitter=HW_DIGEST))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-qualified", action="store_true")
    args = parser.parse_args()
    report = qualify(args.directory, args.download)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(report["verdict"], report["supported"])
    raise SystemExit(int(args.require_qualified and report["verdict"] != "qualified"))
