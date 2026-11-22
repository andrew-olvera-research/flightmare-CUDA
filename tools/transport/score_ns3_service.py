"""Score patched ns-3.44 wifi-bianchi logs from the one-second common-sink fixture."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path


def parse(path, senders):
    start, end = 10_000_000_000, 11_000_000_000
    trials = []
    with path.open() as stream:
        for line in stream:
            if line.startswith("Trial "):
                offers, transmitted, waits = {}, set(), defaultdict(list)
            elif line.startswith(("packet-offer ", "packet-air ")):
                kind, t, sender, uid = line.split()
                t, sender = int(t), int(sender)
                key = sender, int(uid)
                if kind == "packet-offer":
                    if key in offers:
                        raise ValueError("duplicate application packet")
                    offers[key] = t
                elif start <= t < end:
                    if key not in offers or key in transmitted or offers[key] > t:
                        raise ValueError("unmatched, repeated or noncausal transmit start")
                    transmitted.add(key)
                    waits[sender].append(t-offers[key])
            elif line.startswith("Total throughput:"):
                if set(waits) != set(range(1, senders+1)):
                    raise ValueError("incomplete station coverage")
                per_sender = []
                for sender, values in sorted(waits.items()):
                    count = sum(node == sender and start <= t < end for (node,_),t in offers.items())
                    if count != 10000:
                        raise ValueError(f"expected 10000 offers at sender {sender}, saw {count}")
                    values.sort()
                    per_sender.append(dict(sender=sender, offered=count, transmitted=len(values),
                                           wait_ns_p50_p95=[values[math.ceil(q*len(values))-1] for q in (.5,.95)]))
                trials.append(dict(reported_goodput_mbps=float(line.split()[-2]), per_sender=per_sender))
    if len(trials) != 3:
        raise ValueError("expected three complete trials")
    return trials


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="contains service-{1,4,8}.log")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = {n: parse(args.directory / f"service-{n}.log",n) for n in (1,4,8)}
    args.output.write_text(json.dumps(results, indent=2) + "\n")
