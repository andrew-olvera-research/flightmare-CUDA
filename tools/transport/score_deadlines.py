"""Score a common-clock packet capture with known relative TX/RX clock uncertainty.

CSV rows: session,source,attempt,receiver,tx_ns,rx_ns,bytes.
One row per eligible receiver per actual send. Empty rx_ns means missed.
The capture producer must establish the receiver set, TX denominator and
an upper bound on relative clock error independently from the outcomes.
"""
import argparse
import csv
import json
import math
from collections import defaultdict


def quantiles(values):
    if not values:
        return {}
    xs=sorted(values)
    return {str(q): xs[math.ceil(q*len(xs))-1] for q in (.5,.95,.99)}


def score(rows, deadline, error):
    if deadline<0 or error<0:
        raise ValueError("negative deadline or clock uncertainty")
    attempts=defaultdict(dict)
    metadata={}
    links=defaultdict(list)
    for row in rows:
        key=(row["session"],row["source"],row["attempt"])
        rx=row["receiver"]
        tx=int(row["tx_ns"])
        size=int(row["bytes"])
        if size<=0 or tx<0 or any(not k for k in (*key,rx)):
            raise ValueError("invalid packet identity, time or size")
        if metadata.setdefault(key,(tx,size))!=(tx,size) or rx in attempts[key]:
            raise ValueError("ambiguous packet identity or duplicate receiver")
        delay=None if row["rx_ns"]=="" else int(row["rx_ns"])-tx
        if delay is not None and delay < -error:
            raise ValueError("capture violates supplied clock uncertainty")
        attempts[key][rx]=delay
        links[(key[0],key[1],rx)].append((tx,delay))
    if not attempts:
        raise ValueError("empty capture")
    delays=[v for receivers in attempts.values() for v in receivers.values() if v is not None]
    denominator=sum(map(len,attempts.values()))
    group=[max(rx.values()) for rx in attempts.values() if all(v is not None for v in rx.values())]
    runs=[]
    for events in links.values():
        count=0
        for _,delay in sorted(events,key=lambda e:e[0]):
            if delay is None:
                count+=1
            elif count:
                runs.append(count)
                count=0
        if count:
            runs.append(count)
    def bounds(values,total):
        return [sum(v+error<=deadline for v in values)/total,
                sum(v-error<=deadline for v in values)/total]
    return dict(attempts=len(attempts),receiver_opportunities=denominator,
                receiver_deliveries=len(delays),deadline_ns=deadline,clock_bound_ns=error,
                delivery_by_deadline=bounds(delays,denominator),
                all_receivers_by_deadline=bounds(group,len(attempts)),
                measured_delay_ns=quantiles(delays),group_completion_ns=quantiles(group),
                loss_run_packets=quantiles(runs))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture")
    parser.add_argument("--deadline-ns",type=int,required=True)
    parser.add_argument("--clock-bound-ns",type=int,required=True)
    args=parser.parse_args()
    with open(args.capture,newline="") as stream:
        print(json.dumps(score(csv.DictReader(stream),args.deadline_ns,args.clock_bound_ns),indent=2))
