"""Score a common-clock packet capture with known relative TX/RX clock uncertainty.

CSV rows: session,source,attempt,receiver,tx_ns,rx_ns,bytes.
Optional generated_ns gives source creation time; its default is tx_ns.
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


def age_at_offers(events, clock_offset):
    """Age in source time just before each offer, using the freshest generation seen."""
    arrivals = sorted((max(tx, tx+delay+clock_offset), gen) for tx,gen,delay in events if delay is not None)
    ages = []
    latest = None
    i = 0
    for tx,_,_ in sorted(events, key=lambda row: row[0]):
        while i < len(arrivals) and arrivals[i][0] < tx:
            gen = arrivals[i][1]
            latest = gen if latest is None else max(latest, gen)
            i += 1
        if latest is not None:
            ages.append(tx-latest)
    return dict(known=len(ages), unknown=len(events)-len(ages), quantiles_ns=quantiles(ages))


def score(rows, deadline, error, quorum=None):
    if deadline<0 or error<0:
        raise ValueError("negative deadline or clock uncertainty")
    if quorum is not None and quorum < 1:
        raise ValueError("quorum must be positive")
    attempts=defaultdict(dict)
    metadata={}
    links=defaultdict(list)
    for row in rows:
        key=(row["session"],row["source"],row["attempt"])
        rx=row["receiver"]
        tx=int(row["tx_ns"])
        size=int(row["bytes"])
        generated = row.get("generated_ns")
        gen = tx if generated in (None, "") else int(generated)
        if not 0 <= gen <= tx:
            raise ValueError("generation must precede submission")
        if size<=0 or tx<0 or any(not k for k in (*key,rx)):
            raise ValueError("invalid packet identity, time or size")
        if metadata.setdefault(key,(tx,size,gen))!=(tx,size,gen) or rx in attempts[key]:
            raise ValueError("ambiguous packet identity or duplicate receiver")
        delay=None if row["rx_ns"]=="" else int(row["rx_ns"])-tx
        if delay is not None and delay < -error:
            raise ValueError("capture violates supplied clock uncertainty")
        attempts[key][rx]=delay
        links[(key[0],key[1],rx)].append((tx,gen,delay))
    if not attempts:
        raise ValueError("empty capture")
    delays=[v for receivers in attempts.values() for v in receivers.values() if v is not None]
    denominator=sum(map(len,attempts.values()))
    group=[max(rx.values()) for rx in attempts.values() if all(v is not None for v in rx.values())]
    runs=[]
    for events in links.values():
        count=0
        for _,_,delay in sorted(events,key=lambda e:e[0]):
            if delay is None:
                count+=1
            elif count:
                runs.append(count)
                count=0
        if count:
            runs.append(count)
    def bounds(values, total):
        # Each pair supplies a measured time and its causal lower bound.
        return [sum(max(floor, v+error) <= deadline for v, floor in values)/total,
                sum(max(floor, v-error) <= deadline for v, floor in values)/total]
    first, required, everyone = [], [], []
    for key, receivers in attempts.items():
        needed = quorum if quorum is not None else len(receivers)//2 + 1
        if needed > len(receivers):
            raise ValueError("quorum exceeds eligible receivers")
        tx,_,gen = metadata[key]
        times = sorted((tx-gen+delay, tx-gen) for delay in receivers.values() if delay is not None)
        if times: first.append(times[0])
        if len(times) >= needed: required.append(times[needed-1])
        if len(times) == len(receivers): everyone.append(times[-1])
    per_link = []
    for (session,source,receiver), events in sorted(links.items()):
        values = [tx-gen+delay for tx,gen,delay in events if delay is not None]
        per_link.append(dict(session=session, source=source, receiver=receiver, offers=len(events),
                             by_generation_deadline=bounds([(tx-gen+delay, tx-gen) for tx,gen,delay in events
                                                            if delay is not None],len(events)),
                             delay_since_generation_ns=quantiles(values),
                             age_before_offer_earliest_rx=age_at_offers(events,-error),
                             age_before_offer_latest_rx=age_at_offers(events,error)))
    return dict(per_link=per_link, quorum=quorum if quorum is not None else "strict majority",
                first_by_generation_deadline=bounds(first,len(attempts)),
                quorum_by_generation_deadline=bounds(required,len(attempts)),
                all_by_generation_deadline=bounds(everyone,len(attempts)), attempts=len(attempts),receiver_opportunities=denominator,
                receiver_deliveries=len(delays),deadline_ns=deadline,clock_bound_ns=error,
                delivery_by_deadline=bounds([(v,0) for v in delays],denominator),
                all_receivers_by_deadline=bounds([(v,0) for v in group],len(attempts)),
                measured_delay_ns=quantiles(delays),group_completion_ns=quantiles(group),
                loss_run_packets=quantiles(runs))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture")
    parser.add_argument("--deadline-ns",type=int,required=True)
    parser.add_argument("--clock-bound-ns",type=int,required=True)
    parser.add_argument("--quorum", type=int, help="required peer count (default: strict majority)")
    args=parser.parse_args()
    with open(args.capture,newline="") as stream:
        print(json.dumps(score(csv.DictReader(stream),args.deadline_ns,args.clock_bound_ns,args.quorum),indent=2))
