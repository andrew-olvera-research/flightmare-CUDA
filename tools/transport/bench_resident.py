"""Benchmark complete public environment graphs including receiver inbox materialization."""
import argparse
import json
import random
import statistics

import torch
from flightlib_cuda import Environment, NetworkConfig


def measure(domains, stations, mode, senders=1, fanin=None):
    fanin = senders if fanin is None else fanin
    persistent = mode in ("persistent", "edge-rates")
    loaded = mode in ("loaded-fixed", "loaded-ack")
    acknowledged = mode in ("loaded-ack", "unicast-ack")
    unicast = loaded or mode in ("unicast-fixed", "unicast-cw", "unicast-ack")
    sampled = mode in ("sampled", "sampled-loss") or persistent
    outage = dict(outage_period_ns=10000000, outage_enter=.05, outage_recover=.45) if persistent else {}
    net_config = None if mode == "plant" else NetworkConfig(fanin=fanin, range_edges_m=() if sampled else None,
                                                            edge_outage_rates=mode == "edge-rates",
                                                            offsets_ns=tuple(range(0,1000000,100000)) if loaded else (0,),
                                                            unicast_ack_backoff=acknowledged or mode == "unicast-cw",
                                                            unicast_success_hold_ns=46000 if acknowledged else 0,
                                                            unicast_failed_hold_ns=41000 if acknowledged else 0,
                                                            **outage)
    env = Environment(domains, stations, control_dt=.001 if loaded else .01, network=net_config, auto_reset=False)
    env.reset_state[2].fill_(5)
    env.reset()
    if env.network:
        net=env.network
        net.sources.copy_(torch.arange(fanin, device=env.device).expand_as(net.sources))
        if mode == "sampled-loss":
            net.loss_after_rx.fill_(.1)
            net.loss_after_miss.fill_(.1)
        if mode == "edge-rates":
            net.outage_rates[...,0].fill_(.05)
            net.outage_rates[...,1].fill_(.45)
        net.size[..., :senders].fill_(1000)
        net.broadcast[..., :senders].fill_(1)
        if stations==1 or unicast:
            net.broadcast.zero_()
            net.receiver_base.fill_(stations-1)
            net.receivers.fill_(1)
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph=torch.cuda.CUDAGraph()
    with env.access(stream), torch.cuda.graph(graph, stream=stream):
        env.record_step(random=False)
    samples=[]
    with env.access(stream):
        for _ in range(64):
            graph.replay()
        events=[torch.cuda.Event(enable_timing=True) for _ in range(129)]
        for event in events:
            event.record()
        stream.synchronize()
        for event in events[:-1]:
            event.record()
            graph.replay()
        events[-1].record()
        stream.synchronize()
        samples=[1000*a.elapsed_time(b) for a,b in zip(events,events[1:])]
        deliveries=0 if net_config is None else int(net.inbox[...,0].to(torch.int64).sum().item())
        successful = 0
        if net_config:
            ev = net.evaluator
            completed = ev["completed"][..., :senders].bool()
            if loaded:
                admission = ev["admission"][..., :senders]
                balance = (admission == 1).sum(0) - completed.sum(0)
                if not ((admission == 1) | (admission == 2)).all().item() or balance.abs().max().item() > 1:
                    raise RuntimeError("loaded sender accounting failed")
            elif not completed.all().item():
                raise RuntimeError("benchmark tick too short for all submissions")
            successful = (completed[..., :fanin] & (ev["collided"][..., :fanin] == 0)).sum().item()
        possible = successful if unicast else successful * max(1, stations-1)
        if deliveries > possible or ((unicast or mode in ("shared", "sampled")) and deliveries != possible):
            raise RuntimeError("final inbox failed coverage check")
        # Count on a separate replay window so diagnostics stay outside the step timer.
        total = torch.zeros((), dtype=torch.int64, device=env.device)
        if net_config:
            for _ in range(64):
                graph.replay()
                total += net.inbox[...,0].sum()
        rate = total.item()/64*1e6/statistics.mean(samples)
    return dict(fanin=fanin, useful_deliveries_per_second=rate, domains=domains,stations=stations,mode=mode,senders_per_medium=senders,median_us=statistics.median(samples),
                p95_us=sorted(samples)[121],observations_final_step=deliveries,
                arena_bytes=env._native.arena_bytes)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds",type=int,default=3)
    parser.add_argument("--senders", type=int, nargs="+", default=[1])
    parser.add_argument("--session-service", action="store_true",
                        help="compare resident edge rates and acknowledged unicast against their matched controls")
    args=parser.parse_args()
    if min(args.senders) < 1 or max(args.senders) > 1000 or args.rounds < 1:
        parser.error("positive rounds and 1 to 1000 senders required")
    rng=random.Random(22)
    cases = []
    for e, s in ((1,1),(1,100),(1,1000),(100,10)):
        counts = sorted({n for n in args.senders if n <= s})
        if args.session_service:
            if s in (10,1000):
                cases.extend((e,s,m,n,n) for n in counts for m in
                             ("persistent","edge-rates","unicast-fixed","unicast-cw","unicast-ack","loaded-fixed","loaded-ack"))
        elif counts:
            cases.append((e,s,"plant",0,0))
            cases.extend((e,s,mode,n,fan) for n in counts for fan in sorted({1,n})
                         for mode in ("shared","sampled","sampled-loss","persistent"))
    for repeat in range(args.rounds):
        rng.shuffle(cases)
        for case in cases:
            print(json.dumps(dict(measure(*case),round=repeat)),flush=True)
