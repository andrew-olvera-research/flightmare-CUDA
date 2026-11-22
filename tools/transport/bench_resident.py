"""Benchmark complete public environment graphs including receiver inbox materialization."""
import argparse
import json
import random
import statistics

import torch
from flightlib_cuda import Environment, NetworkConfig


def measure(domains, stations, mode, senders=1, fanin=None):
    fanin = senders if fanin is None else fanin
    sampled = mode in ("sampled", "sampled-loss", "persistent")
    outage = dict(outage_period_ns=10000000, outage_enter=.05, outage_recover=.45) if mode == "persistent" else {}
    net_config = None if mode == "plant" else NetworkConfig(fanin=fanin, range_edges_m=() if sampled else None,
                                                            **outage)
    env = Environment(domains, stations, network=net_config, auto_reset=False)
    env.reset_state[2].fill_(5)
    env.reset()
    if env.network:
        net=env.network
        net.sources.copy_(torch.arange(fanin, device=env.device).expand_as(net.sources))
        if mode == "sampled-loss":
            net.loss_after_rx.fill_(.1)
            net.loss_after_miss.fill_(.1)
        net.size[:, :senders].fill_(1000)
        net.broadcast[:, :senders].fill_(1)
        if stations==1:
            net.broadcast.zero_()
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
            if not ev["completed"][:, :senders].all().item():
                raise RuntimeError("benchmark tick too short for all submissions")
            successful = (ev["collided"][:, :fanin] == 0).sum().item()
        possible = successful * max(1, stations-1)
        if deliveries > possible or (mode in ("shared", "sampled") and deliveries != possible):
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
    args=parser.parse_args()
    if min(args.senders) < 1 or max(args.senders) > 1000 or args.rounds < 1:
        parser.error("positive rounds and 1 to 1000 senders required")
    rng=random.Random(22)
    cases = []
    for e, s in ((1,1),(1,100),(1,1000),(100,10)):
        counts = sorted({n for n in args.senders if n <= s})
        if counts:
            cases.append((e,s,"plant",0,0))
            cases.extend((e,s,mode,n,fan) for n in counts for fan in sorted({1,n})
                         for mode in ("shared","sampled","sampled-loss","persistent"))
    for repeat in range(args.rounds):
        rng.shuffle(cases)
        for case in cases:
            print(json.dumps(dict(measure(*case),round=repeat)),flush=True)
