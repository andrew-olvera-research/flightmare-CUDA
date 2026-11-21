"""Benchmark complete public environment graphs including receiver inbox materialization."""
import argparse
import json
import random
import statistics

import torch
from flightlib_cuda import Environment, NetworkConfig


def measure(domains, stations, mode):
    # One source per medium, with every peer listening. Isolates the cost
    # of useful high fan-out from collision-heavy saturation benchmarks.
    net_config = None if mode=="plant" else NetworkConfig(range_edges_m=() if mode=="sampled" else None)
    env = Environment(domains, stations, network=net_config, auto_reset=False)
    env.reset_state[2].fill_(5)
    env.reset()
    if env.network:
        net=env.network
        net.sources.zero_()
        net.size[:,0].fill_(1000)
        net.broadcast[:,0].fill_(1)
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
        if net_config and deliveries!=domains*max(1,stations-1):
            raise RuntimeError("final inbox failed coverage check")
    return dict(domains=domains,stations=stations,mode=mode,median_us=statistics.median(samples),
                p95_us=sorted(samples)[121],observations_final_step=deliveries,
                arena_bytes=env._native.arena_bytes)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds",type=int,default=3)
    args=parser.parse_args()
    rng=random.Random(22)
    cases=[(e,s,m) for e,s in ((1,1),(1,100),(1,1000),(100,10))
           for m in ("plant","shared","sampled")]
    for repeat in range(args.rounds):
        rng.shuffle(cases)
        for case in cases:
            print(json.dumps(dict(measure(*case),round=repeat)),flush=True)
