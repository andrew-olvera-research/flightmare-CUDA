"""Public GPU common-sink unicast diagnostic for the controlled ns-3 comparison."""
import argparse
import json
from pathlib import Path

import torch
from flightlib_cuda import Environment, NetworkConfig


def measure(senders, seed, workload="saturated", unicast_ack_backoff=False, ack_hold=False):
    # 1000-byte offers every 100 us. Each 1 ms plant tick contains ten offers per sender.
    offsets = tuple(range(0, 1000000, 100000))
    env = Environment(1, senders + 1, control_dt=.001, auto_reset=False,
                      network=NetworkConfig(offsets_ns=offsets, fanin=senders, seed=seed,
                                            unicast_ack_backoff=unicast_ack_backoff,
                                            unicast_success_hold_ns=46000 if ack_hold else 0,
                                            unicast_failed_hold_ns=41000 if ack_hold else 0))
    env.reset_state[2].fill_(5)
    env.reset()
    net = env.network
    sizes = ([128 if i % 2 else 1000 for i in range(senders)]
             if workload in ("bursty-mixed","asymmetric") else [1000]*senders)
    offers = torch.tensor(sizes).expand(len(offsets), senders).clone()
    if workload == "periodic":
        offers[1:] = 0
    elif workload == "bursty-mixed":
        offers[4:] = 0
    elif workload == "asymmetric":
        offers[1:, -1] = 0
    offered = offers.count_nonzero(dim=0).tolist()
    with env.access():
        net.sources.copy_(torch.arange(senders, device=env.device).expand_as(net.sources))
        net.size[:, 0, :senders].copy_(offers)
        net.receivers[:, 0, :senders].fill_(1 << senders)
    env.capture(random=False)
    for _ in range(100):
        env.step(graph=True, random=False)

    ticks = 1000
    ev = net.evaluator
    waits = torch.empty((ticks, len(offsets), senders), device=env.device, dtype=torch.int64)
    counts = torch.zeros((4, senders), device=env.device, dtype=torch.int64)
    for tick in range(ticks):
        env.step(graph=True, random=False)
        with env.access():
            complete = ev["completed"][:, 0, :senders].bool()
            admission = ev["admission"][:, 0, :senders]
            waits[tick] = torch.where(complete, ev["air_start_ns"][:, 0, :senders] -
                                      ev["submitted_ns"][:, 0, :senders], -1)
            counts[0] += (admission == 1).sum(0)
            counts[1] += (admission == 2).sum(0)
            counts[2] += ev["collided"][:, 0, :senders].sum(0)
            counts[3] += net.inbox[:, 0, senders, :, 0].sum(0)
    per_sender = []
    waits = waits.cpu()
    for sender, (accepted, rejected, collided, delivered) in enumerate(counts.T.tolist()):
        values = waits[:, :, sender].flatten()
        values = values[values >= 0].float()
        if (accepted + rejected != ticks * offered[sender] or
                len(values) != collided + delivered or abs(accepted - len(values)) > 1):
            raise RuntimeError(f"sender {sender} offer/completion accounting failed")
        per_sender.append(dict(sender=sender, bytes=sizes[sender], offered=ticks*offered[sender], accepted=accepted, full=rejected,
                               completed=len(values), collisions=collided, delivered=delivered,
                               wait_ns_p50_p95=torch.quantile(values, torch.tensor([.5,.95])).tolist()))
    return dict(senders=senders, seed=seed, seconds=1, workload=workload, offset_interval_us=100,
                unicast_ack_backoff=unicast_ack_backoff, ack_hold=ack_hold,
                goodput_mbps=sum(row["delivered"]*row["bytes"] for row in per_sender)*8e-6, per_sender=per_sender)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--unicast-ack-backoff", action="store_true")
    parser.add_argument("--ack-hold", action="store_true", help="add the measured service fixture's ACK admission hold")
    args = parser.parse_args()
    results = [measure(n, seed, unicast_ack_backoff=args.unicast_ack_backoff, ack_hold=args.ack_hold)
               for n in (1, 4, 8) for seed in (22, 23, 24)]
    results.extend(measure(8, 22, load, args.unicast_ack_backoff, args.ack_hold)
                   for load in ("periodic","bursty-mixed","asymmetric"))
    args.output.write_text(json.dumps(results, indent=2) + "\n")
