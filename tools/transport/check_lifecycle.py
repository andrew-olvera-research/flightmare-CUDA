"""Check live Torch allocations after repeated public network graph teardown."""
import gc
import json

import torch

from flightlib_cuda import Environment, NetworkConfig, ReceiverOutcome


def cycle(persistent=False):
    config = (NetworkConfig(window_slots=1, outage_period_ns=10000000, outage_enter=.05,
                            outage_recover=.45, record_outcomes=True, dynamic_subscriptions=True, edge_outage_rates=True)
              if persistent else NetworkConfig(window_slots=1))
    domains, slots = (256,16) if persistent else (32,8)
    env = Environment(domains, slots, network=config, auto_reset=False)
    env.reset_state[2].fill_(5)
    env.reset()
    net = env.network
    if persistent:
        net.outage_rates[...,0].fill_(.05)
        net.outage_rates[...,1].fill_(.45)
    net.sources.zero_()
    net.broadcast[:, 0] = 1
    net.size[:, 0] = 100
    env.capture(random=False)
    for _ in range(64 if persistent else 1):
        env.step(graph=True, random=False)
    if persistent:
        outcome = net.evaluator["receiver_outcomes"][:,1:,0]
        received = outcome == ReceiverOutcome.DELIVERED
        if not (received | (outcome == ReceiverOutcome.OUTAGE)).all().item():
            raise RuntimeError("unaccounted broadcast receiver")
        if not torch.equal(net.inbox[:,1:,0,0].bool(), received):
            raise RuntimeError("outcome/inbox disagreement")
    elif net.inbox[..., 0].count_nonzero().item() != domains*(slots-1):
        raise RuntimeError("expected seven broadcast deliveries per environment")


def sample():
    gc.collect()
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated(), torch.cuda.mem_get_info()[0]


def main():
    cycle()
    live, free = sample()
    for iteration in range(24):
        cycle(persistent=bool(iteration % 2))
        current, remaining = sample()
        if current != live:
            raise RuntimeError(f"cycle {iteration + 1}: expected {live} live Torch bytes, got {current}")
    print(json.dumps(dict(cycles=24, torch_live_bytes=live, free_before=free, free_after=remaining)))


if __name__ == "__main__":
    main()
