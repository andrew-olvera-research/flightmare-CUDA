"""Check live Torch allocations after repeated public network graph teardown."""
import gc
import json

import torch

from flightlib_cuda import Environment, NetworkConfig


def cycle():
    env = Environment(32, 8, network=NetworkConfig(window_slots=1), auto_reset=False)
    env.reset_state[2].fill_(5)
    env.reset()
    net = env.network
    net.sources.zero_()
    net.broadcast[:, 0] = 1
    net.size[:, 0] = 100
    env.capture(random=False)
    env.step(graph=True, random=False)
    if net.inbox[..., 0].count_nonzero().item() != 32 * 7:
        raise RuntimeError("expected seven broadcast deliveries per environment")


def sample():
    gc.collect()
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated(), torch.cuda.mem_get_info()[0]


def main():
    cycle()
    live, free = sample()
    for iteration in range(24):
        cycle()
        current, remaining = sample()
        if current != live:
            raise RuntimeError(f"cycle {iteration + 1}: expected {live} live Torch bytes, got {current}")
    print(json.dumps(dict(cycles=24, torch_live_bytes=live, free_before=free, free_after=remaining)))


if __name__ == "__main__":
    main()
