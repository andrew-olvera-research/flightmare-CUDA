"""Slot-by-slot oracle for simultaneous offers, backoff, collision and expiry."""
import random

import torch

from flightlib_cuda import Environment, NetworkConfig


def draw(sender, sequence, seed, window):
    mask = 2**64-1
    x = (sender + (seed << 32) + sequence*0x9e3779b97f4a7c15) & mask
    for shift, factor in ((30,0xbf58476d1ce4e5b9), (27,0x94d049bb133111eb)):
        x = ((x ^ (x >> shift))*factor) & mask
    return (x ^ (x >> 31)) % window


def test_independent_slot_clock():
    n, tick = 5, 1_000_000
    offsets = (0, 13_000, 29_123, 101_000)
    p = NetworkConfig(fanin=n, offsets_ns=offsets, window_slots=7)
    env = Environment(1,n,control_dt=.001,network=p,auto_reset=False)
    env.reset_state[2].fill_(5)
    env.reset()
    net = env.network
    net.sources.copy_(torch.arange(n,device=env.device).expand(1,n,n))
    net.broadcast.fill_(1)
    seq = [0]*n
    pending = [None]*n
    cursor = 0
    rng = random.Random(123)
    for step in range(25):
        expected = []
        for segment,(offset, stop) in enumerate(zip(offsets, offsets[1:]+(tick,))):
            now, end = step*tick+offset, step*tick+stop
            received = {}
            for sender in range(n):
                size = rng.choice((0,50,400,1000))
                ident = (step*len(offsets)+segment)*n+sender
                expiry = now+rng.randrange(1000,500000)
                net.size[segment,0,sender] = size
                net.packet_id[segment,0,sender] = ident
                net.expiry_ns[segment,0,sender] = expiry
                if size and pending[sender] is None:
                    seq[sender] += 1
                    pending[sender] = dict(backoff=draw(sender,seq[sender],p.seed,p.window_slots),
                                           size=size, expiry=expiry, id=ident, submitted=now)
            while cursor < end:
                waiting = [q for q in pending if q and "finish" not in q]
                if not waiting:
                    cursor=end
                    break
                winners = [q for q in waiting if q["backoff"]==0]
                if winners:
                    for q in winners:
                        q.update(finish=cursor+p.preamble_ns+q["size"]*p.ns_per_byte,
                                 collision=len(winners)>1)
                    cursor=max(q["finish"] for q in winners)+p.recovery_ns
                else:
                    for q in waiting:
                        q["backoff"] -= 1
                    cursor += p.slot_ns
            for sender,q in enumerate(pending):
                if q and q.get("finish", end+1)<=end:
                    if not q["collision"] and q["finish"]<=q["expiry"]:
                        received[sender]=[1,q["id"],q["submitted"],q["finish"],0,0]
                    pending[sender]=None
            expected.append([[received.get(s,[0]*6) if s!=rx else [0]*6 for s in range(n)]
                             for rx in range(n)])
        env.step(random=False)
        inbox = net.inbox[:, 0].tolist()
        for segment, receivers in enumerate(expected):
            for receiver, senders in enumerate(receivers):
                for sender, event in enumerate(senders):
                    actual = inbox[segment][receiver][sender]
                    assert actual == event, (
                        f"traffic_seed=123 backoff_seed={p.seed} env=0 tick={step} "
                        f"segment={segment} sender={sender} receiver={receiver} "
                        f"expected={event} actual={actual}"
                    )
