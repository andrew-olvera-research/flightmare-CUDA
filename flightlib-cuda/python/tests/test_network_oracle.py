"""Slot-by-slot oracle for simultaneous offers, backoff, collision and expiry."""
import random

import pytest
import torch

from flightlib_cuda import Environment, NetworkConfig


def test_offset_and_recent_packet_history_in_graph():
    config = NetworkConfig(offsets_ns=(0, 200000), window_slots=1, loss_history_bits=3)
    env = Environment(2, 2, network=config, control_dt=.001, auto_reset=False)
    env.reset_state[2].fill_(5)
    env.reset()
    net = env.network
    rng = random.Random(41)
    table = [[rng.randrange(2) for _ in range(15)] for _ in range(2)]
    net.context_loss.copy_(torch.tensor(table, device=env.device).expand_as(net.context_loss))
    env.capture(random=False)
    histories = [[], []]
    for tick in range(12):
        if tick in (3, 7):
            # Reset just the sender, then just the receiver in environment zero.
            env.selected.zero_()
            env.selected[int(tick == 7)] = 1
            env.reset()
            histories[0].clear()
        net.sources.zero_()
        net.size[:, :, 0].fill_(32)
        net.broadcast[:, :, 0].fill_(1)
        env.step(graph=True, random=False)
        got = net.inbox[:, :, 1, 0, 0].tolist()
        for ordinal in range(2):
            for lane, history in enumerate(histories):
                value = 0
                for bit in history[-3:]:
                    value = 2*value + bit
                loss = table[ordinal][(1 << min(3, len(history))) - 1 + value]
                assert got[ordinal][lane] == 1-loss
                history.append(loss)


def test_loss_context_follows_queued_submission_offset():
    config = NetworkConfig(offsets_ns=(0, 200000), window_slots=1, loss_history_bits=1)
    env = Environment(1, 2, network=config, control_dt=.001, auto_reset=False)
    env.reset_state[2].fill_(5)
    env.reset()
    net = env.network
    net.sources.zero_()
    net.broadcast.fill_(1)
    net.context_loss[..., 0, :].fill_(1)
    net.context_loss[..., 1, :].zero_()
    # Both frames finish in a later segment than the one in which they were offered.
    net.size[0, 0, 0] = 2000
    env.step(random=False)
    assert net.evaluator["completed"][1, 0, 0].item() == 1
    assert net.inbox[1, 0, 1, 0, 0].item() == 0
    net.size.zero_()
    net.size[1, 0, 0] = 5500
    env.step(random=False)
    net.size.zero_()
    env.step(random=False)
    assert net.inbox[0, 0, 1, 0, 0].item() == 1


def draw(sender, sequence, seed, window):
    mask = 2**64-1
    x = (sender + (seed << 32) + sequence*0x9e3779b97f4a7c15) & mask
    for shift, factor in ((30,0xbf58476d1ce4e5b9), (27,0x94d049bb133111eb)):
        x = ((x ^ (x >> shift))*factor) & mask
    return (x ^ (x >> 31)) % window


@pytest.mark.parametrize(("ack_backoff", "hold"), [(False, False), (True, False), (True, True)])
def test_independent_slot_clock(ack_backoff, hold):
    n, tick = 5, 1_000_000
    offsets = (0, 13_000, 29_123, 101_000)
    p = NetworkConfig(fanin=n, offsets_ns=offsets, window_slots=7, unicast_ack_backoff=ack_backoff,
                      unicast_success_hold_ns=46000 if hold else 0, unicast_failed_hold_ns=41000 if hold else 0)
    env = Environment(1,n,control_dt=.001,network=p,auto_reset=False)
    env.reset_state[2].fill_(5)
    env.reset()
    net = env.network
    net.sources.copy_(torch.arange(n,device=env.device).expand(1,n,n))
    peers = [{(s+1)%n} if ack_backoff else set(range(n))-{s} for s in range(n)]
    net.broadcast.fill_(1)
    if ack_backoff:
        net.broadcast.zero_()
        for sender in range(n):
            net.receivers[:,:,sender].fill_(1 << ((sender+1)%n))
    seq = [0]*n
    window = [p.window_slots]*n
    released = [0]*n
    pending = [None]*n
    cursor = 0
    rng = random.Random(123)
    for step in range(25):
        expected = []
        admissions = []
        for segment,(offset, stop) in enumerate(zip(offsets, offsets[1:]+(tick,))):
            now, end = step*tick+offset, step*tick+stop
            received = {}
            admitted = []
            for sender in range(n):
                size = rng.choice((0,50,400,1000))
                ident = (step*len(offsets)+segment)*n+sender
                expiry = now+rng.randrange(1000,500000)
                net.size[segment,0,sender] = size
                net.packet_id[segment,0,sender] = ident
                net.expiry_ns[segment,0,sender] = expiry
                available = pending[sender] is None and now >= released[sender]
                admitted.append(0 if not size else 1 if available else 2)
                if size and available:
                    seq[sender] += 1
                    pending[sender] = dict(backoff=draw(sender,seq[sender],p.seed,window[sender]),
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
                    if ack_backoff:
                        window[sender] = 2*p.window_slots if q["collision"] and window[sender]==p.window_slots else p.window_slots
                    released[sender] = q["finish"] + (p.unicast_failed_hold_ns if q["collision"] else p.unicast_success_hold_ns)
                    if not q["collision"] and q["finish"]<=q["expiry"]:
                        received[sender]=[1,q["id"],q["submitted"],q["finish"],0,0]
                    pending[sender]=None
            expected.append([[received.get(s,[0]*6) if rx in peers[s] else [0]*6 for s in range(n)]
                             for rx in range(n)])
            admissions.append(admitted)
        env.step(random=False)
        assert net.evaluator["admission"][:,0].tolist() == admissions
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


@pytest.mark.parametrize("edge_rates", [False, True])
def test_persistent_receiver_across_clock_gaps_and_resets(edge_rates):
    # Two offers per period must observe the same latent state. Advance the CPU
    # probabilities one period at a time across idle gaps, independently of CUDA.
    cfg = NetworkConfig(window_slots=1, offsets_ns=(0, 200000), outage_period_ns=1000000,
                        outage_enter=.125, outage_recover=.25, edge_outage_rates=edge_rates)
    env = Environment(32, 3, control_dt=.001, auto_reset=False, network=cfg)
    env.reset_state[2].fill_(5)
    env.reset()
    net = env.network
    bad = [None] * 32
    last_tick = [None] * 32
    mask = 2**64-1

    def mix(x):
        for shift, factor in ((30, 0xbf58476d1ce4e5b9), (27, 0x94d049bb133111eb)):
            x = ((x ^ (x >> shift))*factor) & mask
        return x ^ (x >> 31)

    env.capture(random=False)
    for tick in range(20):
        if edge_rates:
            net.outage_rates[::2,...,0].fill_(1.0 if tick < 13 else .25)
            net.outage_rates[::2,...,1].fill_(1.0 if tick < 13 else .125)
        source = 1 if 8 <= tick < 12 else 0
        net.sources.fill_(source)
        net.size.zero_()
        net.broadcast.fill_(1)
        if tick not in (2, 3, 6, 8, 9):
            net.size[:, :, source].fill_(32)
        if tick == 14:
            env.selected.zero_()
            env.selected[2] = 1  # receiver in env 0
            env.selected[3] = 1  # sender in env 1
            env.reset()
            net.size[:, :, source].fill_(32)
            net.broadcast.fill_(1)
            bad[:2] = [None, None]
        if tick in (8, 12):
            bad = [None] * 32
        env.step(graph=True, random=False)
        actual = net.inbox[:, :, 2, 0, 0].tolist()
        if tick in (2, 3, 6, 8, 9):
            assert not any(actual[0]) and not any(actual[1])
            continue
        for e in range(32):
            enter, recover = cfg.outage_enter, cfg.outage_recover
            if edge_rates and e%2 == 0:
                enter, recover = (1.0, 1.0) if tick < 13 else (.25, .125)
            p = enter / (enter + recover)
            if bad[e] is not None:
                p = float(bad[e])
                for _ in range(tick-last_tick[e]):
                    p = p*(1-recover) + (1-p)*enter
            key = mix(cfg.seed ^ mix(3*e+source) ^ mix(3*e+2+0x9e3779b97f4a7c15))
            bad[e] = (mix(key ^ mix(tick) ^ 0xd1b54a32d192ed03) >> 40)/2**24 < p
            last_tick[e] = tick
            assert actual[0][e] == actual[1][e] == int(not bad[e]), (tick, e, source)
