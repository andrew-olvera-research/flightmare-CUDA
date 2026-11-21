import torch
import pytest
from flightlib_cuda import Environment, NetworkConfig


@pytest.mark.parametrize("graph", [False, True])
def test_resident_channel(graph):
    env = Environment(2, 3, network=NetworkConfig(window_slots=1), auto_reset=False)
    stream = torch.cuda.Stream()
    with env.access(stream):
        env.reset_state[2].fill_(5)
    env.reset(stream=stream)
    if graph:
        env.capture(random=False)
    with env.access(stream):
        net = env.network
        net.sources.zero_()
        net.packet_id[:, 0].fill_(73)
        net.size[:, 0].fill_(100)
        net.broadcast[:, 0].fill_(1)
        net.erased[1, 0] = 2
    env.step(graph=graph, random=False, stream=stream)
    with env.access(stream):
        rows = net.inbox.cpu().tolist()
    assert rows[0][0][0] == [0]*4
    assert rows[0][1][0] == [1,73,0,46800]
    assert rows[1][1][0] == [0]*4
    assert rows[1][2][0] == rows[0][1][0]


def test_receiver_reset_during_transmission():
    env = Environment(1, 3, network=NetworkConfig(window_slots=1), auto_reset=False)
    env.reset_state[2].fill_(5)
    env.reset()
    net = env.network
    net.sources.zero_()
    net.packet_id[0,0] = 91
    net.size[0,0] = 100000
    net.broadcast[0,0] = 1
    env.step(random=False)
    env.selected.zero_()
    env.selected[1] = 1
    env.reset()
    net.size.zero_()
    env.step(random=False)
    rows = net.inbox.cpu().tolist()[0]
    assert rows[1][0] == [0]*4
    assert rows[2][0] == [1,91,0,14832000]

def test_full_reset_cancels_pending_airtime():
    env = Environment(1, 2, network=NetworkConfig(window_slots=1), auto_reset=False)
    env.reset_state[2].fill_(5)
    env.reset()
    net = env.network
    net.sources.zero_()
    net.size[0,0] = 100000
    net.broadcast[0,0] = 1
    env.step(random=False)
    env.reset()
    net.size[0,0] = 10
    net.broadcast[0,0] = 1
    env.step(random=False)
    assert net.inbox[0,1,0].tolist() == [1,0,10000000,10033480]
