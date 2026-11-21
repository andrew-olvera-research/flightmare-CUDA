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
    assert rows[0][0][0] == [0]*6
    assert rows[0][1][0] == [1,73,0,46800,0,0]
    assert rows[1][1][0] == [0]*6
    assert rows[1][2][0] == rows[0][1][0]


def test_broadcast_payload_lifetime():
    env = Environment(2, 3, network=NetworkConfig(window_slots=1), auto_reset=False)
    env.reset_state[2].fill_(5)
    env.reset()
    net = env.network
    net.sources.zero_()
    net.size[:, 0] = 100000
    net.handle[:, 0] = torch.tensor([1, 2], device=env.device)
    net.broadcast[:, 0] = 1
    payload = torch.tensor([0, 101, 202, 303], device=env.device)
    env.step(random=False)
    assert net.evaluator["admission"][:, 0].tolist() == [1, 1]
    assert net.evaluator["completed"][:, 0].tolist() == [0, 0]

    # Cancel sender 0 in env 0 and reset only receiver 1 in env 1.
    env.selected.zero_()
    env.selected[0] = env.selected[4] = 1
    env.reset()
    net.size.zero_()
    net.size[1, 0] = 100
    net.handle[1, 0] = 3     # rejected while handle 2 is still in flight
    env.step(random=False)
    assert net.evaluator["admission"][1, 0].item() == 2
    assert net.evaluator["completed"][:, 0].tolist() == [0, 1]
    assert net.evaluator["completed_handle"][1, 0].item() == 2
    assert net.inbox[..., 0].tolist() == [[[0], [0], [0]], [[0], [0], [1]]]
    consumer = torch.cuda.Stream()
    consumed = torch.cuda.Event()
    with env.access(consumer):
        handle = net.inbox[1, 2, 0, 5].long().reshape(1)
        observed = payload.index_select(0, handle)
        consumed.record()
    # The owner may recycle this slot after all queued receiver reads finish.
    with env.access():
        torch.cuda.current_stream().wait_event(consumed)
        payload[2] = 404
        net.size.zero_()
        net.size[0, 0] = 100
        net.broadcast[0, 0] = 1
        net.handle[0, 0] = 2
        net.expiry_ns[0, 0] = 20000001
    env.step(random=False)
    assert observed.item() == 202 and payload[2].item() == 404
    assert net.evaluator["admission"][0, 0].item() == 1
    assert net.evaluator["expired"][0, 0].item() == 1
    assert net.evaluator["completed_handle"][0, 0].item() == 2
    assert net.inbox[..., 0].count_nonzero().item() == 0


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
    assert net.inbox[0,1,0].tolist() == [1,0,10000000,10033480,0,0]

def test_timestamped_offers_with_expiry_and_opaque_handle():
    config = NetworkConfig(window_slots=1, offsets_ns=(0, 1_000_000, 2_000_000))
    env = Environment(1, 2, network=config, auto_reset=False)
    env.reset_state[2].fill_(5)
    env.reset()
    net = env.network
    net.sources.zero_()
    net.size[:,0,0].fill_(100)
    net.broadcast[:,0,0].fill_(1)
    net.packet_id[:,0,0].fill_(10)
    net.attempt_id[:,0,0].copy_(torch.tensor([1,2,3], device=env.device))
    net.handle[:,0,0].fill_(42)
    net.fixture_label[2,0,0] = 7
    net.expiry_ns[1,0,0] = 1_000_100
    env.capture(random=False)
    env.step(graph=True, random=False)
    assert net.evaluator["completed_handle"][:,0,0].tolist() == [42,42,42]
    assert net.evaluator["fixture_label"][:,0,0].tolist() == [0,0,7]
    assert net.evaluator["expired"][:,0,0].tolist() == [0,1,0]
    rows = net.inbox[:,0,1,0].tolist()
    assert rows == [[1,10,0,46800,1,42], [0]*6, [1,10,2000000,2046800,3,42]]

def test_directional_distance_and_loss_memory():
    env = Environment(1,3,network=NetworkConfig(window_slots=1, range_edges_m=(5.0,)),
                      auto_reset=False)
    env.reset_state[2].fill_(5)
    env.reset_state[0,2] = 10
    env.reset()
    net = env.network
    net.sources.zero_()
    net.broadcast[0,0] = 1
    net.size[0,0] = 100
    net.loss_after_rx[:,:,:,1].fill_(1)
    net.loss_after_miss.zero_()
    env.capture(random=False)
    for far in (0,1,0,1):
        env.step(graph=True,random=False)
        assert net.inbox[0,:,0,0].tolist() == [0,1,far]

def test_hidden_sender_overlap_survives_tick_boundary():
    env=Environment(1,4,network=NetworkConfig(window_slots=1, carrier_group_size=2, hidden_per_edge=1),
                    auto_reset=False)
    env.reset_state[2].fill_(5)
    env.reset()
    net=env.network
    net.sources.zero_()
    net.hidden_sources[0,1,0,0]=2
    net.size[0,0]=100000
    net.size[0,2]=100
    net.broadcast.fill_(1)
    env.step(random=False)
    net.size.zero_()
    env.step(random=False)
    assert net.inbox[0,:,0,0].tolist()==[0,0,1,1]

def test_versioned_network_profile():
    from pathlib import Path
    profile = Path(__file__).resolve().parents[3]/"tools/transport/shared_basic_v1.json"
    config = NetworkConfig.from_profile(profile, fanin=2)
    assert config.ns_per_byte == 148 and config.fanin == 2
