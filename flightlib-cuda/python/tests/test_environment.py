import gc
import pytest
import torch
from flightlib_cuda import Environment, TaskConfig


@pytest.mark.parametrize('legacy', [False, True])
@pytest.mark.parametrize('random', [False, True])
def test_native_graph_transitions(legacy, random):
    a = Environment(3, 1 if legacy else 11, control_dt=.006125, seed=17, legacy=legacy)
    b = Environment(3, 1 if legacy else 11, control_dt=.006125, seed=17, legacy=legacy)
    for e in (a, b):
        e.reset_state[2, 1::2].fill_(5)
        e.reset_time.fill_(7)
        e.reset()
    before = b.state.clone()
    b.capture(random=random)
    assert torch.equal(before, b.state)  # capture must not advance RNG/state
    for i in range(30):
        a.actions.fill_((i % 7 - 3) * .1)
        b.actions.copy_(a.actions)
        if i == 4:
            a.actions.flatten()[0] = float('nan')
            b.actions.flatten()[0] = float('nan')
        if i == 8:
            for e in (a, b):
                e.selected[::2].zero_()
                e.reset(random=True)
                e.selected.fill_(1)
        a.step(random=random)
        b.step(graph=True, random=random)
        for key in a.outputs:
            assert torch.equal(a.outputs[key], b.outputs[key]), key
        assert torch.equal(a.state, b.state)
        assert torch.equal(a.time, b.time)
        assert torch.equal(a.status, b.status)
    assert a.outputs['observation'].shape == ((3, 12) if legacy else (3, 11, 12))


def test_rejection_recapture_and_snapshot():
    e = Environment(3)
    e.reset_state[2].fill_(5)
    e.reset()
    e.capture(random=False)
    with pytest.raises(ValueError):
        e.step(graph=True, random=True)
    e.configure_task(TaskConfig())
    with pytest.raises(ValueError, match='stale'):
        e.step(graph=True, random=False)
    e.capture(random=False)
    with pytest.raises(ValueError):
        e.configure(.006125)  # different allocated substep count
    e.step(graph=True, random=False)  # rejected setup did not invalidate/poison
    original = e.outputs['observation'].data_ptr()
    snapshot = e.snapshot()
    assert snapshot['observation'].device.type == 'cpu'
    e.step(graph=True, random=False)
    assert e.outputs['observation'].data_ptr() == original
    assert not torch.equal(snapshot['observation'], e.snapshot()['observation'])
    e.configure(.009)  # same schedule capacity, new metadata: recapture required
    with pytest.raises(ValueError, match='stale'):
        e.step(graph=True, random=False)
    e.capture(random=False)
    e.step(graph=True, random=False)


def test_invalid_tensors_and_rebinding():
    e = Environment(3, 11)
    for action in (e.actions.cpu(), e.actions.double(), e.actions.transpose(0, 1),
                   torch.zeros_like(e.actions, requires_grad=True), e.actions[:, :, :3],
                   e.outputs['observation'].view(-1)[:132].view(3, 11, 4)):
        with pytest.raises((ValueError, RuntimeError)):
            e.bind(action, e.selected, e.reset_state, e.reset_time)
    e.reset()
    e.capture()
    e.bind(torch.zeros_like(e.actions), e.selected, e.reset_state, e.reset_time)
    with pytest.raises(RuntimeError, match='capture'):
        e.step(graph=True)
    e.capture()
    e.step(graph=True)
    e.synchronize()
    e.actions.resize_(1000000)  # unsupported storage mutation must be caught
    with pytest.raises(ValueError, match='resized'):
        e.step(graph=True)


def test_stream_handoffs_and_export_lifetime():
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    e = Environment(3, 11)
    saved = torch.empty_like(e.outputs['observation'])
    with torch.cuda.stream(producer):
        e.wait()
        e.reset_state[2].fill_(5)
        e.reset()
        e.actions.fill_(.5)
        e.capture(random=False)
        e.step(graph=True, random=False)
    with e.access(consumer):
        saved.copy_(e.outputs['observation'])
    with torch.cuda.stream(producer):
        e.wait()
        e.actions.fill_(-.5)
        e.step(graph=True, random=False)
    e.wait()
    latest = e.outputs['observation'].clone()
    e.record_usage()
    state = e.state  # DLPack export retains the arena and registered inputs
    e.synchronize()
    del e
    gc.collect()
    assert torch.isfinite(state).all()
    assert torch.isfinite(saved).all()
    assert not torch.equal(saved, latest)


def test_wrong_device_when_available():
    if torch.cuda.device_count() < 2:
        pytest.skip('one GPU; cross-device Torch binding not exercised')
    e = Environment(1)
    with pytest.raises(ValueError):
        e.bind(e.actions.to('cuda:1'), e.selected, e.reset_state, e.reset_time)


def test_gpu_policy_loop_and_external_capture_rejection():
    a, b = Environment(33), Environment(33)
    for e in (a, b):
        e.reset_state[2].fill_(5)
        e.reset()
    b.capture(random=False)
    weights = torch.full((12, 4), .001, device='cuda')
    with torch.no_grad():
        for _ in range(20):
            for e, replay in ((a, False), (b, True)):
                torch.mm(e.outputs['observation'].view(33, 12), weights, out=e.actions.view(33, 4))
                e.step(graph=replay, random=False)
    assert torch.equal(a.state, b.state)
    assert torch.equal(a.outputs['reward'], b.outputs['reward'])
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        with pytest.raises(ValueError, match='capture'):
            b.step(graph=True, random=False, stream=stream)

@pytest.mark.parametrize("graph", [False, True])
def test_ctbr_mix_and_plant_gains(graph):
    from flightlib_cuda import ModelConfig, ctbr_task_config, legacy_task_config
    plant = ModelConfig()
    disabled = ModelConfig()
    disabled.rate_gain = [0, 0, 0]
    env = Environment(5, models=[plant, disabled], model_ids=[0, 0, 0, 1, 0])
    env.configure_task([legacy_task_config(plant.mass)] + [ctbr_task_config(plant)] * 4)
    env.reset_state[2].fill_(5)
    env.reset()
    if graph:
        env.capture(random=False)
    env.actions.zero_()
    env.actions[1:, 0, 1] = torch.tensor([1., 3., 1., -1.], device=env.device)
    for _ in range(5):
        assert env.step(graph=graph, random=False)["valid"].all()
    roll = env.state[10]
    assert roll[1] > 0 and roll[4] < 0
    torch.testing.assert_close(env.state[:, 1], env.state[:, 2])
    torch.testing.assert_close(env.state[:, 0], env.state[:, 3], atol=1e-5, rtol=1e-5)
