import gc
import json
import os
from pathlib import Path
import subprocess
import pytest
import torch
from flightlib_cuda import Core, ModelConfig, CommandMode, LaneStatus, CoreOperation, load_model

ROOT = Path(__file__).resolve().parents[3]


def test_frozen_cpu_traces_through_public_api():
    trace = json.loads(subprocess.check_output(
        ['/build/flightmare-parity-tests/flightmare_oracle_trace'],
        env=dict(os.environ, FLIGHTMARE_PATH=str(ROOT)), text=True))
    # Constant-schedule scenarios replay end to end. The variable-capacity motor
    # scenario remains in native parity; resizing is not a frozen-runtime API.
    for scenario in trace['physics']:
        if scenario['name'] == 'motor_transient':
            continue
        model = load_model(str(ROOT/'flightlib/configs/quadrotor_env.yaml')) if 'yaml' in scenario['dynamics'] else ModelConfig()
        n = 6
        e = Core(2, 3, control_dt=scenario['steps'][0]['input']['dt'], models=[model],
                 bounds=[v for pair in scenario['world_box'] for v in pair for _ in range(n)])
        e.state_input.copy_(torch.tensor(scenario['initial_state'], device=e.device)[:, None])
        e.time_input.fill_(scenario['initial_time']); e.reset()
        e.capture(commands=True)
        for i, sample in enumerate(scenario['steps']):
            command = sample['input']
            rates = command['mode'] != 'rotor_thrust'
            e.command_mode.fill_(CommandMode.COLLECTIVE_BODY_RATES if rates else CommandMode.ROTOR_THRUST)
            payload = [command['collective_thrust'], *command['body_rates']] if rates else command['thrust']
            e.command_payload.copy_(torch.tensor(payload, device=e.device)[:, None])
            e.command_time.fill_(command['command_time'])
            e.step(commands=True, graph=i % 2 == 0)
            expected = torch.tensor(sample['state'], device=e.device)[:, None].expand(25, n)
            error = (e.state-expected).abs() / expected.abs().clamp_min(1)
            # CUDA-7's documented long-rollout budget includes acceleration.
            assert (error[:16] <= 3e-4).all(), (scenario['name'], i, error.max().item())
            assert (error[16:] <= 1e-4).all(), scenario['name']
            for field, atol in [('motor_omega', .02), ('motor_thrust', 1e-4)]:
                target = torch.tensor(sample[field], device=e.device)[:, None].expand(4, n)
                torch.testing.assert_close(getattr(e, field), target, rtol=0, atol=atol)
            assert (e.time == sample['time']).all()
            assert (e.status == LaneStatus.OK).all()


@pytest.mark.parametrize('graph', [False, True])
def test_partial_lifecycle_rejection_and_independent_models(graph):
    normal = ModelConfig(); normal.motor_tau = .0001
    m = ModelConfig(); m.mass = 1.7; m.motor_tau = .0001
    e = Core(3, 11, models=[normal, m], model_ids=[i % 2 for i in range(33)], active=[1]*32+[0])
    e.state_input[2].fill_(5); e.time_input.fill_(2); e.reset()
    e.command_payload.fill_(3); e.set_command()
    e.capture(commands=True)
    before = e.state.clone(); motors = e.motor_omega.clone()
    e.command_payload[0, 0] = float('nan')
    e.command_selected[3] = 0; e.command_payload[:, 3] = float('nan')
    e.step(commands=True, graph=graph)
    assert torch.equal(e.state[:, 0], before[:, 0])
    assert torch.equal(e.motor_omega[:, 0], motors[:, 0])
    assert e.time[0] == 2 and e.status[0] == LaneStatus.INVALID_COMMAND
    assert e.time[1] > 2 and e.status[32] == LaneStatus.INACTIVE
    assert e.time[3] > 2 and e.status[3] == LaneStatus.OK  # unselected proposal ignored
    assert e.acceleration[2, 2] > e.acceleration[2, 1]  # same thrust, different mass
    assert (e.accepted_command['payload'][:, 0] == 3).all()
    e.step()  # previous accepted command still usable after rejected run
    assert e.time[0] > 2
    motors = e.motor_omega.clone(); before = e.state.clone()
    e.state_selected.zero_(); e.state_selected[1] = 1; e.state_selected[2] = 1
    e.state_input[2, 1] = 7; e.state_input[0, 2] = float('nan')
    e.set_state()
    assert e.state[2, 1] == 7 and e.time[1] == 2
    assert torch.equal(e.state[:, 2], before[:, 2])
    assert e.status[2] == LaneStatus.INVALID_STATE
    assert torch.equal(e.motor_omega, motors)
    e.reset()
    assert (e.motor_omega[:, 1] == 0).all()
    assert torch.equal(e.motor_omega[:, 2], motors[:, 2])
    assert e.status[2] == LaneStatus.INVALID_RESET
    assert (e.accepted_command['payload'][:, :32] == 3).all()
    assert (e.accepted_command['payload'][:, 32] == 0).all()


def test_setup_transaction_graphs_and_snapshots():
    e = Core(2, 3); e.state_input[2].fill_(5); e.reset()
    e.size[0, 1] = .7; e.collision[2] = True
    e.command_payload.fill_(3); e.set_command(); e.capture()
    before = e.snapshot()
    bad = ModelConfig(); bad.mass = 0
    for kwargs in ({'models': [bad]}, {'model_ids': [2]*6}, {'control_dt': .02}, {'bounds': [0]*36}):
        with pytest.raises(ValueError): e.configure(**kwargs)
        assert e.models[0].mass == 1 and e.control_dt == .01
    e.step(graph=True)
    speeds = e.motor_omega.clone(); state = e.state.clone(); command = e.accepted_command['payload'].clone()
    model = e.models[0]; model.mass = 1.5
    assert e.models[0].mass == 1  # returned host configs are copies
    e.configure(models=[model])
    assert torch.equal(e.state, state) and torch.equal(e.motor_omega, speeds)
    assert torch.equal(e.accepted_command['payload'], command)
    assert e.size[0, 1] == .7 and e.collision[2]
    assert e.parameters['mass'][0] == 1.5
    with pytest.raises(ValueError, match='stale'): e.step(graph=True)
    e.capture(); e.step(graph=True)
    assert before['state'].device.type == 'cpu'
    assert not torch.equal(before['state'], e.snapshot()['state'])
    e.reset()
    assert e.size[0, 1] == .7 and e.collision[2]
    # Config-only dt=0 remains a supported no-integration core operation.
    zero = Core(1, control_dt=0); zero.command_payload.fill_(3); zero.step(commands=True)
    assert zero.substep_dt is None and zero.time[0] == 0 and zero.status[0] == LaneStatus.OK
    dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        fixed = Core(1)
        assert fixed.state_input.dtype == torch.float32 and fixed.command_payload.dtype == torch.float32
    finally:
        torch.set_default_dtype(dtype)


def test_bindings_streams_lifetimes_and_graph_modes():
    e = Core(3, 11)
    for payload in (e.command_payload.cpu(), e.command_payload.double(), e.command_payload.t(),
                    e.motor_thrust, torch.zeros_like(e.command_payload, requires_grad=True)):
        with pytest.raises((ValueError, RuntimeError)):
            e.bind_command(e.command_selected, e.command_mode, e.command_time, payload)
    with pytest.raises(ValueError): e.bind_state(e.state_selected, e.state_input, e.state_input[0])
    e.capture(commands=True)
    with pytest.raises(ValueError): e.step(graph=True)  # must match captured workload
    e.bind_command(e.command_selected, e.command_mode, e.command_time, torch.full_like(e.command_payload, 3))
    with pytest.raises(RuntimeError, match='capture'): e.step(commands=True, graph=True)
    p, c = torch.cuda.Stream(), torch.cuda.Stream()
    saved = torch.empty_like(e.state)
    with torch.cuda.stream(p):
        e.wait(); e.state_input[2].fill_(5); e.reset(); e.capture(commands=True)
        e.step(commands=True, graph=True)
    with e.access(c): saved.copy_(e.state)
    with torch.cuda.stream(p): e.step(commands=True, graph=True)
    e.wait(); assert not torch.equal(saved, e.state)
    e.record_usage(); e.synchronize()
    state = e.state
    del e; gc.collect()
    assert torch.isfinite(state).all() and torch.isfinite(saved).all()
    x = Core(1); x.command_payload.resize_(1000000)
    with pytest.raises(ValueError, match='resized'): x.step(commands=True)


def test_external_capture_and_wrong_device():
    e = Core(1)
    graph = torch.cuda.CUDAGraph(); stream = torch.cuda.Stream()
    with torch.cuda.graph(graph, stream=stream):
        with pytest.raises(ValueError, match='capture'): e.step(stream=stream)
    if torch.cuda.device_count() > 1:
        with pytest.raises(ValueError): e.bind_state(e.state_selected.to('cuda:1'), e.state_input, e.time_input)


def test_optional_helpers_bound_storage_and_motor_side_effects():
    e = Core(2, 3); x = e.state_input.clone(); out = torch.full_like(x, -99)
    x[7:10].fill_(2); x[13:16].fill_(-3)
    derivative = e.bind_operation(CoreOperation.DERIVATIVE, x, out)
    derivative.run()
    assert (out[:3] == 2).all() and (out[7:10] == -3).all() and (out[13:] == 0).all()
    with pytest.raises(ValueError): e.bind_operation(CoreOperation.DERIVATIVE, x, x)
    with pytest.raises(ValueError): e.bind_operation(CoreOperation.DERIVATIVE, x, e.state)
    forces, telemetry = torch.full((4, 6), 3., device=e.device), torch.empty((8, 6), device=e.device)
    motor = e.bind_operation(CoreOperation.MOTORS, forces, telemetry)
    before = e.state.clone(); clock = e.time.clone()
    motor.run(.01)
    assert torch.equal(e.state, before) and torch.equal(e.time, clock)
    assert torch.equal(e.motor_omega, telemetry[:4]) and torch.equal(e.motor_thrust, telemetry[4:])
    assert (e.motor_omega > 0).all()
    with pytest.raises(ValueError): motor.run(-1)
    x.resize_(1000000)
    with pytest.raises(ValueError, match='resized'): derivative.run()
