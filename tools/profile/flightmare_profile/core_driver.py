"""Production resident core probe; identical physical work to cpu_baseline."""
import json
import sys
import time
import torch
from flightlib_cuda import Core, _native


def main():
    E, A, iterations, samples, warmup = map(int, sys.argv[1:6])
    workload = sys.argv[6]
    active_percent, trace = map(int, sys.argv[7:9])
    dt, reset_percent = float(sys.argv[9]), int(sys.argv[10])
    if reset_percent:
        raise ValueError('core probe has no automatic resets; use reset-percent=0')
    n = E*A
    active = [int(i % 100 < active_percent) for i in range(n)]
    e = Core(E, A, control_dt=dt, active=active)
    e.state_input[2].fill_(5); e.time_input.fill_(2); e.reset()
    e.command_payload.fill_(3); e.set_command()
    commands = '-run' in workload
    graph = workload.endswith('-graph')
    if graph: e.capture(commands=commands)
    ptrs = [t.data_ptr() for t in (e.state, e.time, e.motor_omega, e.motor_thrust, e.status)]
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record(); end.record(); end.synchronize()
    for _ in range(warmup): e.step(commands=commands, graph=graph)
    e.synchronize()
    before = torch.cuda.memory_stats(); timings = []
    for _ in range(samples):
        if trace: _native.profile_start()
        begin = time.perf_counter(); start.record(); submit = time.perf_counter()
        for _ in range(iterations): e.step(commands=commands, graph=graph)
        submitted = time.perf_counter(); end.record(); end.synchronize(); complete = time.perf_counter()
        if trace: _native.profile_stop()
        timings.append(dict(submit_ms=(submitted-submit)*1000, wall_ms=(complete-begin)*1000, gpu_timeline_ms=start.elapsed_time(end)))
    after = torch.cuda.memory_stats()
    assert ptrs == [t.data_ptr() for t in (e.state, e.time, e.motor_omega, e.motor_thrust, e.status)]
    expected = torch.tensor([0 if a else 1 for a in active], device=e.device, dtype=torch.uint32)
    assert torch.equal(e.status, expected) and torch.isfinite(e.state).all()
    # No readback in the measured loop. All active lanes deliberately match.
    snap = e.snapshot(); mask = torch.tensor(active, dtype=torch.bool)
    for key in ('state', 'motor_omega', 'motor_thrust'):
        data = snap[key][:, mask]
        if data.numel(): assert torch.equal(data, data[:, :1].expand_as(data))
    keys = ('num_device_alloc', 'num_device_free', 'allocation.all.allocated', 'allocation.all.freed',
            'allocated_bytes.all.current', 'reserved_bytes.all.current')
    delta = {k: after[k]-before[k] for k in keys}
    assert not any(delta.values()), delta
    first = active.index(1) if any(active) else 0
    telemetry = {k: snap[k][:, first].tolist() for k in ('state', 'motor_omega', 'motor_thrust')}
    telemetry['time'] = snap['time'][first].item()
    print(json.dumps(dict(schema_version=1, validated=True, samples=timings, allocator_delta=delta,
        torch=torch.__version__, torch_cuda=torch.version.cuda, active_lanes=sum(active),
        arena_bytes=e._native.arena_bytes, external_input_bytes=(127+13)*n,
        telemetry=telemetry, core_workload_version=1,
        scope='resident core; default model, initial z=5/time=2, constant rotor force=3; final readback outside timing')))


if __name__ == '__main__': main()
