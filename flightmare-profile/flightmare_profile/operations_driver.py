"""Explicit nine-helper batch, outside the normal fused stepping path."""
import json
import sys
import time
import torch
from flightlib_cuda import Core, CoreOperation, CommandMode, _native


def main():
    E, A, iterations, samples, warmup = map(int, sys.argv[1:6])
    active_percent, trace = map(int, sys.argv[7:9])
    if int(sys.argv[10]): raise ValueError('helper probe has no automatic resets')
    n = E*A
    active = [int(i % 100 < active_percent) for i in range(n)]
    e = Core(E, A, active=active); e.reset()
    e.command_mode.fill_(CommandMode.COLLECTIVE_BODY_RATES)
    e.command_payload.zero_(); e.command_payload[0].fill_(9.81); e.set_command()
    ops = []
    for kind, nr, no in ((CoreOperation.DERIVATIVE,25,25),(CoreOperation.CONTROLLER,3,4),
                        (CoreOperation.MOTORS,4,8),(CoreOperation.OMEGA_TO_THRUST,4,4),
                        (CoreOperation.THRUST_TO_OMEGA,4,4),(CoreOperation.CLAMP_THRUST,4,4),
                        (CoreOperation.CLAMP_MOTOR_OMEGA,4,4),(CoreOperation.CLAMP_BODY_RATES,3,3),
                        (CoreOperation.CONSTRAIN_WORLD,25,25)):
        values = torch.full((nr,n), 1., device=e.device)
        out = torch.zeros((no,n), device=e.device)
        ops.append(e.bind_operation(kind, values, out))
    def step():
        for op in ops: op.run(dt=.0025)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record(); end.record(); end.synchronize()
    for _ in range(warmup): step()
    e.synchronize(); before = torch.cuda.memory_stats(); timings=[]
    for _ in range(samples):
        if trace: _native.profile_start()
        begin=time.perf_counter(); start.record(); submit=time.perf_counter()
        for _ in range(iterations): step()
        submitted=time.perf_counter(); end.record(); end.synchronize(); complete=time.perf_counter()
        if trace: _native.profile_stop()
        timings.append(dict(submit_ms=(submitted-submit)*1000,wall_ms=(complete-begin)*1000,gpu_timeline_ms=start.elapsed_time(end)))
    after=torch.cuda.memory_stats()
    assert torch.equal(e.status,torch.tensor([0 if a else 1 for a in active],device=e.device,dtype=torch.uint32))
    assert all(torch.isfinite(op.output).all() for op in ops)
    keys=('num_device_alloc','num_device_free','allocation.all.allocated','allocation.all.freed','allocated_bytes.all.current','reserved_bytes.all.current')
    delta={k:after[k]-before[k] for k in keys};assert not any(delta.values())
    print(json.dumps(dict(validated=True,schema_version=1,samples=timings,allocator_delta=delta,
        torch=torch.__version__,torch_cuda=torch.version.cuda,arena_bytes=e._native.arena_bytes,
        external_input_bytes=140*n+sum((op.input.numel()+op.output.numel())*4 for op in ops),
        helper_workload_version=1,scope='nine explicit independent helper calls per iteration; not fused simulation stepping')))


if __name__ == '__main__': main()
