"""Production Python binding probe; all allocations occur before warmup."""
import json
import sys
import time
import torch
from flightlib_cuda import Environment, _native


def main():
    E, A, iterations, samples, warmup = map(int, sys.argv[1:6])
    workload = sys.argv[6]
    active_percent, trace = map(int, sys.argv[7:9])
    dt, reset_percent = float(sys.argv[9]), int(sys.argv[10])
    n = E * A
    begin = time.perf_counter()
    active = [int(i % 100 < active_percent) for i in range(n)]
    env = Environment(E, A, control_dt=dt, active=active,
                      bounds=[v for v in (-100, 100, -100, 100, 0, 100) for _ in range(n)])
    terminal = torch.arange(n, device='cuda') % 100 < reset_percent
    env.reset_state[2].copy_(torch.where(terminal, 0., 50.))
    env.actions.view(n, 4).copy_(torch.where(terminal[:, None], -1., 0.).expand(n, 4))
    env.reset()
    captured = workload.endswith('-graph')
    policy = workload.startswith('policy')
    start_capture = time.perf_counter()
    if captured:
        env.capture(random=False)
    capture_ms = (time.perf_counter() - start_capture) * 1000 if captured else 0
    weights = torch.full((12, 4), .001, device='cuda')
    # Aggregate slots per environment using persistent output storage.
    total = torch.empty(E, device='cuda')
    observation = env.outputs['observation'].view(n, 12)
    action = env.actions.view(n, 4)
    reward = env.outputs['reward']
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record(); end.record(); end.synchronize()  # initialize lazy CUDA events outside measured window
    setup_ms = (time.perf_counter() - begin) * 1000

    def step():
        if policy:
            torch.mm(observation, weights, out=action)
        env.step(graph=captured, random=False)
        if policy:
            torch.sum(reward, dim=1, out=total)

    with torch.no_grad():
        for _ in range(warmup):
            step()
        torch.cuda.synchronize()
        before = torch.cuda.memory_stats()
        timings = []
        for _ in range(samples):
            if trace:
                _native.profile_start()
            begin = time.perf_counter()
            start.record()
            submit = time.perf_counter()
            for _ in range(iterations):
                step()
            submitted = time.perf_counter()
            end.record()
            end.synchronize()
            complete = time.perf_counter()
            if trace:
                _native.profile_stop()
            timings.append({'submit_ms': (submitted-submit)*1000, 'wall_ms': (complete-begin)*1000,
                            'gpu_timeline_ms': start.elapsed_time(end)})
        after = torch.cuda.memory_stats()
    expected = torch.tensor(active, device='cuda', dtype=torch.uint8).view(E, A)
    assert torch.equal(env.outputs['valid'], expected)
    assert torch.isfinite(env.outputs['observation']).all() and torch.isfinite(reward).all()
    if policy:
        assert torch.isfinite(total).all()
    keys = ('num_device_alloc', 'num_device_free', 'allocation.all.allocated', 'allocation.all.freed',
            'allocated_bytes.all.current', 'reserved_bytes.all.current')
    delta = {k: after[k]-before[k] for k in keys}
    if any(delta.values()):
        raise RuntimeError(f'Torch allocator changed inside warmed loop: {delta}')
    print(json.dumps({'schema_version': 1, 'validated': True, 'samples': timings, 'setup_ms': setup_ms,
                      'capture_ms': capture_ms, 'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
                      'allocator_delta': delta, 'active_lanes': sum(active), 'arena_bytes': env._native.arena_bytes,
                      'external_input_bytes': 121*n,
                      'policy_workload_version': 2 if policy else None,
                      'scope': 'production binding with persistent tensors; policy is preallocated mm + per-environment reward reduction' if policy else 'production Python environment step'}))


if __name__ == '__main__':
    main()
