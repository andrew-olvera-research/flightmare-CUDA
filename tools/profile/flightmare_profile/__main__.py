import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sqlite3
import statistics
import subprocess
import struct
import sys


def run(command, cwd, log=None, env=None):
    result = subprocess.run(list(map(str, command)), cwd=cwd, capture_output=True, text=True, timeout=900, env=env)
    if log:
        log.write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f'{command[0]} exited {result.returncode}: {result.stderr[-3000:]} (see {log})')
    return result.stdout


def fingerprint(repo):
    paths = run(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], repo).split('\0')
    digest = hashlib.sha256()
    for name in sorted(set(filter(None, paths))):
        path = repo / name
        value = path.read_bytes() if path.is_file() else b'MISSING'
        if b'\0' not in value:
            value = value.replace(b'\r\n', b'\n')
        digest.update(name.encode() + b'\0' + hashlib.sha256(value).digest())
    return digest.hexdigest()


CORE_WORKLOADS = {'core', 'core-graph', 'core-run', 'core-run-graph'}
PYTHON_WORKLOADS = {'python', 'python-graph', 'policy', 'policy-graph', 'helpers', *CORE_WORKLOADS}


def kernel_nodes(workload):
    if workload == 'helpers': return 9
    if workload in CORE_WORKLOADS:
        return 2 if '-run' in workload else 1
    return 3 if workload in PYTHON_WORKLOADS or workload == 'environment-graph' else launches(workload)


def launches(workload):
    return {'dispatch': 0, 'environment-graph': 0, 'mixed': 3, 'environment': 3, 'environment-split': 4, 'environment-random': 3}.get(workload, 1)


def validate_core_telemetry(gpu, cpu):
    """CUDA-7 numerical budgets; no angle-equivalence or relaxed time check."""
    if not math.isfinite(cpu['time']) or gpu['time'] != struct.unpack('f', struct.pack('f', cpu['time']))[0]:
        raise ValueError('Core CPU/GPU clock mismatch')
    for key, count in (('state', 25), ('motor_omega', 4), ('motor_thrust', 4)):
        if len(gpu[key]) != count or len(cpu[key]) != count:
            raise ValueError('Incomplete core telemetry')
        for i, (a, b) in enumerate(zip(gpu[key], cpu[key])):
            tol = .02 if key == 'motor_omega' else 1e-4
            if key == 'state': tol = (3e-4 if i < 16 else 1e-4)*max(1, abs(b))
            if not math.isfinite(a) or not math.isfinite(b) or abs(a-b) > tol:
                raise ValueError(f'Core CPU/GPU {key}[{i}] mismatch: {a} versus {b}')


def performance_gate(report, baseline, limit):
    if baseline.get('status') != 'passed':
        raise ValueError('Performance baseline did not pass')
    def hardware(r):
        rows = r['gpu_before'].strip().splitlines()[1:]
        return [row.split(',')[:4] for row in rows], r['cuda_architectures'], r['cuda'], r['compiler']
    if hardware(report) != hardware(baseline):
        raise ValueError('Performance baseline hardware/toolchain mismatch')
    for key, default in [('iterations', 1000), ('samples', 7), ('warmup', 100), ('active_percent', 100), ('control_dt', .01), ('reset_percent', 0)]:
        if report['config'].get(key, default) != baseline['config'].get(key, default):
            raise ValueError('Performance baseline configuration mismatch: ' + key)
    key = lambda c: (c['environments'], c['slots'], c['workload'])
    old = {key(c): c for c in baseline['cases']}
    ratios = []
    for c in report['cases']:
        if key(c) not in old:
            raise ValueError('Missing workload in performance baseline')
        if c['native'].get('torch') != old[key(c)]['native'].get('torch') or c['native'].get('torch_cuda') != old[key(c)]['native'].get('torch_cuda'):
            raise ValueError('Performance baseline Torch version mismatch')
        if c['native'].get('policy_workload_version') != old[key(c)]['native'].get('policy_workload_version'):
            raise ValueError('Performance baseline policy workload mismatch')
        if c['native'].get('core_workload_version') != old[key(c)]['native'].get('core_workload_version'):
            raise ValueError('Performance baseline core workload mismatch')
        if c['native'].get('helper_workload_version') != old[key(c)]['native'].get('helper_workload_version'):
            raise ValueError('Performance baseline helper workload mismatch')
        prior = old[key(c)]['timing']['wall_us_per_iteration']['median']
        if not math.isfinite(prior) or prior <= 0:
            raise ValueError('Invalid baseline wall timing')
        ratio = c['timing']['wall_us_per_iteration']['median'] / prior
        ratios.append({'case': key(c), 'wall_ratio': ratio})
    report['performance_gate'] = {'limit': limit, 'ratios': ratios, 'passed': all(r['wall_ratio'] <= limit for r in ratios)}
    if not report['performance_gate']['passed']:
        raise ValueError('Wall-time performance gate failed; inspect ratios and rerun under matched load before changing the budget')


def summarize(result, iterations, samples, workload):
    if not result.get('validated') or len(result['samples']) != samples:
        raise ValueError('Missing functional validation or sample data')
    measured = iterations * samples
    calls = result.get('calls', {})
    if workload in PYTHON_WORKLOADS:
        delta = result['allocator_delta']
        if set(delta) != {'num_device_alloc', 'num_device_free', 'allocation.all.allocated', 'allocation.all.freed', 'allocated_bytes.all.current', 'reserved_bytes.all.current'} or any(delta.values()):
            raise ValueError('Python allocator residency invariant violated')
    else:
        expected_launches = measured * launches(workload)
        if calls['kernel_launches'] != expected_launches:
            raise ValueError(f'Launch count mismatch: {calls["kernel_launches"]} != {expected_launches}')
        expected_graphs = measured if workload == 'environment-graph' else 0
        if calls.get('graph_launches', 0) != expected_graphs:
            raise ValueError('Graph replay count mismatch')
        forbidden = {k: calls[k] for k in ('allocations', 'frees', 'copies', 'host_syncs') if calls[k]}
        if forbidden:
            raise ValueError(f'Residency invariant violated: {forbidden}')
    timing = {}
    for key in ('submit_ms', 'wall_ms', 'gpu_timeline_ms'):
        values = sorted(s[key] * 1000 / iterations for s in result['samples'])
        if not all(math.isfinite(v) and v >= 0 for v in values):
            raise ValueError('Invalid timing sample')
        timing[key.replace('_ms', '_us_per_iteration')] = {
            'median': statistics.median(values), 'min': values[0], 'max': values[-1],
            'p95_nearest_rank': values[math.ceil(.95 * len(values)) - 1]}
    return {'timing': timing, 'calls_per_iteration': {k: v / measured for k, v in calls.items()}}


def validate_memory_trace(trace, workload, iterations):
    # cuBLAS uses a zeroed 1-KiB persistent workspace for some skinny matmul
    # shapes. Keep that measured policy cost distinct from simulator transfers.
    policy = workload in ('policy', 'policy-graph')
    events = trace['memory_events']
    if (events['CUPTI_ACTIVITY_KIND_MEMCPY'] or
            events['CUPTI_ACTIVITY_KIND_MEMSET'] > (iterations if policy else 0) or
            trace.get('memset_bytes', 0) > (1024 * iterations if policy else 0) or
            trace.get('memset_nonzero_events', 0)):
        raise ValueError('Unexpected memory activity inside warmed trace window')
    trace['policy_scratch_clear_budget_bytes_per_iteration'] = 1024 if policy else 0


def trace_summary(path):
    with sqlite3.connect(path) as db:
        tables = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
        required = {'CUPTI_ACTIVITY_KIND_RUNTIME', 'CUPTI_ACTIVITY_KIND_KERNEL', 'StringIds'}
        if not required <= tables:
            raise ValueError('Trace lacks required CUDA runtime/kernel data')
        api = [{'name': name, 'count': count, 'total_us': ns / 1000} for name, count, ns in db.execute(
            'select s.value,count(*),sum(r.end-r.start) from CUPTI_ACTIVITY_KIND_RUNTIME r '
            'join StringIds s on s.id=r.nameId group by s.value')]
        kernels = [{'name': name, 'count': count, 'total_us': ns / 1000,
                    'max_registers_per_thread': regs, 'max_local_bytes_per_thread': local}
                   for name, count, ns, regs, local in db.execute(
                       'select s.value,count(*),sum(k.end-k.start),max(k.registersPerThread),'
                       'max(k.localMemoryPerThread) from CUPTI_ACTIVITY_KIND_KERNEL k '
                       'join StringIds s on s.id=k.demangledName group by s.value')]
        count, first, last, active = db.execute(
            'select count(*),min(start),max(end),sum(end-start) from CUPTI_ACTIVITY_KIND_KERNEL').fetchone()
        if not count:
            raise ValueError('No measured kernels in trace')
        # Interval union remains valid for producer/simulation/consumer streams.
        union = 0; interval_end = None
        for begin, end in db.execute('select start,end from CUPTI_ACTIVITY_KIND_KERNEL order by start'):
            union += end - max(begin, interval_end) if interval_end is not None and end > interval_end else end-begin if interval_end is None else 0
            interval_end = max(end, interval_end) if interval_end is not None else end
        span = last - first
        memory = {}
        for table in ('CUPTI_ACTIVITY_KIND_MEMCPY', 'CUPTI_ACTIVITY_KIND_MEMSET'):
            memory[table] = db.execute(f'select count(*) from {table}').fetchone()[0] if table in tables else 0
        memset_bytes, nonzero = db.execute('select coalesce(sum(bytes),0),coalesce(sum(value != 0),0) from CUPTI_ACTIVITY_KIND_MEMSET').fetchone() if 'CUPTI_ACTIVITY_KIND_MEMSET' in tables else (0, 0)
        errors = db.execute('select count(*) from CUPTI_ACTIVITY_KIND_RUNTIME where returnValue != 0').fetchone()[0]
        return {'api': api, 'kernels': kernels, 'kernel_count': count, 'cuda_api_errors': errors,
                'memory_events': memory, 'memset_bytes': memset_bytes, 'memset_nonzero_events': nonzero, 'kernel_span_us': span / 1000,
                'kernel_active_us': active / 1000, 'kernel_union_us': union / 1000,
                'idle_gap_fraction_between_first_last_kernel': 1 - union / span if span else 0}


def markdown(report):
    lines = ['# Flightmare systems profile', '', f'Status: **{report["status"]}**', '',
             'Native workloads use C++ loops; python workloads measure the production binding; policy workloads add preallocated GPU matmul/reduction.',
             'GPU timeline includes idle/queue gaps; it is not kernel execution time.', '',
             '| E x A | Workload | Submit us/iteration | Wall us/iteration | GPU timeline us/iteration | Arena MiB |',
             '| --- | --- | ---: | ---: | ---: | ---: |']
    for c in sorted(report['cases'], key=lambda x: (x['environments'] * x['slots'], x['workload'])):
        t = c['timing']
        values = [t[k]['median'] for k in ('submit_us_per_iteration', 'wall_us_per_iteration', 'gpu_timeline_us_per_iteration')]
        lines.append(f'| {c["environments"]} x {c["slots"]} | {c["workload"]} | ' +
                     ' | '.join(f'{v:.3f}' for v in values) + f' | {c["native"]["arena_bytes"] / 2**20:.3f} |')
    lines += ['', 'Raw samples, call counts, source identity, and separate trace metrics are in report.json.',
              'Trace timing is instrumented; do not compare it directly with the untraced timing samples.',
              'Memory figures exclude driver/context overhead. External inputs are reported separately.']
    cpu = [c for c in report['cases'] if 'cpu_baseline' in c]
    if cpu:
        lines += ['', 'Sequential O3 strict-FP CPU comparison (same interval counts; no task/policy):', '',
                  '| Lanes | Workload | CPU us/interval | CPU / GPU wall |', '| ---: | --- | ---: | ---: |']
        for c in sorted(cpu, key=lambda c: c['environments']*c['slots']):
            b = c['cpu_baseline']
            lines.append(f'| {c["environments"]*c["slots"]} | {c["workload"]} | {b["median_us"]:.3f} | {b["sequential_cpu_over_gpu_wall"]:.2f}x |')
    for c in report['cases']:
        if 'compute' in c:
            lines += ['', f'Nsight Compute {c["environments"]}x{c["slots"]}: {c["compute"]["status"]}; see compute logs.']
    if 'performance_gate' in report:
        lines += ['', f'Wall-time gate: {"passed" if report["performance_gate"]["passed"] else "FAILED"}; limit {report["performance_gate"]["limit"]:.2f}x baseline.']
    if 'error' in report:
        lines += ['', 'Failure: ' + report['error']]
    return '\n'.join(lines) + '\n'


def main():
    p = argparse.ArgumentParser(description='Local runtime and dynamics systems measurements.')
    p.add_argument('--repo', type=Path, default=Path.cwd())
    p.add_argument('--build-dir', type=Path, default=Path('/build/v3-profile'))
    p.add_argument('--python-interpreter', type=Path, default=Path('/build/flightlib-cuda-python/bin/python'))
    p.add_argument('--python-build-dir', type=Path, default=Path('/build/flightlib-cuda-python-native'))
    p.add_argument('--cases', default='1x1,3x11,1024x1,1024x16,4096x16')
    p.add_argument('--workloads', default='dispatch,direct,runtime,state,command,mixed,dynamics,environment')
    p.add_argument('--iterations', type=int, default=1000)
    p.add_argument('--samples', type=int, default=7)
    p.add_argument('--warmup', type=int, default=100)
    p.add_argument('--reset-percent', type=int, default=0, help='Initial terminal-lane fraction for environment reset-density studies')
    p.add_argument('--baseline', type=Path, help='Prior report.json for a matched hardware/workload wall-time gate')
    p.add_argument('--max-wall-ratio', type=float, default=1.15)
    p.add_argument('--active-percent', type=int, default=100)
    p.add_argument('--compute', action='store_true', help='Attempt separate Nsight Compute capture; report unavailable counters explicitly')
    p.add_argument('--control-dt', type=float, default=.01)
    p.add_argument('--cpu-baseline', action='store_true', help='Compare dynamics to sequential O3 strict-FP CPU build')
    p.add_argument('--trace', action='store_true', help='Trace supported workloads at smallest/largest cases separately')
    p.add_argument('--trace-all', action='store_true', help='Trace every supported workload and shape, including intermediate library algorithm choices')
    args = p.parse_args()
    repo = args.repo.resolve()
    if not math.isfinite(args.control_dt) or args.control_dt < 0 or (args.cpu_baseline and args.control_dt != .01):
        p.error('Invalid control_dt; CPU comparison currently requires .01')
    if not 0 <= args.reset_percent <= 100 or not math.isfinite(args.max_wall_ratio) or args.max_wall_ratio < 1:
        p.error('Invalid reset percent or wall ratio')
    if args.iterations < 1 or args.samples < 1 or args.warmup < 0 or not 0 <= args.active_percent <= 100:
        p.error('Invalid measurement dimensions')
    cases = [tuple(map(int, case.split('x'))) for case in args.cases.split(',')]
    if any(len(c) != 2 or min(c) <= 0 for c in cases):
        p.error('Cases must be positive ExA pairs')
    if any(e*a >= 2**32-1 for e, a in cases):
        p.error('Case exceeds runtime lane capacity')
    if args.warmup + args.iterations * args.samples >= 2**24:
        p.error('Too many touch increments for exact float32 functional validation')
    if set(args.workloads.split(',')) - {'direct', 'runtime', 'dispatch', 'state', 'command', 'mixed', 'dynamics', 'environment', 'environment-split', 'environment-random', 'environment-graph', *PYTHON_WORKLOADS}:
        p.error('Unknown workload')
    directory = repo / '.local-ci' / ('profile-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ'))
    directory.mkdir(parents=True)
    report = {'schema_version': 1, 'status': 'failed', 'config': vars(args).copy(), 'cases': []}
    report['config'] = {k: str(v) if isinstance(v, Path) else v for k, v in report['config'].items()}
    try:
        # Reuse the local build-volume lock to avoid profiling during verification.
        import fcntl
        with open('/build/flightmare-local-ci.lock', 'w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            report['source_fingerprint'] = fingerprint(repo)
            report['commit'] = run(['git', 'rev-parse', 'HEAD'], repo).strip()
            report['dirty'] = run(['git', '-c', 'core.filemode=false', '-c', 'core.autocrlf=true', 'status', '--short'], repo)
            report['cpu'] = {'logical_processors': os.cpu_count(), 'affinity': sorted(os.sched_getaffinity(0)),
                             'models': sorted({line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')})}
            report['version'] = (repo / 'VERSION').read_text().strip()
            for name, cmd in [('cuda', ['nvcc', '--version']), ('compiler', ['c++', '--version']),
                              ('gpu_before', ['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total,temperature.gpu,clocks.sm,power.draw', '--format=csv'])]:
                report[name] = run(cmd, repo)
            arch = os.environ.get('FLIGHTMARE_CUDA_ARCHITECTURES', '120-real')
            report['cuda_architectures'] = arch
            run(['cmake', '-S', repo / 'tools/profile/native', '-B', args.build_dir, '-G', 'Ninja',
                 '-DCMAKE_BUILD_TYPE=Release', f'-DCMAKE_CUDA_ARCHITECTURES={arch}', '-DPROFILE_CPU_BASELINE=' + ('ON' if args.cpu_baseline else 'OFF')], repo, directory / 'configure.log')
            run(['cmake', '--build', args.build_dir, '--parallel', '4'], repo, directory / 'build.log')
            binary = args.build_dir / 'profile_driver'
            if args.cpu_baseline:
                run(['env', f'FLIGHTMARE_PATH={repo}', args.build_dir / 'cpu_baseline'], repo, directory / 'cpu-optimized-parity.log')
                run(['env', f'FLIGHTMARE_PATH={repo}', args.build_dir / 'cpu_environment_parity'], repo, directory / 'cpu-optimized-environment-parity.log')
                report['cpu_binary_sha256'] = hashlib.sha256((args.build_dir / 'cpu_baseline').read_bytes()).hexdigest()
            report['binary_sha256'] = hashlib.sha256(binary.read_bytes()).hexdigest()
            python_work = bool(set(args.workloads.split(',')) & PYTHON_WORKLOADS)
            probe_env = dict(os.environ, PYTHONPATH=str(args.python_build_dir / 'python'))
            if python_work:
                run(['cmake', '-S', repo / 'flightlib-cuda', '-B', args.python_build_dir, '-G', 'Ninja',
                     '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_TESTING=OFF', '-DFLIGHTLIB_CUDA_BUILD_RUNTIME=ON',
                     '-DFLIGHTLIB_CUDA_BUILD_PYTHON=ON', f'-DPython_EXECUTABLE={args.python_interpreter}',
                     f'-DCMAKE_CUDA_ARCHITECTURES={arch}'], repo, directory / 'python-configure.log')
                run(['cmake', '--build', args.python_build_dir, '--parallel', '4'], repo, directory / 'python-build.log')
                extension = next((args.python_build_dir / 'python/flightlib_cuda').glob('_native*.so'))
                report['python_extension_sha256'] = hashlib.sha256(extension.read_bytes()).hexdigest()
            def probe(e, a, iterations, samples, workload, trace):
                driver = 'operations_driver.py' if workload == 'helpers' else 'core_driver.py' if workload in CORE_WORKLOADS else 'python_driver.py'
                prefix = [args.python_interpreter, repo / ('tools/profile/flightmare_profile/' + driver)] if workload in PYTHON_WORKLOADS else [binary]
                return [*prefix, e, a, iterations, samples, args.warmup, workload, args.active_percent, trace, args.control_dt, args.reset_percent]
            jobs = [(e, a, w) for e, a in cases for w in args.workloads.split(',')]
            random.Random(0).shuffle(jobs)
            extremes = {min(e*a for e, a in cases), max(e*a for e, a in cases)}
            for e, a, workload in jobs:
                name = f'{e}x{a}-{workload}'
                command = probe(e, a, args.iterations, args.samples, workload, 0)
                print(f'Measuring {name}', flush=True)
                result = json.loads(run(command, repo, directory / f'{name}.json', env=probe_env))
                summary = summarize(result, args.iterations, args.samples, workload)
                item = {'environments': e, 'slots': a, 'workload': workload, 'native': result, **summary}
                report['cases'].append(item)
                if args.cpu_baseline and workload in {'dynamics', *CORE_WORKLOADS}:
                    if args.active_percent != 100:
                        raise ValueError('CPU comparison requires all lanes active')
                    cpu_result = json.loads(run([args.build_dir / 'cpu_baseline', e*a, args.iterations, args.warmup, args.samples], repo,
                        directory / f'{name}-cpu.json'))
                    cpu_samples = cpu_result['samples_us']
                    if workload in CORE_WORKLOADS:
                        validate_core_telemetry(result['telemetry'], cpu_result)
                        item['cpu_behavior_validated'] = True
                    item['cpu_baseline'] = {'samples_us': cpu_samples, 'median_us': statistics.median(cpu_samples),
                        'sequential_cpu_over_gpu_wall': statistics.median(cpu_samples) / summary['timing']['wall_us_per_iteration']['median'],
                        'scope': 'Sequential O3 strict-FP CPU (frozen source, separate build); identical initial state, constant rotor commands, dt=.01, warmup and interval counts. No policy/task. Not an optimized parallel CPU comparison.'}
                if args.compute and workload == 'dynamics' and e*a == max(extremes):
                    try:
                        run(['ncu', '--profile-from-start', 'off', '--set', 'full', '--launch-count', '1', '--csv',
                             '--log-file', directory / f'{name}-compute.csv', binary, e, a, 1, 1, args.warmup,
                             workload, args.active_percent, 1, args.control_dt, args.reset_percent], repo, directory / f'{name}-compute.log')
                        item['compute'] = {'status': 'captured', 'metrics_file': f'{name}-compute.csv'}
                    except (RuntimeError, FileNotFoundError) as error:
                        item['compute'] = {'status': 'unavailable', 'reason': str(error), 'metrics_file': f'{name}-compute.csv'}
                if (args.trace or args.trace_all) and (args.trace_all or e*a in extremes) and workload in ('direct', 'runtime', 'mixed', 'dynamics', 'environment', 'environment-split', 'environment-random', 'environment-graph', *PYTHON_WORKLOADS):
                    tracefile = directory / name
                    trace_command = probe(e, a, min(args.iterations, 256), 1, workload, 1)
                    run(['nsys', 'profile', '--trace=cuda', '--cuda-graph-trace=node', '--sample=none', '--cpuctxsw=none',
                         '--capture-range=cudaProfilerApi', '--capture-range-end=stop', f'--output={tracefile}',
                         *trace_command], repo, directory / f'{name}-trace.log', env=probe_env)
                    run(['nsys', 'export', '--type=sqlite', f'--output={tracefile}.sqlite', f'{tracefile}.nsys-rep'],
                        repo, directory / f'{name}-export.log')
                    item['trace'] = trace_summary(Path(str(tracefile) + '.sqlite'))
                    expected = min(args.iterations, 256) * kernel_nodes(workload)
                    observed = item['trace']['kernel_count']
                    if workload in PYTHON_WORKLOADS:
                        observed = sum(k['count'] for k in item['trace']['kernels'] if any(n in k['name'] for n in ('prepareActions', 'dynamicsKernel', 'taskKernel', 'commandKernel', 'operationKernel')))
                        forbidden_api = [a for a in item['trace']['api'] if a['name'].startswith(('cudaMalloc', 'cudaFree', 'cudaMemcpy', 'cudaDeviceSynchronize', 'cudaStreamSynchronize'))]
                        syncs = sum(a['count'] for a in item['trace']['api'] if a['name'].startswith('cudaEventSynchronize'))
                        if forbidden_api or syncs != 1:
                            raise ValueError(f'Python traced residency/synchronization failure: {forbidden_api}, event syncs={syncs}')
                    graph_calls = sum(a['count'] for a in item['trace']['api'] if a['name'].startswith('cudaGraphLaunch'))
                    if graph_calls != (min(args.iterations, 256) if workload.endswith('-graph') else 0):
                        raise ValueError('Traced graph launch count mismatch')
                    if observed != expected or item['trace']['cuda_api_errors']:
                        raise ValueError('Incomplete or failed CUDA trace')
                    validate_memory_trace(item['trace'], workload, min(args.iterations, 256))
            report['gpu_after'] = run(['nvidia-smi', '--query-gpu=temperature.gpu,clocks.sm,power.draw', '--format=csv'], repo)
            if fingerprint(repo) != report['source_fingerprint']:
                raise ValueError('Sources changed during measurement; results invalidated')
            if args.baseline:
                report['baseline_sha256'] = hashlib.sha256(args.baseline.read_bytes()).hexdigest()
                performance_gate(report, json.loads(args.baseline.read_text()), args.max_wall_ratio)
            report['status'] = 'passed'
    except Exception as error:
        report['error'] = str(error)
        print(str(error), file=sys.stderr)
    finally:
        (directory / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        (directory / 'summary.md').write_text(markdown(report))
        print(f'{report["status"].upper()}: {directory / "report.json"}')
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    sys.exit(main())
