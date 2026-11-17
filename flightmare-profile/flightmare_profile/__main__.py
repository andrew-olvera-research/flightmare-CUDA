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
import sys


def run(command, cwd, log=None):
    result = subprocess.run(list(map(str, command)), cwd=cwd, capture_output=True, text=True, timeout=900)
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


def launches(workload):
    return {'dispatch': 0, 'mixed': 3, 'environment': 3, 'environment-split': 4, 'environment-random': 3}.get(workload, 1)


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
    calls = result['calls']
    expected_launches = measured * launches(workload)
    if calls['kernel_launches'] != expected_launches:
        raise ValueError(f'Launch count mismatch: {calls["kernel_launches"]} != {expected_launches}')
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
        # Valid here because these workloads execute sequentially on one stream.
        span = last - first
        memory = {}
        for table in ('CUPTI_ACTIVITY_KIND_MEMCPY', 'CUPTI_ACTIVITY_KIND_MEMSET'):
            memory[table] = db.execute(f'select count(*) from {table}').fetchone()[0] if table in tables else 0
        errors = db.execute('select count(*) from CUPTI_ACTIVITY_KIND_RUNTIME where returnValue != 0').fetchone()[0]
        return {'api': api, 'kernels': kernels, 'kernel_count': count, 'cuda_api_errors': errors,
                'memory_events': memory, 'kernel_span_us': span / 1000,
                'kernel_active_us': active / 1000,
                'idle_gap_fraction_between_first_last_kernel': 1 - active / span if span else 0}


def markdown(report):
    lines = ['# Flightmare systems profile', '', f'Status: **{report["status"]}**', '',
             'Native C++ loops orchestrated by Python; dynamics measures physics; environment includes task/reset. Neither includes Python-per-step/policy costs.',
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
                  '| Lanes | CPU us/interval | CPU / GPU wall |', '| ---: | ---: | ---: |']
        for c in sorted(cpu, key=lambda c: c['environments']*c['slots']):
            b = c['cpu_baseline']
            lines.append(f'| {c["environments"]*c["slots"]} | {b["median_us"]:.3f} | {b["sequential_cpu_over_gpu_wall"]:.2f}x |')
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
    p.add_argument('--build-dir', type=Path, default=Path('/build/flightmare-profile'))
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
    p.add_argument('--trace', action='store_true', help='Trace direct/runtime/mixed for smallest/largest cases separately')
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
    if set(args.workloads.split(',')) - {'direct', 'runtime', 'dispatch', 'state', 'command', 'mixed', 'dynamics', 'environment', 'environment-split', 'environment-random'}:
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
            run(['cmake', '-S', repo / 'flightmare-profile/native', '-B', args.build_dir, '-G', 'Ninja',
                 '-DCMAKE_BUILD_TYPE=Release', f'-DCMAKE_CUDA_ARCHITECTURES={arch}', '-DPROFILE_CPU_BASELINE=' + ('ON' if args.cpu_baseline else 'OFF')], repo, directory / 'configure.log')
            run(['cmake', '--build', args.build_dir, '--parallel', '4'], repo, directory / 'build.log')
            binary = args.build_dir / 'profile_driver'
            if args.cpu_baseline:
                run(['env', f'FLIGHTMARE_PATH={repo}', args.build_dir / 'cpu_baseline'], repo, directory / 'cpu-optimized-parity.log')
                report['cpu_binary_sha256'] = hashlib.sha256((args.build_dir / 'cpu_baseline').read_bytes()).hexdigest()
            report['binary_sha256'] = hashlib.sha256(binary.read_bytes()).hexdigest()
            jobs = [(e, a, w) for e, a in cases for w in args.workloads.split(',')]
            random.Random(0).shuffle(jobs)
            extremes = {min(e*a for e, a in cases), max(e*a for e, a in cases)}
            for e, a, workload in jobs:
                name = f'{e}x{a}-{workload}'
                command = [binary, e, a, args.iterations, args.samples, args.warmup, workload, args.active_percent, 0, args.control_dt, args.reset_percent]
                print(f'Measuring {name}', flush=True)
                result = json.loads(run(command, repo, directory / f'{name}.json'))
                summary = summarize(result, args.iterations, args.samples, workload)
                item = {'environments': e, 'slots': a, 'workload': workload, 'native': result, **summary}
                report['cases'].append(item)
                if args.cpu_baseline and workload == 'dynamics':
                    if args.active_percent != 100:
                        raise ValueError('CPU comparison requires all lanes active')
                    cpu_samples = json.loads(run([args.build_dir / 'cpu_baseline', e*a, args.iterations, args.warmup, args.samples], repo,
                        directory / f'{name}-cpu.json'))['samples_us']
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
                if args.trace and e*a in extremes and workload in ('direct', 'runtime', 'mixed', 'dynamics', 'environment', 'environment-split', 'environment-random'):
                    tracefile = directory / name
                    trace_command = [binary, e, a, min(args.iterations, 256), 1, args.warmup, workload, args.active_percent, 1, args.control_dt, args.reset_percent]
                    run(['nsys', 'profile', '--trace=cuda', '--sample=none', '--cpuctxsw=none',
                         '--capture-range=cudaProfilerApi', '--capture-range-end=stop', f'--output={tracefile}',
                         *trace_command], repo, directory / f'{name}-trace.log')
                    run(['nsys', 'export', '--type=sqlite', f'--output={tracefile}.sqlite', f'{tracefile}.nsys-rep'],
                        repo, directory / f'{name}-export.log')
                    item['trace'] = trace_summary(Path(str(tracefile) + '.sqlite'))
                    expected = min(args.iterations, 256) * launches(workload)
                    if item['trace']['kernel_count'] != expected or item['trace']['cuda_api_errors']:
                        raise ValueError('Incomplete or failed CUDA trace')
                    if any(item['trace']['memory_events'].values()):
                        raise ValueError('Unexpected memory activity inside warmed trace window')
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
