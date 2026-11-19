"""Controlled workload sensitivity and kernel attribution, without kernel variants."""
import argparse
import datetime
import hashlib
import json
import math
import random
import subprocess
import sys
from pathlib import Path
from .__main__ import fingerprint
from .kernel_resources import inspect


def attribute(report):
    if report.get('status') != 'passed':
        raise ValueError('Cannot diagnose a failed measurement')
    rows = []
    for case in report['cases']:
        trace = case.get('trace')
        if not trace or trace['cuda_api_errors'] or not trace['kernels']:
            raise ValueError('A complete successful trace is required')
        count = report['config']['iterations']
        expected = min(count, 256)
        kernels = trace['kernels']
        if len(kernels) != 3 or any(k['count'] != expected for k in kernels):
            raise ValueError('Expected three task kernels on every traced interval')
        total = sum(k['total_us'] for k in kernels)
        if not math.isfinite(total) or total <= 0:
            raise ValueError('Invalid kernel timing')
        stages = {}
        for token, name in [('prepareActions','prepare'), ('dynamicsKernel','dynamics'), ('taskKernel','task_reset')]:
            matched = [k for k in kernels if token in k['name']]
            if len(matched) != 1: raise ValueError('Ambiguous kernel attribution')
            k = matched[0]
            stages[name] = dict(us=k['total_us']/expected, share=k['total_us']/total,
                                registers=k['max_registers_per_thread'],
                                ideal_total_speedup_if_halved=1/(1-.5*k['total_us']/total))
        n = case['environments']*case['slots']
        wall = case['timing']['wall_us_per_iteration']
        rows.append(dict(lanes=n, dt=report['config']['control_dt'],
                         initial_reset_percent=report['config']['reset_percent'],
                         wall_us=wall['median'], wall_min_us=wall['min'], wall_max_us=wall['max'],
                         million_agent_steps_s=n/wall['median'], stages=stages,
                         kernel_sum_us=total/expected,
                         trace_span_us=trace['kernel_span_us']/expected,
                         nonkernel_span_fraction=trace['idle_gap_fraction_between_first_last_kernel'],
                         observed_resets_including_warmup=case['native']['auto_resets_including_warmup'],
                         final_done_lanes=case['native']['final_done_lanes'],
                         trace_scope='Independent instrumented window; do not subtract from untraced wall median'))
    return rows


def stability(rows):
    first = {r['lanes']:r for r in rows if r['run']==0}
    repeats = [r for r in rows if r['anchor_repeat']]
    if {r['lanes'] for r in repeats} != set(first):
        raise ValueError('Incomplete repeated anchor')
    result = {}
    for r in repeats:
        a = first[r['lanes']]
        ratios = {s:r['stages'][s]['us']/a['stages'][s]['us'] for s in a['stages']}
        result[r['lanes']] = dict(wall_ratio=r['wall_us']/a['wall_us'], stage_ratios=ratios,
                                 stage_attribution_stable=all(1/1.15<=v<=1.15 for v in ratios.values()))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lanes', default='65536,1048576')
    p.add_argument('--timesteps', default='0,.0025,.01,.02')
    p.add_argument('--reset-percent', default='0,100')
    p.add_argument('--iterations', type=int, default=40)
    p.add_argument('--samples', type=int, default=5)
    p.add_argument('--warmup', type=int, default=10)
    args = p.parse_args()
    lanes = list(map(int,args.lanes.split(',')))
    dts = list(map(float,args.timesteps.split(',')))
    resets = list(map(int,args.reset_percent.split(',')))
    if min(lanes+[args.iterations,args.samples])<1 or args.warmup<0 or any(not math.isfinite(t) or t<0 for t in dts) or any(r<0 or r>100 for r in resets):
        p.error('Invalid workload dimensions')
    root = Path.cwd()
    out = root/'.local-ci'/('diagnose-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ'))
    out.mkdir(parents=True)
    result = dict(status='failed', source_fingerprint=fingerprint(root), config=vars(args),
                  measurements=[], rows=[], limitations=[
                      'Kernel durations and nonkernel span are not hardware occupancy or DRAM traffic.',
                      'Timestep/reset interventions also change trajectories; they are workload sensitivity, not isolated instruction costs.',
                      'Reset percent is an initial fixture setting; observed reset counts include warmup.',
                      'Synthetic rotor-command, single-model workload; no policy or learner.'])
    try:
        jobs = [(dt,r) for dt in dts for r in resets]
        random.Random(19).shuffle(jobs)
        # Repeat one identical anchor at the end to detect campaign drift.
        anchor = jobs[0]
        jobs.append(anchor)
        for i,(dt,r) in enumerate(jobs):
            print(f'Diagnosing dt={dt} reset={r} ({i+1}/{len(jobs)})',flush=True)
            command = [sys.executable,'-m','flightmare_profile','--workloads','environment-graph',
                       '--cases',','.join(f'{n}x1' for n in lanes),'--iterations',str(args.iterations),
                       '--samples',str(args.samples),'--warmup',str(args.warmup),
                       '--control-dt',str(dt),'--reset-percent',str(r),'--trace-all']
            run = subprocess.run(command,capture_output=True,text=True,timeout=900)
            (out/f'run-{i}.log').write_text(run.stdout+run.stderr)
            if run.returncode: raise RuntimeError(f'Probe failed; see run-{i}.log')
            paths = [line.removeprefix('PASSED: ') for line in run.stdout.splitlines() if line.startswith('PASSED: ')]
            if len(paths)!=1: raise RuntimeError('Missing unique probe report')
            path = Path(paths[0]); report = json.loads(path.read_text())
            rows = attribute(report)
            for row in rows: row.update(run=i,report=str(path),anchor_repeat=i==len(jobs)-1)
            result['rows'].extend(rows)
            result['measurements'].append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                               binary_sha256=report['binary_sha256']))
        # Match by N: native reports randomize case order, so positional pairing is unsafe.
        result['anchor_stability'] = stability(result['rows'])
        ratios = {n:r['wall_ratio'] for n,r in result['anchor_stability'].items()}
        result['anchor_wall_ratios'] = ratios
        if any(not 1/1.15<=v<=1.15 for v in ratios.values()): raise ValueError('Campaign anchor drift exceeds 1.15x')
        if len({r['binary_sha256'] for r in result['measurements']})!=1: raise ValueError('Probe binary changed during campaign')
        import fcntl
        with open('/build/flightmare-local-ci.lock','w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            result['resources'] = inspect('/build/v3-profile/profile_driver')
        if result['resources']['sha256']!=result['measurements'][0]['binary_sha256']:
            raise ValueError('Resource probe does not match measured binary')
        if fingerprint(root)!=result['source_fingerprint']: raise ValueError('Source changed during campaign')
        result['status']='passed'
    except Exception as error:
        result['error']=str(error)
    finally:
        (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
        lines=['# Kernel diagnostic campaign', '', 'Status: '+result['status'], '',
               '| N | dt | Reset fixture % | Wall us | Prepare us | Dynamics us | Task/reset us |',
               '| ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for r in sorted(result['rows'],key=lambda r:(r['lanes'],r['dt'],r['initial_reset_percent'],r['run'])):
            lines.append(f"| {r['lanes']} | {r['dt']} | {r['initial_reset_percent']} | {r['wall_us']:.2f} | "+
                         ' | '.join(f"{r['stages'][s]['us']:.2f}" for s in ('prepare','dynamics','task_reset'))+' |')
        lines += ['',*result['limitations'],'', 'Repeated anchor ratios: '+str(result.get('anchor_wall_ratios',{}))]
        lines += ['', 'Per-stage stability: '+str(result.get('anchor_stability',{})),
                  'Unstable stage attribution must be remeasured before choosing a stage-specific optimization.']
        if 'error' in result: lines += ['',result['error']]
        (out/'summary.md').write_text('\n'.join(lines)+'\n')
        print(result['status'].upper()+': '+str(out/'report.json'),flush=True)
    return 0 if result['status']=='passed' else 1


if __name__=='__main__':
    sys.exit(main())
