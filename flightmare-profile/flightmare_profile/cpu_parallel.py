"""Matched CPU process/thread and production GPU environment benchmark."""
import argparse
import ctypes as ct
import datetime
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import statistics
import time
from .__main__ import fingerprint, run


class Batch:
    def __init__(self, library, config, lanes, action, threads):
        self.lib = ct.CDLL(str(library))
        self.lib.batch_threads.argtypes = [ct.c_int]
        self.lib.batch_threads(threads)
        self.lib.batch_create.argtypes = [ct.c_int, ct.c_char_p, ct.c_float]
        self.lib.batch_create.restype = ct.c_void_p
        self.lib.batch_step.argtypes = [ct.c_void_p, ct.c_int]
        self.lib.batch_snapshot.argtypes = [ct.c_void_p, ct.POINTER(ct.c_float), ct.POINTER(ct.c_ulonglong)]
        self.lib.batch_destroy.argtypes = [ct.c_void_p]
        self.lib.batch_reset.argtypes = [ct.c_void_p,ct.c_float]
        self.ptr = self.lib.batch_create(lanes, str(config).encode(), action)
        if not self.ptr:
            raise RuntimeError('CPU construction failed')

    def step(self, count):
        if self.lib.batch_step(self.ptr, count):
            raise RuntimeError('CPU step failed')

    def snapshot(self):
        out, resets = (ct.c_float * 26)(), ct.c_ulonglong()
        if self.lib.batch_snapshot(self.ptr, out, ct.byref(resets)):
            raise RuntimeError('CPU lane validation failed')
        return {'output': list(out), 'resets': resets.value}

    def close(self):
        self.lib.batch_destroy(self.ptr)


def worker(conn, library, config, lanes, action, threads, affinity):
    try:
        os.sched_setaffinity(0, affinity)
        batch = Batch(library, config, lanes, action, threads)
        conn.send({'ready': True, 'affinity': sorted(os.sched_getaffinity(0))})
        while True:
            command = conn.recv()
            if command == 'stop':
                batch.close(); break
            if command == 'snapshot':
                conn.send(batch.snapshot())
            else:
                batch.step(command); conn.send('done')
    except BaseException as error:
        conn.send({'error': repr(error)})
    finally:
        conn.close()


def receive(conn):
    if not conn.poll(300):
        raise TimeoutError('CPU worker timed out')
    value = conn.recv()
    if isinstance(value, dict) and 'error' in value:
        raise RuntimeError(value['error'])
    return value


def cpu_case(library, config, lanes, action, workers, mode, chunk, iterations, samples, warmup, affinity, cache=None):
    if mode == 'thread':
        start = time.perf_counter()
        if cache is not None and lanes in cache:
            batch=cache[lanes]
            batch.lib.batch_threads(workers)
            batch.lib.batch_reset(batch.ptr,action)
        else:
            batch = Batch(library, config, lanes, action, workers)
            if cache is not None: cache[lanes]=batch
        setup = time.perf_counter()-start
        try:
            batch.step(warmup)
            values = []
            for _ in range(samples):
                begin = time.perf_counter()
                for k in range(0,iterations,chunk): batch.step(min(chunk,iterations-k))
                values.append((time.perf_counter()-begin)*1e6/iterations)
            return dict(batch.snapshot(),samples_us=values,setup_seconds=setup)
        finally:
            if cache is None: batch.close()
    ctx = mp.get_context('spawn')  # no inherited CUDA/OpenMP contexts
    processes, connections = [], []
    start = time.perf_counter()
    try:
        count = workers if mode == 'process' else 1
        for i in range(count):
            parent, child = ctx.Pipe()
            size = lanes // count + (i < lanes % count)
            process = ctx.Process(target=worker, args=(child, library, config, size, action, 1, [affinity[i % len(affinity)]]))
            process.start(); child.close()
            processes.append(process); connections.append(parent)
        worker_affinity = [receive(conn)['affinity'] for conn in connections]
        setup = time.perf_counter() - start
        def step(n):
            for conn in connections: conn.send(n)
            for conn in connections: assert receive(conn) == 'done'
        step(warmup)
        values = []
        for _ in range(samples):
            begin = time.perf_counter()
            for k in range(0, iterations, chunk): step(min(chunk, iterations-k))
            values.append((time.perf_counter()-begin)*1e6/iterations)
        for conn in connections: conn.send('snapshot')
        snapshots = [receive(conn) for conn in connections]
        return {'samples_us': values, 'setup_seconds': setup, 'output': snapshots[0]['output'],
                'resets': sum(s['resets'] for s in snapshots), 'worker_outputs': snapshots, 'worker_affinity':worker_affinity}
    finally:
        for conn in connections:
            try: conn.send('stop')
            except (BrokenPipeError, OSError): pass
        for process in processes:
            process.join(5)
            if process.is_alive(): process.terminate(); process.join()
        for conn in connections: conn.close()


def gpu_case(config, lanes, action, chunk, iterations, samples, warmup, dt=.05):
    import torch
    from flightlib_cuda import Environment, load_model
    env = Environment(lanes, control_dt=dt, models=[load_model(str(config))])
    env.actions.copy_(torch.tensor([action+.1,action-.1,action+.05,action-.05],device=env.device))
    env.reset(random=False); env.capture(random=False)
    for _ in range(warmup): env.step(graph=True, random=False)
    env.synchronize()
    before = torch.cuda.memory_stats()
    values = []
    for _ in range(samples):
        begin = time.perf_counter()
        for k in range(0, iterations, chunk):
            for _ in range(min(chunk, iterations-k)): env.step(graph=True, random=False)
            env.synchronize()
        values.append((time.perf_counter()-begin)*1e6/iterations)
    after = torch.cuda.memory_stats()
    keys = ('num_device_alloc', 'num_device_free', 'allocation.all.allocated', 'allocation.all.freed')
    delta = {k: after[k]-before[k] for k in keys}
    if any(delta.values()): raise RuntimeError(f'GPU allocation churn: {delta}')
    out = env.snapshot()
    for value in out.values():
        assert torch.equal(value, value[0:1].expand_as(value))
    result = torch.cat([out['observation'].flatten()[:12], out['transition_observation'].flatten()[:12],
                        out['reward'].flatten()[:1], out['done'].flatten()[:1].float()]).tolist()
    episodes = torch.utils.dlpack.from_dlpack(env._native.tensor('episodes.episode'))
    resets = int(episodes.to(torch.int64).sum().item()) - lanes
    return {'samples_us': values, 'output': result, 'resets': resets, 'allocator_delta': delta,
            'torch': torch.__version__, 'torch_cuda': torch.version.cuda}


def validate(reference, actual):
    if actual['resets'] != reference['resets']: raise ValueError('Reset count mismatch')
    if actual['output'][25] != reference['output'][25]: raise ValueError('Done mismatch')
    for a, b in zip(actual['output'][:25], reference['output'][:25]):
        if not abs(a-b) <= 5e-4*max(1, abs(b)): raise ValueError(f'Output mismatch: {a} vs {b}')
    for worker_result in actual.get('worker_outputs', []):
        if worker_result['output'] != actual['output']: raise ValueError('Worker partition mismatch')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lanes', default='33,1024,16384')
    p.add_argument('--workers', default='1,2,4,8,16')
    p.add_argument('--chunks', default='1,100')
    p.add_argument('--actions', default='-1,1')
    p.add_argument('--iterations', type=int, default=100)
    p.add_argument('--samples', type=int, default=5)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--control-dt',type=float,default=.05)
    p.add_argument('--modes',default='thread,process')
    args = p.parse_args()
    allowed = sorted(os.sched_getaffinity(0))
    # Physical cores first, then SMT siblings. Child processes explicitly restore
    # affinity instead of inheriting the OpenMP-pinned coordinator core.
    cores={}
    for cpu in allowed:
        topology=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        key=tuple(int((topology/name).read_text()) for name in ('physical_package_id','core_id'))
        cores.setdefault(key,[]).append(cpu)
    affinity=[group[i] for i in range(max(map(len,cores.values()))) for _,group in sorted(cores.items()) if i<len(group)]
    lanes, workers, chunks = [list(map(int, x.split(','))) for x in (args.lanes,args.workers,args.chunks)]
    actions = list(map(float,args.actions.split(',')))
    if set(args.modes.split(','))-{'thread','process'} or not 0<args.control_dt<=1: p.error('Invalid modes or dt')
    if min(lanes+workers+chunks+[args.iterations,args.samples]) < 1 or args.warmup < 0: p.error('Invalid dimensions')
    repo = Path.cwd(); build = Path('/build/flightmare-profile')
    directory = repo/'.local-ci'/('cpu-parallel-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ'))
    directory.mkdir(parents=True)
    config = directory/'environment.yaml'
    config.write_text((repo/'flightlib/configs/quadrotor_env.yaml').read_text().replace('sim_dt: 0.02',f'sim_dt: {args.control_dt}'))
    report = {'status':'failed','config':vars(args),'cases':[]}
    cache={}
    try:
        with open('/build/flightmare-local-ci.lock','w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            report.update(source_fingerprint=fingerprint(repo),commit=run(['git','rev-parse','HEAD'],repo).strip(),
                          cpu=run(['lscpu'],repo),affinity=sorted(os.sched_getaffinity(0)),
                          quota=Path('/sys/fs/cgroup/cpu.max').read_text(),
                          gpu=run(['nvidia-smi','--query-gpu=name,driver_version','--format=csv'],repo),
                          compiler=run(['c++','--version'],repo))
            report['thread_environment']={k:os.environ.get(k) for k in ('OMP_PROC_BIND','OMP_PLACES','OMP_WAIT_POLICY','OPENBLAS_NUM_THREADS')}
            report['worker_cpu_order']=affinity
            report['dirty']=run(['git','status','--short'],repo)
            python_build=Path('/build/flightlib-cuda-python-native')
            run(['cmake','-S',repo/'flightlib-cuda','-B',python_build,'-G','Ninja','-DCMAKE_BUILD_TYPE=Release',
                 '-DBUILD_TESTING=OFF','-DFLIGHTLIB_CUDA_BUILD_RUNTIME=ON','-DFLIGHTLIB_CUDA_BUILD_PYTHON=ON',
                 '-DPython_EXECUTABLE='+os.sys.executable,'-DCMAKE_CUDA_ARCHITECTURES='+os.environ.get('FLIGHTMARE_CUDA_ARCHITECTURES','120-real')],repo,directory/'python-configure.log')
            run(['cmake','--build',python_build,'--parallel','4'],repo,directory/'python-build.log')
            os.sys.path.insert(0,str(python_build/'python'))
            report['gpu_extension_sha256']=hashlib.sha256(next((python_build/'python/flightlib_cuda').glob('_native*.so')).read_bytes()).hexdigest()
            run(['cmake','-S',repo/'flightmare-profile/native','-B',build,'-G','Ninja','-DCMAKE_BUILD_TYPE=Release',
                 '-DPROFILE_CPU_BASELINE=ON','-DCMAKE_CUDA_ARCHITECTURES='+os.environ.get('FLIGHTMARE_CUDA_ARCHITECTURES','120-real')],repo,directory/'configure.log')
            run(['cmake','--build',build,'--parallel','4'],repo,directory/'build.log')
            run([build/'cpu_baseline'],repo,directory/'optimized-parity.log',env=dict(os.environ,FLIGHTMARE_PATH=str(repo)))
            run([build/'cpu_environment_parity'],repo,directory/'optimized-environment-parity.log',env=dict(os.environ,FLIGHTMARE_PATH=str(repo)))
            report['optimized_parity'] = {'dynamics': True, 'environment': True,
                'scope': 'Full differential fixtures, including saturated symmetric Euler/reward regression and randomized model rollouts'}
            library=build/'libcpu_environment.so'
            report['cpu_sha256']=hashlib.sha256(library.read_bytes()).hexdigest()
            report['preflight']=[]
            for action in actions:
                reference=gpu_case(config,1,action,args.iterations,args.iterations,args.samples,args.warmup,args.control_dt)
                candidate=cpu_case(library,config,1,action,1,'thread',args.iterations,args.iterations,args.samples,args.warmup,affinity)
                validate(reference,candidate)
                report['preflight'].append({'action':action,'validated':True})
            jobs=[(n,a,c) for n in lanes for a in actions for c in chunks]
            random.Random(0).shuffle(jobs)
            for n,a,c in jobs:
                gpu=gpu_case(config,n,a,c,args.iterations,args.samples,args.warmup,args.control_dt)
                gpu.update(lanes=n,action=a,chunk=c,mode='gpu-graph',workers=0)
                report['cases'].append(gpu)
                configs=[(m,w) for m in args.modes.split(',') for w in workers if w<=n]
                random.Random(n+c).shuffle(configs)
                for mode,w in configs:
                    print(f'{n} lanes action={a} chunk={c} {mode}={w}',flush=True)
                    result=cpu_case(library,config,n,a,w,mode,c,args.iterations,args.samples,args.warmup,affinity,cache)
                    validate(gpu,result)
                    result.update(lanes=n,action=a,chunk=c,mode=mode,workers=w)
                    report['cases'].append(result)
                    (directory/'progress.json').write_text(json.dumps(report,indent=2))
            if fingerprint(repo)!=report['source_fingerprint']: raise ValueError('Sources changed during measurement')
            report['status']='passed'
    except Exception as error:
        report['error']=repr(error)
    finally:
        for batch in cache.values(): batch.close()
        for case in report['cases']:
            case['median_us']=statistics.median(case['samples_us'])
            case['lane_steps_per_second']=case['lanes']*1e6/case['median_us']
        (directory/'report.json').write_text(json.dumps(report,indent=2))
        print(report['status'].upper(),directory/'report.json',report.get('error',''))
    if report['status']!='passed': raise SystemExit(1)


if __name__ == '__main__': main()
