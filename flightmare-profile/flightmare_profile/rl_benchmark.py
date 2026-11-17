"""Matched closed-loop PPO iteration: production CUDA vs original parallel CPU.

CPU workers share actions and outputs; pipes carry commands only. Both simulators
feed the same PyTorch collector/learner. No GPU/CPU speedup is accepted without
rollout differential preflight. Setup, reset and fixed takeoff are outside timing.
"""
import argparse
import copy
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
import sqlite3
import time
import numpy as np
import torch
from flightrl_cuda import Config, TensorEnv, Transition, ActorCritic, Rollout, ppo_update
from .cpu_parallel import Batch, receive
from .__main__ import fingerprint, run, trace_summary

AFFINITY = sorted(os.sched_getaffinity(0))
_cores={}
for _cpu in AFFINITY:
    _topology=Path(f'/sys/devices/system/cpu/cpu{_cpu}/topology')
    _key=tuple(int((_topology/k).read_text()) for k in ('physical_package_id','core_id'))
    _cores.setdefault(_key,[]).append(_cpu)
WORKER_CPUS=[group[i] for i in range(max(map(len,_cores.values()))) for _,group in sorted(_cores.items()) if i<len(group)]
LIBRARY = Path('/build/flightmare-profile/libcpu_environment.so')
CONFIG = Path('/workspace/flightlib/configs/quadrotor_env.yaml')


def views(raw, dones, n):
    arr = np.frombuffer(raw, dtype=np.float32)
    return (arr[:4*n].reshape(n,4), arr[4*n:16*n].reshape(n,12),
            arr[16*n:28*n].reshape(n,12), arr[28*n:], np.frombuffer(dones,dtype=np.uint8))


class NativeBatch(Batch):
    def __init__(self, n, threads, arrays, library=LIBRARY, config=CONFIG):
        previous=torch.get_num_threads()
        super().__init__(library,config,n,0.,threads)
        self.lib.batch_rl_threads.argtypes=[ct.c_void_p,ct.c_int]
        self.lib.batch_rl_threads(self.ptr,threads)
        # RL OpenMP teams use explicit num_threads clauses; leave Torch's own
        # learner thread setting intact rather than inheriting a previous case.
        torch.set_num_threads(previous)
        f = ct.POINTER(ct.c_float)
        self.lib.batch_rl_reset.argtypes = [ct.c_void_p,ct.c_int,f]
        self.lib.batch_rl_step.argtypes = [ct.c_void_p,f,f,f,f,ct.POINTER(ct.c_ubyte)]
        self.arrays = arrays
        self.pointers = [a.ctypes.data_as(f if i<4 else ct.POINTER(ct.c_ubyte)) for i,a in enumerate(arrays)]
    def reset_rl(self, spin):
        if self.lib.batch_rl_reset(self.ptr,spin,self.pointers[1]): raise RuntimeError('CPU reset failed')
    def step_rl(self):
        if self.lib.batch_rl_step(self.ptr,*self.pointers): raise RuntimeError('CPU rollout invalid')


def cpu_worker(conn, raw, dones, n, lo, hi, cpu, library, config):
    try:
        os.sched_setaffinity(0,[cpu])
        batch = NativeBatch(hi-lo,1,[a[lo:hi] for a in views(raw,dones,n)],library,config)
        conn.send('ready')
        while True:
            command, value = conn.recv()
            if command == 'stop': break
            if command == 'reset': batch.reset_rl(value)
            else: batch.step_rl()
            conn.send('done')
        batch.close()
    except BaseException as error:
        conn.send({'error':repr(error)})
    finally: conn.close()


class CpuEnv(TensorEnv):
    """Benchmark adapter only; native simulation is untouched CPU Flightlib."""
    def __init__(self, config, mode='thread', workers=1, policy_device='cuda:0', library=LIBRARY, config_path=CONFIG):
        self.config, self.mode = config, mode
        self.num_envs = config.num_envs*config.slots
        self.device = torch.device(policy_device)
        self._closed, self._pending, self.auto_reset = False, False, True
        self.processes, self.connections = [], []
        n = self.num_envs
        if workers<1 or workers>n: raise ValueError('workers must be between 1 and N')
        ctx = mp.get_context('spawn')
        self.raw, self.dones = ctx.RawArray(ct.c_float,n*29), ctx.RawArray(ct.c_ubyte,n)
        arrays = views(self.raw,self.dones,n)
        self.host = [torch.from_numpy(a) for a in arrays]
        self.actions = torch.empty((n,4),device=self.device)
        if self.device.type == 'cpu':
            obs, trans, rew, done = self.host[1:]
        else:
            obs, trans, rew, done = [torch.empty_like(a,device=self.device) for a in self.host[1:]]
        self.running_return = torch.zeros(n,device=self.device)
        self.running_length = torch.zeros(n,device=self.device,dtype=torch.int64)
        self._valid = torch.empty(n,device=self.device,dtype=torch.bool)
        self._done = torch.empty_like(self._valid)
        self._reward = torch.empty_like(self.running_return)
        self._zero = torch.zeros_like(self.running_return)
        self._zero_length = torch.zeros_like(self.running_length)
        self.result = Transition(obs,trans,rew,done,torch.ones(n,device=self.device,dtype=torch.uint8),
                                 torch.zeros_like(self.running_return),torch.zeros_like(self.running_length))
        self.stats_graph = None
        self.bind_statistics()
        self.capture_statistics()
        os.sched_setaffinity(0,AFFINITY)
        if mode == 'thread': self.batch = NativeBatch(n,workers,arrays,library,config_path)
        elif mode == 'process':
            for i in range(workers):
                lo, hi = n*i//workers, n*(i+1)//workers
                parent, child = ctx.Pipe()
                p = ctx.Process(target=cpu_worker,args=(child,self.raw,self.dones,n,lo,hi,WORKER_CPUS[i%len(WORKER_CPUS)],library,config_path))
                p.start(); child.close(); self.processes.append(p); self.connections.append(parent)
            for conn in self.connections:
                if receive(conn) != 'ready': raise RuntimeError('worker initialization failed')
        else: raise ValueError('unknown CPU mode')

    def _command(self, command, value=0):
        for conn in self.connections: conn.send((command,value))
        for conn in self.connections:
            if receive(conn) != 'done': raise RuntimeError('worker failed')

    def reset(self, spin=0):
        self._ready()
        if self.mode == 'thread': self.batch.reset_rl(spin)
        else: self._command('reset',spin)
        if self.device.type != 'cpu': self.result.observation.copy_(self.host[1])
        self.running_return.zero_(); self.running_length.zero_()
        self.result.episode_return.zero_(); self.result.episode_length.zero_()
        return self.result.observation

    def step_async(self, actions=None):
        self._ready()
        self.host[0].copy_(self.actions if actions is None else actions)
        if self.mode == 'thread': self.batch.step_rl()
        else: self._command('step')
        if self.device.type != 'cpu':
            for output, source in zip((self.result.observation,self.result.transition_observation,self.result.reward,self.result.done), self.host[1:]):
                output.copy_(source)
        self._pending = True

    def close(self):
        if self._closed: return
        if self.device.type == 'cuda': torch.cuda.synchronize(self.device)
        if self.mode == 'thread': self.batch.close()
        else:
            for c in self.connections:
                try: c.send(('stop',0))
                except (BrokenPipeError,OSError): pass
            for p in self.processes:
                p.join(5)
                if p.is_alive(): p.terminate(); p.join()
            for c in self.connections: c.close()
        self._closed = True


def warm_start(env, steps):
    env.set_auto_reset(False); env.reset()
    env.actions.copy_(torch.tensor([.8,.82,.79,.81],device=env.device))
    for _ in range(steps): env.step()
    env.running_return.zero_(); env.running_length.zero_()
    env.result.episode_return.zero_(); env.result.episode_length.zero_()
    env.set_auto_reset(True); env.capture()


def euler_rotation(angles):
    """Independent double-precision ZYX reconstruction for qualification only."""
    z,y,x=angles.double().unbind(-1)
    cz,sz,cy,sy,cx,sx=z.cos(),z.sin(),y.cos(),y.sin(),x.cos(),x.sin()
    return torch.stack((cz*cy,cz*sy*sx-sz*cx,cz*sy*cx+sz*sx,
                        sz*cy,sz*sy*sx+cz*cx,sz*sy*cx-cz*sx,-sy,cy*sx,cy*cx),dim=-1)


def compare_rollouts(reference, actual, euler_gate='strict'):
    if euler_gate not in ('strict','conditioned'): raise ValueError('unknown Euler gate')
    for name in ('done','valid'):
        if not torch.equal(getattr(reference,name).cpu(),getattr(actual,name).cpu()):
            raise ValueError('rollout discrete mismatch: '+name)
    if not bool(reference.valid.all()) or not bool(actual.valid.all()): raise ValueError('invalid rollout')
    errors = {}
    for name, tolerance in (('observation',5e-4),('reward',2e-4),('action',5e-4),('value',5e-4),('logp',5e-4),('advantage',1e-3),('returns',1e-3)):
        a,b = getattr(reference,name).cpu(),getattr(actual,name).cpu()
        error = ((a-b).abs()/b.abs().clamp_min(1)).max().item()
        finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        permitted=False
        if finite and name=='observation' and error>tolerance and euler_gate=='conditioned':
            failures=(a-b).abs()>tolerance*b.abs().clamp_min(1)
            rows=failures[...,3:6].any(-1)
            non_euler=failures.clone(); non_euler[...,3:6]=False
            ea,eb=a[...,3:6][rows],b[...,3:6][rows]
            near=(ea[:,1].cos().abs()<.01)&(eb[:,1].cos().abs()<.01)
            rotation_error=(euler_rotation(ea)-euler_rotation(eb)).abs().amax(-1)
            # A separate, tightly bounded qualification for ill-conditioned
            # Euler coordinates, never an allowance for branch flips or reward drift.
            permitted=bool(not non_euler.any() and near.all() and
                           ((ea-eb).abs()<=.005).all() and (rotation_error<=5e-5).all())
            if permitted:
                errors['conditioned_euler_rows']=int(rows.sum())
                errors['orientation_matrix_max']=float(rotation_error.max())
                errors['raw_euler_max_rad']=float((ea-eb).abs().max())
        if not finite or (error>tolerance and not permitted):
            raise ValueError(f'rollout {name} mismatch: {error} > {tolerance}')
        errors[name] = error
    return errors


def sync(device):
    if device.type == 'cuda': torch.cuda.synchronize(device)


def validate_bracket(before, after, limit=1.15):
    """Do not turn materially different before/after timings into one headline."""
    for field in ('total_s','collection_s'):
        ratio=after[field]/before[field]
        if not 1/limit<=ratio<=limit:
            raise ValueError(f'GPU bracketing drift in {field}: {ratio:.3f} exceeds {limit}x stability budget')


def case(cfg, mode, workers, policy_device, steps, samples, warmup, epochs, spin, state, noise, learner_threads=8, config_path=CONFIG):
    torch.set_num_threads(learner_threads)
    start = time.perf_counter()
    env = TensorEnv(cfg,random_reset=False) if mode=='gpu' else CpuEnv(cfg,mode,workers,policy_device,config_path=config_path)
    policy = ActorCritic().to(policy_device); policy.load_state_dict(state)
    history = Rollout(steps,env,policy)
    history.noise.copy_(noise)
    optimizer = torch.optim.Adam(policy.parameters(),lr=3e-4)
    setup = time.perf_counter()-start
    values=[]; last_metrics=None
    try:
        for sample in range(warmup+samples):
            # Identical weights, noise and initial state for every measured iteration.
            policy.load_state_dict(state)
            optimizer.state.clear()
            if mode=='gpu': warm_start(env,spin)
            else: env.reset(spin)
            # Initialize Adam buffers outside timing, then reset their values.
            for p in policy.parameters():
                optimizer.state[p] = dict(step=torch.tensor(0.),exp_avg=torch.zeros_like(p),exp_avg_sq=torch.zeros_like(p))
            sync(env.device)
            if sample==warmup and env.device.type=='cuda': torch.cuda.reset_peak_memory_stats(env.device)
            before=torch.cuda.memory_stats() if env.device.type=='cuda' else {}
            begin=time.perf_counter(); history.collect(resample=False); sync(env.device)
            collection=time.perf_counter()-begin
            after=torch.cuda.memory_stats() if env.device.type=='cuda' else {}
            delta={k:after[k]-before[k] for k in ('num_device_alloc','num_device_free','allocation.all.allocated','allocation.all.freed')} if before else {}
            if sample>=warmup and mode=='gpu' and any(delta.values()): raise ValueError(f'resident collection allocates: {delta}')
            begin=time.perf_counter(); history.finish(); sync(env.device); gae=time.perf_counter()-begin
            begin=time.perf_counter(); last_metrics=ppo_update(policy,optimizer,history,epochs=epochs); sync(env.device); update=time.perf_counter()-begin
            if not bool(history.valid.all()) or not bool(torch.isfinite(last_metrics).all()): raise ValueError('invalid RL iteration')
            if sample>=warmup: values.append(dict(collection_s=collection,gae_s=gae,update_s=update,total_s=collection+gae+update))
        telemetry={name:getattr(history,name).cpu().clone() for name in ('observation','reward','done','valid','action','value','logp','advantage','returns')}
        metrics=last_metrics.detach().cpu().tolist()
        result=dict(mode=mode,workers=workers,policy_device=policy_device,lanes=env.num_envs,spin=spin,
                    setup_s=setup,samples=values,collection_allocator_delta=delta,losses=metrics,
                    torch_threads=torch.get_num_threads(),host_buffers='shared pageable' if mode!='gpu' else None,
                    torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(env.device) if env.device.type=='cuda' else None,
                    arena_bytes=env.engine._native.arena_bytes if mode=='gpu' else None,
                    completed_episodes=int(history.done.sum().item()),history_bytes=sum(v.numel()*v.element_size() for v in vars(history).values() if isinstance(v,torch.Tensor)))
        for field in values[0]: result[field]=statistics.median(v[field] for v in values)
        result['transitions_per_second']=steps*env.num_envs/result['total_s']
        return result,telemetry
    finally: env.close()


def validate_trace(path, lanes, steps):
    trace=trace_summary(path)
    api={x['name']:x['count'] for x in trace['api']}
    forbidden=('cudaMalloc','cudaFree','cudaHostAlloc','cudaHostRegister','cudaDeviceSynchronize','cudaStreamSynchronize')
    if trace['cuda_api_errors'] or any(count for name,count in api.items() if name.startswith(forbidden)):
        raise ValueError('hot-loop CUDA allocation/synchronization or API error')
    if sum(c for n,c in api.items() if n.startswith('cudaEventSynchronize'))!=1:
        raise ValueError('expected only one completion boundary')
    if sum(c for n,c in api.items() if n.startswith('cudaGraphLaunch'))!=2*steps+1:
        raise ValueError('policy/environment graph launch mismatch')
    if sum(k['count'] for k in trace['kernels'] if k['name'].startswith('stats('))!=steps:
        raise ValueError('episode statistics kernel launch mismatch')
    physics=sum(k['count'] for k in trace['kernels'] if any(s in k['name'] for s in ('prepareActions','dynamicsKernel','taskKernel')))
    if physics!=3*steps: raise ValueError('missing or extra simulation kernels')
    with sqlite3.connect(path) as db:
        tables={r[0] for r in db.execute("select name from sqlite_master where type='table'")}
        copies=list(db.execute('select copyKind,count(*),sum(bytes) from CUPTI_ACTIVITY_KIND_MEMCPY group by copyKind')) if 'CUPTI_ACTIVITY_KIND_MEMCPY' in tables else []
    # CUPTI_ACTIVITY_MEMCPY_KIND_DTOD = 8; every other kind is prohibited here.
    if any(kind!=8 for kind,_,_ in copies): raise ValueError('resident rollout transferred host/device data')
    budget=(130*steps+20)*lanes  # history, policy noise, action staging; includes mask copy allowance
    if sum(size for _,_,size in copies)>budget: raise ValueError('unexpected device copy volume')
    if trace['memset_nonzero_events'] or trace['memset_bytes']>1024*(steps+1):
        raise ValueError('unexpected scratch clearing')
    trace.update(copy_kinds=copies,device_copy_budget_bytes=budget,validated=True)
    return trace


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lanes',default='33,1024,16384'); p.add_argument('--workers',default='1,2,4,8,16')
    p.add_argument('--modes',default='thread,process'); p.add_argument('--cpu-policy-devices',default='cuda:0')
    p.add_argument('--steps',type=int,default=32); p.add_argument('--samples',type=int,default=5)
    p.add_argument('--warmup',type=int,default=2); p.add_argument('--epochs',type=int,default=2)
    p.add_argument('--spin',type=int,default=20); p.add_argument('--torch-threads',type=int,default=8)
    p.add_argument('--cpu-learner-threads',default='1,8')
    p.add_argument('--trace',action='store_true')
    p.add_argument('--euler-gate',choices=('strict','conditioned'),default='conditioned')
    args=p.parse_args()
    lanes=list(map(int,args.lanes.split(','))); workers=list(map(int,args.workers.split(',')))
    learner_threads=list(map(int,args.cpu_learner_threads.split(',')))
    if min(lanes+workers+[args.steps,args.samples,args.epochs,args.torch_threads])<1 or min(args.spin,args.warmup)<0: p.error('invalid dimensions')
    root=Path('/workspace'); build=Path('/build/flightmare-profile')
    directory=root/'.local-ci'/('rl-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ'))
    directory.mkdir(parents=True)
    report=dict(status='failed',config=vars(args),cases=[],comparisons=[])
    try:
        with open('/build/flightmare-local-ci.lock','w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            report.update(source_fingerprint=fingerprint(root),commit=run(['git','rev-parse','HEAD'],root).strip(),
                          dirty=run(['git','status','--short'],root),cpu=run(['lscpu'],root),
                          gpu=run(['nvidia-smi','--query-gpu=name,driver_version','--format=csv'],root),
                          torch=torch.__version__,torch_cuda=torch.version.cuda,affinity=AFFINITY,
                          worker_cpu_order=WORKER_CPUS,
                          cpu_quota=Path('/sys/fs/cgroup/cpu.max').read_text(),
                          thread_environment={k:os.environ.get(k) for k in ('OMP_PROC_BIND','OMP_PLACES','OMP_WAIT_POLICY')})
            run(['cmake','-S',root/'flightmare-profile/native','-B',build,'-G','Ninja','-DCMAKE_BUILD_TYPE=Release',
                 '-DPROFILE_CPU_BASELINE=ON','-DCMAKE_CUDA_ARCHITECTURES='+os.environ.get('FLIGHTMARE_CUDA_ARCHITECTURES','120-real')],root,directory/'configure.log')
            run(['cmake','--build',build,'--parallel','4','--target','cpu_environment','cpu_baseline','cpu_environment_parity'],root,directory/'build.log')
            for name in ('cpu_baseline','cpu_environment_parity'):
                run([build/name],root,directory/(name+'.log'),env=dict(os.environ,FLIGHTMARE_PATH=str(root)))
            report['cpu_binary_sha256']=hashlib.sha256(LIBRARY.read_bytes()).hexdigest()
            # Read exactly the frozen YAML, but place the repeated constructor
            # reads on the Linux build volume rather than the Windows bind mount.
            # This only reduces excluded setup I/O; all CPU source is unchanged.
            cpu_config=build/'rl-environment.yaml'
            cpu_config.write_bytes(CONFIG.read_bytes())
            report['cpu_config_sha256']=hashlib.sha256(cpu_config.read_bytes()).hexdigest()
            import flightlib_cuda._native as native
            report['gpu_binary_sha256']=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()
            import flightrl_cuda
            report['rl_binary_sha256']=hashlib.sha256(next(Path(flightrl_cuda.__file__).parent.glob('_kernels*.so')).read_bytes()).hexdigest()
            torch.set_num_threads(args.torch_threads); torch.set_num_interop_threads(1)
            torch.backends.cuda.matmul.allow_tf32=False
            torch.manual_seed(17)
            state=ActorCritic().state_dict()
            report['policy_sha256']=hashlib.sha256(b''.join(t.numpy().tobytes() for t in state.values())).hexdigest()
            # Each measured configuration is validated, including process partition.
            from types import SimpleNamespace
            for n in lanes:
                cfg=Config(num_envs=n)
                generator=torch.Generator().manual_seed(91)
                noise=torch.randn((args.steps,n,4),generator=generator)
                base,reference=case(cfg,'gpu',0,'cuda:0',args.steps,args.samples,args.warmup,args.epochs,args.spin,state,noise)
                report['cases'].append(base)
                base['position']='before'
                jobs=[(m,w,d,t) for m in args.modes.split(',') for w in workers if w<=n
                      for d in args.cpu_policy_devices.split(',') for t in (learner_threads if d=='cpu' else [args.torch_threads])]
                random.Random(42).shuffle(jobs)
                for mode,worker_count,device,threads in jobs:
                    row,actual=case(cfg,mode,worker_count,device,args.steps,args.samples,args.warmup,args.epochs,args.spin,state,noise,threads,cpu_config)
                    row['differential_errors']=compare_rollouts(SimpleNamespace(**reference),SimpleNamespace(**actual),args.euler_gate)
                    loss_error=max(abs(a-b)/max(1,abs(b)) for a,b in zip(row['losses'],base['losses']))
                    if loss_error>2e-3: raise ValueError('PPO loss mismatch: '+str(loss_error))
                    row['ppo_loss_error']=loss_error
                    report['cases'].append(row)
                    print(json.dumps({k:row[k] for k in ('lanes','mode','workers','policy_device','collection_s','total_s')}),flush=True)
                    (directory/'report.json').write_text(json.dumps(report,indent=2))
                repeated,repeat_telemetry=case(cfg,'gpu',0,'cuda:0',args.steps,args.samples,args.warmup,args.epochs,args.spin,state,noise)
                compare_rollouts(SimpleNamespace(**reference),SimpleNamespace(**repeat_telemetry),args.euler_gate)
                repeated['position']='after'; report['cases'].append(repeated)
                validate_bracket(base,repeated)
                gpu_total=statistics.median(s['total_s'] for row in (base,repeated) for s in row['samples'])
                gpu_collect=statistics.median(s['collection_s'] for row in (base,repeated) for s in row['samples'])
                candidates=[c for c in report['cases'] if c['lanes']==n and c['mode']!='gpu']
                best=min(candidates,key=lambda c:c['total_s'])
                best_collect=min(candidates,key=lambda c:c['collection_s'])
                report['comparisons'].append(dict(lanes=n,best_cpu={k:best[k] for k in ('mode','workers','policy_device','torch_threads')},
                    total_speedup=best['total_s']/gpu_total,gpu_transitions_per_second=args.steps*n/gpu_total,
                    gpu_total_s=gpu_total,gpu_collection_s=gpu_collect,
                    gpu_repeat_ratio=repeated['total_s']/base['total_s'],
                    collection_speedup=best_collect['collection_s']/gpu_collect,
                    best_collection_cpu={k:best_collect[k] for k in ('mode','workers','policy_device','torch_threads')}))
            if args.trace:
                report['traces']=[]
                for n in sorted(set((min(lanes),max(lanes)))):
                    prefix=directory/f'trace-{n}'
                    run(['nsys','profile','--trace=cuda','--cuda-graph-trace=node','--sample=none',
                         '--capture-range=cudaProfilerApi','--capture-range-end=stop','--output='+str(prefix),
                         os.sys.executable,'-m','flightmare_profile.rl_trace','--lanes',str(n),'--steps',str(args.steps)],root,directory/f'trace-{n}.log')
                    run(['nsys','export','--type=sqlite','--output='+str(prefix)+'.sqlite',str(prefix)+'.nsys-rep'],root,directory/f'export-{n}.log')
                    report['traces'].append(dict(lanes=n,summary=validate_trace(Path(str(prefix)+'.sqlite'),n,args.steps)))
            if fingerprint(root)!=report['source_fingerprint']: raise RuntimeError('source changed during benchmark')
            report['status']='passed'
    except BaseException as error:
        report['error']=repr(error); raise
    finally:
        (directory/'report.json').write_text(json.dumps(report,indent=2))
        print(directory,flush=True)


if __name__=='__main__': main()
