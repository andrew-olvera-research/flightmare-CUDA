"""Independent Nsight capture of the production resident rollout collector."""
import argparse
import json
import torch
from flightlib_cuda import _native
from flightrl_cuda import Config, TensorEnv, ActorCritic, Rollout
from .rl_benchmark import warm_start


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lanes',type=int,required=True); p.add_argument('--steps',type=int,default=32)
    p.add_argument('--agents',type=int,default=1)
    args=p.parse_args()
    torch.manual_seed(17)
    if args.agents<1 or args.lanes%args.agents: p.error('lanes must divide by agents')
    env=TensorEnv(Config(num_envs=args.lanes//args.agents,slots=args.agents),random_reset=False)
    warm_start(env,20)
    policy=ActorCritic().cuda(); history=Rollout(args.steps,env,policy)
    history.noise.normal_(); history.collect(resample=False)
    warm_start(env,20)
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    start.record(); end.record(); end.synchronize()
    before=torch.cuda.memory_stats()
    _native.profile_start()
    start.record(); history.collect(resample=False); end.record(); end.synchronize()
    _native.profile_stop()
    after=torch.cuda.memory_stats()
    delta={k:after[k]-before[k] for k in ('num_device_alloc','num_device_free','allocation.all.allocated','allocation.all.freed')}
    if any(delta.values()) or not bool(history.valid.all()): raise RuntimeError('invalid resident trace workload')
    print(json.dumps(dict(validated=True,allocator_delta=delta,gpu_timeline_ms=start.elapsed_time(end))))
    env.close()


if __name__=='__main__': main()
