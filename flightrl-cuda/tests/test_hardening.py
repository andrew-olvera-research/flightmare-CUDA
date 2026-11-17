import pytest
import torch
from flightrl_cuda import Config, TensorEnv, ActorCritic, Rollout, ppo_update


def test_configuration_cannot_lie_about_setup():
    env=TensorEnv(Config(num_envs=3)); env.reset(); env.capture()
    before=env.engine.state.clone(); cfg=env.config; cfg.control_dt=.01
    with pytest.raises(AttributeError): env.config=cfg
    cfg=env.config; cfg.seed=123
    with pytest.raises(ValueError,match='seed'): env.configure(cfg)
    assert env.config.seed==1 and env.graph
    assert torch.equal(before,env.engine.state)
    env.seed(123); env.reset()
    fresh=TensorEnv(Config(num_envs=3,seed=123)); fresh.reset()
    assert torch.equal(env.engine.state,fresh.engine.state)
    fresh.close(); env.close()


def test_binding_rejects_same_shape_stride_mutation():
    env=TensorEnv(Config(num_envs=3)); env.reset()
    h=Rollout(3,env,ActorCritic().cuda()); h.reward.transpose_(0,1)
    with pytest.raises(ValueError,match='storage'): h.finish()
    env.close()


@pytest.mark.parametrize('failure',['invalid','nan_return'])
def test_bad_rollout_rejected_before_optimizer_mutation(failure):
    env=TensorEnv(Config(num_envs=2)); env.reset()
    p=ActorCritic().cuda(); h=Rollout(4,env,p); h.noise.zero_()
    if failure=='invalid': h.noise[0,0,0]=float('nan')
    h.collect(resample=False); h.finish()
    if failure=='nan_return': h.returns[0,0]=float('nan')
    before=[x.detach().clone() for x in p.parameters()]
    opt=torch.optim.Adam(p.parameters())
    with pytest.raises(ValueError): ppo_update(p,opt,h,epochs=1)
    assert not opt.state
    assert all(torch.equal(a,b) for a,b in zip(before,p.parameters()))
    env.close()

@pytest.mark.parametrize('target',['policy','history','workspace'])
def test_replaced_capture_storage_is_rejected(target):
    env=TensorEnv(Config(num_envs=3)); env.reset()
    p=ActorCritic().cuda(); h=Rollout(3,env,p)
    if target=='policy':
        old=p.actor.bias.detach()
        p.actor.bias.data=torch.ones_like(p.actor.bias)
    elif target=='history': h.returns=torch.full_like(h.returns,99)
    else: h.work['action']=torch.ones_like(h.work['action'])
    with pytest.raises(ValueError,match='storage'):
        if target=='history': h.finish()
        else: h.collect()
    env.close()


def test_autoreset_mode_cannot_diverge_from_native():
    env=TensorEnv(Config(num_envs=1),random_reset=False); env.reset()
    with pytest.raises(AttributeError): env.auto_reset=False
    env.set_auto_reset(False); env.actions.fill_(-1)
    result=env.step()
    assert torch.equal(result.observation,result.transition_observation)
    env.close()

@pytest.mark.parametrize('field',['actions','reward'])
def test_environment_replacement_rejected_before_step(field):
    env=TensorEnv(Config(num_envs=3)); env.reset(); before=env.engine.time.clone()
    if field=='actions': env.actions=torch.ones_like(env.actions)
    else: env.result.reward=torch.ones_like(env.result.reward)
    with pytest.raises(ValueError,match='storage'): env.step()
    assert torch.equal(before,env.engine.time)
    env.close()


def test_pending_step_rejects_output_replacement():
    env=TensorEnv(Config(num_envs=3)); env.reset(); env.step_async()
    env.result.reward=env.result.reward.clone()
    with pytest.raises(ValueError,match='storage'): env.step_wait()
    env.close()


def test_invalid_gradient_limit_rejected_before_update():
    env=TensorEnv(Config(num_envs=2)); env.reset()
    p=ActorCritic().cuda(); h=Rollout(2,env,p); h.collect(); h.finish()
    opt=torch.optim.Adam(p.parameters()); before=[x.detach().clone() for x in p.parameters()]
    with pytest.raises(ValueError): ppo_update(p,opt,h,max_grad_norm=-1)
    assert not opt.state and all(torch.equal(a,b) for a,b in zip(before,p.parameters()))
    env.close()


@pytest.mark.parametrize('name', ['rl_stats', 'rl_gae'])
def test_rl_launch_dimensions_reject_before_native_submission(name):
    from flightrl_cuda.kernels import BoundKernel
    if name == 'rl_stats':
        types = (torch.float32,torch.uint8,torch.uint8,torch.float32,torch.int64,torch.float32,torch.int64)
        shapes = [(3,)] * 7
        bad = [(4,True), (2**32+3,True), (0,True), (-1,True), (3.0,True), ()]
        good = (1,True)  # lane-zero test_step is explicitly supported
    else:
        types = (torch.float32,torch.float32,torch.uint8,torch.uint8,torch.float32,torch.float32,torch.float32)
        shapes = [(2,3)]*4+[(3,)]+[(2,3)]*2
        bad = [(4,2,.99,.95), (2,2,.99,.95), (3,3,.99,.95),
               (3,2**32+2,.99,.95), (3,2,float('nan'),.95), ()]
        good = (3,2,.99,.95)
    tensors = [torch.zeros(s,dtype=t,device='cuda') for s,t in zip(shapes,types)]
    kernel = BoundKernel(name,tensors,types,shapes)
    # Never launch deliberately invalid pointers/extents, even on the unfixed
    # implementation. The negative control asserts rejection before FFI entry.
    calls = []
    kernel.fn = lambda *args: calls.append(args) or 0
    for args in bad:
        with pytest.raises(ValueError,match='launch dimensions'): kernel.run(*args)
    assert calls == []
    kernel.run(*good)
    assert len(calls) == 1
