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
