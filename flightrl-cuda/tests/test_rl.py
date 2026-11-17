"""Acceptance targets: real task transitions, history ownership and learning."""
import copy
import torch
import pytest
from flightrl_cuda import TensorEnv, Config, ActorCritic, Rollout, ppo_update, save, load


def test_wrapper_terminal_history_and_partial_reset():
    env = TensorEnv(Config(num_envs=3, slots=11), random_reset=False)
    env.reset()
    env.actions.fill_(-1)
    result = env.step()
    assert result.observation.shape == (33, 12)
    assert env.grouped.observation.shape==(3,11,12)
    assert env.grouped.observation.data_ptr()==result.observation.data_ptr()
    assert result.done.all() and result.valid.all()
    assert torch.equal(result.episode_return, result.reward)
    assert (result.episode_length == 1).all()
    assert not torch.equal(result.transition_observation, result.observation)
    env.engine.selected.zero_(); env.engine.selected[1] = 1
    env.engine.reset_state[2, 1] = 5
    env.reset()
    env.engine.selected.fill_(1)
    env.step()
    assert result.done[1] == 0 and result.episode_length[1] == 0
    assert result.done[0] == 1
    env.step_async()
    with pytest.raises(RuntimeError): env.step_async()
    env.step_wait()
    exported = result.observation
    env.close(); env.close()
    with pytest.raises(RuntimeError): env.step()
    assert torch.isfinite(exported).all()


def test_reseed_and_rejected_configuration_preserve_state():
    cfg=Config(num_envs=3)
    env=TensorEnv(cfg)
    env.reset(); initial=env.engine.state.clone()
    env.reset(); assert not torch.equal(env.engine.state,initial)
    env.seed(cfg.seed); env.reset(); assert torch.equal(env.engine.state,initial)
    env.capture()
    invalid=copy.copy(cfg); invalid.reward_coeff=[float('nan')]*5
    with pytest.raises(ValueError): env.configure(invalid)
    env.step(); assert env.graph
    updated=copy.copy(cfg); updated.reward_coeff=[0.]*5
    env.configure(updated); assert not env.graph
    updated.reward_coeff[0]=-1.; updated.model.mass=1.5
    assert env.config.reward_coeff==[0.]*5 and env.config.model.mass==pytest.approx(.73)
    out=env.step()
    torch.testing.assert_close(out.reward,torch.full_like(out.reward,.1))


def test_nondefault_stream_and_rejected_binding_resize():
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        env=TensorEnv(Config(num_envs=3)); env.reset(); env.capture()
        policy=ActorCritic().cuda(); history=Rollout(3,env,policy)
        history.collect(); history.finish()
    with pytest.raises(ValueError,match='stream'): env.step()
    with torch.cuda.stream(stream):
        history.reward.resize_(1000)
        with pytest.raises(ValueError,match='storage'): history.finish()
    env.close()


def test_float32_contract_independent_of_framework_default():
    old=torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        env=TensorEnv(Config(num_envs=3)); env.reset()
        policy=ActorCritic().cuda(); history=Rollout(3,env,policy)
        history.collect(); history.finish()
        assert history.observation.dtype==torch.float32
    finally:
        torch.set_default_dtype(old)


def test_strict_gate_retains_observed_gimbal_limit():
    from types import SimpleNamespace
    from flightmare_profile.rl_benchmark import compare_rollouts
    # Actual lane 6315, t=30 from the 16,384-lane/32-step qualification failure.
    # Physical orientation is close, but raw Euler outputs must NOT silently
    # bypass the existing strict observation gate.
    a=torch.zeros((1,1,12)); b=a.clone()
    a[0,0,3:6]=torch.tensor([1.4561834335,-1.5723339319,-1.1779060364])
    b[0,0,3:6]=torch.tensor([1.4574944973,-1.5723326206,-1.1792175770])
    shared=dict(done=torch.zeros((1,1),dtype=torch.uint8),valid=torch.ones((1,1),dtype=torch.uint8))
    with pytest.raises(ValueError,match='observation'):
        compare_rollouts(SimpleNamespace(observation=a,**shared),SimpleNamespace(observation=b,**shared))
    for key in ('reward','action','value','logp','advantage','returns'):
        shared[key]=torch.zeros((1,1))
    result=compare_rollouts(SimpleNamespace(observation=a,**shared),SimpleNamespace(observation=b,**shared),'conditioned')
    assert result['conditioned_euler_rows']==1 and result['orientation_matrix_max']<5e-5
    bad=dict(shared,reward=torch.ones((1,1)))
    with pytest.raises(ValueError,match='reward'):
        compare_rollouts(SimpleNamespace(observation=a,**shared),SimpleNamespace(observation=b,**bad),'conditioned')
    b[0,0,5]+=.01
    with pytest.raises(ValueError,match='observation'):
        compare_rollouts(SimpleNamespace(observation=a,**shared),SimpleNamespace(observation=b,**shared),'conditioned')


def test_performance_gate_rejects_bracketing_drift():
    from flightmare_profile.rl_benchmark import validate_bracket
    validate_bracket(dict(total_s=1.,collection_s=.5),dict(total_s=1.1,collection_s=.52))
    for after in (dict(total_s=2.64,collection_s=.5),dict(total_s=.7,collection_s=.5),
                  dict(total_s=1.,collection_s=.7)):
        with pytest.raises(ValueError,match='drift'):
            validate_bracket(dict(total_s=1.,collection_s=.5),after)


def test_no_auto_reset_graph_and_bad_action():
    env = TensorEnv(Config(num_envs=3), random_reset=False, auto_reset=False)
    env.reset(); env.capture()
    env.actions.fill_(-1)
    out = env.step()
    assert torch.equal(out.observation, out.transition_observation)
    env.actions[1, 0] = float('nan')
    out = env.step()
    assert out.valid.tolist() == [1, 0, 1]
    assert out.episode_length[1] == 0


def test_lane_zero_execution_preserves_other_lanes():
    env=TensorEnv(Config(num_envs=3,slots=11),random_reset=False)
    single=TensorEnv(Config(num_envs=1),random_reset=False)
    env.reset(); single.reset(); env.capture(); single.capture()
    env.actions.fill_(-1); single.actions.fill_(-1)
    before=[x.clone() for x in (env.engine.state,env.engine.time,env.result.observation,
                                env.result.reward,env.result.done,env.running_length)]
    actual=env.test_step(); expected=single.step()
    torch.testing.assert_close(actual.observation[:1],expected.observation,rtol=0,atol=0)
    torch.testing.assert_close(actual.reward[:1],expected.reward,rtol=0,atol=0)
    assert actual.episode_length[0]==1
    for old,new in zip(before,(env.engine.state,env.engine.time,env.result.observation,
                               env.result.reward,env.result.done,env.running_length)):
        assert torch.equal(old[:,1:] if old.shape[0]==25 else old[1:],
                           new[:,1:] if new.shape[0]==25 else new[1:])
    env.step()  # Existing full-batch graph still works after lane-zero submission.
    assert env.result.episode_length.eq(1).all()
    env.close(); single.close()


def test_host_adapter_preserves_snapshots_and_partial_step():
    import numpy as np
    from flightrl_cuda import NumpyEnv
    env=TensorEnv(Config(num_envs=3),random_reset=False)
    host=NumpyEnv(env); initial=host.reset()
    assert host.observation_space.contains(initial[0])
    assert host.sample_actions().shape==(3,4)
    action=np.full((3,4),-1,dtype=np.float32)
    obs,rew,done,info=host.testStep(action)
    assert done.tolist()==[True,False,False] and info[0]['episode']['l']==1
    saved=obs.copy(); host.step(action)
    np.testing.assert_array_equal(obs,saved)
    np.testing.assert_array_equal(initial,np.zeros_like(initial))
    host.close()
    with pytest.raises(RuntimeError): host.step(action)


def test_scalar_task_has_no_vector_terminal_bonus():
    import numpy as np
    from flightrl_cuda import SingleNumpyEnv,NumpyEnv
    single=SingleNumpyEnv(TensorEnv(Config(num_envs=1),random_reset=False,auto_reset=False))
    vector=NumpyEnv(TensorEnv(Config(num_envs=1),random_reset=False))
    assert single.reset().shape==(12,); vector.reset()
    action=np.full(4,-1,dtype=np.float32)
    obs,reward,done,info=single.step(action)
    _,vector_reward,vector_done,_=vector.step(action[None,:])
    assert done and vector_done[0] and info['valid']
    assert reward==pytest.approx(vector_reward[0]+.02,abs=1e-7)
    assert np.array_equal(obs,single.env.result.transition_observation[0].cpu().numpy())
    single.close(); vector.close()


def test_gae_terminal_horizon_invalid_and_history():
    env = TensorEnv(Config(num_envs=3), random_reset=False)
    policy = ActorCritic().cuda()
    history = Rollout(3, env, policy)
    # Independent arithmetic fixture: terminal excludes V(next), horizon includes it.
    history.reward.copy_(torch.tensor([[1., 2., 8.], [3., 4., 9.], [5., 6., 10.]], device='cuda'))
    history.value.fill_(.5); history.done.zero_(); history.valid.fill_(1)
    history.done[1, 0] = 1; history.valid[1, 2] = 0
    history.last_value.fill_(2)
    history.finish(gamma=1., lam=1.)
    torch.testing.assert_close(history.returns[:, 0], torch.tensor([4., 3., 7.], device='cuda'))
    torch.testing.assert_close(history.returns[:, 1], torch.tensor([14., 12., 8.], device='cuda'))
    assert history.advantage[1, 2] == 0
    # An invalid next transition cannot leak an arbitrary value through GAE.
    assert history.returns[0, 2] == 8
    env.reset(); history.collect()
    original = history.observation.clone()
    env.step()
    assert torch.equal(history.observation, original)


def test_ppo_update_checkpoint_and_replay(tmp_path):
    torch.manual_seed(123)
    env = TensorEnv(Config(num_envs=8), random_reset=True)
    env.reset()
    policy = ActorCritic().cuda(); opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    history = Rollout(8, env, policy)
    history.collect(); history.finish()
    before = [p.detach().clone() for p in policy.parameters()]
    metrics = ppo_update(policy, opt, history, epochs=2)
    assert torch.isfinite(metrics).all()
    assert any(not torch.equal(a, b) for a, b in zip(before, policy.parameters()))
    path = tmp_path/'policy.pt'
    save(path, policy, opt, env.config, updates=1)
    restored, optimizer, metadata = load(path, device='cuda')
    assert metadata['updates'] == 1
    with torch.no_grad():
        torch.testing.assert_close(restored(history.observation[0])[0], policy(history.observation[0])[0], rtol=0, atol=0)
    assert optimizer.state_dict()['state']


def test_configuration_and_resident_collection_allocation():
    cfg = Config.from_yaml(task_text='quadrotor_env: {sim_dt: 0.02, max_t: 5}\nrl: {pos_coeff: -0.003}', num_envs=3, slots=11)
    assert cfg.control_dt == .02 and cfg.model.mass == pytest.approx(.73)
    assert cfg.reward_coeff[0] == -.003
    with pytest.raises(ValueError): Config.from_yaml(vector_text='env: {render: yes}')
    env = TensorEnv(cfg, random_reset=True)
    policy = ActorCritic().cuda(); history = Rollout(8, env, policy)
    env.reset(); env.capture(); history.collect()
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()
    history.collect()
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats()
    for key in ('num_device_alloc', 'num_device_free', 'allocation.all.allocated', 'allocation.all.freed'):
        assert after[key] == before[key], (key, after[key]-before[key])


@pytest.mark.parametrize('mode,workers', [('thread',1), ('thread',4), ('process',2)])
def test_parallel_cpu_gpu_policy_rollout(mode, workers):
    from flightmare_profile.rl_benchmark import CpuEnv, warm_start, compare_rollouts
    torch.manual_seed(7)
    cfg = Config(num_envs=33)
    gpu = TensorEnv(cfg, random_reset=False)
    cpu = CpuEnv(cfg, mode=mode, workers=workers, policy_device='cuda:0')
    try:
        warm_start(gpu, 20); cpu.reset(spin=20)
        policy = ActorCritic().cuda()
        a, b = Rollout(16,gpu,policy), Rollout(16,cpu,policy)
        a.noise.normal_(); b.noise.copy_(a.noise)
        a.collect(resample=False); b.collect(resample=False)
        a.finish(); b.finish()
        compare_rollouts(a,b)
        # Prove a corrupted terminal/history field cannot pass the benchmark.
        b.done[0,0] ^= 1
        with pytest.raises(ValueError): compare_rollouts(a,b)
    finally:
        cpu.close(); gpu.close()
