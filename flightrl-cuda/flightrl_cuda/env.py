from dataclasses import dataclass
import torch
from flightlib_cuda import Environment
from .config import Config
from .kernels import BoundKernel, storage_signature


@dataclass
class Transition:
    observation: torch.Tensor
    transition_observation: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    valid: torch.Tensor
    episode_return: torch.Tensor
    episode_length: torch.Tensor


class TensorEnv:
    """Borrowed tensor outputs. All submitted storage and shapes are fixed.

    Construct and submit on one chosen Torch stream (including a nondefault one).
    Cross-stream submission is rejected; the underlying Core API remains flexible.
    Wrapper episode statistics are separate from the frozen simulation arena.
    """
    @property
    def config(self): return self._config.copy()

    @property
    def auto_reset(self): return self._auto_reset

    def __init__(self, config=None, *, random_reset=True, auto_reset=True):
        self._config = (config or Config()).copy()
        c = self.config
        self.num_envs, self.device = c.num_envs*c.slots, torch.device('cuda', c.device)
        self.engine = Environment(c.num_envs, c.slots, control_dt=c.control_dt, device=c.device,
                                  seed=c.seed, models=[c.model])
        self.engine.configure_task(c.task())
        self.random_reset, self._auto_reset = random_reset, bool(auto_reset)
        self.engine._native.auto_reset(auto_reset)
        self.actions = self.engine.actions.view(self.num_envs, 4)
        self.running_return = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.running_length = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self._selected = torch.empty(self.num_envs, dtype=torch.bool, device=self.device)
        out = self.engine.outputs
        self.result = Transition(out['observation'].view(-1,12), out['transition_observation'].view(-1,12),
                                 out['reward'].view(-1), out['done'].view(-1), out['valid'].view(-1),
                                 torch.zeros_like(self.running_return), torch.zeros_like(self.running_length))
        self.grouped = Transition(*(getattr(self.result,name).view(c.num_envs,c.slots,*getattr(self.result,name).shape[1:])
                                    for name in Transition.__dataclass_fields__))
        self._pending, self._closed, self.graph = False, False, False
        self.stats_graph = None
        self.bind_statistics()
        entries=[(self,k) for k in ('actions','running_return','running_length')]
        entries.extend((self.result,k) for k in Transition.__dataclass_fields__)
        self._fixed_tensors=[('self' if owner is self else 'result',k,getattr(owner,k).detach(),storage_signature(getattr(owner,k))) for owner,k in entries]

    def bind_statistics(self):
        self.stream = torch.cuda.current_stream(self.device) if self.device.type=='cuda' else None
        self.stats_kernel = None
        if self.device.type=='cuda':
            out=self.result
            self.stats_kernel=BoundKernel('rl_stats',
                (out.reward,out.done,out.valid,self.running_return,self.running_length,out.episode_return,out.episode_length),
                (torch.float32,torch.uint8,torch.uint8,torch.float32,torch.int64,torch.float32,torch.int64),
                [(self.num_envs,)]*7)

    def _ready(self):
        self._check_stream()
        if self._closed: raise RuntimeError('environment is closed')
        if self._pending: raise RuntimeError('collect the pending step first')
        self._check_storage()

    def _check_storage(self):
        for source,k,retained,signature in getattr(self,'_fixed_tensors',()):
            owner=self if source=='self' else self.result
            if storage_signature(getattr(owner,k))!=signature:
                raise ValueError('environment bound storage changed; reconstruct wrapper')

    def _check_stream(self):
        if self.stream is not None and torch.cuda.current_stream(self.device)!=self.stream:
            raise ValueError('submit on the wrapper construction stream')

    def reset(self):
        self._ready()
        self.engine.reset(random=self.random_reset)
        torch.eq(self.engine.selected,1,out=self._selected)
        self._selected.logical_and_(self.engine.status==0)
        self.running_return.masked_fill_(self._selected, 0)
        self.running_length.masked_fill_(self._selected, 0)
        self.result.episode_return.masked_fill_(self._selected, 0)
        self.result.episode_length.masked_fill_(self._selected, 0)
        return self.result.observation

    def capture(self):
        self._ready(); self.engine.capture(random=self.random_reset); self.graph = True
        self.capture_statistics()

    def capture_statistics(self):
        # One fused native launch replaces the former multi-node statistics graph.
        self.stats_graph = None

    def set_auto_reset(self, enabled):
        self._ready(); self.engine._native.auto_reset(enabled)
        self._auto_reset, self.graph, self.stats_graph = bool(enabled), False, None

    def seed(self, seed):
        self._ready()
        self.engine.record_usage()
        self.engine._native.seed(seed)
        self.engine.wait()
        self._config.seed=seed; self.graph=False

    def configure(self, config):
        self._ready()
        config=config.copy()
        if config.seed!=self._config.seed:
            raise ValueError('use seed() explicitly to restart RNG streams')
        if (config.num_envs,config.slots,config.device)!=(self.config.num_envs,self.config.slots,self.config.device):
            raise ValueError('fixed environment capacity/device; construct another wrapper')
        # Native configure validates dynamics and schedule before upload. Task
        # coefficients are checked here before either setup is submitted.
        import math
        if len(config.reward_coeff)!=5 or not all(math.isfinite(x) for x in config.reward_coeff):
            raise ValueError('invalid reward coefficients')
        task=config.task()
        if not all(math.isfinite(x) for values in (task.reward_coeff,task.action_mean,task.action_scale,task.goal) for x in values):
            raise ValueError('task values exceed float32 domain')
        self.engine.record_usage()
        self.engine._native.configure([config.model],self.engine.model_ids,self.engine.active,self.engine.bounds,config.control_dt)
        self.engine.configure_task(task)
        self.engine.models=[config.model]; self.engine.wait()
        self._config=config; self.graph=False

    def step_async(self, actions=None):
        self._ready()
        if actions is not None:
            if actions.shape != self.actions.shape or actions.dtype != torch.float32 or actions.device != self.device or not actions.is_contiguous() or actions.requires_grad:
                raise ValueError('actions must be dense float32 CUDA [N,4] without autograd')
            if actions.data_ptr() != self.actions.data_ptr(): self.actions.copy_(actions)
        self.engine.step(graph=self.graph, random=self.random_reset, stream=self.stream)
        self._pending = True

    def step_wait(self):
        self._check_stream()
        if self._closed or not self._pending: raise RuntimeError('no pending step')
        self._check_storage()
        if self.stats_graph is None: self._statistics()
        else: self.stats_graph.replay()
        self._pending = False
        return self.result

    def _statistics(self):
        if self.stats_kernel is not None:
            self.stats_kernel.run(self.num_envs,self.auto_reset)
            return
        out = self.result
        self._valid.copy_(out.valid); self._done.copy_(out.done)
        self._done.logical_and_(self._valid)
        torch.where(self._valid, out.reward, self._zero, out=self._reward)
        self.running_return.add_(self._reward)
        self.running_length.add_(self._valid)
        torch.where(self._done, self.running_return, self._zero, out=out.episode_return)
        torch.where(self._done, self.running_length, self._zero_length, out=out.episode_length)
        if self.auto_reset:
            self.running_return.masked_fill_(self._done, 0)
            self.running_length.masked_fill_(self._done, 0)

    def step(self, actions=None):
        self.step_async(actions); return self.step_wait()

    def test_step(self):
        """Legacy vector testStep: consume lane-zero action and update only it.

        All other output rows remain stale by design. This is not a rollout step.
        Uses original vector terminal reward/autoreset; no single-env semantics.
        """
        self._ready()
        if not self.auto_reset: raise ValueError('test_step requires vector autoreset mode')
        self.engine._native.test_step(self.engine._stream(),self.random_reset)
        self.stats_kernel.run(1,True)
        return self.result

    def close(self):
        if not self._closed:
            self.stream.synchronize()
            self.engine.synchronize(); self._closed = True; self._pending = False

    def __del__(self):
        try: self.close()
        except (AttributeError,RuntimeError): pass


class NumpyEnv:
    """Explicit synchronous host adapter; does not claim a framework subclass."""
    def __init__(self, env): self.env = env
    def reset(self): return self.env.reset().cpu().numpy().copy()
    def reset_and_update_info(self):
        self.env._ready()
        returns=self.env.running_return.cpu().tolist(); lengths=self.env.running_length.cpu().tolist()
        return self.reset(), [{'episode':{'r':r,'l':l}} for r,l in zip(returns,lengths)]
    def _submit(self, action, first=False):
        import numpy as np
        self.env._ready()
        a = np.asarray(action)
        if a.dtype != np.float32 or a.shape != tuple(self.env.actions.shape):
            raise ValueError('expected float32 [N,4]')
        self.env.actions.copy_(torch.from_numpy(a))
        r = self.env.test_step() if first else self.env.step()
        obs, rew, done, valid, ret, length = [x.cpu().numpy().copy() for x in
            (r.observation, r.reward, r.done, r.valid, r.episode_return, r.episode_length)]
        info = [{'valid': bool(valid[i]), **({'episode': {'r': float(ret[i]), 'l': int(length[i])}} if done[i] and valid[i] else {})}
                for i in range(self.env.num_envs)]
        if first:
            for i in range(1,self.env.num_envs): info[i] = {}
        return obs, rew, done.astype(bool), info
    def step(self, action): return self._submit(action)
    def testStep(self, action): return self._submit(action,True)
    @property
    def num_envs(self): return self.env.num_envs
    @property
    def observation_space(self):
        from gymnasium.spaces import Box
        import numpy as np
        return Box(-np.inf,np.inf,(12,),dtype=np.float32)
    @property
    def action_space(self):
        from gymnasium.spaces import Box
        import numpy as np
        return Box(-1.,1.,(4,),dtype=np.float32)
    def sample_actions(self):
        import numpy as np
        space=self.action_space
        return np.stack([space.sample() for _ in range(self.num_envs)])
    def close(self): self.env.close()
    def seed(self, seed=0): self.env.seed(seed)
    def setSeed(self, seed=0): self.seed(seed)
    def getNumOfEnvs(self): return self.env.num_envs
    def getObsDim(self): return 12
    def getActDim(self): return 4
    def getExtraInfoNames(self): return []
    def curriculum_callback(self): pass  # Original QuadrotorEnv has no curriculum update.
    def curriculumUpdate(self): self.curriculum_callback()
    def connectUnity(self): raise NotImplementedError('rendering is outside this package')
    def disconnectUnity(self): raise NotImplementedError('rendering is outside this package')


class SingleNumpyEnv(NumpyEnv):
    """Scalar task interface: no vector terminal reward or automatic reset."""
    def __init__(self, env):
        if env.num_envs!=1 or env.auto_reset:
            raise ValueError('single adapter requires one lane and auto_reset=False')
        super().__init__(env)
    def reset(self): return super().reset()[0]
    def step(self, action):
        import numpy as np
        a=np.asarray(action)
        if a.shape!=(4,): raise ValueError('expected action [4]')
        obs,reward,done,info=super().step(a[None,:])
        return obs[0],float(reward[0]),bool(done[0]),info[0]
    def testStep(self, action):
        raise NotImplementedError('testStep is a vector interface; use step')
