"""Resident Flightmare environment. Values may change; bound storage must not resize."""
from contextlib import contextmanager
import torch
from . import _native

__version__ = '0.1.0'
ModelConfig = _native.ModelConfig
TaskConfig = _native.TaskConfig
load_model = _native.load_model
legacy_task_config = _native.legacy_task_config


class Environment:
    def __init__(self, environments, slots=1, control_dt=.01, device=0, seed=0,
                 models=None, model_ids=None, active=None, bounds=None, legacy=False):
        if legacy and slots != 1:
            raise ValueError('legacy shapes require slots=1')
        self.environments, self.slots, self.device = environments, slots, torch.device('cuda', device)
        self.legacy = legacy
        n = environments * slots
        self.models = list(models) if models is not None else [ModelConfig()]
        self.model_ids = list(model_ids) if model_ids is not None else [0] * n
        self.active = list(active) if active is not None else [1] * n
        self.bounds = list(bounds) if bounds is not None else [v for v in (-20, 20, -20, 20, 0, 20) for _ in range(n)]
        self._native = _native.NativeEnvironment(environments, slots, control_dt, device, seed,
                                               self.models, self.model_ids, self.active, self.bounds)
        with torch.cuda.device(self.device):
            actions = torch.zeros((environments, 4) if legacy else (environments, slots, 4), dtype=torch.float32, device=self.device)
            selected = torch.ones(n, dtype=torch.uint8, device=self.device)
            state = torch.zeros((25, n), dtype=torch.float32, device=self.device)
            state[3].fill_(1)
            time = torch.zeros(n, dtype=torch.float32, device=self.device)
        self.bind(actions, selected, state, time)
        self.outputs = {name: torch.utils.dlpack.from_dlpack(self._native.tensor('task_output.' + name))
                        for name in ('observation', 'transition_observation', 'reward', 'done', 'valid')}
        if legacy:
            self.outputs = {k: v.squeeze(1) for k, v in self.outputs.items()}
        self.state = torch.utils.dlpack.from_dlpack(self._native.tensor('state.x'))
        self.time = torch.utils.dlpack.from_dlpack(self._native.tensor('state.time'))
        self.status = torch.utils.dlpack.from_dlpack(self._native.tensor('diagnostics.status'))
        self.wait()

    def _stream(self, stream=None):
        stream = torch.cuda.current_stream(self.device) if stream is None else stream
        if stream.device != self.device:
            raise ValueError('stream device mismatch')
        return stream.cuda_stream

    def bind(self, actions, selected, reset_state, reset_time):
        if self.legacy:
            if tuple(actions.shape) != (self.environments, 4) or not actions.is_contiguous():
                raise ValueError('legacy actions must be contiguous [E,4]')
            native_actions = actions.view(self.environments, 1, 4)
        else:
            native_actions = actions
        self._native.bind(native_actions, selected, reset_state, reset_time)
        self.actions, self.selected, self.reset_state, self.reset_time = actions, selected, reset_state, reset_time

    def reset(self, random=False, stream=None):
        self._native.reset(self._stream(stream), random)
        return self.outputs

    def capture(self, random=True):
        """Setup boundary; drains outstanding simulation work but does not step."""
        if torch.cuda.is_current_stream_capturing():
            raise ValueError('nested framework capture is unsupported')
        self._native.capture(random)

    def step(self, graph=False, random=True, stream=None):
        """Outputs are borrowed and overwritten by subsequent operations."""
        self._native.step(self._stream(stream), graph, random)
        return self.outputs

    def wait(self, stream=None):
        """Order a different consumer stream before reading borrowed outputs."""
        self._native.wait(self._stream(stream))

    def record_usage(self, stream=None):
        """After enqueuing external reads/writes, join their lifetime to the owner."""
        self._native.record_usage(self._stream(stream))

    @contextmanager
    def access(self, stream=None):
        """Order and retain external tensor use on a consumer/producer stream."""
        stream = torch.cuda.current_stream(self.device) if stream is None else stream
        self._stream(stream)  # reject a wrong-device stream before entering it
        with torch.cuda.stream(stream):
            self.wait(stream)
            try:
                yield self
            finally:
                self.record_usage(stream)

    def configure_task(self, configs):
        self._native.configure_task([configs] if isinstance(configs, TaskConfig) else configs)

    def configure(self, control_dt):
        """Apply host model/topology metadata; schedule capacity must stay fixed."""
        self._native.configure(self.models, self.model_ids, self.active, self.bounds, control_dt)

    def snapshot(self):
        """Explicit synchronized CPU export; never used by resident step."""
        self.wait()
        return {k: v.cpu() for k, v in self.outputs.items()}

    def synchronize(self):
        self._native.synchronize()
