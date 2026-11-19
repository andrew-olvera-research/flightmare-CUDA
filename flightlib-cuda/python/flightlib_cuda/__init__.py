"""Resident Flightmare environment. Values may change; bound storage must not resize."""
from contextlib import contextmanager
from enum import IntEnum
import torch
from . import _native
from .sensors import ImuConfig, OdometryConfig, SensorPackets, settings as _sensor_settings
from .transport import Backpressure, Subscription, Transport, step_nanoseconds

__version__ = '0.1.2'
ModelConfig = _native.ModelConfig
TaskConfig = _native.TaskConfig
load_model = _native.load_model
legacy_task_config = _native.legacy_task_config
prepare_schedule = _native.prepare_schedule
CoreOperation = _native.CoreOperation


def ctbr_task_config(model):
    """Normalized actions: zero is hover; +/-1 requests +/- the model rate limits."""
    task = TaskConfig()
    task.ctbr = True
    task.action_mean = [9.81, 0., 0., 0.]
    task.action_scale = [9.81, *model.rate_limit]
    return task


class Operation:
    """Bound optional helper; run writes output in place with no allocations."""
    def __init__(self, core, kind, input, output):
        self._native = _native.NativeOperation(core._native, kind, input, output)
        self.core, self.input, self.output = core, input, output

    def run(self, dt=0, stream=None):
        self._native.run(self.core._stream(stream), dt)
        return self.output


class CommandMode(IntEnum):
    INVALID = 0
    ROTOR_THRUST = 1
    COLLECTIVE_BODY_RATES = 2


class LaneStatus(IntEnum):
    OK = 0
    INACTIVE = 1
    INVALID_COMMAND = 2
    INVALID_STATE = 3
    INVALID_RESET = 4
    NUMERICAL_FAILURE = 5


class _Resident:
    def _stream(self, stream=None):
        stream = torch.cuda.current_stream(self.device) if stream is None else stream
        if stream.device != self.device:
            raise ValueError('stream device mismatch')
        return stream.cuda_stream

    def wait(self, stream=None):
        """Order a consumer stream before accessing borrowed storage."""
        self._native.wait(self._stream(stream))

    def record_usage(self, stream=None):
        """Join external tensor accesses to subsequent simulation work."""
        self._native.record_usage(self._stream(stream))

    @contextmanager
    def access(self, stream=None):
        stream = torch.cuda.current_stream(self.device) if stream is None else stream
        self._stream(stream)
        with torch.cuda.stream(stream):
            self.wait(stream)
            try:
                yield self
            finally:
                self.record_usage(stream)

    def synchronize(self):
        self._native.synchronize()


class Core(_Resident):
    """Batched Flightlib physical state. No task, observation, reward or auto-reset.

    Inputs and returned views are persistent tensors; configure/bind/capture are
    setup operations. Update values in place and never resize bound storage.
    Model/topology/accepted-command views are read-only by API contract.
    """
    def __init__(self, environments, slots=1, control_dt=.01, device=0,
                 models=None, model_ids=None, active=None, bounds=None):
        self.environments, self.slots = environments, slots
        self.device = torch.device('cuda', device)
        n = environments * slots
        models = list(models) if models is not None else [ModelConfig()]
        self._ids = list(model_ids) if model_ids is not None else [0] * n
        self._active = list(active) if active is not None else [1] * n
        self._bounds = list(bounds) if bounds is not None else [v for v in (-100, 100, -100, 100, -100, 100) for _ in range(n)]
        self._native = _native.NativeCore(environments, slots, control_dt, device, models,
                                        self._ids, self._active, self._bounds)
        self.control_dt = control_dt
        def view(name):
            return torch.utils.dlpack.from_dlpack(self._native.tensor(name))
        self.state, self.time = view('state.x'), view('state.time')
        self.motor_omega, self.motor_thrust = view('state.motor_omega'), view('state.motor_thrust')
        self.status = view('diagnostics.status')
        self.position, self.quaternion = self.state[:3], self.state[3:7]
        self.velocity, self.body_rates = self.state[7:10], self.state[10:13]
        self.acceleration, self.torque = self.state[13:16], self.state[16:19]
        self.angular_bias, self.acceleration_bias = self.state[19:22], self.state[22:25]
        self.accepted_command = {k: view('commands.' + k) for k in ('mode', 'time', 'payload')}
        self.parameters = {k: view('models.' + k) for k in (
            'mass', 'inertia', 'inverse_inertia', 'allocation', 'inverse_allocation',
            'thrust_polynomial', 'motor_tau_inverse', 'motor_speed_bounds', 'thrust_bounds',
            'body_rate_limit', 'rate_gain', 'gravity')}
        self.topology = {k: view('topology.' + k) for k in ('model_id', 'active', 'world_bounds')}
        self.substep_dt = view('schedule.substep_dt') if prepare_schedule(control_dt) else None
        with torch.cuda.device(self.device):
            x = torch.zeros((25, n), dtype=torch.float32, device=self.device); x[3].fill_(1)
            t = torch.zeros(n, dtype=torch.float32, device=self.device)
            selected = torch.ones(n, dtype=torch.uint8, device=self.device)
            mode = torch.full((n,), int(CommandMode.ROTOR_THRUST), dtype=torch.uint8, device=self.device)
            payload = torch.zeros((4, n), dtype=torch.float32, device=self.device)
            # Legacy scene metadata, owned separately from frozen physics/query
            # tensors. Neither stepping nor reset synthesizes or consumes it.
            self.size = torch.ones((3, n), dtype=torch.float32, device=self.device)
            self.collision = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.bind_state(selected, x, t)
        self.bind_command(selected.clone(), mode, t.clone(), payload)
        self.wait()

    @property
    def models(self):
        """Copies of accepted host configuration, not mutable device parameters."""
        return self._native.model_configs

    def bind_state(self, selected, state, time):
        self._native.bind_state(selected, state, time)
        self.state_selected, self.state_input, self.time_input = selected, state, time

    def bind_operation(self, kind, input, output):
        """Setup an optional derivative/controller/motor/conversion/clamp helper."""
        return Operation(self, kind, input, output)

    def bind_command(self, selected, mode, time, payload):
        self._native.bind_command(selected, mode, time, payload)
        self.command_selected, self.command_mode = selected, mode
        self.command_time, self.command_payload = time, payload

    def reset(self, stream=None):
        """Apply selected state_input/time_input and clear their motors only."""
        self._native.state(self._stream(stream), True)

    def set_state(self, stream=None):
        """Apply selected state_input/time_input, retaining motors and commands."""
        self._native.state(self._stream(stream), False)

    def set_command(self, stream=None):
        """Accept/clamp selected commands; rejected lanes keep prior commands."""
        self._native.set_command(self._stream(stream))

    def step(self, commands=False, graph=False, stream=None):
        """Advance accepted commands, or accept+step with commands=True.

        commands=True rejects invalid selected lanes without advancing them.
        Unselected active lanes run their previously accepted commands.
        """
        self._native.step(self._stream(stream), commands, graph)

    def capture(self, commands=False):
        if torch.cuda.is_current_stream_capturing():
            raise ValueError('nested framework capture is unsupported')
        self._native.capture(commands)

    def configure(self, control_dt=None, models=None, model_ids=None, active=None, bounds=None):
        """Transactional host setup; capacities fixed, successful changes stale graphs."""
        dt = self.control_dt if control_dt is None else control_dt
        m = self.models if models is None else list(models)
        ids = self._ids if model_ids is None else list(model_ids)
        mask = self._active if active is None else list(active)
        box = self._bounds if bounds is None else list(bounds)
        self._native.configure(m, ids, mask, box, dt)
        self.control_dt, self._ids, self._active, self._bounds = dt, ids, mask, box

    def snapshot(self):
        """Explicit CPU copy of physical state, motors, status and accepted commands."""
        with self.access():
            result = {k: getattr(self, k).cpu() for k in ('state', 'time', 'motor_omega', 'motor_thrust', 'status', 'size', 'collision')}
            result['command'] = {k: v.cpu() for k, v in self.accepted_command.items()}
        return result


class Environment(_Resident):
    def __init__(self, environments, slots=1, control_dt=.01, device=0, seed=0,
                 models=None, model_ids=None, active=None, bounds=None, legacy=False,
                 scene=None, collision_radius=.15, terminate_on_collision=False,
                 agent_collisions=False, candidate_capacity=None, imu=None, odometry=None,
                 subscriptions=(), auto_reset=True):
        if legacy and slots != 1:
            raise ValueError('legacy shapes require slots=1')
        self.environments, self.slots, self.device = environments, slots, torch.device('cuda', device)
        self.legacy = legacy
        n = environments * slots
        self.models = list(models) if models is not None else [ModelConfig()]
        self.model_ids = list(model_ids) if model_ids is not None else [0] * n
        self.active = list(active) if active is not None else [1] * n
        self.bounds = list(bounds) if bounds is not None else [v for v in (-20, 20, -20, 20, 0, 20) for _ in range(n)]
        if agent_collisions:
            if not 2**-150 < collision_radius <= 1e6:
                raise ValueError("moving sphere radius must be positive in float32 and at most 1e6 m")
            if candidate_capacity is None:
                candidate_capacity = max(1, min(8 * slots, slots * (slots - 1) // 2))
            if type(candidate_capacity) is not int or not 0 < candidate_capacity < 2**32-1:
                raise ValueError("candidate_capacity must be a positive uint32 capacity")
        elif candidate_capacity is not None:
            raise ValueError("candidate_capacity requires agent_collisions")
        self._native = _native.NativeEnvironment(environments, slots, control_dt, device, seed,
                                               self.models, self.model_ids, self.active, self.bounds,
                                               candidate_capacity or 0, collision_radius if agent_collisions else 0.,
                                               terminate_on_collision, _sensor_settings(imu, ImuConfig),
                                               _sensor_settings(odometry, OdometryConfig))
        self._native.auto_reset(auto_reset)
        self.imu = SensorPackets(self._native,"imu") if imu is not None else None
        self.odometry = SensorPackets(self._native,"odometry") if odometry is not None else None
        with torch.cuda.device(self.device):
            actions = torch.zeros((environments, 4) if legacy else (environments, slots, 4), dtype=torch.float32, device=self.device)
            selected = torch.ones(n, dtype=torch.uint8, device=self.device)
            state = torch.zeros((25, n), dtype=torch.float32, device=self.device)
            state[3].fill_(1)
            time = torch.zeros(n, dtype=torch.float32, device=self.device)
        self._scene_bindings = ()
        self.collision_hit = self.collision_valid = None
        if scene is not None:
            from .scene import bind_scene
            self._scene_bindings = bind_scene(scene, collision_radius, terminate_on_collision, self.device)
        self.agent_collision_hit = self.candidate_pairs = self.candidate_count = self.candidate_overflow = None
        if agent_collisions:
            self.agent_collision_hit, self.candidate_pairs, self.candidate_count, self.candidate_overflow = (
                torch.utils.dlpack.from_dlpack(self._native.tensor(name)) for name in
                ('swarm.hit', 'candidates.instance_pair', 'candidates.count', 'candidates.overflow'))
        if scene is not None or agent_collisions:
            self.collision_hit, self.collision_valid = (
                torch.utils.dlpack.from_dlpack(self._native.tensor('collisions.' + name))
                for name in ('hit', 'valid'))
        elif terminate_on_collision:
            raise ValueError("collision termination requires a scene or agent_collisions")
        self.bind(actions, selected, state, time)
        self.outputs = {name: torch.utils.dlpack.from_dlpack(self._native.tensor('task_output.' + name))
                        for name in ('observation', 'transition_observation', 'reward', 'done', 'valid')}
        if legacy:
            self.outputs = {k: v.squeeze(1) for k, v in self.outputs.items()}
        self.state = torch.utils.dlpack.from_dlpack(self._native.tensor('state.x'))
        self.time = torch.utils.dlpack.from_dlpack(self._native.tensor('state.time'))
        self.status = torch.utils.dlpack.from_dlpack(self._native.tensor('diagnostics.status'))
        self._transport = Transport(self, subscriptions, control_dt) if subscriptions else None
        self.subscriptions = self._transport.channels if self._transport else ()
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
        self._native.bind(native_actions, selected, reset_state, reset_time, self._scene_bindings)
        self.actions, self.selected, self.reset_state, self.reset_time = actions, selected, reset_state, reset_time

    def reset(self, random=False, stream=None):
        self._native.reset(self._stream(stream), random)
        return self.outputs

    def capture(self, random=True):
        """Setup boundary; drains outstanding simulation work but does not step."""
        if torch.cuda.is_current_stream_capturing():
            raise ValueError('nested framework capture is unsupported')
        self._native.capture(random, any(c.kind == "pose" and not c.closed for c in self.subscriptions))

    def step(self, graph=False, random=True, stream=None):
        """Outputs are borrowed and overwritten by subsequent operations."""
        if self._transport is None:
            self._native.step(self._stream(stream), graph, random)
        else:
            stream = torch.cuda.current_stream(self.device) if stream is None else stream
            self._stream(stream)
            self._transport.step(self._native, stream, graph, random)
        return self.outputs

    def record_step(self, random=True, stream=None):
        """Append a step to an active external capture. See caller-owned graph lifetime rules."""
        if self._transport is not None:
            raise ValueError("external graph recording requires an environment without subscriptions")
        self._native.record_step(self._stream(stream), random)
        return self.outputs

    def record_reset(self, random=False, stream=None):
        """Append a reset using the bound device selection mask at replay time."""
        if self._transport is not None:
            raise ValueError("external graph recording requires an environment without subscriptions")
        self._native.record_reset(self._stream(stream), random)
        return self.outputs

    def close_subscriptions(self):
        for channel in self.subscriptions:
            channel.close()
        self._transport = None

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
        self._check_transport_period(control_dt)
        self._native.configure(self.models, self.model_ids, self.active, self.bounds, control_dt)

    def _check_transport_period(self, dt):
        if self._transport and step_nanoseconds(dt) != self._transport.step_ns:
            raise ValueError("reconstruct subscriptions to change the control period")

    def snapshot(self):
        """Explicit synchronized CPU export; never used by resident step."""
        self.wait()
        return {k: v.cpu() for k, v in self.outputs.items()}

    def synchronize(self):
        self._native.synchronize()
