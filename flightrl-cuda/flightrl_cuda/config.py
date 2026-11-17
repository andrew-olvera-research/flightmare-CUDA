from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
import yaml
from flightlib_cuda import ModelConfig, legacy_task_config


def defaults():
    return yaml.safe_load(files('flightrl_cuda').joinpath('defaults.yaml').read_text())


def model_from(data):
    model = ModelConfig()
    for source, target in dict(mass='mass', arm_l='arm_length', motor_omega_min='motor_min',
                               motor_omega_max='motor_max', motor_tau='motor_tau',
                               thrust_map='thrust_map', kappa='kappa', omega_max='rate_limit').items():
        setattr(model, target, data[source])
    return model


@dataclass
class Config:
    num_envs: int = 100
    slots: int = 1
    seed: int = 1
    device: int = 0
    control_dt: float = .02
    max_t: float = 5.0  # metadata, not a time-limit wrapper
    model: object = field(default_factory=lambda: model_from(defaults()['quadrotor_dynamics']))
    reward_coeff: list = field(default_factory=lambda: list(defaults()['rl'].values()))

    def copy(self):
        """Own a snapshot, including the pybind model and mutable coefficients."""
        from dataclasses import replace
        model=ModelConfig()
        for key,value in self.metadata()['model'].items(): setattr(model,key,value)
        return replace(self,model=model,reward_coeff=list(self.reward_coeff))

    @classmethod
    def from_yaml(cls, vector=None, task=None, vector_text=None, task_text=None, **overrides):
        data = defaults()
        for path, text in ((vector, vector_text), (task, task_text)):
            if path is not None and text is not None:
                raise ValueError('provide a path or YAML text, not both')
            incoming = yaml.safe_load(Path(path).read_text() if path is not None else text or '') or {}
            for section, fields in incoming.items():
                if section not in data or not isinstance(fields, dict):
                    raise ValueError('unsupported configuration section: '+section)
                for key, value in fields.items():
                    if key not in data[section]: raise ValueError('unsupported configuration key: '+key)
                    data[section][key] = value
        if data['env']['render'] or data['quadrotor_env']['camera']:
            raise ValueError('rendering and sensors are not supported')
        # add_camera is unused by the original headless constructor; retain it as inert metadata.
        config = dict(num_envs=data['env']['num_envs'], seed=data['env']['seed'],
                      control_dt=data['quadrotor_env']['sim_dt'], max_t=data['quadrotor_env']['max_t'],
                      model=model_from(data['quadrotor_dynamics']),
                      reward_coeff=[data['rl'][k] for k in defaults()['rl']])
        config.update(overrides)
        return cls(**config)

    def task(self):
        task = legacy_task_config(self.model.mass)
        task.reward_coeff = self.reward_coeff
        return task

    def metadata(self):
        return dict(num_envs=self.num_envs, slots=self.slots, seed=self.seed, device=self.device,
                    control_dt=self.control_dt, max_t=self.max_t, reward_coeff=list(self.reward_coeff),
                    model={k: getattr(self.model, k) for k in ('mass', 'arm_length', 'motor_min', 'motor_max',
                           'motor_tau', 'kappa', 'thrust_map', 'rate_limit')})
