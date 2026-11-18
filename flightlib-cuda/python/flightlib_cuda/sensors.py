"""Configuration and borrowed device packets for the headless sensor API."""

from dataclasses import dataclass
import math
from numbers import Real
import torch


def _axes(value, size=3):
    values = (
        (float(value),) * size if isinstance(value, Real) else tuple(map(float, value))
    )
    if len(values) != size or not all(map(math.isfinite, values)):
        raise ValueError(f"expected {size} finite sensor components")
    return values


@dataclass(frozen=True)
class _Schedule:
    rate_hz: float = 200.0
    phase_s: float = 0.0
    latency_s: float = 0.0
    seed: int = 0
    dropout: float = 0.0
    mount_wxyz: tuple = (1.0, 0.0, 0.0, 0.0)
    offset_m: tuple = (0.0, 0.0, 0.0)

    def _settings(self):
        if not math.isfinite(self.rate_hz) or self.rate_hz <= 0:
            raise ValueError("sensor rate must be positive and finite")
        if not all(math.isfinite(s) and s >= 0 for s in (self.phase_s, self.latency_s)):
            raise ValueError("sensor phase and latency must be finite and nonnegative")
        period = round(1e9 / self.rate_hz)
        phase, latency = (round(s * 1e9) for s in (self.phase_s, self.latency_s))
        if (
            not 0 < period < 2**62
            or not 0 <= phase <= period
            or not 0 <= latency < 2**62
        ):
            raise ValueError("invalid sensor period, phase or latency")
        if type(self.seed) is not int or not 0 <= self.seed < 2**64:
            raise ValueError("sensor seed must fit uint64")
        mount = _axes(self.mount_wxyz, 4)
        norm = math.sqrt(sum(x * x for x in mount))
        if abs(norm - 1) > 1e-5:
            raise ValueError("mount_wxyz must be a unit quaternion")
        return dict(
            period_ns=period,
            phase_ns=phase or period,
            latency_ns=latency,
            seed=self.seed,
            dropout=self.dropout,
            mount=tuple(x / norm for x in mount),
            offset=_axes(self.offset_m),
            bias=(0.0,) * 6,
            walk=(0.0,) * 6,
            limit=(0.0,) * 6,
            attitude_std=(0.0,) * 3,
            increment_width=(0.0,) * 6,
        )


@dataclass(frozen=True)
class ImuConfig(_Schedule):
    """Body-origin IMU. Noise densities use the Kalibr continuous-time convention.

    Each error parameter accepts a scalar or XYZ triple in sensor axes.
    A positive limit clips that axis in rad/s or m/s².
    """

    gyro_noise_density: float | tuple = 0.0
    accel_noise_density: float | tuple = 0.0
    gyro_random_walk: float | tuple = 0.0
    accel_random_walk: float | tuple = 0.0
    gyro_bias: float | tuple = 0.0
    accel_bias: float | tuple = 0.0
    gyro_limit: float | tuple = 0.0
    accel_limit: float | tuple = 0.0
    gyro_increment_width: float | tuple = (
        0.0  # rad, peak-to-peak cumulative readout error
    )
    accel_increment_width: float | tuple = 0.0  # m/s

    def _settings(self):
        d = super()._settings()
        if any(d["offset"]):
            raise ValueError("IMU requires a body-origin mount")
        d.update(
            noise=_axes(self.gyro_noise_density) + _axes(self.accel_noise_density),
            walk=_axes(self.gyro_random_walk) + _axes(self.accel_random_walk),
            bias=_axes(self.gyro_bias) + _axes(self.accel_bias),
            limit=_axes(self.gyro_limit) + _axes(self.accel_limit),
        )
        d["increment_width"] = _axes(self.gyro_increment_width) + _axes(
            self.accel_increment_width
        )
        return d


@dataclass(frozen=True)
class OdometryConfig(_Schedule):
    """Local-world position, sensor-to-world quaternion and world velocity."""

    rate_hz: float = 50.0
    position_std: float | tuple = 0.0
    velocity_std: float | tuple = 0.0
    attitude_std: float | tuple = 0.0

    def _settings(self):
        d = super()._settings()
        d.update(
            noise=_axes(self.position_std) + _axes(self.velocity_std),
            attitude_std=_axes(self.attitude_std),
        )
        return d


def settings(config, expected):
    if config is None:
        return {}
    if not isinstance(config, expected):
        raise TypeError(f"expected {expected.__name__}")
    return config._settings()


class SensorPackets:
    """Delivered-only tensors, [packet slot, flattened lane, channel].

    sequence == 0 denotes padding. valid == 0 with nonzero sequence is a
    dropped or invalid acquisition. Values and metadata are borrowed views.
    """

    def __init__(self, native, name):
        def view(field):
            return torch.utils.dlpack.from_dlpack(native.tensor(name + "." + field))

        self.values = view("values")
        self.valid = view("valid")
        (
            self.sequence,
            self.episode,
            self.scheduled_ns,
            self.acquired_ns,
            self.available_ns,
            self.delivered_ns,
        ) = view("metadata").view(torch.int64).unbind(0)
