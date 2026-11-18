"""A batched virtual range measurement and planar yaw estimator using delivered packets."""

import torch
from flightlib_cuda import ImuConfig, OdometryConfig
from . import Config, TensorEnv


def range_to_beacon(odometry, beacon):
    """Custom sensor using delayed local position. Inherits odometry timing and validity."""
    distance = torch.linalg.vector_norm(odometry.values[..., :3] - beacon, dim=-1)
    return torch.where(odometry.valid.bool(), distance, torch.nan)


class PlanarGyro:
    """Integrate delivered yaw rate on a level vehicle, resetting on episode changes."""

    def __init__(self, lanes, device):
        self.yaw = torch.zeros(lanes, device=device)
        self.time = torch.zeros(lanes, device=device, dtype=torch.int64)
        self.episode = torch.full_like(self.time, -1)

    def consume(self, imu):
        for slot in range(len(imu.values)):
            valid = imu.valid[slot].bool()
            same = imu.episode[slot] == self.episode
            dt = (imu.acquired_ns[slot] - self.time) * 1e-9
            integrated = self.yaw + imu.values[slot, :, 2] * dt
            self.yaw = torch.where(valid, torch.where(same, integrated, 0.0), self.yaw)
            self.time = torch.where(valid, imu.acquired_ns[slot], self.time)
            self.episode = torch.where(valid, imu.episode[slot], self.episode)


def main():
    env = TensorEnv(
        Config(num_envs=2, slots=16),
        random_reset=False,
        imu=ImuConfig(rate_hz=200, latency_s=0.003),
        odometry=OdometryConfig(rate_hz=50, latency_s=0.01),
    )
    try:
        env.engine.reset_state[2] = 5
        env.engine.reset_state[12] = 0.5
        env.reset()
        env.capture()
        estimator = PlanarGyro(env.num_envs, env.device)
        beacon = torch.tensor([2.0, 0.0, 5.0], device=env.device)
        seen = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        for _ in range(12):
            env.step(env.actions)
            estimator.consume(env.imu)
            seen |= torch.isfinite(range_to_beacon(env.odometry, beacon)).any(dim=0)
        assert seen.all() and (estimator.yaw > 0).all()
        print(
            "Validated delivered IMU integration and virtual beacon range across 32 lanes."
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
