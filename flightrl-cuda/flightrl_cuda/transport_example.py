"""A slow GPU viewer and a small pinned-host IMU subscription in one swarm."""

import torch
from flightlib_cuda import ImuConfig, Subscription
from . import Config, TensorEnv


def main():
    env = TensorEnv(
        Config(num_envs=1, slots=1000, control_dt=0.01),
        random_reset=False,
        imu=ImuConfig(rate_hz=200),
        subscriptions=(
            Subscription("pose", depth=1),
            Subscription("imu", ((0, 7), (0, 12)), destination="host"),
        ),
    )
    viewer, telemetry = env.subscriptions
    render = torch.cuda.Stream(device=env.device)
    try:
        env.engine.reset_state[2] = 5
        env.reset()
        env.capture()
        env.step()
        with torch.cuda.stream(render), viewer.acquire(render) as poses:
            first = poses.values.clone()
        # Hold the next snapshot across simulation steps while consuming host IMU.
        render.synchronize()
        with telemetry.acquire(wait=True) as imu:
            assert imu.count == 4
        env.step()
        frame = viewer.acquire(render)
        try:
            for _ in range(4):
                with telemetry.acquire(wait=True) as imu:
                    assert set(imu.lane.tolist()) == {7, 12}
                env.step()
            with torch.cuda.stream(render):
                assert frame.values.shape == first.shape == (1000, 7)
                assert torch.isfinite(frame.values).all()
        finally:
            frame.release()
        assert viewer.stats["dropped"] == 4
        print(
            "Collected IMU from two actors while a viewer held a 1,000-actor pose frame."
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
