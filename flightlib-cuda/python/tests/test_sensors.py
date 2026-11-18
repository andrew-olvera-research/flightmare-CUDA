import math
import numpy as np
import pytest
import torch
from flightlib_cuda import Environment, ImuConfig, ModelConfig, OdometryConfig


def hover_model():
    m = ModelConfig()
    m.motor_tau = 1e-5
    return m


@pytest.mark.parametrize("graph", [False, True])
def test_ideal_imu_and_mounted_odometry(graph):
    h = math.sqrt(0.5)
    env = Environment(
        1,
        3,
        models=[hover_model()],
        imu=ImuConfig(mount_wxyz=(h, h, 0, 0)),
        odometry=OdometryConfig(
            rate_hz=100, offset_m=(1, 0, 0), mount_wxyz=(h, 0, 0, h)
        ),
    )
    env.reset_state[2] = 5
    env.reset_state[12, 2] = 2
    env.actions[0, 1].fill_(-1)
    env.reset()
    if graph:
        env.capture(random=False)
    env.step(graph=graph, random=False)
    imu = env.imu
    assert imu.valid[:2].all() and not imu.valid[2:].any()
    np.testing.assert_allclose(
        imu.values[:2, 0].cpu(), [[0, 0, 0, 0, 9.81, 0]] * 2, atol=3e-5
    )
    np.testing.assert_allclose(imu.values[:2, 1].cpu(), 0, atol=1e-6)
    np.testing.assert_allclose(imu.values[:2, 2, 1].cpu(), 2, atol=1e-4)
    od = env.odometry.values[0, 2].cpu().numpy()
    theta = 0.02
    np.testing.assert_allclose(od[:3], [math.cos(theta), math.sin(theta), 5], atol=1e-5)
    np.testing.assert_allclose(
        od[3:7],
        [math.cos(theta / 2 + math.pi / 4), 0, 0, math.sin(theta / 2 + math.pi / 4)],
        atol=1e-5,
    )
    np.testing.assert_allclose(
        od[7:], [-2 * math.sin(theta), 2 * math.cos(theta), 0], atol=1e-4
    )


def test_noncommensurate_clocks_and_delay():
    env = Environment(
        1,
        1,
        imu=ImuConfig(rate_hz=173, phase_s=0.001, latency_s=0.013),
        odometry=OdometryConfig(rate_hz=37, latency_s=0.009),
    )
    env.reset_state[2] = 5
    env.reset()
    env.capture()
    rows = [[], []]
    for step in range(1, 61):
        env.step(graph=True)
        for j, p in enumerate((env.imu, env.odometry)):
            for row in (
                torch.stack(
                    (
                        p.sequence,
                        p.scheduled_ns,
                        p.acquired_ns,
                        p.available_ns,
                        p.delivered_ns,
                        p.valid,
                    ),
                    -1,
                )[:, 0]
                .cpu()
                .tolist()
            ):
                if row[0]:
                    assert row[-1] and row[4] == step * 10_000_000
                    rows[j].append(row[:5])
    for got, rate, phase, delay in zip(
        rows, (173, 37), (1_000_000, None), (13_000_000, 9_000_000)
    ):
        period = round(1e9 / rate)
        expected = []
        for i in range(200):
            deadline = (phase or period) + i * period
            acquired = math.ceil(deadline / 2_500_000) * 2_500_000
            delivered = math.ceil((acquired + delay) / 10_000_000) * 10_000_000
            if delivered > 600_000_000:
                break
            expected.append([i + 1, deadline, acquired, acquired + delay, delivered])
        assert got == expected


def test_delayed_reset_terminal_and_invalid_lanes():
    env = Environment(
        1,
        3,
        active=[1, 1, 0],
        scene="room",
        terminate_on_collision=True,
        imu=ImuConfig(rate_hz=100),
        odometry=OdometryConfig(rate_hz=100, latency_s=0.015),
    )
    env.reset_state[2] = 5
    env.reset_state[0, 0] = 16
    env.reset()
    env.reset_state[0, 0] = 0
    env.capture(random=False)
    out = env.step(graph=True, random=False)
    assert out["done"].tolist() == [[1, 0, 0]]
    assert env.imu.episode[0].tolist() == [1, 1, 0]
    assert env.imu.valid[0].tolist() == [1, 1, 0]
    for i in range(2):
        env.step(graph=True, random=False)
    assert env.odometry.sequence[0].tolist() == [0, 1, 0]
    env.step(graph=True, random=False)
    assert env.odometry.episode[0].tolist() == [2, 1, 0]
    assert env.odometry.acquired_ns[0].tolist() == [20_000_000, 20_000_000, 0]
    env.selected.zero_()
    env.selected[0] = 1
    env.reset()
    assert env.odometry.sequence[0].tolist() == [0, 2, 0]
    env.actions[0, 1, 0] = float("nan")
    env.step(graph=True, random=False)
    assert env.imu.valid[0].tolist() == [1, 0, 0]


def test_bounds_correction_masks_endpoint_only():
    env = Environment(1, 1, imu=ImuConfig(rate_hz=400))
    env.reset_state[2] = 19
    env.reset_state[9] = 200
    env.reset()
    env.step()
    assert env.imu.valid[:4, 0].tolist() == [1, 1, 1, 0]
    assert env.imu.sequence[:4, 0].tolist() == [1, 2, 3, 4]


def test_noise_walk_dropout_and_replay():
    slots = 8192
    for graph in (False, True):
        env = Environment(
            1,
            slots,
            control_dt=0.005,
            imu=ImuConfig(
                seed=71,
                rate_hz=200,
                gyro_noise_density=(0.01, 0, 0),
                gyro_random_walk=(0, 0.3, 0),
                gyro_bias=(0.2, 0, 2),
                gyro_limit=(0, 0, 1),
                dropout=0.12,
            ),
        )
        env.reset_state[2] = 5
        env.reset()
        if graph:
            env.capture()
        env.step(graph=graph)
        first = env.imu.values[0, :, :3].cpu().numpy()
        valid = env.imu.valid[0].cpu().numpy().astype(bool)
        env.step(graph=graph)
        second = env.imu.values[0, :, :3].cpu().numpy()
        both = valid & env.imu.valid[0].cpu().numpy().astype(bool)
        assert 0.10 < 1 - valid.mean() < 0.14
        assert np.all(first[~valid] == 0) and np.all(first[valid, 2] == 1)
        assert first[valid, 0].mean() == pytest.approx(0.2, abs=0.006)
        assert first[valid, 0].std() == pytest.approx(0.01 * math.sqrt(200), rel=0.045)
        assert (second[both, 1] - first[both, 1]).std() == pytest.approx(
            0.3 / math.sqrt(200), rel=0.045
        )
        if graph:
            np.testing.assert_array_equal(first, previous)
            np.testing.assert_array_equal(second, previous_second)
        previous, previous_second = first, second


def test_stream_safe_borrow_and_invalid_configuration():
    env = Environment(1, 1, imu=ImuConfig())
    env.reset_state[2] = 5
    env.reset()
    ptr = env.imu.values.data_ptr()
    other = torch.cuda.Stream()
    env.capture()
    env.step(graph=True)
    with env.access(other):
        saved = env.imu.sequence.clone()
    env.step(graph=True)
    other.synchronize()
    assert (
        saved[0].item() == 1
        and env.imu.sequence[0].item() == 3
        and ptr == env.imu.values.data_ptr()
    )
    for cfg in [
        ImuConfig(rate_hz=1000),
        ImuConfig(offset_m=(1, 0, 0)),
        ImuConfig(dropout=1.1),
        ImuConfig(gyro_random_walk=-1),
        ImuConfig(mount_wxyz=(0, 0, 0, 0)),
    ]:
        with pytest.raises(ValueError):
            Environment(1, imu=cfg)
    with pytest.raises(ValueError):
        env.configure(0.009)


def test_readout_increments_and_odometry_dropout():
    n = 8192
    dt = 0.005
    width = 0.0001
    env = Environment(
        1,
        n,
        control_dt=dt,
        imu=ImuConfig(gyro_increment_width=(width, 0, 0)),
        odometry=OdometryConfig(rate_hz=200, dropout=1),
    )
    env.reset_state[2] = 5
    env.reset()
    env.capture()
    env.step(graph=True)
    first = env.imu.values[0, :, 0].clone()
    env.step(graph=True)
    second = env.imu.values[0, :, 0]
    assert first.std().item() == pytest.approx(width / (math.sqrt(6) * dt), rel=0.05)
    assert torch.corrcoef(torch.stack((first, second)))[0, 1].item() == pytest.approx(
        -0.5, abs=0.04
    )
    assert torch.all(env.odometry.sequence[0] == 2) and not env.odometry.valid.any()
    assert not env.odometry.values.any()


def test_intermediate_truth_and_reactivation_ids():
    env = Environment(1, 1, odometry=OdometryConfig(rate_hz=400))
    env.reset_state[2] = 5
    env.reset_state[7] = 2
    env.reset()
    env.step()
    np.testing.assert_allclose(
        env.odometry.values[:4, 0, 0].cpu(), [0.005, 0.010, 0.015, 0.020], atol=1e-6
    )
    env.active[0] = 0
    env.configure(0.01)
    env.step()
    assert not env.odometry.sequence.any()
    env.active[0] = 1
    env.configure(0.01)
    env.step()
    assert env.odometry.sequence[:4, 0].tolist() == [9, 10, 11, 12]


def test_swarm_contact_preserves_delivered_terminal_pose():
    env = Environment(
        1, 2, agent_collisions=True, terminate_on_collision=True,
        imu=ImuConfig(), odometry=OdometryConfig(rate_hz=100),
    )
    env.reset_state[2] = 5
    env.reset()
    # Both agents are coincident this step. Autoreset will put them apart.
    env.reset_state[0, 1] = 4
    env.capture(random=False)
    env.step(graph=True, random=False)
    assert env.outputs["done"].all() and env.agent_collision_hit.bool().all()
    assert env.odometry.valid[0].all() and env.imu.valid[:2].all()
    assert env.odometry.values[0, :, 0].tolist() == [0, 0]
    episode = env.odometry.episode[0].clone()
    env.step(graph=True, random=False)
    assert not env.outputs["done"].any()
    assert torch.all(env.odometry.episode[0] == episode + 1)
    assert env.odometry.values[0, :, 0].tolist() == [0, 4]
