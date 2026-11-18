from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from flightlib_cuda import (
    Backpressure,
    Environment,
    ImuConfig,
    OdometryConfig,
    Subscription,
)


def start(*specs, worlds=1, actors=4, **options):
    env = Environment(worlds, actors, subscriptions=specs, **options)
    env.reset_state[2] = 5
    env.reset()
    env.capture(random=False)
    return env


def step(env, graph=True):
    return env.step(graph=graph, random=False)


@pytest.mark.parametrize("graph", [True, False])
def test_capture_terminal_pose_and_episode_before_partial_autoreset(graph):
    spec = Subscription(
        "pose", ((0, 1), (1, 0)), destination="host", period_ns=15_000_000
    )
    env = start(spec, worlds=2, actors=2)
    baseline = start(worlds=2, actors=2)
    (ch,) = env.subscriptions
    for e in (env, baseline):
        step(e, graph)
        e.state[2, 1] = 0.01
        e.state[0, 1] = 3
        e.reset_state[0, 1] = -4
    assert ch.acquire() is None
    step(env, graph)
    step(baseline, graph)
    for field in env.outputs:
        assert torch.equal(env.outputs[field], baseline.outputs[field])
    assert torch.equal(env.state, baseline.state)
    assert env.outputs["done"].flatten().tolist() == [0, 1, 0, 0]
    with ch.acquire(wait=True) as frame:
        assert (frame.sequence, frame.requested_ns, frame.acquired_ns) == (
            1,
            15_000_000,
            20_000_000,
        )
        assert frame.lane.tolist() == [1, 2] and frame.valid.tolist() == [1, 1]
        assert frame.episode.tolist() == [1, 1]
        assert frame.values[:, 0].tolist() == [3, 0]
        assert frame.values[:, 3:].tolist() == [[1, 0, 0, 0]] * 2
    step(env, graph)
    with ch.acquire(wait=True) as frame:
        assert frame.episode.tolist() == [2, 1] and frame.values[:, 0].tolist() == [
            -4,
            0,
        ]
    assert ch.stats["host_bytes"] == 2 * 2 * 48


def test_slow_viewer_does_not_stall_world_or_sensor_ring():
    env = start(
        Subscription("pose", depth=1),
        Subscription("imu", ((0, 9), (0, 30)), destination="host", depth=2),
        actors=1000,
        imu=ImuConfig(latency_s=0.003),
    )
    viewer, telemetry = env.subscriptions
    reader = torch.cuda.Stream()
    step(env)
    frame = viewer.acquire(reader)
    with torch.cuda.stream(reader):
        saved = frame.records.clone()
    for tick in range(1, 7):
        if tick > 1:
            step(env)
        with telemetry.acquire(wait=True) as samples:
            assert set(samples.lane.tolist()) == {9, 30}
            assert samples.count == (2 if tick == 1 else 4)
            assert samples.delivered_ns.unique().item() == tick * 10_000_000
            assert (samples.available_ns <= samples.delivered_ns).all()
            for lane in (9, 30):
                seqs = samples.packet_sequence[samples.lane == lane].tolist()
                assert seqs == ([1] if tick == 1 else [2 * tick - 2, 2 * tick - 1])
    with torch.cuda.stream(reader):
        assert torch.equal(saved, frame.records)
        # Releasing a lease may precede completion of the submitted reads.
        torch.cuda._sleep(300_000_000)
        done = torch.cuda.Event()
        done.record()
        frame.release()
    step(env)
    env.synchronize()
    assert not done.query() and viewer.acquire() is None
    done.synchronize()
    step(env)
    with viewer.acquire() as latest:
        assert latest.sequence == 8
    assert viewer.stats["dropped"] == 6 and viewer.stats["host_bytes"] == 0
    assert telemetry.stats["host_bytes"] >= 6 * 4 + (2 + 5 * 4) * 80
    env.close_subscriptions()


def test_required_backpressure_is_atomic_and_retry_reuses_sequence():
    env = start(
        Subscription("pose", depth=3), Subscription("pose", depth=1, required=True)
    )
    other, required = env.subscriptions
    step(env)
    held = required.acquire()
    before = env.time.clone()
    with pytest.raises(Backpressure):
        step(env)
    assert torch.equal(before, env.time) and other.stats["published"] == 1
    held.release()
    torch.cuda.current_stream().synchronize()
    step(env)
    with required.acquire() as result:
        assert result.sequence == 2 and result.acquired_ns == 20_000_000
    assert required.stats["backpressure"] == 1 and required.stats["dropped"] == 0


def test_delayed_odometry_wrap_disconnect_and_shutdown():
    env = start(
        Subscription("odometry", destination="host", depth=2),
        Subscription("pose"),
        actors=2,
        odometry=OdometryConfig(rate_hz=50, latency_s=0.01, dropout=1),
    )
    sensor, viewer = env.subscriptions
    step(env)
    outstanding = viewer.acquire()
    viewer.close()
    for tick in range(1, 9):
        if tick == 3:
            env.selected[0] = 0
            env.reset()
        if tick > 1:
            step(env)
        with sensor.acquire(wait=True) as frame:
            assert (frame.valid == 0).all()
            if tick == 3:
                assert frame.lane.tolist() == [0] and frame.episode.tolist() == [1]
            if tick == 5:
                assert sorted(zip(frame.lane.tolist(), frame.episode.tolist())) == [
                    (0, 1),
                    (1, 2),
                ]
        assert viewer.acquire() is None
    outstanding.release()
    step(env)
    env.close_subscriptions()  # queued header copy in flight
    assert sensor.acquire() is None
    step(env)


def test_device_staging_reused_while_host_frame_is_leased():
    env = start(
        Subscription(
            "imu",
            ((0, 1),),
            destination="host",
            depth=2,
            staging_depth=1,
            required=True,
        ),
        imu=ImuConfig(),
    )
    (channel,) = env.subscriptions
    step(env)
    with channel.acquire(wait=True) as first:
        step(env)
        with channel.acquire(wait=True) as second:
            assert first.packet_sequence.tolist() == [1, 2]
            assert second.packet_sequence.tolist() == [3, 4]
            env.close_subscriptions()
            assert first.values.is_pinned() and second.episode.tolist() == [1, 1]


def test_host_worker_disconnect_with_held_frame_and_no_autoreset():
    env = start(Subscription("pose", destination="host"), actors=2)
    (ch,) = env.subscriptions
    env._native.auto_reset(False)
    env.capture(random=False)
    env.state[2, 0] = 0.01
    env.state[3:7, 1] = 0  # finite but has no valid orientation
    step(env)
    with ThreadPoolExecutor(1) as worker:
        frame = worker.submit(ch.acquire, wait=True).result(timeout=5)
        assert frame.valid.tolist() == [1, 0] and env.outputs["done"].flatten()[0]
        assert frame.episode.tolist() == [1, 1] and env.state[2, 0] < 1
        step(env)
        worker.submit(ch.close).result(timeout=5)
        step(env)
        assert frame.sequence == 1 and frame.values[1].count_nonzero() == 0
        worker.submit(frame.release).result(timeout=5)


@pytest.mark.parametrize(
    "spec",
    [
        Subscription("imu"),
        Subscription("pose", ((1, 0),)),
        Subscription("pose", ((0, 0), (0, 0))),
        Subscription("pose", depth=0),
        Subscription("pose", period_ns=1),
    ],
)
def test_invalid_subscription(spec):
    with pytest.raises(ValueError):
        start(spec)
