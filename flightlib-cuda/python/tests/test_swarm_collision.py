import numpy as np
import pytest
import torch

from flightlib_cuda import Environment, ModelConfig


@pytest.mark.parametrize("graph", [False, True])
def test_crossing_and_reset(graph):
    env = Environment(
        1,
        3,
        control_dt=0.0025,
        agent_collisions=True,
        collision_radius=0.125,
        terminate_on_collision=True,
    )
    env.reset_state[0] = torch.tensor([-1.0, 1.0, 0.0])
    env.reset_state[2] = 4
    env.reset_state[1, 2] = 1
    env.reset_state[7] = torch.tensor([800.0, -800.0, 0.0])
    env.reset()
    env.reset_state[0] = torch.tensor([-5.0, 5.0, 0.0])
    env.reset_state[7] = 0
    if graph:
        env.capture(random=False)
    out = env.step(graph=graph, random=False)
    assert env.collision_valid.tolist() == [1, 1, 1]
    assert out["done"].tolist() == [[1, 1, 0]]
    assert env.agent_collision_hit.tolist() == [1, 1, 0]
    assert env.candidate_count.tolist() == [1]
    assert env.candidate_pairs[0, 0].tolist() == [0, 1]
    assert out["transition_observation"][0, 0, 0] > 0.9
    assert out["observation"][0, 0, 0] == -5
    env.step(graph=graph, random=False)
    assert env.candidate_count.tolist() == [0]
    assert env.collision_hit.tolist() == [0, 0, 0]


def test_temporal_near_miss_and_tangency():
    env = Environment(
        3, 2, control_dt=2**-9, agent_collisions=True, collision_radius=0.125
    )
    env.reset_state[2] = 4
    # Both paths go through (0,0), at different times. The next two
    # environments distinguish tangency from its adjacent float32 miss.
    env.reset_state[0] = torch.tensor([-1.0, 0.0, -1.0, 1.0, -1.0, 1.0])
    env.reset_state[1] = torch.tensor(
        [0.0, -1.75, 0.0, 0.25, 0.0, np.nextafter(np.float32(0.25), np.float32(1))]
    )
    env.reset_state[7] = torch.tensor([1024.0, 0.0, 1024.0, -1024.0, 1024.0, -1024.0])
    env.reset_state[8, 1] = 1024
    env.reset()
    env.step()
    assert env.collision_hit.tolist() == [0, 0, 1, 1, 0, 0]
    assert env.collision_valid.all()


def test_curved_relative_motion():
    dt = 1.0 / 512
    model = ModelConfig()
    model.motor_tau = 0.0001
    env = Environment(
        1,
        2,
        control_dt=dt,
        models=[model],
        agent_collisions=True,
        collision_radius=0.125,
    )
    env.reset_state[2] = 4
    env.reset_state[3, 1] = 0
    env.reset_state[4, 1] = 1  # inverted thrust on the other sphere
    env.actions.fill_(1)
    env.reset()
    env.step()
    accel = float(env.state[15, 0] - env.state[15, 1])
    assert accel > 1
    env.reset_state[2, 0] = 4.25 + accel * dt**2 / 16
    env.reset_state[9, 0] = -accel * dt / 2
    env.reset()
    assert env.state[2, 0] - env.state[2, 1] > 0.25
    env.step()
    assert env.state[2, 0] - env.state[2, 1] > 0.25
    assert env.agent_collision_hit.tolist() == [1, 1]


def test_random_curved_pairs_against_polynomial_reference():
    worlds, dt, radius = 384, 2.0**-9, 1.0e-5
    rng = np.random.default_rng(570)
    model = ModelConfig()
    model.motor_tau = 0.0001
    env = Environment(
        worlds,
        2,
        models=[model],
        control_dt=dt,
        agent_collisions=True,
        collision_radius=radius,
    )
    start = (rng.uniform(-3, 3, (3, worlds * 2)) * radius).astype("f")
    start[2] += 4
    velocity = rng.uniform(-0.035, 0.035, (3, worlds * 2)).astype("f")
    quat = rng.normal(size=(4, worlds * 2))
    quat /= np.linalg.norm(quat, axis=0)
    env.reset_state[:3].copy_(torch.from_numpy(start))
    env.reset_state[3:7].copy_(torch.from_numpy(quat))
    env.reset_state[7:10].copy_(torch.from_numpy(velocity))
    env.actions.fill_(1)
    env.reset()
    env.step()

    def relative(x):
        p = x.T.astype(float)
        return p[::2] - p[1::2]

    a = relative(env.state[13:16].cpu().numpy()) * (dt**2 / 2)
    c = relative(start)
    end = relative(env.state[:3].cpu().numpy())
    b = end - c - a
    expected = []
    interior = 0
    for curve, slope, origin, stop in zip(a, b, c, end):
        # d ||origin + slope*t + curve*t²||² / dt, divided by two.
        roots = np.polynomial.polynomial.polyroots(
            [
                origin @ slope,
                slope @ slope + 2 * origin @ curve,
                3 * slope @ curve,
                2 * curve @ curve,
            ]
        )
        times = [
            0.0,
            1.0,
            *(r.real for r in roots if abs(r.imag) < 1e-10 and 0 < r.real < 1),
        ]
        hit = min(
            np.linalg.norm(origin + t * (slope + t * curve)) for t in times
        ) <= 2 * np.float32(radius)
        expected.append(hit)
        interior += int(
            hit and min(np.linalg.norm(origin), np.linalg.norm(stop)) > 2 * radius
        )
    assert interior > 0 and env.collision_valid.all()
    np.testing.assert_array_equal(env.agent_collision_hit.cpu(), np.repeat(expected, 2))


def test_dense_overflow_isolation_and_recovery():
    env = Environment(
        2,
        4,
        control_dt=0,
        agent_collisions=True,
        candidate_capacity=2,
        active=[1, 1, 1, 1, 1, 0, 1, 1],
    )
    env.reset_state[2] = 4
    env.reset_state[0, 6:] = torch.tensor([3.0, 3.0])
    env.reset()
    env.capture(random=False)
    out = env.step(graph=True, random=False)
    assert env.candidate_count.tolist() == [
        2,
        1,
    ] and env.candidate_overflow.tolist() == [1, 0]
    assert env.collision_valid.tolist() == [0, 0, 0, 0, 1, 0, 1, 1]
    assert out["valid"].flatten().tolist() == env.collision_valid.tolist()
    assert env.agent_collision_hit.tolist() == [0, 0, 0, 0, 0, 0, 1, 1]
    env.reset_state[0, :4] = torch.tensor([0.0, 1.0, 2.0, 3.0])
    env.reset()
    env.step(graph=True, random=False)
    assert env.candidate_overflow.tolist() == [
        0,
        0,
    ] and env.candidate_count.tolist() == [0, 1]
    env.actions[0, 1, 0] = float("nan")
    env.step(graph=True, random=False)
    assert env.collision_valid.tolist() == [0, 0, 0, 0, 1, 0, 1, 1]
    assert env.candidate_overflow.tolist() == [0, 0]


def test_course_and_peer_union_and_bounds_jump():
    env = Environment(1, 4, control_dt=0.0025, scene="split_s", agent_collisions=True)
    env.reset_state[:3].copy_(
        torch.tensor(
            [[-0.1, 2.0, 2.0, 2.0], [-1.27, 4.0, 4.0, 4.0], [3.48, 4.0, 4.0, 19.0]]
        )
    )
    env.reset_state[9, 3] = (
        1500  # projects to the floor without flying through the other two
    )
    env.reset()
    env.step()
    assert env.collision_valid.all()
    assert env.agent_collision_hit.tolist() == [0, 1, 1, 0]
    assert env.collision_hit.tolist() == [1, 1, 1, 1]


def test_exact_capacity_and_partial_reset_jump():
    env = Environment(1, 3, control_dt=0, agent_collisions=True, candidate_capacity=1)
    env.reset_state[0] = torch.tensor([0.0, 0.0, 1.0])
    env.reset_state[2] = 4
    env.reset()
    env.step()
    assert env.candidate_count.item() == 1 and not env.candidate_overflow.item()
    env.selected.zero_()
    env.selected[0] = 1
    env.reset_state[0, 0] = 2
    env.reset()
    assert env.agent_collision_hit.tolist() == [0, 1, 0]
    env.step()
    assert env.collision_valid.all() and env.candidate_count.item() == 0
    assert env.agent_collision_hit.tolist() == [0, 0, 0]
    with pytest.raises(ValueError, match="simultaneous"):
        env._native.test_step(torch.cuda.current_stream(env.device).cuda_stream, False)


@pytest.mark.parametrize("graph", [False, True])
def test_all_pairs_reference_across_sorted_runs(graph):
    rng = np.random.default_rng(19)
    slots = 1031
    env = Environment(
        2,
        slots,
        control_dt=0.0025,
        agent_collisions=True,
        collision_radius=0.125,
        candidate_capacity=slots * (slots - 1) // 2,
    )
    n = 2 * slots
    start = rng.uniform(-1, 1, (n, 3)).astype("f")
    start[:, 2] += 4
    env.reset_state[:3].copy_(torch.from_numpy(start.T))
    env.reset_state[7:10].copy_(
        torch.from_numpy(rng.uniform(-200, 200, (3, n)).astype("f"))
    )
    env.reset()
    if graph:
        env.capture()
    env.step(graph=graph)
    end = env.state[:3].cpu().numpy().T.astype(float)
    # Identical acceleration on both bodies makes each relative arc linear.
    assert (env.state[13:16] == env.state[13:16, :1]).all()
    for world in range(2):
        base = world * slots
        a, b = np.triu_indices(slots, 1)
        a += base
        b += base
        r = start[a].astype(float) - start[b]
        delta = end[a] - end[b] - r
        t = np.clip(
            -np.sum(r * delta, axis=1) / np.maximum(np.sum(delta**2, axis=1), 1e-30),
            0,
            1,
        )
        hit = np.sum((r + t[:, None] * delta) ** 2, axis=1) <= 0.25**2
        confirmed = set(zip(a[hit], b[hit]))
        count = int(env.candidate_count[world])
        candidates = set(map(tuple, env.candidate_pairs[world, :count].cpu().tolist()))
        assert len(candidates) == count and confirmed.issubset(candidates)
        assert all(base <= i < j < base + slots for i, j in candidates)
        expected = set(a[hit]) | set(b[hit])
        assert env.collision_hit[base : base + slots].tolist() == [
            i in expected for i in range(base, base + slots)
        ]
    assert env.collision_valid.all()
