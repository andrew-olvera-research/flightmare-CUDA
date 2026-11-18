"""Continuous-contact regressions through the public CUDA environment."""

import json
import math
from importlib.resources import files

import numpy as np
import pytest
import torch

from flightlib_cuda import Environment
from flightlib_cuda.scene import load_boxes


def single_box(tmp_path, center, size, yaw=0):
    path = tmp_path / "obstacle.flightcourse"
    path.write_text(
        json.dumps(
            {
                "schema": "flightmare.boxes.v1",
                "assets": [
                    {"name": "obstacle", "boxes": [{"size": size, "center": [0, 0, 0]}]}
                ],
                "instances": [
                    {"asset": "obstacle", "position": center, "yaw_deg": yaw}
                ],
            }
        )
    )
    return path


def test_degenerate_float32_geometry_is_rejected(tmp_path):
    path = single_box(tmp_path, [0, 0, 1], [2.0**-149, 1, 1])
    with pytest.raises(ValueError, match="dimensions"):
        load_boxes(path.read_bytes())
    with pytest.raises(ValueError, match="radius"):
        Environment(1, scene="room", collision_radius=2.0**-151)


@pytest.mark.parametrize("graph", [False, True])
def test_arc_apex_between_physics_samples(tmp_path, graph):
    # With motors off both RK endpoints are below 2.000001 m. The center
    # reaches about 2.000007 m inside the substep. The second lane stays clear.
    path = single_box(tmp_path, [0, 0, 2.000006], [1, 1, 0.000001])
    env = Environment(
        1,
        2,
        control_dt=0.0025,
        scene=path,
        collision_radius=0.000001,
        terminate_on_collision=True,
    )
    env.reset_state[2] = 2
    env.reset_state[9, 0] = 0.012
    env.actions.fill_(-1)
    env.reset()
    env.reset_state[2] = 3
    if graph:
        env.capture(random=False)
    out = env.step(graph=graph, random=False)
    assert env.collision_valid.tolist() == [1, 1]
    assert env.collision_hit.tolist() == [1, 0]
    assert out["done"].tolist() == [[1, 0]]
    assert out["transition_observation"][0, 0, 2] < 2.000001
    assert out["observation"][0, 0, 2] == 3
    env.step(graph=graph, random=False)
    assert env.collision_hit.tolist() == [0, 0]


def test_rounded_corner_tangent_and_ulp_neighbors(tmp_path):
    # In XY the closest point is the box corner (.25,.25). The lanes pass
    # vertically through its Z span entirely between two physics samples.
    r = 0.15625
    path = single_box(tmp_path, [0, 0, 3], [0.5, 0.5, 0.001])
    env = Environment(1, 4, control_dt=0.0025, scene=path, collision_radius=r)
    tangent = np.float32(0.25 + 0.09375)
    env.reset_state[0].copy_(
        torch.tensor(
            [
                tangent,
                np.nextafter(tangent, np.float32(1)),
                np.nextafter(tangent, np.float32(0)),
                0.39,
            ]
        )
    )
    env.reset_state[1] = 0.25 + 0.125
    env.reset_state[2] = 2
    env.reset_state[9] = 800
    env.reset()
    env.step()
    assert env.collision_valid.tolist() == [1] * 4
    assert env.collision_hit.tolist() == [1, 0, 1, 0]


def test_bounds_correction_is_discontinuous(tmp_path):
    # Exceeding the upper Z world bound projects Z to zero. The moving
    # sphere crosses the box on lane one. Only the artificial jump crosses
    # the box on lane zero.
    path = single_box(tmp_path, [0, 0, 8], [2, 2, 0.01])
    env = Environment(1, 2, control_dt=0.0025, scene=path, collision_radius=0.02)
    env.reset_state[2].copy_(torch.tensor([19.0, 7.0]))
    env.reset_state[9] = 6000
    env.reset()
    out = env.step(random=False)
    assert out["transition_observation"][0, :, 2].tolist() == [0, 0]
    assert env.collision_valid.tolist() == [1, 1]
    assert env.collision_hit.tolist() == [0, 1]


def test_isolated_tangency_inside_interval(tmp_path):
    # Closest approach is strictly between face crossings. The normal at the
    # rounded corner is (3,4)/5 and velocity is perpendicular to that normal.
    path = single_box(tmp_path, [0, 0, 3], [0.5, 0.5, 1])
    env = Environment(1, 3, control_dt=2.0**-9, scene=path, collision_radius=0.15625)
    x = np.float32(0.25 + 0.09375 - 0.25 * 0.375)
    env.reset_state[0].copy_(
        torch.from_numpy(
            np.array(
                [np.nextafter(x, np.float32(0)), x, np.nextafter(x, np.float32(1))]
            )
        )
    )
    env.reset_state[1] = 0.25 + 0.125 + 0.1875 * 0.375
    env.reset_state[2] = 3
    env.reset_state[7] = 128
    env.reset_state[8] = -96
    env.reset()
    env.step()
    assert env.collision_valid.tolist() == [1, 1, 1]
    assert env.collision_hit.tolist() == [1, 1, 0]


def test_unresolvable_sweep_is_invalid(tmp_path):
    path = single_box(tmp_path, [0, 0, 3], [0.01, 0.1, 0.1])
    env = Environment(1, 1, control_dt=0.0025, scene=path, collision_radius=0.0001)
    env.reset_state[0] = -1
    env.reset_state[2] = 3
    env.reset_state[7] = 1.0e12
    env.reset()
    out = env.step()
    assert not env.collision_valid.item() and not out["valid"].item()


def test_first_lane_and_nonreset_step(tmp_path):
    path = single_box(tmp_path, [0, 0, 3], [0.01, 0.1, 1])
    env = Environment(
        1,
        2,
        control_dt=0.0025,
        scene=path,
        collision_radius=0.01,
        terminate_on_collision=True,
    )
    env.reset_state[0] = -0.25
    env.reset_state[2] = 3
    env.reset_state[7] = 200
    env.reset()
    env._native.test_step(torch.cuda.current_stream(env.device).cuda_stream, False)
    assert env.collision_hit.tolist() == [1, 0]
    assert env.collision_valid.tolist() == [1, 0]
    env.reset()
    env._native.auto_reset(False)
    env.capture(random=False)
    out = env.step(graph=True, random=False)
    assert out["done"].tolist() == [[1, 1]] and env.collision_hit.tolist() == [1, 1]
    assert (out["observation"][0, :, 0] > 0.24).all()


def bounded_reference(start, end, bend, half, radius):
    """Independent certificate via the Lipschitz bound on distance to a box.

    Any occupied midpoint certifies a hit. An interval whose midpoint
    distance minus the maximum travel exceeds the radius is entirely clear.
    Fail the test if either label cannot be certified.
    """
    linear = end - start - bend
    speed_bound = max(np.linalg.norm(linear), np.linalg.norm(linear + 2 * bend))
    stack = [(0.0, 1.0)]
    while stack:
        lo, hi = stack.pop()
        mid = (lo + hi) / 2
        p = start + mid * (linear + mid * bend)
        distance = np.linalg.norm(np.maximum(np.abs(p) - half, 0.0))
        if distance <= radius:
            return True
        if distance - speed_bound * (hi - lo) / 2 > radius:
            continue
        assert hi - lo > 1e-12, "reference could not certify this trajectory"
        stack.extend([(lo, mid), (mid, hi)])
    return False


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("length_scale", [1.0, 2.0**-15])
def test_random_sweeps_against_bounded_reference(tmp_path, graph, length_scale):
    n = 1024
    center = np.array([1.0, -2.0, 4.0])
    half = np.array([0.03125, 0.0625, 0.5]) * length_scale
    yaw = 37
    path = single_box(tmp_path, center.tolist(), (2 * half).tolist(), yaw)
    dt, radius = 0.0025, 0.125 * length_scale
    env = Environment(2, n // 2, control_dt=dt, scene=path, collision_radius=radius)
    rng = np.random.default_rng(9681)
    start = (rng.uniform(-0.7, 0.7, (n, 3)) * length_scale + center).astype(np.float32)
    velocity = (rng.uniform(-450, 450, (n, 3)) * length_scale).astype(np.float32)
    env.reset_state[:3].copy_(torch.from_numpy(start.T))
    env.reset_state[7:10].copy_(torch.from_numpy(velocity.T))
    env.reset()
    if graph:
        env.capture(random=False)
    env.step(graph=graph, random=False)
    assert env.collision_valid.all()
    end = env.state[:3].cpu().numpy().T.astype(float)
    acceleration = env.state[13:16].cpu().numpy().T.astype(float)
    c, s = np.float32(
        [math.cos(math.radians(yaw)), math.sin(math.radians(yaw))]
    ).astype(float)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, np.hypot(c, s)]]) / np.hypot(c, s)
    begin = (start - center) @ rot
    finish = (end - center) @ rot
    bend = 0.5 * float(np.float32(dt)) ** 2 * acceleration @ rot
    expected = [
        bounded_reference(a, b, k, half, radius) for a, b, k in zip(begin, finish, bend)
    ]
    assert 0 < sum(expected) < n
    np.testing.assert_array_equal(env.collision_hit.cpu(), expected)


def test_all_split_s_gate_clips():
    course = json.loads(
        files("flightlib_cuda").joinpath("assets/split_s.flightcourse").read_bytes()
    )
    starts, velocities, expected = [], [], []
    for gate in course["instances"]:
        if gate["asset"] != "Gate":
            continue
        angle = math.radians(gate["yaw_deg"])
        normal = np.array([math.cos(angle), math.sin(angle), 0])
        horizontal = np.array([-normal[1], normal[0], 0])
        for sign in [-1, 1]:
            for distance in [0.31, 0.53, 0.89, 1.33]:
                for offset, hit in [
                    (0, False),
                    (0.624, False),
                    (0.626, True),
                    (0.8, True),
                    (0.976, False),
                ]:
                    starts.append(
                        gate["position"]
                        - sign * distance * normal
                        + offset * horizontal
                    )
                    velocities.append(sign * 190 * normal)
                    expected.append(hit)
    env = Environment(
        1, len(starts), control_dt=0.01, scene="split_s", collision_radius=0.125
    )
    env.reset_state[:3].copy_(torch.from_numpy(np.stack(starts).T))
    env.reset_state[7:10].copy_(torch.from_numpy(np.stack(velocities).T))
    env.reset()
    env.capture(random=False)
    env.step(graph=True, random=False)
    assert len(expected) == 280 and env.collision_valid.all()
    np.testing.assert_array_equal(env.collision_hit.cpu(), expected)
