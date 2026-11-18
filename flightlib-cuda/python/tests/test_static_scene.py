import math
import json
import numpy as np
import pytest
import torch
from flightlib_cuda import Environment
from flightlib_cuda.scene import load_boxes
from importlib.resources import files


def oracle(points, boxes, radius):
    result = np.zeros(len(points), dtype=bool)
    for x, y, z, hx, hy, hz, c, s in boxes:
        d = points - [x, y, z]
        local = d @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        excess = np.maximum(np.abs(local) - [hx, hy, hz], 0)
        result |= np.sum(excess**2, axis=1) <= radius**2
    return result


@pytest.mark.parametrize("graph", [False, True])
def test_course_overlap(graph):
    boxes = load_boxes(
        files("flightlib_cuda").joinpath("assets/split_s.flightcourse").read_bytes()
    )
    assert len(boxes) == 34
    rng = np.random.default_rng(57)
    pts = rng.uniform([-17, -13, 0.01], [17, 13, 8.3], (1000, 3))
    # Explicitly exercise the opening, post, tangency, and rounded box corner.
    pts[:6] = [
        [-0.9, -1.27, 3.48],
        [-0.1, -1.27, 3.48],
        [1, 1, 0.125],
        [1, 1, 0.126],
        [16.29, 12.29, 8.09],
        [-4.48, -5.94, 1.05],
    ]
    for i, box in enumerate(boxes):
        pts[10 + i] = box[:3]
    pts[10, 2] = 0.01  # stay within world bounds for the floor probe
    pts = pts.astype(np.float32)
    active = [1] * 1000
    active[-1] = 0
    env = Environment(
        2,
        500,
        control_dt=0,
        scene="split_s",
        collision_radius=0.125,
        active=active,
        bounds=[v for v in (-21, 21, -21, 21, -2, 12) for _ in active],
    )
    env.reset_state[:3].copy_(torch.from_numpy(pts.T))
    env.reset()
    if graph:
        env.capture(random=False)
    env.step(graph=graph, random=False)
    expected = oracle(pts, boxes, 0.125)
    expected[-1] = False
    np.testing.assert_array_equal(env.collision_hit.cpu(), expected)
    np.testing.assert_array_equal(env.collision_valid.cpu(), active)
    assert not expected[0] and expected[1] and expected[2] and not expected[3]
    assert not env.outputs["done"].flatten()[1]  # query-only post hit
    env.selected.zero_()
    env.selected[1] = 1
    env.reset()
    assert not env.collision_valid[1] and env.collision_valid[2]


def test_cached_room_and_terminal_transition():
    env = Environment(1, 2, control_dt=0, scene="room", terminate_on_collision=True)
    other = Environment(1, 1, scene="room")
    assert env._scene_bindings[0].data_ptr() == other._scene_bindings[0].data_ptr()
    env.reset_state[2].fill_(3)
    env.reset_state[0, 0] = 16
    env.reset()
    env.reset_state[0].zero_()
    env.capture(random=False)
    out = env.step(graph=True, random=False)
    assert out["done"].tolist() == [[1, 0]]
    assert env.collision_hit.tolist() == [1, 0]
    assert (
        out["transition_observation"][0, 0, 0] == 16
        and out["observation"][0, 0, 0] == 0
    )
    env.step(graph=True, random=False)
    assert env.collision_hit.tolist() == [0, 0]


def test_collision_invalid_lane():
    env = Environment(1, 1, scene="room")
    env.reset()
    env.actions.fill_(math.nan)
    env.step()
    assert not env.collision_hit.item() and not env.collision_valid.item()


def test_contact_at_control_boundary(tmp_path):
    # The vehicle crosses a thin box between control samples. Only endpoint
    # overlap is supported, including at nonzero control dt.
    box = dict(name="Post", center=[0, 0, 2], size=[0.02, 0.2, 1])
    path = tmp_path / "post.flightcourse"
    path.write_text(
        json.dumps(
            dict(
                schema="flightmare.boxes.v1",
                assets=[dict(name="Post", boxes=[box])],
                instances=[dict(asset="Post", position=[0, 0, 0], yaw_deg=0)],
            )
        )
    )
    env = Environment(1, 2, control_dt=0.01, scene=path, collision_radius=0.025)
    env.reset_state[0] = torch.tensor([-0.1, -0.2], device=env.device)
    env.reset_state[2] = 2
    env.reset_state[7] = 20
    env.reset()
    env.step()
    assert env.state[0, 0] > 0.08 and abs(env.state[0, 1]) < 0.02
    assert env.collision_valid.tolist() == [1, 1]
    assert env.collision_hit.tolist() == [0, 1]
