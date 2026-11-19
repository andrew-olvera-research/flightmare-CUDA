"""Rendered axial depth versus independent box intersections, including first frame after a swap."""

from importlib.resources import files
import numpy as np
import pytest
import torch
import warp as wp
from flightlib_cuda.scene import load_boxes
from flightlib_cuda.camera import TiledCamera


def expected_depth(boxes, pose, w, h):
    x, y, z, qw, qx, qy, qz = pose
    xyz = np.array([qx, qy, qz])

    def rotate(v):
        cross = np.cross(xyz, v)
        return v + 2 * (qw * cross + np.cross(xyz, cross))

    u, v = np.meshgrid(
        (np.arange(w) + 0.5 - w / 2) / (w * 0.625),
        (np.arange(h) + 0.5 - h / 2) / (w * 0.625),
    )
    origin = np.array([x, y, z]) + rotate(np.array([0.16, 0, 0]))
    rays = rotate(np.stack([np.ones_like(u), -u, -v], axis=-1))
    nearest = np.full((h, w), np.inf)
    for bx, by, bz, a, b, d, co, si in boxes:
        rotation = np.array([[co, -si, 0], [si, co, 0], [0, 0, 1]])
        offset = (origin - [bx, by, bz]) @ rotation
        direction = rays @ rotation
        half = np.array([a, b, d])
        direction = np.where(abs(direction) < 1e-12, 1e-12, direction)
        lo, hi = (-half - offset) / direction, (half - offset) / direction
        near, far = np.minimum(lo, hi).max(-1), np.maximum(lo, hi).min(-1)
        nearest = np.minimum(
            nearest, np.where((far >= near) & (near > 0), near, np.inf)
        )
    return nearest


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="requires CUDA"
            ),
        ),
    ],
)
def test_view_swap(device):
    boxes = load_boxes(
        files("flightlib_cuda").joinpath("assets/split_s.flightcourse").read_bytes()
    )
    # Exercise padded tiles at both image edges.
    width, height = 49, 33
    camera = TiledCamera(3, width, height, device=device, depth=True)
    poses = torch.tensor(
        [
            [-0.9, -4.27, 3.48, 0.5**0.5, 0, 0, 0.5**0.5],
            [-4, 2, 4, 1, 0, 0, 0],
            [1, 1, 5, 0.9, 0.2, -0.3, 0.1],
        ]
    )
    poses[:, 3:] /= poses[:, 3:].norm(dim=1, keepdim=True)
    expected = np.stack(
        [expected_depth(boxes, p.numpy(), width, height) for p in poses]
    )
    poses = poses.to(device)
    reference = camera.draw(poses).clone()
    assert reference.flatten(1).float().std(1).min() > 1
    for indices in ([0, 1, 2], [2, 0, 1], [1, 2, 0]):
        image = camera.draw(poses[indices])
        assert torch.equal(image, reference[indices])
        np.testing.assert_allclose(
            wp.to_torch(camera.depth)[0].cpu().numpy(),
            expected[indices],
            atol=0.002,
            rtol=0.0001,
        )


def test_delayed_cuda_reader_and_partial_reset():
    import pytest
    from flightlib_cuda import Environment, Subscription

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    env = Environment(2, subscriptions=(Subscription("pose", required=True, depth=2),))
    reader = torch.cuda.Stream()
    try:
        env.reset_state[0, 0] = -4
        env.reset_state[1].fill_(8)
        env.reset_state[2].fill_(4)
        env.reset()
        env.state[0, 0] = 3
        env.state[2, 0] = 0.01
        env.capture(random=False)
        for _ in range(2):
            env.step(graph=True, random=False)
        with torch.cuda.stream(reader):
            camera = TiledCamera(2, 49, 33, depth=True)
            first = camera.consume(env.subscriptions[0], 1, 10_000_000)
            first_depth = wp.to_torch(camera.depth)[0, :, 16, 24].clone()
            second = camera.consume(env.subscriptions[0], 2, 20_000_000)
            second_depth = wp.to_torch(camera.depth)[0, :, 16, 24]
            assert first.lane.tolist() == second.lane.tolist() == [0, 1]
            assert first.episode.tolist() == [1, 1] and second.episode.tolist() == [
                2,
                1,
            ]
            assert first.valid.all() and second.valid.all()
            torch.testing.assert_close(
                second_depth - first_depth,
                torch.tensor([7.0, 0.0], device=env.device),
                atol=0.002,
                rtol=0,
            )
            assert not torch.equal(first.rgb[0], second.rgb[0])
            assert env.subscriptions[0].stats["host_bytes"] == 0
    finally:
        env.close_subscriptions()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_recorded_tick_matches_pre_reset_subscription():
    from flightlib_cuda import Environment, Subscription
    from torch.nn.functional import adaptive_avg_pool2d

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        kwargs = dict(
            control_dt=0.02,
            scene="split_s",
            collision_radius=0.125,
            terminate_on_collision=True,
        )
        direct = Environment(2, 1, **kwargs)
        reference = Environment(
            2, 1, subscriptions=(Subscription("pose", required=True),), **kwargs
        )
        try:
            for env in (direct, reference):
                env.reset_state[0].fill_(-4)
                env.reset_state[1].fill_(8)
                env.reset_state[2].fill_(4)
                env.reset()
                env.state[:3, 0] = torch.tensor([-0.1, -1.60, 3.48], device=env.device)
                env.state[8, 0] = 70
            cam, ref = (TiledCamera(2, 64, 48, depth=True) for _ in range(2))
            cam.draw(direct.state[:7].T)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                obs, reward, done, valid = cam.record_tick(direct)

            for tick in range(1, 5):
                direct.actions.fill_(0.05 * tick)
                reference.actions.copy_(direct.actions)
                graph.replay()
                expected = reference.step(random=False)
                exposure = ref.consume(
                    reference.subscriptions[0], tick, tick * 20_000_000
                )
                assert exposure.episode.tolist() == ([1, 1] if tick == 1 else [2, 1])
                if tick == 1:
                    assert direct.collision_hit.tolist() == [1, 0]
                    assert done.tolist() == [1, 0]
                assert torch.equal(cam.rgb, exposure.rgb)
                torch.testing.assert_close(
                    wp.to_torch(cam.depth), wp.to_torch(ref.depth)
                )
                assert torch.equal(valid, exposure.valid)
                pooled = adaptive_avg_pool2d(
                    exposure.rgb.permute(0, 3, 1, 2).float(), 16
                )
                pooled = (
                    pooled.flatten(1).div_(255).masked_fill_(done.bool()[:, None], 0)
                )
                torch.testing.assert_close(obs, pooled)
                torch.testing.assert_close(reward, expected["reward"].flatten())
                assert torch.equal(done, expected["done"].flatten())
                assert torch.equal(direct.state, reference.state)
        finally:
            reference.close_subscriptions()
