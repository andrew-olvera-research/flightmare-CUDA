"""Installed API and transport checks that run without Kit. RTX checks live in tools/isaac."""

import sys
from unittest.mock import Mock

import pytest
import torch
from flightlib_cuda import Environment, Subscription
from flightmare_isaac import Camera, Image, IsaacRenderer


def test_headless_import_and_image_delivery():
    assert "omni.replicator.core" not in sys.modules
    camera = Camera(2, latency_ns=3)
    image = Image(camera, 1, 4, 8, 10, object(), object())
    assert image.observation(episode=4, time_ns=13, acquired_ns=10) == (
        image.rgb,
        image.depth_m,
    )
    for episode, now, capture in ((4, 12, 10), (3, 14, 10), (4, 14, 9)):
        with pytest.raises(ValueError):
            image.observation(episode=episode, time_ns=now, acquired_ns=capture)


@pytest.mark.parametrize(
    "change",
    [
        dict(fx=float("nan")),
        dict(width=0),
        dict(depth="false"),
        dict(near=101),
        dict(latency_ns=-1),
        dict(mount_wxyz=(1, 0, 0, 1)),
    ],
)
def test_invalid_calibration(change):
    with pytest.raises(ValueError):
        Camera(0, **change)


def test_consume_real_pose_channel_and_release_on_mismatch():
    env = Environment(
        2,
        1,
        subscriptions=(
            Subscription("pose", destination="host", depth=1, required=True),
        ),
    )
    env.reset_state[2] = 5
    env.reset()
    (channel,) = env.subscriptions
    draw = Mock()
    try:
        assert IsaacRenderer.consume(draw, channel) is None
        with pytest.raises(RuntimeError, match="missing"):
            IsaacRenderer.consume(draw, channel, expected_ns=10_000_000)
        env.step()
        with pytest.raises(ValueError, match="tick"):
            IsaacRenderer.consume(draw, channel, expected_ns=20_000_000)
        draw.render.assert_not_called()
        env.step()
        IsaacRenderer.consume(draw, channel, expected_ns=20_000_000)
        args = draw.render.call_args.kwargs
        assert args["poses"].shape == (2, 7) and args["lanes"].tolist() == [0, 1]
        assert args["episodes"].tolist() == [1, 1] and args["sequence"] == 2
    finally:
        env.close_subscriptions()


def test_slow_viewer_skips_capture_and_resumes_on_current_pose():
    env = Environment(
        2,
        1,
        subscriptions=(
            Subscription("pose", actors=((1, 0),), destination="host", depth=1),
        ),
    )
    (channel,) = env.subscriptions
    draw = Mock()
    try:
        env.reset()
        env.step()
        env.step()  # single slot still contains tick one
        assert channel.stats["dropped"] == 1
        torch.cuda.synchronize(env.device)
        IsaacRenderer.consume(draw, channel)
        assert draw.render.call_args.kwargs["sequence"] == 1
        env.step()
        torch.cuda.synchronize(env.device)
        IsaacRenderer.consume(draw, channel)
        actual = draw.render.call_args.kwargs
        assert actual["sequence"] == 3 and actual["lanes"].tolist() == [1]
        assert channel.stats["host_bytes"] == 96
    finally:
        env.close_subscriptions()
