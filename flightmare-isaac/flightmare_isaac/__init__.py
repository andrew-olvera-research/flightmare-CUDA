"""Visual adapters. Construct IsaacRenderer after starting SimulationApp."""

from .renderer import Camera, Image, IsaacRenderer


def launch_visual_app():
    """Start a dedicated synchronous RTX consumer on GPU 0."""
    from isaacsim import SimulationApp

    return SimulationApp(
        dict(
            headless=True,
            multi_gpu=False,
            active_gpu=0,
            anti_aliasing=0,
            extra_args=["--/rtx/hydra/supportMultiTickRate=false"],
        )
    )


__all__ = ["Camera", "Image", "IsaacRenderer", "launch_visual_app"]
