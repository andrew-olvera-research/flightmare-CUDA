"""Compare Newton kernel traversal on identical camera poses using CUDA events."""

import argparse
import json
from statistics import median

import torch
from newton.sensors import SensorTiledCamera
from tiled import TiledCamera


def sweep(count, repeats, ticks):
    if min(count, repeats, ticks) < 1:
        raise ValueError("envs, repeats and ticks must be positive")
    camera = TiledCamera(count)
    # Cover different parts of the room and different headings at every batch size.
    i = torch.arange(count, device="cuda")
    yaw = i * 0.17
    pose = torch.zeros(count, 7, device="cuda")
    pose[:, 0] = (i % 11) - 5
    pose[:, 1] = (i % 17) * 0.5 - 4
    pose[:, 2] = 2 + i % 3
    pose[:, 3] = yaw.cos()
    pose[:, 6] = yaw.sin()

    cfg = camera.camera.default_render_config
    order = SensorTiledCamera.RenderOrder
    graphs = []
    reference = None
    # Each graph has the same input and output addresses.
    for mode, width, height, shadows in [
        (order.PIXEL_PRIORITY, 16, 8, True),
        (order.VIEW_PRIORITY, 16, 8, True),
        (order.TILED, 8, 8, True),
        (order.TILED, 16, 8, True),
        (order.TILED, 32, 4, True),
        (order.TILED, 16, 8, False),
    ]:
        cfg.render_order, cfg.tile_width, cfg.tile_height = mode, width, height
        cfg.enable_shadows = shadows
        for block in (32, 64, 128, 256) if shadows else (64,):

            def launch():
                camera.camera.update(
                    camera.state,
                    camera.poses,
                    camera.rays,
                    color_image=camera.color,
                    kernel_block_dim=block,
                )

            with camera._scope():
                camera.draw(pose)
                launch()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=camera.stream):
                    launch()
            camera.rgb.zero_()
            graph.replay()
            if reference is None:
                reference = camera.rgb.clone()
            elif shadows and not torch.equal(camera.rgb, reference):
                raise RuntimeError(
                    f"RGB differs for {mode.name}, tile {width}x{height}, block {block}"
                )
            label = f"{mode.name}/{width}x{height}/{block}"
            graphs.append((label if shadows else label + "/unshadowed", graph, []))
    for rep in range(repeats):
        for name, graph, samples in graphs if rep % 2 == 0 else reversed(graphs):
            for _ in range(4):
                graph.replay()
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            for _ in range(ticks):
                graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / ticks)
    return {name: round(median(samples), 3) for name, _, samples in graphs}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--ticks", type=int, default=16)
    args = parser.parse_args()
    with torch.cuda.stream(torch.cuda.Stream()):
        print(json.dumps(sweep(args.envs, args.repeats, args.ticks), indent=2))
