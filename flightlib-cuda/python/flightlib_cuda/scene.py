"""Static upright box courses. Host BVHs are immutable and shared across lanes."""

from functools import lru_cache
from importlib.resources import files
import json
import math
from pathlib import Path
from threading import Lock
from weakref import WeakValueDictionary

import torch


def _xyz(value, positive=False):
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("expected a three-component vector")
    if any(
        type(x) not in (float, int)
        or not math.isfinite(x)
        or abs(x) > 1e6
        or (positive and x <= 2**-149)  # half extent must survive float32 rounding
        for x in value
    ):
        raise ValueError("invalid course dimensions or coordinates")
    return value


def load_boxes(data):
    """Expand instanced box assets in metres, right handed, Z up. Size is full extent."""
    scene = json.loads(data)
    if scene["schema"] != "flightmare.boxes.v1":
        raise ValueError("unsupported course schema")
    assets = {a["name"]: a["boxes"] for a in scene["assets"]}
    if len(assets) != len(scene["assets"]):
        raise ValueError("duplicate asset name")
    boxes = []
    for instance in scene["instances"]:
        x, y, z = _xyz(instance["position"])
        yaw = instance["yaw_deg"]
        if type(yaw) not in (int, float) or not math.isfinite(yaw):
            raise ValueError("invalid yaw")
        c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        for box in assets[instance["asset"]]:
            u, v, w = _xyz(box["center"])
            sx, sy, sz = _xyz(box["size"], positive=True)
            boxes.append(
                (
                    x + c * u - s * v,
                    y + s * u + c * v,
                    z + w,
                    sx / 2,
                    sy / 2,
                    sz / 2,
                    c,
                    s,
                )
            )
    if not boxes or len(boxes) >= 2**23:
        raise ValueError("course must fit float32 exact BVH indices")
    return boxes


@lru_cache(maxsize=16)
def _compile(data):
    boxes = load_boxes(data)
    rows = []

    def build(group):
        index = len(rows)
        rows.append(None)
        leaf = len(group) == 1
        if leaf:
            bounds = group[0]
        else:
            spans = [
                (
                    abs(b[6]) * b[3] + abs(b[7]) * b[4],
                    abs(b[7]) * b[3] + abs(b[6]) * b[4],
                    b[5],
                )
                for b in group
            ]
            low = [min(b[i] - d[i] for b, d in zip(group, spans)) for i in range(3)]
            high = [max(b[i] + d[i] for b, d in zip(group, spans)) for i in range(3)]
            center = [(a + b) / 2 for a, b in zip(low, high)]
            # Conservative bounds after float32 center, extent and yaw rounding.
            half = [
                (b - a) / 2 + 1e-6 * max(1, abs(a), abs(b)) for a, b in zip(low, high)
            ]
            bounds = (*center, *half, 1, 0)
            axis = max(range(3), key=half.__getitem__)
            ordered = sorted(group, key=lambda box: box[axis])
            mid = len(ordered) // 2
            build(ordered[:mid])
            build(ordered[mid:])
        rows[index] = (*bounds, len(rows), int(leaf))

    build(boxes)
    return tuple(rows)


_device_cache = WeakValueDictionary()
_lock = Lock()


def bind_scene(scene, radius, terminate, device):
    """Retain a single read-only traversal table per live course and CUDA device."""
    if not math.isfinite(radius) or not 2**-150 < radius <= 1e6:
        raise ValueError(
            "collision radius must be positive in float32 and at most 1e6 m"
        )
    path = (
        files(__package__).joinpath("assets", scene + ".flightcourse")
        if isinstance(scene, str) and scene in ("room", "split_s")
        else Path(scene)
    )
    data = path.read_bytes()
    key = (device, data)
    with _lock:
        nodes = _device_cache.get(key)
        if nodes is None:
            nodes = torch.tensor(_compile(data), dtype=torch.float32, device=device)
            # Other construction streams may immediately acquire this cache entry.
            torch.cuda.current_stream(device).synchronize()
            _device_cache[key] = nodes
    options = torch.tensor(
        [radius, bool(terminate)], dtype=torch.float32, device=device
    )
    return nodes, options
