"""Static upright box courses. Host BVHs are immutable and shared across lanes."""

from dataclasses import dataclass
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


@dataclass(frozen=True)
class Course:
    colliders: tuple
    visual_boxes: tuple
    visual_meshes: tuple


def _color(value):
    rgb = tuple(_xyz(value))
    if not all(0 <= c <= 1 for c in rgb):
        raise ValueError("display color outside [0,1]")
    return rgb


def _expand(scene):
    if scene["schema"] not in ("flightmare.boxes.v1", "flightmare.scene.v2"):
        raise ValueError("unsupported course schema")
    assets = {a["name"]: a["boxes"] for a in scene["assets"]}
    if len(assets) != len(scene["assets"]):
        raise ValueError("duplicate asset name")
    boxes, visuals = [], []
    for instance in scene["instances"]:
        x, y, z = _xyz(instance["position"])
        yaw = instance["yaw_deg"]
        if type(yaw) not in (int, float) or not math.isfinite(yaw):
            raise ValueError("invalid yaw")
        c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        for box in assets[instance["asset"]]:
            u, v, w = _xyz(box["center"])
            sx, sy, sz = _xyz(box["size"], positive=True)
            geometry = (
                x + c * u - s * v,
                y + s * u + c * v,
                z + w,
                sx / 2,
                sy / 2,
                sz / 2,
                c,
                s,
            )
            if box.get("collision", True):
                boxes.append(geometry)
            if box.get("visible", True):
                default = (
                    [0.5, 0.54, 0.58]
                    if instance["asset"] == "Room"
                    else [0.86, 0.25, 0.08]
                )
                visuals.append((geometry, _color(box.get("color", default))))
    meshes = []
    for mesh in scene.get("visual_meshes", []):
        points = tuple(tuple(_xyz(v)) for v in mesh["points"])
        indices = tuple(mesh["indices"])
        if (
            not indices
            or len(indices) % 3
            or any(type(i) != int or not 0 <= i < len(points) for i in indices)
        ):
            raise ValueError("invalid visual triangle indices")
        meshes.append((points, indices, _color(mesh["color"])))
    if len(boxes) >= 2**23:
        raise ValueError("course must fit float32 exact BVH indices")
    return Course(tuple(boxes), tuple(visuals), tuple(meshes))


def load_boxes(data):
    """Expand collision proxies from a portable course document."""
    boxes = _expand(json.loads(data)).colliders
    if not boxes:
        raise ValueError("course contains no collision boxes")
    return list(boxes)


def load_scene(source):
    """Load a scene once at setup and share it between an environment and its camera."""
    if isinstance(source, Course):
        return source
    if isinstance(source, str) and source in ("room", "split_s"):
        return _portable(
            files(__package__).joinpath("assets", source + ".flightcourse").read_bytes()
        )
    path = Path(source)
    if path.suffix.lower() in (".usd", ".usda", ".usdc", ".usdz"):
        from .usd_scene import read_usd

        return _expand(read_usd(path))
    if path.suffix.lower() in (".unity", ".prefab"):
        from .unity_scene import read_unity

        return _expand(read_unity(path))
    return _expand(json.loads(path.read_bytes()))


@lru_cache(maxsize=16)
def _portable(data):
    return _expand(json.loads(data))


@lru_cache(maxsize=16)
def _compile(data):
    return _compile_boxes(tuple(load_boxes(data)))


@lru_cache(maxsize=16)
def _compile_boxes(boxes):
    if not boxes:
        raise ValueError("course contains no collision boxes")
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
    boxes = load_scene(scene).colliders
    key = (device, boxes)
    with _lock:
        nodes = _device_cache.get(key)
        if nodes is None:
            nodes = torch.tensor(
                _compile_boxes(boxes), dtype=torch.float32, device=device
            )
            # Other construction streams may immediately acquire this cache entry.
            torch.cuda.current_stream(device).synchronize()
            _device_cache[key] = nodes
    options = torch.tensor(
        [radius, bool(terminate)], dtype=torch.float32, device=device
    )
    return nodes, options
