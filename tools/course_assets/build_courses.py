"""Regenerate the portable room and Split-S fixtures from the pinned gate poses."""

import json
import math
from pathlib import Path

import yaml

ASSETS = (
    Path(__file__).resolve().parents[2] / "flightlib-cuda/python/flightlib_cuda/assets"
)


def box(name, center, size):
    return dict(name=name, center=center, size=size)


def instance(name, asset, position, yaw=0):
    return dict(name=name, asset=asset, position=position, yaw_deg=yaw)


def main():
    room = dict(
        name="Room",
        boxes=[
            box("Floor", [0, 0, -0.1], [32, 24, 0.2]),
            box("Ceiling", [0, 0, 8.1], [32, 24, 0.2]),
            box("West", [-16.1, 0, 4], [0.2, 24, 8]),
            box("East", [16.1, 0, 4], [0.2, 24, 8]),
            box("South", [0, -12.1, 4], [32, 0.2, 8]),
            box("North", [0, 12.1, 4], [32, 0.2, 8]),
        ],
    )
    gate = dict(
        name="Gate",
        boxes=[
            box("Left", [0, -0.8, 0], [0.12, 0.1, 1.7]),
            box("Right", [0, 0.8, 0], [0.12, 0.1, 1.7]),
            box("Top", [0, 0, 0.8], [0.12, 1.5, 0.1]),
            box("Bottom", [0, 0, -0.8], [0.12, 1.5, 0.1]),
        ],
    )
    scene = dict(
        schema="flightmare.boxes.v1",
        assets=[room],
        instances=[instance("Room", "Room", [0, 0, 0])],
    )

    def save(name):
        (ASSETS / (name + ".flightcourse")).write_text(
            json.dumps(scene, indent=2) + "\n", encoding="utf8"
        )

    save("room")
    track = yaml.safe_load((ASSETS / "SplitS.yaml").read_text())
    scene["assets"].append(gate)
    for i in range(1, track["gates"]["N"] + 1):
        name = f"Gate{i}"
        g = track["gates"][name]
        w, x, y, z = g["rotation"]
        if x or y:
            raise ValueError("fixture supports upright gates")
        scene["instances"].append(
            instance(name, "Gate", g["position"], math.degrees(2 * math.atan2(z, w)))
        )
    save("split_s")


if __name__ == "__main__":
    main()
