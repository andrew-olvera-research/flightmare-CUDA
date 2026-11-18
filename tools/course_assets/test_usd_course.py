"""Pytest with usd-core installed, independent of the CUDA runtime."""

import json
import math
from pathlib import Path

import pytest

pytest.importorskip("pxr")
from pxr import Usd, UsdGeom, UsdPhysics
from usd_course import export_usd, import_usd

ASSETS = (
    Path(__file__).resolve().parents[2] / "flightlib-cuda/python/flightlib_cuda/assets"
)


@pytest.mark.parametrize("name,count", [("room", 6), ("split_s", 34)])
def test_release_usd_matches_course(tmp_path, name, count):
    source = json.loads((ASSETS / (name + ".flightcourse")).read_text())
    target = tmp_path / "roundtrip.flightcourse"
    import_usd(ASSETS / (name + ".usda"), target)
    imported = json.loads(target.read_text())
    actual = {x["source_path"]: x for x in imported["instances"]}
    export_usd(target, tmp_path / "rebuilt.usda")
    assets = {a["name"]: a["boxes"] for a in source["assets"]}
    actual_assets = {a["name"]: a["boxes"][0] for a in imported["assets"]}
    assert len(actual) == count
    for inst in source["instances"]:
        theta = math.radians(inst["yaw_deg"])
        for b in assets[inst["asset"]]:
            x, y, z = b["center"]
            dx, dy, dz = inst["position"]
            pose = actual[f"/World/{inst['name']}/{b['name']}"]
            assert pose["position"] == pytest.approx(
                [
                    dx + x * math.cos(theta) - y * math.sin(theta),
                    dy + x * math.sin(theta) + y * math.cos(theta),
                    dz + z,
                ],
                abs=1e-5,
            )
            assert math.remainder(
                pose["yaw_deg"] - inst["yaw_deg"], 360
            ) == pytest.approx(0, abs=1e-5)
            assert actual_assets[pose["asset"]]["size"] == pytest.approx(b["size"])


def test_unsupported_collider(tmp_path):
    path = tmp_path / "sphere.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdPhysics.CollisionAPI.Apply(UsdGeom.Sphere.Define(stage, "/Obstacle").GetPrim())
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError, match="unsupported collider"):
        import_usd(path, tmp_path / "unused.flightcourse")
