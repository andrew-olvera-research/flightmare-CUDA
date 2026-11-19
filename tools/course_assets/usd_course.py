"""Convert authored static box courses between portable JSON and composed OpenUSD."""

import argparse
import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics


def export_usd(source, target):
    scene = json.loads(source.read_text())
    if scene.get("visual_meshes"):
        raise ValueError("USD course export currently accepts box courses only")
    stage = Usd.Stage.CreateNew(str(target))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    library = UsdGeom.Scope.Define(stage, "/Library").GetPrim()
    for asset in scene["assets"]:
        template = UsdGeom.Xform.Define(stage, "/Library/" + asset["name"])
        for b in asset["boxes"]:
            cube = UsdGeom.Cube.Define(stage, template.GetPath().AppendChild(b["name"]))
            cube.CreateSizeAttr(1.0)
            cube.AddTranslateOp().Set(Gf.Vec3d(*b["center"]))
            cube.AddScaleOp().Set(Gf.Vec3f(*b["size"]))
            cube.CreateDisplayColorAttr(
                [
                    b.get(
                        "color",
                        (0.86, 0.25, 0.08)
                        if asset["name"] == "Gate"
                        else (0.5, 0.54, 0.58),
                    )
                ]
            )
            if b.get("collision", True):
                UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            if not b.get("visible", True):
                cube.MakeInvisible()
    for inst in scene["instances"]:
        obj = UsdGeom.Xform.Define(stage, world.GetPath().AppendChild(inst["name"]))
        obj.GetPrim().GetReferences().AddInternalReference("/Library/" + inst["asset"])
        obj.AddTranslateOp().Set(Gf.Vec3d(*inst["position"]))
        obj.AddRotateZOp().Set(inst["yaw_deg"])
        obj.GetPrim().SetInstanceable(True)
    # Use abstract classes for prototypes so only placed instances render/collide.
    library.SetSpecifier(Sdf.SpecifierClass)
    stage.GetRootLayer().Save()
    if target.suffix == ".usda":
        target.write_text(
            target.read_text(encoding="utf-8").rstrip() + "\n", encoding="utf-8"
        )


def import_usd(source, target):
    # The standalone asset tool also works before building the CUDA extension.
    reader_path = (
        Path(__file__).resolve().parents[2]
        / "flightlib-cuda/python/flightlib_cuda/usd_scene.py"
    )
    spec = spec_from_file_location("usd_scene", reader_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    target.write_text(json.dumps(module.read_usd(source), indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["import", "export"])
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    (export_usd if args.mode == "export" else import_usd)(args.source, args.target)
