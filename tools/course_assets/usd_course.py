"""Convert authored static box courses between portable JSON and composed OpenUSD."""

import argparse
import json
import math
from pathlib import Path
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics


def export_usd(source, target):
    scene = json.loads(source.read_text())
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
                [(0.86, 0.25, 0.08) if asset["name"] == "Gate" else (0.5, 0.54, 0.58)]
            )
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
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
        target.write_text(target.read_text(encoding="utf-8").rstrip() + "\n", encoding="utf-8")


def import_usd(source, target):
    stage = Usd.Stage.Open(str(source))
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        raise ValueError("requires Z-up stage")
    unit = UsdGeom.GetStageMetersPerUnit(stage)
    cache = UsdGeom.XformCache()
    assets = []
    instances = []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if prim.IsAbstract():
            continue
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError(f"dynamic body: {prim.GetPath()}")
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr()
        if enabled.ValueMightBeTimeVarying():
            raise ValueError("animated collision enablement")
        if not enabled.Get():
            continue
        if not prim.IsA(UsdGeom.Cube):
            raise ValueError(f"unsupported collider: {prim.GetPath()}")
        cube = UsdGeom.Cube(prim)
        ancestor = prim
        while ancestor and not ancestor.IsPseudoRoot():
            if UsdGeom.Xformable(ancestor).TransformMightBeTimeVarying():
                raise ValueError("animated collision transform")
            ancestor = ancestor.GetParent()
        if cube.GetSizeAttr().ValueMightBeTimeVarying():
            raise ValueError("animated box")
        m = cache.GetLocalToWorldTransform(prim)
        axes = [m.TransformDir(Gf.Vec3d.Axis(i)) for i in range(3)]
        dims = [v.GetLength() * unit * cube.GetSizeAttr().Get() for v in axes]
        a, b, c = [v.GetNormalized() for v in axes]
        if (
            min(dims) <= 0
            or max(abs(a[2]), abs(b[2]), abs(c[0]), abs(c[1]), abs(Gf.Dot(a, b))) > 1e-6
        ):
            raise ValueError(f"tilted, sheared or degenerate box: {prim.GetPath()}")
        name = f"Box{len(assets)}"
        assets.append(
            dict(name=name, boxes=[dict(name="Shape", center=[0, 0, 0], size=dims)])
        )
        instances.append(
            dict(
                name=name,
                source_path=str(prim.GetPath()),
                asset=name,
                position=list(m.ExtractTranslation() * unit),
                yaw_deg=math.degrees(math.atan2(a[1], a[0])),
            )
        )
    if not assets:
        raise ValueError("no enabled box colliders")
    target.write_text(
        json.dumps(
            dict(schema="flightmare.boxes.v1", assets=assets, instances=instances),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["import", "export"])
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    (export_usd if args.mode == "export" else import_usd)(args.source, args.target)
