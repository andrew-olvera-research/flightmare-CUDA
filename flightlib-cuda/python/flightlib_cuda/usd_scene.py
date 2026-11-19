"""Read composed static USD geometry into the portable course representation."""

import math
from pxr import Gf, Usd, UsdGeom, UsdPhysics


def read_usd(path):
    stage = Usd.Stage.Open(str(path))
    errors = stage.GetCompositionErrors()
    if errors:
        raise ValueError(f"unresolved USD composition in {path}: {errors[0]}")
    y_up = UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.y
    unit = UsdGeom.GetStageMetersPerUnit(stage)
    cache = UsdGeom.XformCache()
    assets, instances, meshes = [], [], []

    def basis(v):
        return Gf.Vec3d(v[0], -v[2], v[1]) if y_up else Gf.Vec3d(v)

    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if prim.IsAbstract():
            continue
        label = str(prim.GetPath())
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError(f"dynamic body: {label}")
        collider = False
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr()
            if enabled.ValueMightBeTimeVarying():
                raise ValueError(f"animated collision enablement: {label}")
            collider = enabled.Get()
        if collider and not prim.IsA(UsdGeom.Cube):
            raise ValueError(f"unsupported collider: {label}; use static box proxies")
        if not prim.IsA(UsdGeom.Gprim):
            if prim.IsA(UsdGeom.PointInstancer):
                raise ValueError(f"expand point instancer: {label}")
            continue
        geom = UsdGeom.Gprim(prim)
        visible = (
            geom.ComputeVisibility() != UsdGeom.Tokens.invisible
            and geom.ComputePurpose()
            in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render)
        )
        if not visible and not collider:
            continue
        ancestor = prim
        while ancestor and not ancestor.IsPseudoRoot():
            xf = UsdGeom.Xformable(ancestor)
            if xf and xf.TransformMightBeTimeVarying():
                raise ValueError(f"animated transform: {label}")
            ancestor = ancestor.GetParent()
        m = cache.GetLocalToWorldTransform(prim)
        colors = geom.GetDisplayColorAttr().Get()
        color = list(colors[0]) if colors else [0.5, 0.54, 0.58]
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            attrs = (
                mesh.GetPointsAttr(),
                mesh.GetFaceVertexCountsAttr(),
                mesh.GetFaceVertexIndicesAttr(),
            )
            if any(a.ValueMightBeTimeVarying() for a in attrs):
                raise ValueError(f"animated mesh: {label}")
            points, counts, indices = (a.Get() for a in attrs)
            if (
                not counts
                or any(n != 3 for n in counts)
                or len(indices) != 3 * len(counts)
                or mesh.GetHoleIndicesAttr().Get()
            ):
                raise ValueError(f"triangulate and bake mesh: {label}")
            indices = list(indices)
            if (mesh.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded) != (
                m.GetDeterminant() < 0
            ):
                indices[1::3], indices[2::3] = indices[2::3], indices[1::3]
            meshes.append(
                dict(
                    points=[
                        list(basis(m.Transform(Gf.Vec3d(p))) * unit) for p in points
                    ],
                    indices=indices,
                    color=color,
                )
            )
            continue
        if not prim.IsA(UsdGeom.Cube):
            raise ValueError(f"unsupported visual geometry: {label}; bake to triangles")
        size = UsdGeom.Cube(prim).GetSizeAttr()
        if size.ValueMightBeTimeVarying():
            raise ValueError(f"animated box: {label}")
        axes = [
            basis(m.TransformDir(Gf.Vec3d.Axis(i)))
            for i in ((0, 2, 1) if y_up else (0, 1, 2))
        ]
        dims = [v.GetLength() * unit * size.Get() for v in axes]
        a, b, c = [v.GetNormalized() for v in axes]
        if (
            min(dims) <= 0
            or max(abs(a[2]), abs(b[2]), abs(c[0]), abs(c[1]), abs(Gf.Dot(a, b))) > 1e-6
        ):
            raise ValueError(f"tilted, sheared or degenerate box: {label}")
        name = f"Box{len(assets)}"
        assets.append(
            dict(
                name=name,
                boxes=[
                    dict(
                        name="Shape",
                        center=[0, 0, 0],
                        size=dims,
                        collision=collider,
                        visible=visible,
                        color=color,
                    )
                ],
            )
        )
        instances.append(
            dict(
                name=name,
                asset=name,
                source_path=label,
                position=list(basis(m.ExtractTranslation()) * unit),
                yaw_deg=math.degrees(math.atan2(a[1], a[0])),
            )
        )
    if not assets and not meshes:
        raise ValueError("no supported geometry")
    return dict(
        schema="flightmare.scene.v2",
        assets=assets,
        instances=instances,
        visual_meshes=meshes,
    )
