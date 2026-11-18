from dataclasses import dataclass
import math
from pathlib import Path


@dataclass(frozen=True)
class Camera:
    """Centered square-pixel pinhole. Mount is FLU optical rig to FLU body."""

    lane: int
    width: int = 256
    height: int = 192
    fx: float = 160.0
    offset: tuple = (0.16, 0.0, 0.0)
    mount_wxyz: tuple = (1.0, 0.0, 0.0, 0.0)
    near: float = 0.15
    far: float = 100.0
    latency_ns: int = 0
    depth: bool = True

    def __post_init__(self):
        if type(self.depth) is not bool:
            raise ValueError("depth must be a boolean")
        if any(type(v) is not int or v < 1 for v in (self.width, self.height)):
            raise ValueError("camera dimensions must be positive integers")
        if not all(map(math.isfinite, (self.fx, self.near, self.far))) or not (
            self.fx > 0 and 0 < self.near < self.far
        ):
            raise ValueError("invalid camera focal length or clipping range")
        if (
            type(self.lane) is not int
            or self.lane < 0
            or type(self.latency_ns) is not int
            or self.latency_ns < 0
        ):
            raise ValueError("invalid camera lane or latency")
        if len(self.offset) != 3 or not all(map(math.isfinite, self.offset)):
            raise ValueError("invalid camera mount position")
        if len(self.mount_wxyz) != 4 or not math.isclose(
            math.hypot(*self.mount_wxyz), 1.0, abs_tol=1e-6
        ):
            raise ValueError("mount_wxyz must be a unit quaternion")


@dataclass(frozen=True)
class Image:
    camera: Camera
    sequence: int
    episode: int
    requested_ns: int
    acquired_ns: int
    rgb: object
    depth_m: object

    @property
    def available_ns(self):
        return self.acquired_ns + self.camera.latency_ns

    def observation(self, *, episode, time_ns, acquired_ns):
        if (
            self.episode != episode
            or self.acquired_ns != acquired_ns
            or time_ns < self.available_ns
        ):
            raise ValueError(
                "image generation, acquisition time or delivery deadline mismatch"
            )
        return self.rgb, self.depth_m


class IsaacRenderer:
    """A shared-world viewer or independent single-aircraft FPV lanes on one static course."""

    def __init__(
        self,
        course,
        *,
        world,
        slots,
        actors,
        cameras,
        device="cuda:0",
        fixture_lighting=True,
    ):
        import numpy as np
        import omni.replicator.core as rep
        import omni.usd
        from pxr import Gf, UsdGeom, UsdLux, Vt

        if (
            type(slots) is not int
            or slots < 1
            or (world is None and slots != 1)
            or (world is not None and (type(world) is not int or world < 0))
        ):
            raise ValueError("invalid world index or slot count")
        self.cameras = tuple(cameras)
        self.lanes = tuple(actors)
        if (
            not self.lanes
            or len(set(self.lanes)) != len(self.lanes)
            or any(
                type(n) is not int
                or n < 0
                or (world is not None and n // slots != world)
                for n in self.lanes
            )
        ):
            raise ValueError(
                "actor selection contains a duplicate or a lane outside the selected world"
            )
        if not self.cameras or any(c.lane not in self.lanes for c in self.cameras):
            raise ValueError("each camera must subscribe to a selected actor")
        path = Path(course).resolve(strict=True)
        from pxr import Usd

        asset = Usd.Stage.Open(str(path))
        if (
            not asset
            or not asset.GetDefaultPrim()
            or UsdGeom.GetStageMetersPerUnit(asset) != 1
            or UsdGeom.GetStageUpAxis(asset) != UsdGeom.Tokens.z
        ):
            raise ValueError("course must have a default prim, Z up and metre units")
        stage = omni.usd.get_context().get_stage()
        root = "/Flightmare"
        if stage.GetPrimAtPath(root):
            raise ValueError("stage already contains a Flightmare renderer")
        self.stage, self.root, self.device = stage, root, device
        self.products = []
        self.warmed_up = False
        self.last_sequence, self.last_acquired_ns = 0, -1
        try:
            UsdGeom.SetStageMetersPerUnit(stage, 1.0)
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            stage.DefinePrim(root, "Xform")
            stage.DefinePrim(root + "/Course", "Xform").GetReferences().AddReference(
                str(path)
            )
            # Interior fill lights for the closed room fixture.
            positions = ((-8, -6), (-8, 6), (8, -6), (8, 6)) if fixture_lighting else ()
            for i, xy in enumerate(positions):
                light = UsdLux.SphereLight.Define(stage, f"{root}/Light{i}")
                light.CreateIntensityAttr(50000)
                light.CreateExposureAttr(4.0)
                light.CreateRadiusAttr(0.15)
                light.AddTranslateOp().Set((*xy, 7.8))
            self.instances = None
            if world is not None:
                self.instances = UsdGeom.PointInstancer.Define(stage, root + "/Actors")
                body = UsdGeom.Sphere.Define(stage, root + "/Actors/Body")
                body.CreateRadiusAttr(1)
                body.AddScaleOp().Set((0.15, 0.12, 0.06))
                body.CreateDisplayColorAttr([(0.03, 0.35, 0.92)])
                self.instances.CreatePrototypesRel().SetTargets([body.GetPath()])
                self.instances.CreateProtoIndicesAttr(
                    Vt.IntArray([0] * len(self.lanes))
                )
                self.instances.CreateIdsAttr(Vt.Int64Array(self.lanes))
            self._indices = {lane: i for i, lane in enumerate(self.lanes)}
            self._lanes_np = np.asarray(self.lanes)
            for i, spec in enumerate(self.cameras):
                rig = UsdGeom.Xform.Define(stage, f"{root}/Rig{i}")
                pos_op, rot_op = rig.AddTranslateOp(), rig.AddOrientOp()
                camera = UsdGeom.Camera.Define(stage, f"{root}/Rig{i}/Camera")
                camera.AddTranslateOp().Set(Gf.Vec3d(*spec.offset))
                # USD camera looks along -Z, +Y up. Body looks along +X, +Z up.
                rig_to_body = Gf.Quatf(
                    spec.mount_wxyz[0], Gf.Vec3f(*spec.mount_wxyz[1:])
                )
                camera.AddOrientOp().Set(rig_to_body * Gf.Quatf(0.5, 0.5, -0.5, -0.5))
                camera.CreateClippingRangeAttr((spec.near, spec.far))
                camera.CreateFocalLengthAttr(20.0)
                camera.CreateHorizontalApertureAttr(20.0 * spec.width / spec.fx)
                camera.CreateVerticalApertureAttr(20.0 * spec.height / spec.fx)
                rp = rep.create.render_product(
                    str(camera.GetPath()), (spec.width, spec.height)
                )
                annotators = []
                self.products.append((rp, annotators, pos_op, rot_op))
                for name in (
                    ("rgb", "distance_to_image_plane") if spec.depth else ("rgb",)
                ):
                    ann = rep.AnnotatorRegistry.get_annotator(
                        name, device=device, do_array_copy=False
                    )
                    ann.attach(rp)
                    annotators.append(ann)
        except BaseException:
            self.close()
            raise

    def consume(self, channel, *, expected_ns=None):
        """Poll a viewer or require the specified policy capture tick."""
        if channel.kind != "pose" or not channel.host:
            raise ValueError("USD pose authoring requires a host pose subscription")
        frame = channel.acquire(wait=expected_ns is not None)
        if frame is None:
            if expected_ns is not None:
                raise RuntimeError("required camera capture is missing")
            return None
        with frame:
            if expected_ns is not None and frame.acquired_ns != expected_ns:
                raise ValueError("unexpected capture tick")
            return self.render(
                sequence=frame.sequence,
                requested_ns=frame.requested_ns,
                acquired_ns=frame.acquired_ns,
                lanes=frame.lane.numpy(),
                episodes=frame.episode.numpy(),
                valid=frame.valid.numpy(),
                poses=frame.values.numpy(),
            )

    def render(
        self, *, sequence, requested_ns, acquired_ns, lanes, episodes, valid, poses
    ):
        """Consume one coherent host snapshot. RGB and axial depth remain on CUDA."""
        import numpy as np
        import torch
        import warp as wp
        import omni.replicator.core as rep
        from pxr import Gf, Vt

        if (
            not self.products
            or sequence <= self.last_sequence
            or not 0 <= requested_ns <= acquired_ns
            or acquired_ns <= self.last_acquired_ns
        ):
            raise ValueError("closed renderer or invalid capture identity")
        if not np.array_equal(lanes, self._lanes_np) or np.shape(poses) != (
            len(lanes),
            7,
        ):
            raise ValueError("capture has a different selection or pose shape")
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != (len(lanes),) or np.shape(episodes) != valid.shape:
            raise ValueError("invalid per-actor metadata")
        if not any(valid[self._indices[c.lane]] for c in self.cameras):
            self.last_sequence, self.last_acquired_ns = sequence, acquired_ns
            return (None,) * len(self.cameras)
        if self.instances is not None:
            self.instances.GetPositionsAttr().Set(
                Vt.Vec3fArray.FromNumpy(
                    np.ascontiguousarray(poses[:, :3], dtype=np.float32)
                )
            )
            # Vt stores quaternion vector components before the scalar component.
            xyzw = np.ascontiguousarray(poses[:, [4, 5, 6, 3]], dtype=np.float16)
            self.instances.GetOrientationsAttr().Set(Vt.QuathArray.FromNumpy(xyzw))
            self.instances.GetInvisibleIdsAttr().Set(
                Vt.Int64Array.FromNumpy(self._lanes_np[~valid].astype(np.int64))
            )
        for spec, (_, _, pos_op, rot_op) in zip(self.cameras, self.products):
            i = self._indices[spec.lane]
            if valid[i]:
                x, y, z, w, qx, qy, qz = map(float, poses[i])
                pos_op.Set(Gf.Vec3d(x, y, z))
                rot_op.Set(Gf.Quatf(w, qx, qy, qz))
        # Shader/annotator initialization may yield empty arrays on the first
        # capture. Keep the exact same poses and clock while warming up.
        for _ in range(1 if self.warmed_up else 8):
            rep.orchestrator.step(
                rt_subframes=1,
                delta_time=0.0,
                pause_timeline=True,
                wait_for_render=True,
            )
            buffers = [
                [a.get_data(device=self.device) for a in anns]
                for _, anns, _, _ in self.products
            ]
            if all(
                getattr(buf, "size", 0)
                and getattr(getattr(buf, "device", None), "is_cuda", False)
                for pair in buffers
                for buf in pair
            ):
                break
        else:
            raise RuntimeError("RTX did not produce complete CUDA camera buffers")
        self.warmed_up = True
        result = []
        for spec, pair in zip(self.cameras, buffers):
            i = self._indices[spec.lane]
            if not valid[i]:
                result.append(None)
                continue
            rgba = wp.to_torch(pair[0])
            h, w = spec.height, spec.width
            if rgba.shape != (h, w, 4) or rgba.dtype != torch.uint8:
                raise RuntimeError("unexpected RTX color shape or dtype")
            rgb = rgba[..., :3].clone(memory_format=torch.contiguous_format)
            depth_m = None
            if spec.depth:
                depth = wp.to_torch(pair[1])
                if depth.numel() != h * w or depth.dtype != torch.float32:
                    raise RuntimeError("unexpected RTX depth shape or dtype")
                depth_m = depth.reshape(h, w).clone()
            result.append(
                Image(
                    spec,
                    sequence,
                    int(episodes[i]),
                    requested_ns,
                    acquired_ns,
                    rgb,
                    depth_m,
                )
            )
        # The next Kit capture can reuse its annotators as soon as these two copies finish.
        torch.cuda.current_stream(self.device).synchronize()
        self.last_sequence, self.last_acquired_ns = sequence, acquired_ns
        return tuple(result)

    def close(self):
        for rp, annotators, _, _ in reversed(self.products):
            for ann in reversed(annotators):
                ann.detach(rp)
        for rp, _, _, _ in reversed(self.products):
            rp.destroy()
        self.products.clear()
        if self.stage.GetPrimAtPath(self.root):
            self.stage.RemovePrim(self.root)
