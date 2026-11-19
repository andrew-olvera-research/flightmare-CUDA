"""Optional Newton tiled FPV camera for static Flightlib courses."""

from dataclasses import dataclass
import math

import newton
from newton.sensors import SensorTiledCamera
import torch
import warp as wp

from flightlib_cuda.scene import load_scene


@wp.kernel
def _camera_poses(
    body: wp.array2d[float],
    camera: wp.array2d[wp.transformf],
    usable: wp.array[wp.bool],
):
    i = wp.tid()
    q = wp.quat(body[i, 4], body[i, 5], body[i, 6], body[i, 3])
    position = wp.vec3(body[i, 0], body[i, 1], body[i, 2])
    q64 = wp.quatd(q)
    norm = wp.length(q64)
    usable[i] = wp.isfinite(norm) and norm > wp.float64(0.0) and wp.isfinite(position)
    if usable[i]:
        q = wp.quatf(q64 / norm)
    else:
        position = wp.vec3(0.0)
        q = wp.quat_identity()
    # Optical -Z forward, +Y up, mounted on a +X forward, +Z up body.
    mount = wp.transform(wp.vec3(0.16, 0.0, 0.0), wp.quat(0.5, -0.5, -0.5, 0.5))
    camera[i, 0] = wp.transform(position, q) * mount


@wp.kernel
def _pool_rgb(
    color: wp.array4d[wp.uint32],
    done: wp.array[wp.uint8],
    valid: wp.array[wp.bool],
    output: wp.array2d[float],
):
    lane, y, x = wp.tid()
    # One thread sums one policy tile. Read the packed renderer output directly.
    index = y * 16 + x
    r = int(0)
    g = int(0)
    b = int(0)
    h = color.shape[2] // 16
    w = color.shape[3] // 16
    if valid[lane] and done[lane] == wp.uint8(0):
        for j in range(h):
            for i in range(w):
                pixel = color[0, lane, y * h + j, x * w + i]
                r += int(pixel & wp.uint32(255))
                g += int((pixel >> wp.uint32(8)) & wp.uint32(255))
                b += int((pixel >> wp.uint32(16)) & wp.uint32(255))
    scale = 1.0 / float(h * w * 255)
    output[lane, index] = float(r) * scale
    output[lane, 256 + index] = float(g) * scale
    output[lane, 512 + index] = float(b) * scale


@dataclass
class Exposure:
    sequence: int
    acquired_ns: int
    lane: torch.Tensor
    episode: torch.Tensor
    valid: torch.Tensor
    rgb: torch.Tensor


class TiledCamera:
    def __init__(
        self,
        count,
        width=256,
        height=192,
        device="cuda:0",
        depth=False,
        *,
        scene="split_s",
    ):
        self.count, self.width, self.height = count, width, height
        self.device = wp.get_device(str(device))
        self.stream = torch.cuda.current_stream(device) if self.device.is_cuda else None
        self.warp_stream = (
            wp.stream_from_torch(self.stream) if self.stream is not None else None
        )
        self.last_sequence = -1
        with self._scope():
            builder = newton.ModelBuilder()
            self.scene = load_scene(scene)
            if not self.scene.visual_boxes and not self.scene.visual_meshes:
                raise ValueError("camera scene contains no visible geometry")
            for (x, y, z, hx, hy, hz, c, s), color in self.scene.visual_boxes:
                angle = math.atan2(s, c) / 2
                xf = wp.transform((x, y, z), (0, 0, math.sin(angle), math.cos(angle)))
                builder.add_shape_box(-1, xform=xf, hx=hx, hy=hy, hz=hz, color=color)
            for points, indices, color in self.scene.visual_meshes:
                builder.add_shape_mesh(
                    -1, mesh=newton.Mesh(points, indices), color=color
                )
            model = builder.finalize(device=self.device)
            self.state = model.state()
            self.camera = SensorTiledCamera(model)
            # Group adjacent rays from one FPV image so a warp writes nearby
            # pixels and traverses similar parts of the scene across its rays.
            if count > 1:
                config = self.camera.default_render_config
                config.render_order = SensorTiledCamera.RenderOrder.TILED
                config.tile_width = 16
                config.tile_height = 8
            self.camera.default_render_config.enable_shadows = True
            self.camera.utils.create_default_light()
            fov = 2 * math.atan(height / (2 * width * 0.625))
            self.base_rays = self.camera.utils.compute_camera_rays_pinhole(
                width, height, camera_fovs=[fov]
            )
            # All lanes have identical intrinsics. Warp honors the zero camera stride.
            ray_view = wp.to_torch(self.base_rays).expand(count, -1, -1, -1, -1)
            self.rays = wp.from_torch(ray_view, dtype=wp.vec3f)
            self.usable = wp.empty(count, dtype=wp.bool, device=self.device)
            self.pose_valid = wp.to_torch(self.usable)
            self.poses = wp.empty((count, 1), dtype=wp.transformf, device=self.device)
            self.color = self.camera.utils.create_color_image_output(
                width, height, count
            )
            self.rgb = (
                wp.to_torch(self.color)[0]
                .view(torch.uint8)
                .reshape(self.count, self.height, self.width, 4)[..., :3]
            )
            self.depth = (
                wp.empty(self.color.shape, dtype=float, device=self.device)
                if depth
                else None
            )

    def _scope(self):
        if self.stream is None:
            return wp.ScopedDevice(self.device)
        if torch.cuda.current_stream(self.stream.device) != self.stream:
            raise ValueError("render and consume on the stream used at construction")
        return wp.ScopedStream(self.warp_stream, sync_enter=False, sync_exit=False)

    def draw(self, pose):
        """Borrowed RGB view, overwritten by the next draw on this stream."""
        if (
            pose.shape != (self.count, 7)
            or pose.dtype != torch.float32
            or str(pose.device) != str(self.device)
        ):
            raise ValueError("expected one device float32 XYZ/WXYZ pose per camera")
        with self._scope():
            wp.launch(
                _camera_poses,
                dim=self.count,
                inputs=[wp.from_torch(pose), self.poses, self.usable],
                device=self.device,
            )
            self.camera.update(
                self.state,
                self.poses,
                self.rays,
                color_image=self.color,
                forward_depth_image=self.depth,
            )
            return self.rgb

    def record_tick(self, env, counts=None):
        """Append a complete FPV tick to the caller's active CUDA graph.

        env.actions is the persistent input. All returned tensors are borrowed.
        Keep env, camera and graph alive, and order all consumers on the replay stream.
        """
        if self.stream is None or not torch.cuda.is_current_stream_capturing():
            raise ValueError("record_tick requires an active CUDA graph")
        if (
            env.state.shape[1] != self.count
            or env.device != self.stream.device
            or env.slots != 1
        ):
            raise ValueError(
                "expected one camera per single-aircraft environment on the same GPU"
            )
        if self.height % 16 or self.width % 16:
            raise ValueError("policy pooling requires image dimensions divisible by 16")
        if (
            env.scene is not None
            and env.scene is not self.scene
            and env.scene != self.scene
        ):
            raise ValueError(
                "camera and collision scenes differ; construct the camera with scene=env.scene"
            )
        out = env.outputs
        done, valid = (out[k].reshape(-1) for k in ("done", "valid"))
        obs = torch.empty((self.count, 768), device=env.device)

        accepted = torch.empty_like(self.pose_valid)

        def exposure():
            self.draw(env.state[:7].T)
            torch.logical_and(self.pose_valid, valid, out=accepted)
            wp.launch(
                _pool_rgb,
                (self.count, 16, 16),
                inputs=[
                    self.color,
                    wp.from_torch(done),
                    wp.from_torch(accepted),
                    wp.from_torch(obs),
                ],
                device=self.device,
            )

        with self._scope():
            env.record_step(random=False, before_reset=exposure)
            if counts is not None:
                counts.add_(env.collision_hit.sum())
        return obs, out["reward"].reshape(-1), done, accepted

    def consume(self, channel, expected_sequence, expected_ns):
        """Acquire a pre-reset device snapshot and return an independently owned batch."""
        if channel.host or channel.kind != "pose" or self.stream is None:
            raise ValueError("expected a device pose channel")
        with self._scope():
            frame = channel.acquire(self.stream)
            if frame is None:
                raise RuntimeError("required pose capture missing")
            with frame:
                if (
                    frame.sequence != expected_sequence
                    or frame.sequence <= self.last_sequence
                    or frame.acquired_ns != expected_ns
                    or frame.requested_ns != expected_ns
                ):
                    raise ValueError("stale or mistimed pose snapshot")
                rgb = self.draw(frame.values)
                valid = frame.valid.bool()
                owned = rgb.masked_fill(~valid[:, None, None, None], 0)
                self.last_sequence = frame.sequence
                return Exposure(
                    frame.sequence,
                    frame.acquired_ns,
                    frame.lane.clone(),
                    frame.episode.clone(),
                    valid,
                    owned,
                )
