"""Shared imported scene exercised by the CUDA BVH and the Newton camera."""

import json
from pathlib import Path

import pytest
import torch
import warp as wp

from flightlib_cuda import Environment
from flightlib_cuda.scene import load_scene
from tiled import TiledCamera

WALL = Path(__file__).resolve().parents[1] / "course_assets/wall.prefab"


def usd_wall(path):
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, "Y")
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    collider = UsdGeom.Cube.Define(stage, "/WallProxy")
    collider.CreateSizeAttr(100)
    collider.AddTranslateOp().Set((200, 200, 0))
    collider.AddScaleOp().Set((1, 4, 4))
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
    collider.MakeInvisible()
    mesh = UsdGeom.Mesh.Define(stage, "/WallFace")
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreatePointsAttr(
        [(350, 0, -200), (350, 0, 200), (350, 400, 200), (350, 400, -200)]
    )
    mesh.CreateFaceVertexCountsAttr([3, 3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 2, 3])
    stage.GetRootLayer().Save()


@pytest.mark.parametrize("kind", ["prefab", "unity", "usdc", "flightcourse"])
def test_visible_wall_has_independent_collision_proxy(tmp_path, kind):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    path = tmp_path / ("wall." + kind)
    if kind == "usdc":
        pytest.importorskip("pxr")
        usd_wall(path)
    elif kind == "flightcourse":
        # Editor bake schema: a hidden box proxy and a colored triangle submesh.
        path.write_text(
            json.dumps(
                dict(
                    schema="flightmare.scene.v2",
                    assets=[
                        dict(
                            name="Wall",
                            boxes=[
                                dict(size=[1, 4, 4], center=[0, 0, 0], visible=False)
                            ],
                        )
                    ],
                    instances=[dict(asset="Wall", position=[2, 0, 2], yaw_deg=0)],
                    visual_meshes=[
                        dict(
                            points=[
                                [3.5, 2, 0],
                                [3.5, -2, 0],
                                [3.5, -2, 4],
                                [3.5, 2, 4],
                            ],
                            indices=[0, 1, 2, 0, 2, 3],
                            color=[1, 0, 0],
                        )
                    ],
                )
            )
        )
    else:
        pytest.importorskip("yaml")
        path.write_bytes(WALL.read_bytes())
    env = Environment(3, 1, control_dt=0, scene=path)
    env.reset_state[0] = torch.tensor([0, 2, 4], device=env.device)
    env.reset_state[2].fill_(2)
    env.reset()
    env.step(random=False)
    assert env.collision_hit.tolist() == [0, 1, 0]
    camera = TiledCamera(3, 49, 33, depth=True, scene=env.scene)
    env.reset()
    camera.draw(env.state[:7].T)
    # First surface at 3.5 m, plus the 0.16 m forward camera mount.
    assert wp.to_torch(camera.depth)[0, :2, 16, 24].tolist() == pytest.approx(
        [3.34, 1.34], abs=2e-4
    )

    # Render the imported asset in an actual caller-owned graph.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with env.access(stream):
        captured_camera = TiledCamera(3, 48, 32, depth=True, scene=env.scene)
        captured_camera.draw(env.state[:7].T)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            _, _, _, valid = captured_camera.record_tick(env)
        graph.replay()
        assert valid.all()
        assert env.collision_hit.tolist() == [0, 1, 0]
        assert wp.to_torch(captured_camera.depth)[0, 0, 16, 24].item() == pytest.approx(
            3.34, abs=2e-4
        )


def test_unity_rotation_active_and_unpacked_constraints(tmp_path):
    pytest.importorskip("yaml")
    path = tmp_path / "course.unity"
    text = WALL.read_text()
    path.write_text(
        text.replace("y: 0, z: 0, w: 1", "y: 0.70710678, z: 0, w: 0.70710678", 1)
    )
    assert load_scene(path).colliders[0] == pytest.approx(
        [1, -1, 2, 0.5, 2, 2, 0, -1], abs=1e-5
    )
    path.write_text(text.replace("m_IsActive: 1", "m_IsActive: 0", 1))
    with pytest.raises(ValueError, match="no active"):
        load_scene(path)
    path.write_text(text + "\n--- !u!1001 &77\nPrefabInstance: {}\n")
    with pytest.raises(ValueError, match="linked prefab"):
        load_scene(path)
    path.write_text(text.replace("BoxCollider:", "SphereCollider:"))
    with pytest.raises(ValueError, match="SphereCollider"):
        load_scene(path)
