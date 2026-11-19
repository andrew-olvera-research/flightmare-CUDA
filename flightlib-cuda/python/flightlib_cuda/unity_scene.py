"""Unity Force Text adapter for unpacked static box scenes and prefabs."""

import math
import re
import numpy as np

_HEADER = re.compile(r"^--- !u!(\d+) &(-?\d+)( stripped)?\s*$", re.MULTILINE)
_NEUTRAL = [0.5, 0.54, 0.58]


def read_unity(path):
    import yaml

    text = path.read_text(encoding="utf-8-sig")
    headers = list(_HEADER.finditer(text))
    if not headers:
        raise ValueError("expected Unity Force Text scene or prefab")
    components = []
    objects, transforms, owners = {}, {}, {}
    ids = set()
    for i, header in enumerate(headers):
        kind, identifier, stripped = header.groups()
        identifier = int(identifier)
        if identifier in ids:
            raise ValueError(f"duplicate Unity fileID {identifier}")
        ids.add(identifier)
        if stripped or kind == "1001":
            raise ValueError(
                "linked prefab: unpack it or use the Flightmare Unity exporter"
            )
        doc = yaml.safe_load(
            text[
                header.end() : headers[i + 1].start() if i + 1 < len(headers) else None
            ]
        )
        name, obj = next(iter(doc.items()))
        if name == "GameObject":
            objects[identifier] = obj
        elif name in ("Transform", "RectTransform"):
            transforms[identifier] = obj
            owners[obj["m_GameObject"]["fileID"]] = identifier
        elif "m_GameObject" in obj:
            components.append((name, obj))

    def vector(data):
        return np.array([data[k] for k in "xyz"], dtype=float)

    placed = {}

    def place(key, visiting):
        if key in placed:
            return placed[key]
        if key in visiting or key not in transforms:
            raise ValueError(f"invalid Unity transform hierarchy at {key}")
        t = transforms[key]
        q = t["m_LocalRotation"]
        xyz, w = vector(q), float(q["w"])
        norm = math.sqrt(xyz @ xyz + w * w)
        if not math.isfinite(norm) or norm == 0:
            raise ValueError("invalid Unity quaternion")
        x, y, z = xyz / norm
        cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        m = np.eye(4)
        m[:3, :3] = (np.eye(3) + 2 * ((w / norm) * cross + cross @ cross)) @ np.diag(
            vector(t["m_LocalScale"])
        )
        m[:3, 3] = vector(t["m_LocalPosition"])
        active = bool(objects[t["m_GameObject"]["fileID"]]["m_IsActive"])
        parent = t["m_Father"]["fileID"]
        if parent:
            pm, pa = place(parent, visiting | {key})
            m, active = pm @ m, active and pa
        placed[key] = m, active
        return placed[key]

    assets, instances = [], []
    renderers = {
        o["m_GameObject"]["fileID"]: o for n, o in components if n == "MeshRenderer"
    }
    for name, obj in components:
        owner = obj["m_GameObject"]["fileID"]
        m, active = place(owners[owner], set())
        if not active or not obj.get("m_Enabled", 1):
            continue
        if name.endswith("Collider") and obj.get("m_IsTrigger"):
            continue
        collision = name == "BoxCollider"
        if collision:
            center, size = vector(obj["m_Center"]), vector(obj["m_Size"])
        elif name == "MeshFilter":
            renderer = renderers.get(owner)
            if renderer is None or not renderer.get("m_Enabled", 1):
                continue
            mesh = obj["m_Mesh"]
            if (
                mesh.get("fileID") != 10202
                or mesh.get("guid") != "0000000000000000e000000000000000"
            ):
                raise ValueError(
                    "external Unity mesh: use the Flightmare Unity exporter"
                )
            center, size = np.zeros(3), np.ones(3)
        elif name.endswith("Collider") or name in (
            "Rigidbody",
            "Terrain",
            "SkinnedMeshRenderer",
            "Animator",
            "Animation",
        ):
            raise ValueError(f"unsupported static Unity component {name}")
        else:
            continue
        # Unity right/up/forward to Flightlib forward/left/up; local box axes z,x,y.
        basis = np.array([[0, 0, 1], [-1, 0, 0], [0, 1, 0]])
        axes = (basis @ m[:3, :3])[:, [2, 0, 1]]
        lengths = np.linalg.norm(axes, axis=0)
        if np.any(lengths == 0):
            raise ValueError("degenerate Unity box")
        a, b, c = (axes / lengths).T
        if max(abs(a[2]), abs(b[2]), abs(c[0]), abs(c[1]), abs(a @ b)) > 1e-6:
            raise ValueError("tilted or sheared Unity box")
        asset = f"Box{len(assets)}"
        assets.append(
            dict(
                name=asset,
                boxes=[
                    dict(
                        center=[0, 0, 0],
                        size=(size[[2, 0, 1]] * lengths).tolist(),
                        collision=collision,
                        visible=not collision,
                        color=_NEUTRAL,
                    )
                ],
            )
        )
        instances.append(
            dict(
                name=asset,
                asset=asset,
                position=(basis @ (m[:3, :3] @ center + m[:3, 3])).tolist(),
                yaw_deg=math.degrees(math.atan2(a[1], a[0])),
            )
        )
    if not assets:
        raise ValueError("no active Unity box geometry")
    return dict(schema="flightmare.scene.v2", assets=assets, instances=instances)
