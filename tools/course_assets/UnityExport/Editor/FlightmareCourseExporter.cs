// Requires Unity's com.unity.nuget.newtonsoft-json package. Install under Assets/Editor.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Newtonsoft.Json;
using UnityEditor;
using UnityEngine;
using UnityEngine.SceneManagement;

public static class FlightmareCourseExporter
{
    static Vector3 Flight(Vector3 p) => new Vector3(p.z, -p.x, p.y);
    static float[] XYZ(Vector3 p) => new[] { p.x, p.y, p.z };

    [MenuItem("Tools/Flightmare/Export active scene")]
    public static void ExportScene()
    {
        var scene = SceneManager.GetActiveScene();
        Export(scene.name, scene.GetRootGameObjects());
    }

    [MenuItem("Assets/Flightmare/Export selected prefab")]
    public static void ExportPrefab()
    {
        var path = AssetDatabase.GetAssetPath(Selection.activeObject);
        if (!path.EndsWith(".prefab", StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException("Select a prefab in the Project window.");
        var root = PrefabUtility.LoadPrefabContents(path);
        try { Export(root.name, new[] { root }); }
        finally { PrefabUtility.UnloadPrefabContents(root); }
    }

    static void Export(string name, GameObject[] roots)
    {
        var path = EditorUtility.SaveFilePanel("Export static Flightmare course", "", name, "flightcourse");
        if (string.IsNullOrEmpty(path)) return;
        var assets = new List<object>();
        var instances = new List<object>();
        var meshes = new List<object>();
        foreach (var root in roots)
        foreach (var t in root.GetComponentsInChildren<Transform>(true))
        {
            if (!t.gameObject.activeInHierarchy) continue;
            if (t.GetComponent<Rigidbody>() || t.GetComponent<Animator>() || t.GetComponent<Animation>() ||
                t.GetComponent<Terrain>() || t.GetComponent<SkinnedMeshRenderer>() || t.GetComponent<LODGroup>())
                throw new InvalidDataException($"Bake dynamic, terrain or LOD geometry first: {t.name}");
            foreach (var collider in t.GetComponents<Collider>())
            {
                if (!collider.enabled || collider.isTrigger) continue;
                if (!(collider is BoxCollider box))
                    throw new InvalidDataException($"Use BoxCollider proxies for {t.name}");
                var a = Flight(t.TransformVector(Vector3.forward));
                var b = Flight(t.TransformVector(Vector3.right));
                var c = Flight(t.TransformVector(Vector3.up));
                if (a.magnitude == 0 || b.magnitude == 0 || c.magnitude == 0 ||
                    Mathf.Abs(a.normalized.z) > 1e-5f || Mathf.Abs(b.normalized.z) > 1e-5f ||
                    Mathf.Abs(c.normalized.x) > 1e-5f || Mathf.Abs(c.normalized.y) > 1e-5f ||
                    Mathf.Abs(Vector3.Dot(a.normalized, b.normalized)) > 1e-5f)
                    throw new InvalidDataException($"Tilted, sheared or degenerate box: {t.name}");
                var id = "Box" + assets.Count;
                var size = new[] { a.magnitude * box.size.z, b.magnitude * box.size.x, c.magnitude * box.size.y };
                assets.Add(new { name = id, boxes = new[] { new { name = t.name, size, center = new float[3], visible = false } } });
                instances.Add(new { name = id, asset = id, position = XYZ(Flight(t.TransformPoint(box.center))),
                                    yaw_deg = Mathf.Atan2(a.y, a.x) * Mathf.Rad2Deg });
            }
            var renderer = t.GetComponent<MeshRenderer>();
            if (!renderer || !renderer.enabled) continue;
            var filter = t.GetComponent<MeshFilter>();
            if (!filter || !filter.sharedMesh) throw new InvalidDataException($"Missing mesh: {t.name}");
            var mesh = filter.sharedMesh;
            var points = mesh.vertices.Select(v => XYZ(Flight(t.TransformPoint(v)))).ToArray();
            var materials = renderer.sharedMaterials;
            for (int sub = 0; sub < mesh.subMeshCount; ++sub)
            {
                if (mesh.GetTopology(sub) != MeshTopology.Triangles)
                    throw new InvalidDataException($"Expected triangles: {t.name}");
                var indices = mesh.GetTriangles(sub);
                if (indices.Length == 0) continue;
                if (t.localToWorldMatrix.determinant > 0)
                    for (int i = 0; i < indices.Length; i += 3)
                        (indices[i+1], indices[i+2]) = (indices[i+2], indices[i+1]);
                var mat = sub < materials.Length ? materials[sub] : null;
                var rgb = mat && mat.HasProperty("_BaseColor") ? mat.GetColor("_BaseColor")
                        : mat && mat.HasProperty("_Color") ? mat.GetColor("_Color") : Color.gray;
                meshes.Add(new { points, indices, color = new[] { rgb.r, rgb.g, rgb.b } });
            }
        }
        File.WriteAllText(path, JsonConvert.SerializeObject(new {
            schema = "flightmare.scene.v2", assets, instances, visual_meshes = meshes
        }));
    }
}
