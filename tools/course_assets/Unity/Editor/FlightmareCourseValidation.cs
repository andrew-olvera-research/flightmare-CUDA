using System;
using System.Linq;
using UnityEditor;
using UnityEngine;

public static class FlightmareCourseValidation
{
    // Run in batch mode with -executeMethod FlightmareCourseValidation.ValidateFixtures.
    public static void ValidateFixtures()
    {
        Validate("room", 6);
        Validate("split_s", 34);
    }

    static void Validate(string name, int count)
    {
        string path = AssetDatabase.GetAllAssetPaths().Single(p =>
            p.StartsWith("Assets/") && p.EndsWith("/" + name + ".flightcourse"));
        AssetDatabase.ImportAsset(path, ImportAssetOptions.ForceSynchronousImport);
        var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(path);
        var root = UnityEngine.Object.Instantiate(prefab);
        try
        {
            Physics.SyncTransforms();
            var boxes = root.GetComponentsInChildren<BoxCollider>();
            if (boxes.Length != count || root.GetComponentsInChildren<MeshRenderer>().Length != count)
                throw new InvalidOperationException("Shape count differs for " + name);
            CheckOverlap(boxes, new Vector3(0, .1f, 0), true);
            CheckOverlap(boxes, new Vector3(0, 4, 0), false);
            if (name == "split_s")
            {
                CheckOverlap(boxes, new Vector3(1.27f, 3.48f, -.9f), false);
                CheckOverlap(boxes, new Vector3(1.27f, 3.48f, -.1f), true);
            }
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(root);
        }
    }

    static void CheckOverlap(BoxCollider[] boxes, Vector3 center, bool expected)
    {
        const float radius = .125f;
        bool hit = boxes.Any(box => (box.ClosestPoint(center) - center).sqrMagnitude <= radius * radius);
        if (hit != expected) throw new InvalidOperationException("Overlap mismatch at " + center);
    }
}
