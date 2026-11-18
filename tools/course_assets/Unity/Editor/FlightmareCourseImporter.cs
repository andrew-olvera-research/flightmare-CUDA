// Install under Assets/Editor and import the .flightcourse fixtures as prefabs.
using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor.AssetImporters;
using UnityEngine;

[ScriptedImporter(1, "flightcourse")]
public class FlightmareCourseImporter : ScriptedImporter
{
    [Serializable] class Box { public string name; public float[] center, size; }
    [Serializable] class Asset { public string name; public Box[] boxes; }
    [Serializable] class Instance { public string name, asset; public float[] position; public float yaw_deg; }
    [Serializable] class Scene { public string schema; public Asset[] assets; public Instance[] instances; }

    static Vector3 Position(float[] xyz) => new Vector3(-xyz[1], xyz[2], xyz[0]);

    public override void OnImportAsset(AssetImportContext ctx)
    {
        var scene = JsonUtility.FromJson<Scene>(File.ReadAllText(ctx.assetPath));
        if (scene.schema != "flightmare.boxes.v1") throw new InvalidDataException("Unknown course schema");
        var assets = new Dictionary<string, Box[]>();
        foreach (var asset in scene.assets) assets.Add(asset.name, asset.boxes);
        var root = new GameObject(Path.GetFileNameWithoutExtension(ctx.assetPath));
        ctx.AddObjectToAsset("course", root);
        ctx.SetMainObject(root);
        foreach (var instance in scene.instances)
        {
            var group = new GameObject(instance.name).transform;
            group.SetParent(root.transform, false);
            group.localPosition = Position(instance.position);
            group.localRotation = Quaternion.Euler(0, -instance.yaw_deg, 0);
            foreach (var box in assets[instance.asset])
            {
                var shape = GameObject.CreatePrimitive(PrimitiveType.Cube);
                shape.name = box.name;
                shape.transform.SetParent(group, false);
                shape.transform.localPosition = Position(box.center);
                shape.transform.localScale = new Vector3(box.size[1], box.size[2], box.size[0]);
            }
        }
    }
}
