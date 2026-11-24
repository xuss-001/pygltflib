"""Demonstrate merging two glTF props that use identically named assets.

Run from the repository root:

    python examples/merge_scenes_demo.py merged_scene.glb

The script creates two temporary model directories, each containing its own
``shared.bin`` and ``texture.png``. It merges the selected scene from each
directory, saves a GLB, reloads it, and verifies resource and geometry mapping.
"""

import argparse
import base64
import hashlib
import struct
import tempfile
from pathlib import Path

from pygltflib import (
    ARRAY_BUFFER,
    ELEMENT_ARRAY_BUFFER,
    FLOAT,
    SCALAR,
    UNSIGNED_SHORT,
    VEC2,
    VEC3,
    Accessor,
    Attributes,
    Buffer,
    BufferView,
    GLTF2,
    Image,
    Material,
    MergeScene,
    Mesh,
    Node,
    PbrMetallicRoughness,
    Primitive,
    Sampler,
    Scene,
    Texture,
    TextureInfo,
    merge_scenes,
)


PNG_A = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
    "/x8AAusB9Y9ZlYQAAAAASUVORK5CYII="
)
PNG_B = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def write_prop(directory, positions, image_data, color):
    directory = Path(directory)
    directory.mkdir()
    indices = struct.pack("<HHH", 0, 1, 2)
    position_data = struct.pack("<9f", *positions)
    texcoords = struct.pack("<6f", 0.0, 0.0, 1.0, 0.0, 0.0, 1.0)
    binary = indices + b"\x00\x00" + position_data + texcoords
    (directory / "shared.bin").write_bytes(binary)
    (directory / "texture.png").write_bytes(image_data)

    xs = positions[0::3]
    ys = positions[1::3]
    gltf = GLTF2(
        buffers=[Buffer(uri="shared.bin", byteLength=len(binary))],
        bufferViews=[
            BufferView(
                buffer=0,
                byteOffset=0,
                byteLength=6,
                target=ELEMENT_ARRAY_BUFFER,
            ),
            BufferView(
                buffer=0,
                byteOffset=8,
                byteLength=36,
                target=ARRAY_BUFFER,
            ),
            BufferView(
                buffer=0,
                byteOffset=44,
                byteLength=24,
                target=ARRAY_BUFFER,
            ),
        ],
        accessors=[
            Accessor(
                bufferView=1,
                componentType=FLOAT,
                count=3,
                type=VEC3,
                min=[min(xs), min(ys), 0.0],
                max=[max(xs), max(ys), 0.0],
            ),
            Accessor(
                bufferView=2,
                componentType=FLOAT,
                count=3,
                type=VEC2,
            ),
            Accessor(
                bufferView=0,
                componentType=UNSIGNED_SHORT,
                count=3,
                type=SCALAR,
                min=[0],
                max=[2],
            ),
        ],
        materials=[
            Material(
                name="material",
                pbrMetallicRoughness=PbrMetallicRoughness(
                    baseColorFactor=color,
                    baseColorTexture=TextureInfo(index=0),
                ),
            )
        ],
        textures=[Texture(name="texture", source=0, sampler=0)],
        images=[Image(name="texture", uri="texture.png")],
        samplers=[Sampler()],
        meshes=[
            Mesh(
                name="mesh",
                primitives=[
                    Primitive(
                        attributes=Attributes(POSITION=0, TEXCOORD_0=1),
                        indices=2,
                        material=0,
                    )
                ],
            )
        ],
        nodes=[
            Node(name="unused", mesh=0),
            Node(name="selected", mesh=0),
        ],
        scenes=[
            Scene(name="unused scene", nodes=[0]),
            Scene(name="selected scene", nodes=[1]),
        ],
        scene=0,
    )
    gltf.save(directory / "model.gltf")
    return directory / "model.gltf"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", nargs="?", default="merged_scene.glb")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        first_path = write_prop(
            tmp / "red",
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            PNG_A,
            [1.0, 0.0, 0.0, 1.0],
        )
        second_path = write_prop(
            tmp / "blue",
            [0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.25, 0.0],
            PNG_B,
            [0.0, 0.0, 1.0, 1.0],
        )
        before = {
            path: digest(path)
            for path in [
                first_path.parent / "shared.bin",
                first_path.parent / "texture.png",
                second_path.parent / "shared.bin",
                second_path.parent / "texture.png",
            ]
        }

        first = GLTF2().load(first_path)
        second = GLTF2().load(second_path)
        merged = merge_scenes(
            [
                MergeScene(
                    first,
                    scene=1,
                    translation=[1.0, 0.0, 0.0],
                    rotation=[0.0, 0.0, 0.0, 1.0],
                    scale=[1.0, 1.0, 1.0],
                    name="red prop",
                ),
                MergeScene(
                    second,
                    scene=1,
                    translation=[-1.0, 0.0, 0.0],
                    rotation=[0.0, 0.0, 0.0, 1.0],
                    scale=[1.0, 1.0, 1.0],
                    name="blue prop",
                ),
            ],
            scene_name="demonstration scene",
        )
        merged.save(args.output)
        assert digest(first_path.parent / "shared.bin") == before[
            first_path.parent / "shared.bin"
        ]
        assert digest(second_path.parent / "texture.png") == before[
            second_path.parent / "texture.png"
        ]

    reloaded = GLTF2().load(args.output)
    roots = reloaded.scenes[reloaded.scene].nodes
    print("Merged scene:", reloaded.scenes[reloaded.scene].name)
    print("Root nodes:", roots)
    for index in roots:
        node = reloaded.nodes[index]
        print(
            "  {}: translation={} rotation={} scale={} children={}".format(
                node.name,
                node.translation,
                node.rotation,
                node.scale,
                node.children,
            )
        )

    print("Material -> texture -> image:")
    for index, material in enumerate(reloaded.materials):
        texture_index = material.pbrMetallicRoughness.baseColorTexture.index
        image_index = reloaded.textures[texture_index].source
        print("  {} -> {} -> {}".format(index, texture_index, image_index))

    assert reloaded.materials[0].pbrMetallicRoughness.baseColorTexture.index == 0
    assert reloaded.materials[1].pbrMetallicRoughness.baseColorTexture.index == 1
    assert reloaded.textures[0].source == 0
    assert reloaded.textures[1].source == 1
    print("Saved and reloaded:", args.output)


if __name__ == "__main__":
    main()
