#!/usr/bin/env python3
"""Demonstration / end-to-end check for pygltflib's merge feature.

It generates two props in *different* directories that deliberately use the
same file/resource names (``prop.gltf``, ``asset.bin``, ``texture.png`` and
even the same material/image names), merges them with per-input scene
selection and root transforms, saves the result as a GLB, reloads it and
checks geometry, materials, textures and index references.

Run from the repository root with the project virtualenv::

    python demo_merge.py
"""

import json
import struct
import zlib
from pathlib import Path

import numpy as np

from pygltflib import GLTF2, merge_gltfs


OUT_ROOT = Path(__file__).resolve().parent / "merge_demo_output"


# ---------------------------------------------------------------------------
# Fixture construction
# ---------------------------------------------------------------------------


def write_png(path, rgba):
    width = height = 4
    row = bytes(rgba) * width
    raw = b"".join(b"\x00" + row for _ in range(height))

    def chunk(kind, data):
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)
    return png


def pad4(blob):
    blob.extend(b"\x00" * ((-len(blob)) % 4))


def build_prop(folder, x0, color, rgba, with_alt_scene=False):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    positions = [(x0, 0, 0), (x0 + 1, 0, 0), (x0, 1, 0)]
    uvs = [(0, 0), (1, 0), (0, 1)]
    index_bytes = struct.pack("<HHH", 0, 1, 2)
    pos_bytes = b"".join(struct.pack("<fff", *p) for p in positions)
    uv_bytes = b"".join(struct.pack("<ff", *u) for u in uvs)

    blob = bytearray()
    off_index = 0
    blob += index_bytes
    pad4(blob)
    off_pos = len(blob)
    blob += pos_bytes
    off_uv = len(blob)
    blob += uv_bytes
    (folder / "asset.bin").write_bytes(blob)
    png_bytes = write_png(folder / "texture.png", rgba)

    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"name": "DefaultScene", "nodes": [0]}],
        "nodes": [{"name": "PropNode", "mesh": 0}],
        "meshes": [
            {
                "name": "SameMeshName",
                "primitives": [
                    {
                        "attributes": {"POSITION": 1, "TEXCOORD_0": 2},
                        "indices": 0,
                        "material": 0,
                    }
                ],
            }
        ],
        # Deliberately identical names across the two inputs.
        "materials": [
            {
                "name": "SameMaterialName",
                "pbrMetallicRoughness": {
                    "baseColorFactor": color,
                    "baseColorTexture": {"index": 0},
                },
            }
        ],
        "textures": [{"name": "SameTextureName", "sampler": 0, "source": 0}],
        "samplers": [{"name": "SameSamplerName"}],
        "images": [
            {
                "name": "SameImageName",
                "uri": "texture.png",
                "mimeType": "image/png",
            }
        ],
        "buffers": [{"uri": "asset.bin", "byteLength": len(blob)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": off_index, "byteLength": 6, "target": 34963},
            {
                "buffer": 0,
                "byteOffset": off_pos,
                "byteLength": len(pos_bytes),
                "target": 34962,
            },
            {
                "buffer": 0,
                "byteOffset": off_uv,
                "byteLength": len(uv_bytes),
                "target": 34962,
            },
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5123, "count": 3, "type": "SCALAR"},
            {
                "bufferView": 1,
                "componentType": 5126,
                "count": 3,
                "type": "VEC3",
                "min": [x0, 0, 0],
                "max": [x0 + 1, 1, 0],
            },
            {"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC2"},
        ],
    }

    if with_alt_scene:
        alt_pos = [(40, 40, 0), (42, 40, 0), (40, 42, 0)]
        alt_index = struct.pack("<HHH", 0, 1, 2)
        alt_pos_bytes = b"".join(struct.pack("<fff", *p) for p in alt_pos)
        data = bytearray((folder / "asset.bin").read_bytes())
        pad4(data)
        alt_off_index = len(data)
        data += alt_index
        pad4(data)
        alt_off_pos = len(data)
        data += alt_pos_bytes
        (folder / "asset.bin").write_bytes(data)
        gltf["buffers"][0]["byteLength"] = len(data)
        gltf["bufferViews"] += [
            {"buffer": 0, "byteOffset": alt_off_index, "byteLength": 6, "target": 34963},
            {
                "buffer": 0,
                "byteOffset": alt_off_pos,
                "byteLength": len(alt_pos_bytes),
                "target": 34962,
            },
        ]
        gltf["accessors"] += [
            {"bufferView": 3, "componentType": 5123, "count": 3, "type": "SCALAR"},
            {
                "bufferView": 4,
                "componentType": 5126,
                "count": 3,
                "type": "VEC3",
                "min": [40, 40, 0],
                "max": [42, 42, 0],
            },
        ]
        gltf["meshes"].append(
            {"name": "AlternateMesh",
             "primitives": [{"attributes": {"POSITION": 4}, "indices": 3}]}
        )
        gltf["nodes"].append({"name": "AlternateNode", "mesh": 1})
        gltf["scenes"].append({"name": "ChosenScene", "nodes": [1]})

    (folder / "prop.gltf").write_text(json.dumps(gltf, indent=2))
    return png_bytes


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------

COMP = {5123: ("H", 2), 5126: ("f", 4)}
NCOMPS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}


def read_accessor(gltf, index):
    acc = gltf.accessors[index]
    view = gltf.bufferViews[acc.bufferView]
    blob = gltf.binary_blob()
    start = (view.byteOffset or 0) + (acc.byteOffset or 0)
    char, size = COMP[acc.componentType]
    count = NCOMPS[acc.type]
    out = []
    for row in range(acc.count):
        vals = struct.unpack_from(
            "<" + char * count, blob, start + row * count * size
        )
        out.append(vals[0] if count == 1 else vals)
    return out


def trs_matrix(translation, rotation, scale):
    x, y, z, w = rotation
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    matrix = np.eye(4)
    matrix[:3, :3] = rot @ np.diag(scale)
    matrix[:3, 3] = translation
    return matrix


def main():
    if OUT_ROOT.exists():
        for child in sorted(OUT_ROOT.rglob("*"), reverse=True):
            if child.is_file():
                child.unlink()
    assets = OUT_ROOT / "assets"

    red_png = build_prop(
        assets / "dir_red" / "models",
        0.0,
        [0.85, 0.15, 0.1, 1.0],
        (230, 40, 40, 255),
    )
    blue_png = build_prop(
        assets / "dir_blue" / "props",
        10.0,
        [0.1, 0.3, 0.9, 1.0],
        (40, 90, 230, 255),
        with_alt_scene=True,
    )
    print("Created same-named fixtures in separate directories:")
    print("  ", (assets / "dir_red" / "models").relative_to(Path.cwd()))
    print("  ", (assets / "dir_blue" / "props").relative_to(Path.cwd()))

    red_path = assets / "dir_red" / "models" / "prop.gltf"
    blue_path = assets / "dir_blue" / "props" / "prop.gltf"

    # Path inputs are loaded and their external bin/png resolve relative to
    # each file's own parent directory.
    merged = merge_gltfs(
        {
            "path": str(red_path),
            "translation": [-2.0, 0.0, 0.0],
            "name": "red_root",
        },
        {
            "path": str(blue_path),
            "scene": 1,  # choose the alternate "ChosenScene"
            "translation": [3.0, 1.0, 0.0],
            "rotation": [0.0, 0.0, 0.7071067811865476, 0.7071067811865476],
            "scale": [1.5, 1.5, 1.5],
            "name": "blue_root",
        },
    )

    glb_path = OUT_ROOT / "merged_scene.glb"
    merged.save_binary(str(glb_path))
    reloaded = GLTF2.load(str(glb_path))
    print("Saved and reloaded GLB:", glb_path.relative_to(Path.cwd()))

    checks = []

    def check(title, condition):
        checks.append(bool(condition))
        print(("  [PASS] " if condition else "  [FAIL] ") + title)
        assert condition, title

    check("exactly one merged scene is the default", reloaded.scene == 0
          and len(reloaded.scenes) == 1)
    roots = reloaded.scenes[0].nodes
    check("merged scene has two independent root nodes", len(roots) == 2)

    red_root = reloaded.nodes[roots[0]]
    blue_root = reloaded.nodes[roots[1]]
    check("red root translation preserved", red_root.translation == [-2.0, 0.0, 0.0])
    check("blue root translation/scale preserved",
          blue_root.translation == [3.0, 1.0, 0.0]
          and blue_root.scale == [1.5, 1.5, 1.5])
    check("blue root rotation is a +90deg Z quaternion",
          np.allclose(blue_root.rotation, [0, 0, 0.70710678, 0.70710678], atol=1e-6))

    # Combined transform: T*R*S maps a unit X point as expected.
    matrix = trs_matrix(blue_root.translation, blue_root.rotation, blue_root.scale)
    point = matrix @ np.array([1.0, 0.0, 0.0, 1.0])
    check("combined T*R*S transform behaves as expected",
          np.allclose(point[:3], [3.0, 2.5, 0.0], atol=1e-5))

    # Scene selection: blue wrapper parents the alternate node (40/42 coords).
    blue_child = reloaded.nodes[blue_root.children[0]]
    blue_mesh = reloaded.meshes[blue_child.mesh]
    positions = read_accessor(reloaded, blue_mesh.primitives[0].attributes.POSITION)
    xs = sorted({round(p[0], 2) for p in positions})
    check("scene=1 selects the alternate scene geometry (40/42)", xs == [40.0, 42.0])

    # Red wrapper parents the default node (0/1 coords).
    red_child = reloaded.nodes[red_root.children[0]]
    red_mesh = reloaded.meshes[red_child.mesh]
    red_positions = read_accessor(
        reloaded, red_mesh.primitives[0].attributes.POSITION
    )
    red_xs = sorted({round(p[0], 2) for p in red_positions})
    check("default scene keeps red geometry (0/1)", red_xs == [0.0, 1.0])

    # Same-named materials remain two distinct resources with own factors.
    check("two same-named materials kept distinct",
          len(reloaded.materials) == 2
          and {m.name for m in reloaded.materials} == {"SameMaterialName"})
    factors = [
        tuple(m.pbrMetallicRoughness.baseColorFactor[:3])
        for m in reloaded.materials
    ]
    check("material factors are not mixed",
          any(np.allclose(f, [0.85, 0.15, 0.1]) for f in factors)
          and any(np.allclose(f, [0.1, 0.3, 0.9]) for f in factors))

    # Material -> texture -> image -> binary bytes correspondence.
    png_matches = {"red": False, "blue": False}
    for material in reloaded.materials:
        tex_index = material.pbrMetallicRoughness.baseColorTexture.index
        image_index = reloaded.textures[tex_index].source
        image = reloaded.images[image_index]
        view = reloaded.bufferViews[image.bufferView]
        data = reloaded.binary_blob()[
            view.byteOffset: view.byteOffset + view.byteLength
        ]
        if data.startswith(red_png):
            png_matches["red"] = True
        if data.startswith(blue_png):
            png_matches["blue"] = True
    check("red material samples red directory texture", png_matches["red"])
    check("blue material samples blue directory texture", png_matches["blue"])

    # Index-zero references are real references, not dropped by truthiness.
    prim0 = reloaded.meshes[0].primitives[0]
    check("index-zero material/indices/accessors still referenced",
          prim0.material == 0 and prim0.indices == 0
          and prim0.attributes.POSITION is not None)

    print()
    print(f"All {len(checks)} demonstration checks passed.")


if __name__ == "__main__":
    main()
