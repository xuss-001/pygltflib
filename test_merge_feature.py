"""Behavioral tests for the pygltflib scene-merge feature.

These tests generate their own tiny fixtures in ``tmp_path`` (including
external ``.bin`` and image files that deliberately share names across
different directories) so they do not need glTF-Sample-Models.
"""

import json
import struct
import zlib
from pathlib import Path

import pytest

from pygltflib import (
    GLTF2,
    Animation,
    Camera,
    Material,
    MergeError,
    Node,
    Scene,
    UnsupportedGltfError,
    InvalidReferenceError,
    Merger,
    merge_gltfs,
)
from pygltflib.utils import add_primitive


# ---------------------------------------------------------------------------
# Tiny binary / PNG helpers
# ---------------------------------------------------------------------------

_COMPONENT = {
    5120: ("<b", 1),
    5121: ("<B", 1),
    5122: ("<h", 2),
    5123: ("<H", 2),
    5125: ("<I", 4),
    5126: ("<f", 4),
}
_TYPE_COMPS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def read_accessor(gltf, accessor_index):
    accessor = gltf.accessors[accessor_index]
    buffer_view = gltf.bufferViews[accessor.bufferView]
    blob = gltf.binary_blob() or b""
    start = (buffer_view.byteOffset or 0) + (accessor.byteOffset or 0)
    prefix, size = _COMPONENT[accessor.componentType]
    char = prefix[1:]
    components = _TYPE_COMPS[accessor.type]
    fmt = "<" + char * components
    rows = []
    for row in range(accessor.count):
        values = struct.unpack_from(fmt, blob, start + row * components * size)
        rows.append(values[0] if components == 1 else values)
    return rows


def make_png(path, rgba):
    width = height = 2
    row = bytes(rgba) * width
    raw = b"".join(b"\x00" + row for _ in range(height))

    def chunk(chunk_type, data):
        crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
        return (
            struct.pack(">I", len(data))
            + chunk_type
            + data
            + struct.pack(">I", crc)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    Path(path).write_bytes(png)
    return png


def _pad4(blob):
    blob.extend(b"\x00" * ((-len(blob)) % 4))


def build_prop(directory, x0, color_factor, png_rgba, alternate_scene=False):
    """Create a small textured triangle prop with external bin + png."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    positions = [(x0, 0.0, 0.0), (x0 + 1.0, 0.0, 0.0), (x0, 1.0, 0.0)]
    uvs = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    indices = (0, 1, 2)

    index_bytes = struct.pack("<HHH", *indices)
    position_bytes = b"".join(struct.pack("<fff", *p) for p in positions)
    uv_bytes = b"".join(struct.pack("<ff", *uv) for uv in uvs)

    blob = bytearray()
    index_offset = 0
    blob += index_bytes
    _pad4(blob)
    position_offset = len(blob)
    blob += position_bytes
    uv_offset = len(blob)
    blob += uv_bytes

    (directory / "scene_data.bin").write_bytes(bytes(blob))
    png_bytes = make_png(directory / "surface.png", png_rgba)

    gltf_json = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"name": "MainScene", "nodes": [0]}],
        "nodes": [{"name": "PropNode", "mesh": 0}],
        "meshes": [
            {
                "name": "SharedMesh",
                "primitives": [
                    {
                        "attributes": {"POSITION": 1, "TEXCOORD_0": 2},
                        "indices": 0,
                        "material": 0,
                    }
                ],
            }
        ],
        "materials": [
            {
                "name": "SharedPaint",
                "pbrMetallicRoughness": {
                    "baseColorFactor": color_factor,
                    "baseColorTexture": {"index": 0},
                },
            }
        ],
        "textures": [{"name": "SharedTexture", "sampler": 0, "source": 0}],
        "samplers": [{"name": "SharedSampler"}],
        "images": [
            {"name": "SharedSurface", "uri": "surface.png", "mimeType": "image/png"}
        ],
        "buffers": [{"uri": "scene_data.bin", "byteLength": len(blob)}],
        "bufferViews": [
            {
                "buffer": 0,
                "byteOffset": index_offset,
                "byteLength": len(index_bytes),
                "target": 34963,
            },
            {
                "buffer": 0,
                "byteOffset": position_offset,
                "byteLength": len(position_bytes),
                "target": 34962,
            },
            {
                "buffer": 0,
                "byteOffset": uv_offset,
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
                "min": [x0, 0.0, 0.0],
                "max": [x0 + 1.0, 1.0, 0.0],
            },
            {"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC2"},
        ],
    }

    if alternate_scene:
        extra_positions = [
            (50.0, 50.0, 0.0),
            (51.0, 50.0, 0.0),
            (50.0, 51.0, 0.0),
        ]
        extra_index_bytes = struct.pack("<HHH", 0, 1, 2)
        extra_position_bytes = b"".join(
            struct.pack("<fff", *p) for p in extra_positions
        )
        data = bytearray((directory / "scene_data.bin").read_bytes())
        _pad4(data)
        extra_index_offset = len(data)
        data += extra_index_bytes
        _pad4(data)
        extra_position_offset = len(data)
        data += extra_position_bytes
        (directory / "scene_data.bin").write_bytes(bytes(data))

        gltf_json["buffers"][0]["byteLength"] = len(data)
        gltf_json["bufferViews"] += [
            {
                "buffer": 0,
                "byteOffset": extra_index_offset,
                "byteLength": 6,
                "target": 34963,
            },
            {
                "buffer": 0,
                "byteOffset": extra_position_offset,
                "byteLength": len(extra_position_bytes),
                "target": 34962,
            },
        ]
        gltf_json["accessors"] += [
            {"bufferView": 3, "componentType": 5123, "count": 3, "type": "SCALAR"},
            {
                "bufferView": 4,
                "componentType": 5126,
                "count": 3,
                "type": "VEC3",
                "min": [50.0, 50.0, 0.0],
                "max": [51.0, 51.0, 0.0],
            },
        ]
        gltf_json["meshes"].append(
            {
                "name": "AlternateMesh",
                "primitives": [
                    {"attributes": {"POSITION": 4}, "indices": 3}
                ],
            }
        )
        gltf_json["nodes"].append({"name": "AlternateNode", "mesh": 1})
        gltf_json["scenes"].append({"name": "AlternateScene", "nodes": [1]})

    (directory / "prop.gltf").write_text(json.dumps(gltf_json, indent=2))
    return png_bytes


@pytest.fixture()
def two_props(tmp_path):
    red_png = build_prop(
        tmp_path / "dir_a_red",
        0.0,
        [0.8, 0.1, 0.1, 1.0],
        (220, 30, 30, 255),
    )
    blue_png = build_prop(
        tmp_path / "dir_b_blue",
        10.0,
        [0.1, 0.2, 0.9, 1.0],
        (30, 60, 220, 255),
        alternate_scene=True,
    )
    red = GLTF2.load(str(tmp_path / "dir_a_red" / "prop.gltf"))
    blue = GLTF2.load(str(tmp_path / "dir_b_blue" / "prop.gltf"))
    return {
        "tmp": tmp_path,
        "red": red,
        "blue": blue,
        "red_png": red_png,
        "blue_png": blue_png,
    }


# ---------------------------------------------------------------------------
# Happy-path merging
# ---------------------------------------------------------------------------


def test_merge_creates_wrapper_roots_and_single_scene(two_props):
    merged = merge_gltfs(
        [two_props["red"], two_props["blue"]],
        translations=[[1.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
    )
    assert merged.scene == 0
    assert len(merged.scenes) == 1
    roots = merged.scenes[0].nodes
    assert len(roots) == 2
    for root_index in roots:
        root = merged.nodes[root_index]
        assert root.children, "wrapper root must parent the selected scene roots"
    assert merged.nodes[roots[0]].translation == [1.0, 0.0, 0.0]
    assert merged.nodes[roots[1]].translation == [5.0, 0.0, 0.0]


def test_scene_selection_picks_alternate_scene(two_props):
    merged = merge_gltfs(
        [
            two_props["red"],
            {
                "gltf": two_props["blue"],
                "scene": 1,
                "translation": [7.0, 0.0, 0.0],
                "scale": [2.0, 2.0, 2.0],
            },
        ]
    )
    blue_wrapper = merged.nodes[merged.scenes[0].nodes[1]]
    assert blue_wrapper.translation == [7.0, 0.0, 0.0]
    assert blue_wrapper.scale == [2.0, 2.0, 2.0]
    child_index = blue_wrapper.children[0]
    child_node = merged.nodes[child_index]
    # AlternateScene -> AlternateNode -> mesh whose POSITION values are at 50/51.
    positions = read_accessor(
        merged, merged.meshes[child_node.mesh].primitives[0].attributes.POSITION
    )
    xs = sorted({round(row[0], 3) for row in positions})
    assert xs == [50.0, 51.0]


def test_glb_roundtrip_geometry_materials_and_images(two_props, tmp_path):
    merged = merge_gltfs(
        [two_props["red"], two_props["blue"]],
        scenes=[0, 0],
        translations=[[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]],
    )
    out = tmp_path / "out" / "merged.glb"
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.save(str(out))
    assert out.is_file()

    reloaded = GLTF2.load(str(out))
    assert len(reloaded.materials) == 2
    assert len(reloaded.images) == 2
    assert len(reloaded.textures) == 2

    # Walk the two wrapper roots to the selected main-scene mesh of each prop.
    all_x = []
    material_factors = []
    image_prefixes = []
    for wrapper_index in reloaded.scenes[0].nodes:
        wrapper = reloaded.nodes[wrapper_index]
        mesh_index = reloaded.nodes[wrapper.children[0]].mesh
        primitive = reloaded.meshes[mesh_index].primitives[0]
        positions = read_accessor(reloaded, primitive.attributes.POSITION)
        all_x.extend(round(row[0], 3) for row in positions)
        material_index = primitive.material
        material = reloaded.materials[material_index]
        material_factors.append(
            tuple(material.pbrMetallicRoughness.baseColorFactor)
        )
        texture_index = (
            material.pbrMetallicRoughness.baseColorTexture.index
        )
        image_index = reloaded.textures[texture_index].source
        image = reloaded.images[image_index]
        assert image.bufferView is not None
        view = reloaded.bufferViews[image.bufferView]
        image_bytes = reloaded.binary_blob()[
            view.byteOffset : view.byteOffset + view.byteLength
        ]
        image_prefixes.append(image_bytes)

    assert any(abs(x - 0.0) < 1e-6 for x in all_x)
    assert any(abs(x - 10.0) < 1e-6 for x in all_x)
    assert [f[:3] for f in material_factors] == [
        pytest.approx([0.8, 0.1, 0.1]),
        pytest.approx([0.1, 0.2, 0.9]),
    ]

    red_png = two_props["red_png"]
    blue_png = two_props["blue_png"]
    assert image_prefixes[0].startswith(red_png)
    assert image_prefixes[1].startswith(blue_png)
    assert not image_prefixes[0].startswith(blue_png)
    assert not image_prefixes[1].startswith(red_png)


def test_same_named_resources_stay_distinct_but_shared_index_preserved():
    first = add_primitive(None)
    second = add_primitive(None)
    # Share material 0 between two primitives inside the first input.
    shared = Material(name="SameName")
    first.materials.append(shared)
    first.meshes[0].primitives[0].material = 0
    second.materials.append(Material(name="SameName"))
    second.meshes[0].primitives[0].material = 0

    merged = merge_gltfs(first, second)
    assert len(merged.materials) == 2
    assert merged.materials[0].name == "SameName"
    assert merged.materials[1].name == "SameName"
    # Index zero must survive as a real reference (truthiness bugs would drop it).
    assert merged.meshes[0].primitives[0].material == 0
    assert merged.meshes[1].primitives[0].material == 1


def test_intra_input_shared_references_remain_shared():
    gltf = add_primitive(None)
    gltf.materials.append(Material(name="Shared"))
    gltf.meshes[0].primitives[0].material = 0
    # Two meshes / primitives referencing the same material and accessors.
    gltf.meshes.append(__import__("pygltflib").Mesh())
    gltf.meshes[1].primitives.append(
        __import__("pygltflib").Primitive(material=0)
    )
    gltf.nodes.append(Node(mesh=1))
    gltf.scenes[0].nodes.append(len(gltf.nodes) - 1)

    merged = merge_gltfs(gltf)
    material_refs = [
        primitive.material
        for mesh in merged.meshes
        for primitive in mesh.primitives
        if primitive.material is not None
    ]
    assert material_refs == [0, 0]


def test_index_zero_everywhere_roundtrips(tmp_path):
    gltf = add_primitive(None)
    # add_primitive leaves indices/material unset; point them at index 0 so
    # the truthiness-sensitive zero references are exercised.
    gltf.meshes[0].primitives[0].indices = 0
    gltf.materials.append(Material(name="Zero"))
    gltf.meshes[0].primitives[0].material = 0
    merged = merge_gltfs(gltf)
    out = tmp_path / "zero.glb"
    merged.save(str(out))
    reloaded = GLTF2.load(str(out))
    assert reloaded.scenes[0].nodes == [len(reloaded.nodes) - 1]
    wrapper = reloaded.nodes[reloaded.scenes[0].nodes[0]]
    original_node = reloaded.nodes[wrapper.children[0]]
    assert original_node.mesh == 0
    primitive = reloaded.meshes[0].primitives[0]
    assert primitive.indices == 0
    assert primitive.material == 0
    assert primitive.attributes.POSITION == 1


# ---------------------------------------------------------------------------
# Transform semantics
# ---------------------------------------------------------------------------


def test_wrapper_trs_values_are_stored_verbatim():
    gltf = add_primitive(None)
    rotation = [0.0, 0.0, 0.70710678, 0.70710678]
    merged = merge_gltfs(
        [gltf],
        translation=[1.0, 2.0, 3.0],
        rotation=rotation,
        scale=[2.0, 2.0, 2.0],
    )
    wrapper = merged.nodes[merged.scenes[0].nodes[0]]
    assert wrapper.translation == [1.0, 2.0, 3.0]
    assert wrapper.rotation == pytest.approx(rotation)
    assert wrapper.scale == [2.0, 2.0, 2.0]
    assert wrapper.matrix is None


def test_instance_and_builder_entry_points_agree():
    a = add_primitive(None)
    b = add_primitive(None)
    m1 = GLTF2.merge([a, b], translations=[[0, 0, 0], [1, 0, 0]])
    m2 = GLTF2().merge(a, b)
    from pygltflib import Merger

    m3 = Merger().add(a).add(b, translation=[1, 0, 0]).merge()
    assert len(m1.scenes[0].nodes) == 2
    assert len(m2.scenes[0].nodes) == 2
    assert len(m3.scenes[0].nodes) == 2


# ---------------------------------------------------------------------------
# Rejection / validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mutate",
    [
        lambda g: setattr(g, "animations", [Animation()]),
        lambda g: setattr(g, "skins", [__import__("pygltflib").Skin()]),
        lambda g: setattr(g, "cameras", [Camera(type="perspective")]),
        lambda g: setattr(g, "extensionsUsed", ["KHR_draco_mesh_compression"]),
        lambda g: setattr(g.nodes[0], "camera", 0)
        or setattr(g, "cameras", [Camera(type="perspective")]),
        lambda g: setattr(g.meshes[0].primitives[0], "targets", [{}]),
        lambda g: setattr(g.nodes[0], "extensions", {"FOO_ext": {}}),
    ],
)
def test_unsupported_features_are_rejected(mutate):
    gltf = add_primitive(None)
    # Give the primitive a material-free static baseline; mutations add features.
    mutate(gltf)
    with pytest.raises((UnsupportedGltfError, MergeError, NotImplementedError)):
        merge_gltfs(gltf)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda g: setattr(g.nodes[0], "mesh", 99),
        lambda g: setattr(g.scenes[0], "nodes", [5]),
        lambda g: setattr(g.meshes[0].primitives[0], "material", 7),
        lambda g: setattr(g.meshes[0].primitives[0], "indices", 40),
        lambda g: setattr(g.accessors[0], "bufferView", 12),
        lambda g: setattr(g.bufferViews[0], "buffer", 3),
    ],
)
def test_invalid_references_raise(mutate):
    gltf = add_primitive(None)
    mutate(gltf)
    with pytest.raises((InvalidReferenceError, MergeError, IndexError, ValueError)):
        merge_gltfs(gltf)


def test_invalid_texture_reference_raises():
    import pygltflib

    gltf = add_primitive(None)
    gltf.textures.append(pygltflib.Texture(source=3))
    gltf.materials.append(
        Material(
            pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                baseColorTexture=pygltflib.TextureInfo(index=0)
            )
        )
    )
    gltf.meshes[0].primitives[0].material = 0
    with pytest.raises((InvalidReferenceError, MergeError, IndexError, ValueError)):
        merge_gltfs(gltf)


def test_missing_external_file_raises(tmp_path):
    build_prop(tmp_path / "p", 0.0, [1, 1, 1, 1], (0, 0, 0, 255))
    gltf = GLTF2.load(str(tmp_path / "p" / "prop.gltf"))
    (tmp_path / "p" / "surface.png").unlink()
    with pytest.raises((InvalidReferenceError, MergeError)):
        merge_gltfs(gltf)


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


def test_original_objects_unchanged_after_merge(two_props):
    red_before = two_props["red"].to_json()
    blue_before = two_props["blue"].to_json()
    merge_gltfs(
        [two_props["red"], two_props["blue"]],
        translations=[[1, 0, 0], [2, 0, 0]],
    )
    assert two_props["red"].to_json() == red_before
    assert two_props["blue"].to_json() == blue_before


def test_original_objects_unchanged_after_failed_merge():
    good = add_primitive(None)
    bad = add_primitive(None)
    bad.nodes[0].mesh = 99
    good_before = good.to_json()
    bad_before = bad.to_json()
    with pytest.raises(MergeError):
        merge_gltfs(good, bad)
    assert good.to_json() == good_before
    assert bad.to_json() == bad_before


def test_source_files_unchanged(two_props):
    red_path = two_props["tmp"] / "dir_a_red" / "prop.gltf"
    before = red_path.read_bytes()
    merge_gltfs(two_props["red"], output=str(two_props["tmp"] / "merged.glb"))
    assert red_path.read_bytes() == before


def test_path_inputs_resolve_per_input_directory(two_props, tmp_path):
    merged = merge_gltfs(
        str(two_props["tmp"] / "dir_a_red" / "prop.gltf"),
        str(two_props["tmp"] / "dir_b_blue" / "prop.gltf"),
        output=str(tmp_path / "path_merge.glb"),
    )
    assert len(merged.scenes[0].nodes) == 2
    reloaded = GLTF2.load(str(tmp_path / "path_merge.glb"))
    assert len(reloaded.images) == 2
    assert len(reloaded.scenes[0].nodes) == 2


def test_empty_inputs_returns_empty_scene():
    merged = merge_gltfs([])
    assert merged.scene == 0
    assert len(merged.scenes) == 1
    assert merged.scenes[0].nodes == []


def test_glb_file_inputs_with_external_images(tmp_path):
    png_a = build_prop(tmp_path / "ga", 3.0, [0.2, 0.3, 0.4, 1.0], (10, 20, 30, 255))
    build_prop(tmp_path / "gb", 7.0, [0.4, 0.5, 0.6, 1.0], (40, 50, 60, 255))
    # Convert the glTF props to GLB in their own directories.  Their images
    # stay external but their geometry buffers become GLB binary blobs.
    for name in ("ga", "gb"):
        gltf = GLTF2.load(str(tmp_path / name / "prop.gltf"))
        gltf.save_binary(str(tmp_path / name / "prop.glb"))

    merged = merge_gltfs(
        str(tmp_path / "ga" / "prop.glb"),
        str(tmp_path / "gb" / "prop.glb"),
    )
    out = tmp_path / "glb_merge.glb"
    merged.save_binary(str(out))
    reloaded = GLTF2.load(str(out))
    assert len(reloaded.scenes[0].nodes) == 2
    assert len(reloaded.images) == 2
    # First wrapper's mesh still carries the x=3 coordinates.
    wrapper = reloaded.nodes[reloaded.scenes[0].nodes[0]]
    mesh_index = reloaded.nodes[wrapper.children[0]].mesh
    positions = read_accessor(
        reloaded, reloaded.meshes[mesh_index].primitives[0].attributes.POSITION
    )
    assert round(positions[0][0], 3) == 3.0
    view = reloaded.bufferViews[reloaded.images[0].bufferView]
    image_bytes = reloaded.binary_blob()[
        view.byteOffset : view.byteOffset + view.byteLength
    ]
    assert image_bytes.startswith(png_a)


def test_index_values_survive_roundtrip(tmp_path):
    from pygltflib.utils import add_indexed_geometry

    gltf = GLTF2()
    gltf.scenes.append(Scene(nodes=[0]))
    gltf.scene = 0
    indices = [(0, 1, 2), (0, 2, 3)]
    vertices = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
    ]
    add_indexed_geometry(gltf, indices, vertices)
    # add_indexed_geometry builds the indices accessor but does not wire it.
    gltf.meshes[0].primitives[0].indices = 0
    # ...and hard-codes the index count to 3; correct it for two triangles.
    gltf.accessors[0].count = 6
    merged = merge_gltfs(gltf)
    out = tmp_path / "indexed.glb"
    merged.save_binary(str(out))
    reloaded = GLTF2.load(str(out))
    primitive = reloaded.meshes[0].primitives[0]
    decoded_indices = read_accessor(reloaded, primitive.indices)
    flat = []
    for row in decoded_indices:
        flat.extend(row if isinstance(row, tuple) else [row])
    assert flat == [0, 1, 2, 0, 2, 3]
    positions = read_accessor(reloaded, primitive.attributes.POSITION)
    assert positions[1] == pytest.approx((1.0, 0.0, 0.0), abs=1e-6)


def test_submodule_and_alias_entrypoints():
    import pygltflib.merge as merge_module
    from pygltflib.merge import merge_gltfs as via_merge_submodule
    from pygltflib.merger import Merger as MergerDirect

    a = add_primitive(None)
    assert via_merge_submodule([a]) is not None
    assert merge_module([a]) is not None  # callable module shim
    assert callable(merge_module.merge_gltf)
    assert MergerDirect is Merger


def test_rotation_and_scale_compose():
    import math

    gltf = add_primitive(None)
    translation = [1.0, 2.0, 3.0]
    angle = math.radians(90.0)
    rotation = [0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2)]
    scale = [2.0, 2.0, 2.0]
    merged = merge_gltfs(
        [gltf], translation=translation, rotation=rotation, scale=scale
    )
    wrapper = merged.nodes[merged.scenes[0].nodes[0]]

    # Rotate the scaled +X basis vector by the quaternion (+90 deg about Z
    # maps +X to +Y), then add the translation.
    x, y, z, w = wrapper.rotation
    vx = wrapper.scale[0]
    # quaternion * (vx,0,0) (rotation only)
    rx = (1 - 2 * (y * y + z * z)) * vx
    ry = (2 * (x * y + z * w)) * vx
    rz = (2 * (x * z - y * w)) * vx
    assert rx == pytest.approx(0.0, abs=1e-6)
    assert ry == pytest.approx(2.0, abs=1e-6)
    assert rz == pytest.approx(0.0, abs=1e-6)
    assert wrapper.translation[0] + rx == pytest.approx(1.0, abs=1e-6)
    assert wrapper.translation[1] + ry == pytest.approx(4.0, abs=1e-6)
    assert wrapper.translation[2] + rz == pytest.approx(3.0, abs=1e-6)
