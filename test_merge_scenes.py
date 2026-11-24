import base64
import hashlib
import struct
from pathlib import Path

import pytest

from pygltflib import (
    ARRAY_BUFFER,
    ELEMENT_ARRAY_BUFFER,
    FLOAT,
    SCALAR,
    UNSIGNED_SHORT,
    VEC2,
    VEC3,
    Accessor,
    Animation,
    Attributes,
    Buffer,
    BufferView,
    Camera,
    GLTF2,
    Image,
    Material,
    MergeError,
    MergeScene,
    Mesh,
    Node,
    PbrMetallicRoughness,
    Perspective,
    Primitive,
    Sampler,
    Scene,
    Skin,
    Texture,
    TextureInfo,
    merge_scenes,
)


PNG_RED = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
    "/x8AAusB9Y9ZlYQAAAAASUVORK5CYII="
)
PNG_BLUE = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _geometry(positions):
    indices = struct.pack("<HHH", 0, 1, 2)
    positions_data = struct.pack("<9f", *positions)
    texcoords = struct.pack("<6f", 0.0, 0.0, 1.0, 0.0, 0.0, 1.0)
    return indices + b"\x00\x00" + positions_data + texcoords


def _write_model(directory, positions, image_data, base_color):
    directory = Path(directory)
    directory.mkdir()
    buffer_data = _geometry(positions)
    (directory / "shared.bin").write_bytes(buffer_data)
    (directory / "texture.png").write_bytes(image_data)

    xs = positions[0::3]
    ys = positions[1::3]
    gltf = GLTF2(
        buffers=[Buffer(uri="shared.bin", byteLength=len(buffer_data))],
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
                    baseColorFactor=base_color,
                    baseColorTexture=TextureInfo(index=0),
                ),
            ),
            Material(
                name="material",
                pbrMetallicRoughness=PbrMetallicRoughness(
                    baseColorFactor=base_color,
                    baseColorTexture=TextureInfo(index=1),
                ),
            ),
        ],
        textures=[
            Texture(name="texture", source=0, sampler=0),
            Texture(name="texture", source=0, sampler=0),
        ],
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
            ),
            Mesh(
                name="mesh",
                primitives=[
                    Primitive(
                        attributes=Attributes(POSITION=0, TEXCOORD_0=1),
                        indices=2,
                        material=1,
                    )
                ],
            ),
        ],
        nodes=[
            Node(name="other", mesh=0),
            Node(name="selected", mesh=1, children=[2]),
            Node(name="child", mesh=0),
        ],
        scenes=[
            Scene(name="other scene", nodes=[0]),
            Scene(name="selected scene", nodes=[1]),
        ],
        scene=0,
    )
    gltf.save(directory / "model.gltf")
    return directory / "model.gltf"


def _load_models(tmp_path):
    first = _write_model(
        tmp_path / "first",
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        PNG_RED,
        [1.0, 0.0, 0.0, 1.0],
    )
    second = _write_model(
        tmp_path / "second",
        [0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.25, 0.0],
        PNG_BLUE,
        [0.0, 0.0, 1.0, 1.0],
    )
    return GLTF2().load(first), GLTF2().load(second), first, second


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _accessor_bytes(gltf, accessor_index):
    accessor = gltf.accessors[accessor_index]
    buffer_view = gltf.bufferViews[accessor.bufferView]
    component_sizes = {
        5120: 1,
        5121: 1,
        5122: 2,
        5123: 2,
        5125: 4,
        5126: 4,
    }
    component_counts = {
        "SCALAR": 1,
        "VEC2": 2,
        "VEC3": 3,
        "VEC4": 4,
        "MAT2": 4,
        "MAT3": 9,
        "MAT4": 16,
    }
    size = (
        component_sizes[accessor.componentType]
        * component_counts[accessor.type]
        * accessor.count
    )
    start = (buffer_view.byteOffset or 0) + (accessor.byteOffset or 0)
    return gltf.binary_blob()[start : start + size]


def _image_bytes(gltf, image_index):
    image = gltf.images[image_index]
    buffer_view = gltf.bufferViews[image.bufferView]
    start = buffer_view.byteOffset or 0
    return gltf.binary_blob()[start : start + buffer_view.byteLength]


def test_merge_scenes_glb_roundtrip(tmp_path):
    first, second, first_path, second_path = _load_models(tmp_path)
    source_state = (
        first.gltf_to_json(),
        second.gltf_to_json(),
        _digest(first_path.parent / "shared.bin"),
        _digest(second_path.parent / "shared.bin"),
        _digest(first_path.parent / "texture.png"),
        _digest(second_path.parent / "texture.png"),
    )

    merged = merge_scenes(
        [
            MergeScene(
                first,
                scene=1,
                translation=[1.0, 2.0, 3.0],
                rotation=[0.0, 0.0, 0.0, 1.0],
                scale=[2.0, 2.0, 2.0],
                name="first root",
            ),
            MergeScene(
                second,
                scene=1,
                translation=[-1.0, 0.0, 0.5],
                rotation=[0.0, 0.0, 0.0, 1.0],
                scale=[0.5, 0.5, 0.5],
                name="second root",
            ),
        ],
        scene_name="combined",
    )

    assert merged.scene == 0
    assert len(merged.scenes) == 1
    assert merged.scenes[0].name == "combined"
    assert len(merged.scenes[0].nodes) == 2
    assert merged.scenes[0].nodes == [6, 7]
    assert merged.nodes[6].children == [1]
    assert merged.nodes[7].children == [4]
    assert merged.nodes[6].translation == [1.0, 2.0, 3.0]
    assert merged.nodes[7].scale == [0.5, 0.5, 0.5]

    output = tmp_path / "merged.glb"
    merged.save(output)
    reloaded = GLTF2().load(output)

    assert len(reloaded.scenes) == 1
    assert reloaded.scenes[0].name == "combined"
    assert reloaded.scenes[0].nodes == [6, 7]
    assert reloaded.nodes[1].children == [2]
    assert reloaded.nodes[4].children == [5]
    assert reloaded.nodes[1].mesh == 1
    assert reloaded.nodes[4].mesh == 3
    assert reloaded.nodes[6].translation == [1.0, 2.0, 3.0]
    assert reloaded.nodes[7].translation == [-1.0, 0.0, 0.5]
    assert reloaded.nodes[7].scale == [0.5, 0.5, 0.5]

    assert len(reloaded.materials) == 4
    assert [m.name for m in reloaded.materials] == ["material"] * 4
    assert [
        m.pbrMetallicRoughness.baseColorTexture.index
        for m in reloaded.materials
    ] == [0, 1, 2, 3]
    assert reloaded.materials[0].pbrMetallicRoughness.baseColorFactor == [
        1.0,
        0.0,
        0.0,
        1.0,
    ]
    assert reloaded.materials[2].pbrMetallicRoughness.baseColorFactor == [
        0.0,
        0.0,
        1.0,
        1.0,
    ]
    assert [(t.source, t.sampler) for t in reloaded.textures] == [
        (0, 0),
        (0, 0),
        (1, 1),
        (1, 1),
    ]
    assert reloaded.meshes[0].primitives[0].material == 0
    assert reloaded.meshes[1].primitives[0].material == 1
    assert reloaded.meshes[1].primitives[0].attributes.POSITION == 0
    assert reloaded.meshes[1].primitives[0].indices == 2

    assert len(reloaded.images) == 2
    assert reloaded.images[0].name == reloaded.images[1].name == "texture"
    assert reloaded.images[0].uri is None
    assert reloaded.images[1].uri is None
    assert reloaded.images[0].bufferView != reloaded.images[1].bufferView
    assert _image_bytes(reloaded, 0) == PNG_RED
    assert _image_bytes(reloaded, 1) == PNG_BLUE

    expected_second_positions = struct.pack(
        "<9f", 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.25, 0.0
    )
    assert _accessor_bytes(reloaded, 3) == expected_second_positions
    assert _accessor_bytes(reloaded, 5) == struct.pack("<HHH", 0, 1, 2)
    assert len(reloaded.buffers) == 1

    assert first.gltf_to_json() == source_state[0]
    assert second.gltf_to_json() == source_state[1]
    assert _digest(first_path.parent / "shared.bin") == source_state[2]
    assert _digest(second_path.parent / "shared.bin") == source_state[3]
    assert _digest(first_path.parent / "texture.png") == source_state[4]
    assert _digest(second_path.parent / "texture.png") == source_state[5]


def test_merge_accepts_paths_and_parallel_options(tmp_path):
    _, _, first_path, second_path = _load_models(tmp_path)

    merged = merge_scenes(
        [first_path, second_path],
        scene_indices=[1, 1],
        translations=[[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
        rotations=[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
        scales=[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        names=["path one", "path two"],
    )

    assert merged.scenes[0].nodes == [6, 7]
    assert merged.nodes[6].name == "path one"
    assert merged.nodes[7].name == "path two"
    assert merged.nodes[6].children == [1]
    assert merged.nodes[7].children == [4]
    assert len(merged.images) == 2


def test_merge_accepts_transform_sequence_and_constructor_aliases(tmp_path):
    first, second, _, _ = _load_models(tmp_path)

    merged = merge_scenes(
        [
            MergeScene(first, scene_index=1, node_name="first alias"),
            MergeScene(second, scene_index=1, node_name="second alias"),
        ],
        [1, 1],
        [
            {
                "translation": [1.0, 0.0, 0.0],
                "rotation": [0.0, 0.0, 0.0, 1.0],
                "scale": [1.0, 1.0, 1.0],
            },
            {
                "translation": [-1.0, 0.0, 0.0],
                "rotation": [0.0, 0.0, 0.0, 1.0],
                "scale": [1.0, 1.0, 1.0],
            },
        ],
    )

    assert merged.nodes[6].name == "first alias"
    assert merged.nodes[7].name == "second alias"
    assert merged.nodes[6].translation == [1.0, 0.0, 0.0]
    assert merged.nodes[7].translation == [-1.0, 0.0, 0.0]


def test_merge_single_input_accepts_bare_vectors(tmp_path):
    first, _, _, _ = _load_models(tmp_path)
    merged = merge_scenes(
        first,
        scene_indices=1,
        translations=[1.0, 2.0, 3.0],
        rotations=[0.0, 0.0, 0.0, 1.0],
        scales=[2.0, 2.0, 2.0],
    )

    root = merged.nodes[merged.scenes[0].nodes[0]]
    assert root.translation == [1.0, 2.0, 3.0]
    assert root.rotation == [0.0, 0.0, 0.0, 1.0]
    assert root.scale == [2.0, 2.0, 2.0]
    assert root.children == [1]


def test_merge_reads_glb_buffers_and_buffer_view_images(tmp_path):
    first, _, first_path, _ = _load_models(tmp_path)
    buffer_data = (first_path.parent / "shared.bin").read_bytes() + PNG_RED
    first.buffers[0].uri = (
        "data:application/octet-stream;base64,"
        + base64.b64encode(buffer_data).decode("ascii")
    )
    first.buffers[0].byteLength = len(buffer_data)
    first.bufferViews.append(
        BufferView(
            buffer=0,
            byteOffset=68,
            byteLength=len(PNG_RED),
        )
    )
    first.images[0].uri = None
    first.images[0].bufferView = 3
    first.images[0].mimeType = None

    source_glb_path = tmp_path / "source.glb"
    first.save(source_glb_path)
    source_glb = GLTF2().load(source_glb_path)
    assert source_glb.images[0].bufferView == 3
    assert source_glb.images[0].mimeType is None

    merged = merge_scenes([MergeScene(source_glb, scene=1)])
    output = tmp_path / "merged_from_glb.glb"
    merged.save(output)
    reloaded = GLTF2().load(output)

    assert reloaded.images[0].mimeType == "image/png"
    assert _image_bytes(reloaded, 0) == PNG_RED
    assert reloaded.textures[0].source == 0
    assert _accessor_bytes(reloaded, 0) == struct.pack(
        "<9f", 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0
    )


@pytest.mark.parametrize(
    "kind",
    [
        "animation",
        "skin",
        "morph",
        "camera",
        "extension",
        "invalid-node",
        "invalid-texture",
    ],
)
def test_merge_rejects_unsupported_or_invalid_inputs(tmp_path, kind):
    first, second, _, _ = _load_models(tmp_path)
    if kind == "animation":
        first.animations.append(Animation())
    elif kind == "skin":
        first.skins.append(Skin())
    elif kind == "morph":
        first.meshes[0].primitives[0].targets = [Attributes(POSITION=0)]
    elif kind == "camera":
        first.cameras.append(
            Camera(
                type="perspective",
                perspective=Perspective(yfov=0.8, znear=0.1),
            )
        )
    elif kind == "extension":
        first.materials[0].extensions["TEST_unsupported"] = {}
    elif kind == "invalid-node":
        first.nodes[0].children = [99]
    elif kind == "invalid-texture":
        first.textures[0].source = 99

    before = first.gltf_to_json()
    with pytest.raises(MergeError):
        merge_scenes([first, second], scene_indices=[1, 1])
    assert first.gltf_to_json() == before


def test_merge_requires_explicit_scene_when_input_has_multiple(tmp_path):
    first, second, _, _ = _load_models(tmp_path)
    first.scene = None
    second.scene = None
    with pytest.raises(MergeError):
        merge_scenes([first, second])


def test_merge_rejects_invalid_scene_selection_without_mutation(tmp_path):
    first, second, _, _ = _load_models(tmp_path)
    before = first.gltf_to_json()
    with pytest.raises(MergeError):
        merge_scenes([first, second], scene_indices=[3, 1])
    assert first.gltf_to_json() == before
