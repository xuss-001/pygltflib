import math

import numpy as np
import pytest

import pygltflib
from pygltflib import (
    Buffer,
    GLTF2,
    InvalidNodeIndexError,
    InvalidNodeTransformError,
    NonInvertibleMatrixError,
    Node,
    Scene,
    SceneGraphStructureError,
    ShearNotAllowedError,
    UnsupportedAssetError,
)


def quaternion_y(angle):
    return [0.0, math.sin(angle / 2.0), 0.0, math.cos(angle / 2.0)]


def trs_matrix(translation=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0, 1.0),
               scale=(1.0, 1.0, 1.0)):
    x, y, z, w = rotation
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    xw, yw, zw = x * w, y * w, z * w
    rotation_matrix = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - zw), 2.0 * (xz + yw)],
            [2.0 * (xy + zw), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - xw)],
            [2.0 * (xz - yw), 2.0 * (yz + xw), 1.0 - 2.0 * (xx + yy)],
        ]
    )
    matrix = np.eye(4)
    matrix[:3, :3] = rotation_matrix @ np.diag(scale)
    matrix[:3, 3] = translation
    return matrix


def world_matrices(gltf, scene_index=0):
    return {
        entry.index: np.array(entry.world_matrix)
        for entry in gltf.traverse_scene(scene_index)
    }


def assert_world_matrices_equal(before, after):
    assert before.keys() == after.keys()
    for index in before:
        np.testing.assert_allclose(after[index], before[index], atol=1e-10)


def test_traverse_scene_preorder_and_matrices():
    matrix_translation = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        1.0, 2.0, 3.0, 1.0,
    ]
    gltf = GLTF2(
        scene=0,
        scenes=[Scene(nodes=[2, 0])],
        nodes=[
            Node(matrix=matrix_translation, children=[1]),
            Node(translation=[0.0, 1.0, 0.0], scale=[2.0, 3.0, 4.0]),
            Node(translation=[10.0, 0.0, 0.0], rotation=quaternion_y(math.pi / 2.0),
                 children=[3]),
            Node(translation=[1.0, 2.0, 3.0], scale=[-1.0, 2.0, 1.0]),
            Node(matrix=[1.0] * 16, translation=[9.0, 9.0, 9.0]),
        ],
    )

    entries = gltf.traverse_scene()

    assert [entry.index for entry in entries] == [2, 3, 0, 1]
    assert [entry.ancestors for entry in entries] == [[], [2], [], [0]]
    assert [entry.ancestor_indices for entry in entries] == [[], [2], [], [0]]
    assert entries[0]["node_index"] == 2

    root_matrix = np.eye(4)
    root_matrix[:3, 3] = [1.0, 2.0, 3.0]
    np.testing.assert_allclose(entries[2].local_matrix, root_matrix)

    expected_node_3 = (
        trs_matrix(translation=[10.0, 0.0, 0.0], rotation=quaternion_y(math.pi / 2.0))
        @ trs_matrix(translation=[1.0, 2.0, 3.0], scale=[-1.0, 2.0, 1.0])
    )
    np.testing.assert_allclose(entries[1].world_matrix, expected_node_3)

    expected_node_1 = root_matrix @ trs_matrix(
        translation=[0.0, 1.0, 0.0], scale=[2.0, 3.0, 4.0]
    )
    np.testing.assert_allclose(entries[3].world_matrix, expected_node_1)
    assert entries[2].local_matrix_gltf == gltf.nodes[0].matrix
    assert entries[0].world_matrix_gltf == entries[0].world_matrix_flat
    index, path, local, world = entries[1]
    assert index == 3
    assert path == [2]
    np.testing.assert_allclose(local, entries[1].local_matrix)
    np.testing.assert_allclose(world, entries[1].world_matrix)


def test_traverse_scene_allows_shared_nodes_across_scenes():
    gltf = GLTF2(
        scene=0,
        scenes=[Scene(nodes=[0, 2]), Scene(nodes=[2])],
        nodes=[
            Node(translation=[1.0, 0.0, 0.0]),
            Node(translation=[0.0, 1.0, 0.0]),
            Node(translation=[0.0, 0.0, 2.0]),
        ],
    )

    first = gltf.traverse_scene(0)
    second = gltf.traverse_scene(1)

    assert [entry.index for entry in first] == [0, 2]
    assert [entry.index for entry in second] == [2]
    assert second[0].ancestors == []
    np.testing.assert_allclose(
        first[1].world_matrix, trs_matrix(translation=[0.0, 0.0, 2.0])
    )


def test_traverse_scene_rejects_invalid_graphs():
    with pytest.raises(InvalidNodeIndexError):
        GLTF2(scenes=[Scene(nodes=[3])], nodes=[Node()]).traverse_scene()

    with pytest.raises(InvalidNodeIndexError):
        GLTF2(
            scenes=[Scene(nodes=[0])],
            nodes=[Node(children=[4])],
        ).traverse_scene()

    with pytest.raises(SceneGraphStructureError):
        GLTF2(
            scenes=[Scene(nodes=[0])],
            nodes=[Node(children=[1]), Node(children=[0])],
        ).traverse_scene()

    with pytest.raises(SceneGraphStructureError):
        GLTF2(
            scenes=[Scene(nodes=[0, 1])],
            nodes=[
                Node(children=[2]),
                Node(children=[2]),
                Node(),
            ],
        ).traverse_scene()

    with pytest.raises(InvalidNodeTransformError):
        GLTF2(
            scenes=[Scene(nodes=[0])],
            nodes=[Node(matrix=[1.0] * 16, translation=[0.0, 0.0, 0.0])],
        ).traverse_scene()


def hierarchy_asset():
    return GLTF2(
        scene=0,
        scenes=[Scene(nodes=[0, 1])],
        buffers=[Buffer(uri="data:application/octet-stream;base64,AAAA", byteLength=3)],
        nodes=[
            Node(
                name="parent-a",
                translation=[1.0, 2.0, 3.0],
                rotation=quaternion_y(0.7),
                scale=[2.0, 1.0, 0.5],
                children=[2],
            ),
            Node(
                name="parent-b",
                translation=[-1.0, 0.0, 2.0],
                rotation=quaternion_y(0.7),
                scale=[-2.0, 1.0, 0.5],
            ),
            Node(
                name="subtree-root",
                translation=[0.5, -1.0, 2.0],
                rotation=quaternion_y(-0.3),
                scale=[1.0, 2.0, 1.0],
                children=[3],
            ),
            Node(
                name="subtree-child",
                translation=[1.0, 0.0, 0.0],
                rotation=quaternion_y(1.1),
                scale=[0.5, 1.0, 2.0],
            ),
        ],
    )


def test_reparent_preserves_subtree_with_rotation_nonuniform_and_negative_scale():
    gltf = hierarchy_asset()
    original_json = gltf.gltf_to_json()
    before = world_matrices(gltf)

    result = gltf.reparent_node(2, 1)

    assert result is not gltf
    assert result.nodes[2] is not gltf.nodes[2]
    assert result.buffers is not gltf.buffers
    assert result.buffers[0].uri == gltf.buffers[0].uri
    assert gltf.gltf_to_json() == original_json
    assert gltf.scenes[0].nodes == [0, 1]
    assert gltf.nodes[0].children == [2]

    assert result.scenes[0].nodes == [0, 1]
    assert result.nodes[0].children == []
    assert result.nodes[1].children == [2]
    assert result.nodes[2].matrix is None
    assert any(scale < 0.0 for scale in result.nodes[2].scale)
    assert_world_matrices_equal(before, world_matrices(result))


def test_reparent_to_root_preserves_subtree_and_other_worlds():
    gltf = GLTF2(
        scene=0,
        scenes=[Scene(nodes=[0])],
        nodes=[
            Node(
                translation=[3.0, -1.0, 2.0],
                rotation=quaternion_y(0.9),
                scale=[-2.0, 0.5, 1.5],
                children=[1],
            ),
            Node(
                translation=[1.0, 2.0, -3.0],
                children=[2],
            ),
            Node(
                translation=[0.25, 0.0, 0.0],
                rotation=quaternion_y(-0.4),
                scale=[2.0, 3.0, 0.5],
            ),
        ],
    )
    before = world_matrices(gltf)

    result = gltf.reparent_node(1, None)

    assert result.scenes[0].nodes == [0, 1]
    assert result.nodes[0].children == []
    assert_world_matrices_equal(before, world_matrices(result))


def test_reparent_rejects_shear_when_world_cannot_be_trs():
    gltf = GLTF2(
        scene=0,
        scenes=[Scene(nodes=[0])],
        nodes=[
            Node(scale=[2.0, 1.0, 1.0], children=[1]),
            Node(rotation=quaternion_y(math.pi / 4.0)),
        ],
    )

    with pytest.raises(ShearNotAllowedError):
        gltf.reparent_node(1, None)


def test_reparent_rejects_singular_new_parent():
    gltf = GLTF2(
        scene=0,
        scenes=[Scene(nodes=[0, 1])],
        nodes=[
            Node(scale=[1.0, 0.0, 1.0]),
            Node(translation=[1.0, 2.0, 3.0]),
        ],
    )

    with pytest.raises(NonInvertibleMatrixError):
        gltf.reparent_node(1, 0)


def test_reparent_rejects_cycle():
    gltf = GLTF2(
        scene=0,
        scenes=[Scene(nodes=[0])],
        nodes=[Node(children=[1]), Node()],
    )

    with pytest.raises(SceneGraphStructureError):
        gltf.reparent_node(0, 1)


def test_reparent_rejects_invalid_and_unreachable_nodes():
    gltf = GLTF2(
        scene=0,
        scenes=[Scene(nodes=[0])],
        nodes=[Node(), Node()],
    )

    with pytest.raises(InvalidNodeIndexError):
        gltf.reparent_node(3, None)

    with pytest.raises(InvalidNodeIndexError):
        gltf.reparent_node(0, 3)

    with pytest.raises(InvalidNodeIndexError):
        gltf.reparent_node(1, None)


def test_reparent_rejects_multiple_parents():
    gltf = GLTF2(
        scene=0,
        scenes=[Scene(nodes=[0, 1])],
        nodes=[Node(children=[2]), Node(children=[2]), Node()],
    )

    with pytest.raises(SceneGraphStructureError):
        gltf.reparent_node(2, None)


@pytest.mark.parametrize(
    "asset",
    [
        GLTF2(scenes=[Scene(nodes=[0]), Scene(nodes=[0])], nodes=[Node()]),
        GLTF2(
            scenes=[Scene(nodes=[0])],
            nodes=[Node()],
            animations=[pygltflib.Animation()],
        ),
        GLTF2(
            scenes=[Scene(nodes=[0])],
            nodes=[Node()],
            skins=[pygltflib.Skin()],
        ),
        GLTF2(scenes=[Scene(nodes=[0])], nodes=[Node(skin=0)]),
    ],
)
def test_reparent_rejects_unsupported_assets(asset):
    with pytest.raises(UnsupportedAssetError):
        asset.reparent_node(0, None)


def test_reparent_same_parent_returns_independent_copy():
    gltf = hierarchy_asset()
    before = world_matrices(gltf)

    result = gltf.reparent_node(2, 0)

    assert result is not gltf
    assert result.nodes is not gltf.nodes
    assert_world_matrices_equal(before, world_matrices(result))


def test_module_level_api_and_aliases():
    gltf = hierarchy_asset()

    assert pygltflib.traverse_scene(gltf) == gltf.traverse_scene()
    assert pygltflib.get_scene_node_transforms(gltf) == gltf.traverse_scene()
    assert pygltflib.get_scene_transforms(gltf) == gltf.traverse_scene()
    assert pygltflib.traverse_scene_nodes(gltf) == gltf.traverse_scene()

    result = pygltflib.reparent_node(gltf, 2, 1)
    assert result.nodes[1].children == [2]
    assert pygltflib.move_node(gltf, 2, parent_index=1).nodes[1].children == [2]
    assert pygltflib.reparent(gltf, 2, 1).nodes[1].children == [2]
    assert pygltflib.reparent_subtree(gltf, 2, 1).nodes[1].children == [2]
    assert pygltflib.set_node_parent(gltf, 2, 1).nodes[1].children == [2]

