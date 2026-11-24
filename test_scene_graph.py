import copy
import math

import numpy as np
import pytest

from pygltflib import (
    Animation,
    Buffer,
    GLTF2,
    InvalidNodeIndexError,
    InvalidSceneIndexError,
    MatrixTRSConflictError,
    MultipleParentsError,
    Node,
    NonInvertibleParentError,
    ReparentUnsupportedError,
    Scene,
    SceneGraphCycleError,
    ShearError,
    Skin,
)


def quat(axis, radians):
    half = radians / 2.0
    value = math.sin(half)
    result = [0.0, 0.0, 0.0, math.cos(half)]
    result[axis] = value
    return result


def rotation_matrix(quaternion):
    x, y, z, w = quaternion
    length = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / length, y / length, z / length, w / length
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def trs_matrix(translation, quaternion, scale):
    result = np.eye(4)
    result[:3, :3] = rotation_matrix(quaternion) @ np.diag(scale)
    result[:3, 3] = translation
    return result


def matrix_from_gltf(values):
    return np.array(values).reshape(4, 4).T


def world_matrices(gltf):
    return {
        item.index: np.asarray(item.world_matrix)
        for item in gltf.traverse_scene(0)
    }


def assert_worlds_unchanged(before, after):
    assert set(before) == set(after)
    for index in before:
        np.testing.assert_allclose(after[index], before[index], atol=1e-6)


def traversal_asset():
    return GLTF2(
        nodes=[
            Node(
                translation=[1.0, 2.0, 3.0],
                rotation=quat(2, math.radians(90.0)),
                scale=[2.0, 3.0, 4.0],
                children=[1, 3],
            ),
            Node(translation=[0.0, 1.0, 0.0], children=[2]),
            Node(scale=[1.0, 2.0, 1.0]),
            Node(
                matrix=[
                    1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    4.0, 5.0, 6.0, 1.0,
                ]
            ),
            Node(name="unreachable"),
        ],
        scenes=[Scene(nodes=[0])],
        scene=0,
    )


def reparent_asset():
    return GLTF2(
        nodes=[
            Node(
                name="root",
                translation=[1.0, 2.0, -1.0],
                rotation=quat(2, math.radians(15.0)),
                scale=[2.0, 1.0, 0.5],
                children=[1, 4],
            ),
            Node(
                name="model",
                translation=[2.0, -1.0, 0.5],
                rotation=quat(1, math.radians(30.0)),
                scale=[2.0, 3.0, 4.0],
                children=[2],
            ),
            Node(
                name="model-child",
                translation=[0.5, 1.0, -0.25],
                rotation=quat(0, math.radians(20.0)),
                scale=[1.0, 2.0, 1.0],
            ),
            Node(
                name="other-root",
                translation=[-2.0, 4.0, 1.0],
                rotation=quat(1, math.radians(-25.0)),
                scale=[0.5, 1.0, 1.5],
            ),
            Node(
                name="mount",
                translation=[-1.0, 0.5, 2.0],
                rotation=quat(2, math.radians(40.0)),
                scale=[1.5, 1.5, 1.5],
            ),
        ],
        scenes=[Scene(nodes=[0, 3])],
        scene=0,
    )


class TestSceneTraversal:
    def test_preorder_paths_and_matrices(self):
        gltf = traversal_asset()
        before = gltf.to_json()

        transforms = gltf.traverse_scene(0)

        assert [item.index for item in transforms] == [0, 1, 2, 3]
        assert [item.path for item in transforms] == [
            (),
            (0,),
            (0, 1),
            (0,),
        ]
        assert transforms[2].full_path == (0, 1, 2)

        root_local = np.asarray(transforms[0].local_matrix)
        np.testing.assert_allclose(
            root_local,
            trs_matrix(
                [1.0, 2.0, 3.0],
                quat(2, math.radians(90.0)),
                [2.0, 3.0, 4.0],
            ),
        )
        np.testing.assert_allclose(
            np.asarray(transforms[2].world_matrix),
            root_local
            @ np.asarray(transforms[1].local_matrix)
            @ np.asarray(transforms[2].local_matrix),
        )

        matrix_local = np.asarray(transforms[3].local_matrix)
        np.testing.assert_allclose(
            matrix_local,
            matrix_from_gltf(
                gltf.nodes[3].matrix
            ),
        )
        assert transforms[3].local_matrix == gltf.nodes[3].matrix
        assert tuple(transforms[3].local_matrix.to_gltf()) == tuple(
            gltf.nodes[3].matrix
        )
        assert gltf.to_json() == before

    def test_scenes_may_share_nodes_when_only_reading(self):
        gltf = GLTF2(
            nodes=[Node(children=[1]), Node()],
            scenes=[Scene(nodes=[0]), Scene(nodes=[1])],
            scene=0,
        )

        assert [item.path for item in gltf.traverse_scene(0)] == [(), (0,)]
        assert [item.path for item in gltf.traverse_scene(1)] == [()]

    def test_only_reachable_nodes_are_validated_and_returned(self):
        gltf = traversal_asset()
        gltf.nodes[4].matrix = [1.0] * 16
        gltf.nodes[4].translation = [0.0, 0.0, 0.0]

        assert 4 not in [item.index for item in gltf.traverse_scene(0)]

    def test_validation_errors(self):
        gltf = traversal_asset()
        gltf.nodes[0].matrix = [1.0] * 16
        with pytest.raises(MatrixTRSConflictError):
            gltf.traverse_scene(0)

        with pytest.raises(InvalidSceneIndexError):
            traversal_asset().traverse_scene(3)
        with pytest.raises(InvalidNodeIndexError):
            invalid = traversal_asset()
            invalid.nodes[0].children = [99]
            invalid.traverse_scene(0)

    def test_cycles_and_multiple_parents_are_rejected(self):
        cycle = GLTF2(
            nodes=[Node(children=[1]), Node(children=[0])],
            scenes=[Scene(nodes=[0])],
            scene=0,
        )
        with pytest.raises(SceneGraphCycleError):
            cycle.traverse_scene(0)

        multiple = GLTF2(
            nodes=[Node(children=[2]), Node(children=[2]), Node()],
            scenes=[Scene(nodes=[0, 1])],
            scene=0,
        )
        with pytest.raises(MultipleParentsError):
            multiple.traverse_scene(0)


class TestReparentNode:
    def test_preserves_subtree_and_other_world_transforms(self):
        gltf = reparent_asset()
        before_worlds = world_matrices(gltf)
        before_children = copy.deepcopy([node.children for node in gltf.nodes])
        gltf.set_binary_blob(b"unchanged-resource")
        gltf.buffers.append(Buffer(uri="model.bin", byteLength=16))
        before_json = gltf.to_json()

        adjusted = gltf.reparent_node(node_index=1, new_parent_index=4)

        assert adjusted is not gltf
        assert adjusted.nodes[1] is not gltf.nodes[1]
        assert adjusted.buffers[0] is not gltf.buffers[0]
        assert adjusted.buffers[0].uri == "model.bin"
        assert gltf.buffers[0].uri == "model.bin"
        assert adjusted.binary_blob() == b"unchanged-resource"
        assert gltf.binary_blob() == b"unchanged-resource"
        assert gltf.to_json() == before_json
        assert [node.children for node in gltf.nodes] == before_children

        assert 1 not in adjusted.nodes[0].children
        assert adjusted.nodes[4].children[-1] == 1
        assert adjusted.nodes[1].children == [2]
        assert adjusted.nodes[1].matrix is None
        assert adjusted.nodes[1].translation is not None
        assert adjusted.nodes[1].rotation is not None
        assert adjusted.nodes[1].scale is not None

        after_worlds = world_matrices(adjusted)
        assert_worlds_unchanged(before_worlds, after_worlds)
        assert adjusted.nodes[1].children == [2]

        roundtrip = GLTF2.gltf_from_json(adjusted.gltf_to_json())
        assert_worlds_unchanged(before_worlds, world_matrices(roundtrip))

    def test_negative_scale_preserves_world_transform(self):
        gltf = GLTF2(
            nodes=[
                Node(children=[1, 2]),
                Node(
                    translation=[1.0, 2.0, -3.0],
                    rotation=quat(2, math.radians(35.0)),
                    scale=[-2.0, 1.5, -0.5],
                    children=[3],
                ),
                Node(
                    translation=[-1.0, 0.0, 2.0],
                    rotation=quat(1, math.radians(-20.0)),
                    scale=[1.25, 1.25, 1.25],
                ),
                Node(
                    translation=[0.0, 1.0, 0.0],
                    rotation=quat(0, math.radians(10.0)),
                    scale=[2.0, 1.0, 3.0],
                ),
            ],
            scenes=[Scene(nodes=[0])],
            scene=0,
        )
        before = world_matrices(gltf)

        adjusted = gltf.reparent_node(1, 2)

        assert adjusted.nodes[2].children == [1]
        assert_worlds_unchanged(before, world_matrices(adjusted))

    def test_can_move_below_an_existing_ancestor(self):
        gltf = GLTF2(
            nodes=[
                Node(children=[1]),
                Node(
                    translation=[1.0, 2.0, 3.0],
                    rotation=quat(2, math.radians(20.0)),
                    children=[2],
                ),
                Node(
                    translation=[0.5, 0.0, -1.0],
                    scale=[2.0, 1.0, 3.0],
                ),
            ],
            scenes=[Scene(nodes=[0])],
            scene=0,
        )
        before = world_matrices(gltf)

        adjusted = gltf.reparent_node(node_index=2, new_parent_index=0)

        assert adjusted.nodes[1].children == []
        assert adjusted.nodes[0].children == [1, 2]
        assert_worlds_unchanged(before, world_matrices(adjusted))

    def test_same_parent_preserves_order(self):
        gltf = GLTF2(
            nodes=[
                Node(children=[1, 2, 3]),
                Node(translation=[1.0, 0.0, 0.0]),
                Node(translation=[0.0, 1.0, 0.0]),
                Node(translation=[0.0, 0.0, 1.0]),
            ],
            scenes=[Scene(nodes=[0])],
            scene=0,
        )
        before = world_matrices(gltf)

        adjusted = gltf.reparent_node(node_index=2, new_parent_index=0)

        assert adjusted.nodes[0].children == [1, 2, 3]
        assert_worlds_unchanged(before, world_matrices(adjusted))

    def test_zero_scale_can_be_preserved(self):
        gltf = GLTF2(
            nodes=[
                Node(children=[1]),
                Node(
                    translation=[1.0, 2.0, 3.0],
                    rotation=quat(2, math.radians(10.0)),
                    scale=[0.0, 2.0, 1.0],
                ),
            ],
            scenes=[Scene(nodes=[0])],
            scene=0,
        )
        before = world_matrices(gltf)

        adjusted = gltf.reparent_node(node_index=1, new_parent_index=None)

        assert_worlds_unchanged(before, world_matrices(adjusted))

    def test_move_to_scene_root(self):
        gltf = GLTF2(
            nodes=[
                Node(children=[1, 2]),
                Node(
                    translation=[1.0, 2.0, -3.0],
                    rotation=quat(2, math.radians(30.0)),
                    scale=[2.0, 3.0, 0.5],
                    children=[3],
                ),
                Node(translation=[4.0, 0.0, 1.0]),
                Node(
                    translation=[0.0, 1.0, 0.0],
                    rotation=quat(1, math.radians(10.0)),
                ),
            ],
            scenes=[Scene(nodes=[0])],
            scene=0,
        )
        before = world_matrices(gltf)

        adjusted = gltf.reparent_node(node_index=1, new_parent_index=None)

        assert adjusted.scenes[0].nodes == [0, 1]
        assert 1 not in adjusted.nodes[0].children
        assert adjusted.nodes[1].children == [3]
        assert_worlds_unchanged(before, world_matrices(adjusted))

    def test_rejects_cycles_shear_and_non_invertible_parent(self):
        cycle = reparent_asset()
        with pytest.raises(SceneGraphCycleError):
            cycle.reparent_node(0, 1)
        with pytest.raises(SceneGraphCycleError):
            cycle.reparent_node(1, 1)

        shear = GLTF2(
            nodes=[
                Node(children=[1, 2]),
                Node(rotation=quat(2, math.radians(30.0)), scale=[2.0, 1.0, 1.0]),
                Node(scale=[2.0, 1.0, 1.0]),
            ],
            scenes=[Scene(nodes=[0])],
            scene=0,
        )
        with pytest.raises(ShearError):
            shear.reparent_node(1, 2)

        singular = GLTF2(
            nodes=[
                Node(children=[1, 2]),
                Node(translation=[1.0, 0.0, 0.0]),
                Node(scale=[1.0, 0.0, 1.0]),
            ],
            scenes=[Scene(nodes=[0])],
            scene=0,
        )
        with pytest.raises(NonInvertibleParentError):
            singular.reparent_node(1, 2)

    def test_rejects_unsupported_assets_and_bad_indices(self):
        multiple_scenes = reparent_asset()
        multiple_scenes.scenes.append(Scene(nodes=[3]))
        with pytest.raises(ReparentUnsupportedError):
            multiple_scenes.reparent_node(1, 4)

        animated = reparent_asset()
        animated.animations.append(Animation())
        with pytest.raises(ReparentUnsupportedError):
            animated.reparent_node(1, 4)

        skinned = reparent_asset()
        skinned.skins.append(Skin())
        with pytest.raises(ReparentUnsupportedError):
            skinned.reparent_node(1, 4)

        with pytest.raises(InvalidNodeIndexError):
            reparent_asset().reparent_node(99, 4)
        with pytest.raises(InvalidNodeIndexError):
            reparent_asset().reparent_node(1, 99)
        with pytest.raises(InvalidNodeIndexError):
            unreachable = reparent_asset()
            unreachable.nodes.append(Node())
            unreachable.reparent_node(5, 4)

    def test_rejects_existing_graph_errors_without_mutation(self):
        multiple = GLTF2(
            nodes=[Node(children=[2]), Node(children=[2]), Node()],
            scenes=[Scene(nodes=[0, 1])],
            scene=0,
        )
        before = multiple.to_json()
        with pytest.raises(MultipleParentsError):
            multiple.reparent_node(2, None)
        assert multiple.to_json() == before
