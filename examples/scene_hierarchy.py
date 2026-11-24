"""Traverse a scene and move a node subtree while preserving world transforms."""

import math

from pygltflib import GLTF2, Node, Scene


def quaternion(axis_y_radians):
    return [0.0, math.sin(axis_y_radians / 2.0), 0.0, math.cos(axis_y_radians / 2.0)]


gltf = GLTF2(
    nodes=[
        Node(name="root", translation=[1.0, 2.0, 0.0], children=[1, 3]),
        Node(
            name="model",
            translation=[2.0, 0.0, 1.0],
            rotation=quaternion(math.radians(30.0)),
            scale=[2.0, 3.0, 0.5],
            children=[2],
        ),
        Node(name="model-child", translation=[0.0, 1.0, 0.0]),
        Node(
            name="mount",
            translation=[-1.0, 0.5, 2.0],
            rotation=quaternion(math.radians(15.0)),
            scale=[1.5, 1.5, 1.5],
        ),
    ],
    scenes=[Scene(name="scene", nodes=[0])],
    scene=0,
)

print("Traversal before reparenting:")
for item in gltf.traverse_scene(0):
    print(
        "node={} ancestors={} world_translation={}".format(
            item.index,
            item.path,
            tuple(item.world_matrix[row][3] for row in range(3)),
        )
    )

adjusted = gltf.reparent_node(node_index=1, new_parent_index=3)

print("\nTraversal after moving node 1 below node 3:")
for item in adjusted.traverse_scene(0):
    print(
        "node={} ancestors={} world_translation={}".format(
            item.index,
            item.path,
            tuple(item.world_matrix[row][3] for row in range(3)),
        )
    )

print("\nThe original asset is still unchanged:")
print("root children =", gltf.nodes[0].children)
print("mount children =", gltf.nodes[3].children)
