"""
Scene graph helpers for pygltflib.

The functions in this module operate on glTF 2.0 node hierarchies.  They do
not mutate the GLTF2 instance they are given; ``reparent_node`` returns a deep
copy with the requested hierarchy change applied.

Matrices returned by ``traverse_scene`` are 4x4 nested lists in row-major
order.  This is Python-friendly and differs from the flattened column-major
representation used by glTF's ``node.matrix`` property.  The
``*_matrix_gltf`` properties on :class:`SceneNodeTransform` expose the
column-major, flattened representation when needed.
"""

import copy
import math
import numbers
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .types import Node


class SceneGraphError(ValueError):
    """Base class for scene traversal and hierarchy errors."""


class InvalidSceneError(SceneGraphError):
    """The requested scene does not exist or cannot be traversed."""


class InvalidNodeIndexError(SceneGraphError):
    """A scene root, child, or requested node index is invalid."""


class InvalidNodeTransformError(SceneGraphError):
    """A node's local transform is malformed."""


class SceneGraphStructureError(SceneGraphError):
    """The scene graph contains a cycle, multiple parents, or a bad edge."""


class NonInvertibleMatrixError(SceneGraphError):
    """A parent world matrix cannot be inverted."""


class ShearNotAllowedError(SceneGraphError):
    """The new local transform cannot be represented by glTF TRS properties."""


class UnsupportedAssetError(SceneGraphError):
    """The asset uses features not supported by hierarchy adjustment."""


_IDENTITY = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)

_AFFINE_TOLERANCE = 1e-9
_SHEAR_TOLERANCE = 1e-6
_INVERSE_TOLERANCE = 1e-12


@dataclass
class SceneNodeTransform:
    """A node visited while traversing a scene.

    Attributes:
        index: Index into ``gltf.nodes``.
        ancestors: Node indices from the scene root down to the parent.  The
            list is empty for a scene root and does not include ``index``.
        local_matrix: The node's local transform as a row-major 4x4 matrix.
        world_matrix: The node's world transform as a row-major 4x4 matrix.
    """

    index: int
    ancestors: List[int] = field(default_factory=list)
    local_matrix: List[List[float]] = field(default_factory=list)
    world_matrix: List[List[float]] = field(default_factory=list)

    @property
    def node_index(self) -> int:
        """Alias for ``index``."""

        return self.index

    @property
    def ancestor_indices(self) -> List[int]:
        """Alias for ``ancestors``."""

        return self.ancestors

    @property
    def ancestor_path(self) -> List[int]:
        """Alias for ``ancestors``."""

        return self.ancestors

    @property
    def path(self) -> List[int]:
        """Alias for ``ancestors``."""

        return self.ancestors

    @property
    def local(self) -> List[List[float]]:
        """Alias for ``local_matrix``."""

        return self.local_matrix

    @property
    def world(self) -> List[List[float]]:
        """Alias for ``world_matrix``."""

        return self.world_matrix

    @property
    def local_matrix_gltf(self) -> List[float]:
        """Return the local matrix in glTF's flat column-major order."""

        return _flatten_column_major(self.local_matrix)

    @property
    def world_matrix_gltf(self) -> List[float]:
        """Return the world matrix in glTF's flat column-major order."""

        return _flatten_column_major(self.world_matrix)

    @property
    def local_matrix_flat(self) -> List[float]:
        """Alias for ``local_matrix_gltf``."""

        return self.local_matrix_gltf

    @property
    def world_matrix_flat(self) -> List[float]:
        """Alias for ``world_matrix_gltf``."""

        return self.world_matrix_gltf

    def as_dict(self) -> Dict[str, Any]:
        """Return the entry as a plain dictionary."""

        return {
            "index": self.index,
            "node_index": self.index,
            "ancestors": self.ancestors,
            "ancestor_indices": self.ancestors,
            "ancestor_path": self.ancestors,
            "local_matrix": self.local_matrix,
            "world_matrix": self.world_matrix,
        }

    def __iter__(self):
        """Allow tuple-style unpacking in index, path, local, world order."""

        return iter(
            (self.index, self.ancestors, self.local_matrix, self.world_matrix)
        )

    def __getitem__(self, key):
        if isinstance(key, int):
            return tuple(self)[key]
        if isinstance(key, str):
            aliases = {
                "index": self.index,
                "node": self.index,
                "node_index": self.index,
                "ancestors": self.ancestors,
                "ancestor_indices": self.ancestors,
                "ancestor_path": self.ancestors,
                "path": self.ancestors,
                "local": self.local_matrix,
                "local_matrix": self.local_matrix,
                "world": self.world_matrix,
                "world_matrix": self.world_matrix,
            }
            try:
                return aliases[key]
            except KeyError:
                raise KeyError(key)
        raise TypeError("SceneNodeTransform indices must be int or str")


def _flatten_column_major(matrix: Sequence[Sequence[float]]) -> List[float]:
    return [float(matrix[row][column]) for column in range(4) for row in range(4)]


def _matrix4(values: Sequence[float], *, name: str) -> List[List[float]]:
    if not isinstance(values, (list, tuple)) or len(values) != 16:
        raise InvalidNodeTransformError(
            "{} must contain exactly 16 numbers".format(name)
        )

    try:
        numbers = [float(value) for value in values]
    except (TypeError, ValueError):
        raise InvalidNodeTransformError(
            "{} must contain only numbers".format(name)
        )

    if not all(math.isfinite(value) for value in numbers):
        raise InvalidNodeTransformError(
            "{} must contain only finite numbers".format(name)
        )

    return [
        [numbers[0], numbers[4], numbers[8], numbers[12]],
        [numbers[1], numbers[5], numbers[9], numbers[13]],
        [numbers[2], numbers[6], numbers[10], numbers[14]],
        [numbers[3], numbers[7], numbers[11], numbers[15]],
    ]


def _vector(
    values: Optional[Sequence[float]],
    length: int,
    default: Sequence[float],
    *,
    name: str,
) -> List[float]:
    if values is None:
        return [float(value) for value in default]
    if not isinstance(values, (list, tuple)) or len(values) != length:
        raise InvalidNodeTransformError(
            "{} must contain exactly {} numbers".format(name, length)
        )
    try:
        numbers = [float(value) for value in values]
    except (TypeError, ValueError):
        raise InvalidNodeTransformError(
            "{} must contain only numbers".format(name)
        )
    if not all(math.isfinite(value) for value in numbers):
        raise InvalidNodeTransformError(
            "{} must contain only finite numbers".format(name)
        )
    return numbers


def _identity_matrix() -> List[List[float]]:
    return [list(row) for row in _IDENTITY]


def _matmul(
    left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]
) -> List[List[float]]:
    return [
        [
            sum(left[row][k] * right[k][column] for k in range(4))
            for column in range(4)
        ]
        for row in range(4)
    ]


def _quaternion_matrix(quaternion: Sequence[float]) -> List[List[float]]:
    x, y, z, w = (float(value) for value in quaternion)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= _INVERSE_TOLERANCE:
        raise InvalidNodeTransformError("rotation quaternion must not be zero")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    xw, yw, zw = x * w, y * w, z * w
    return [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - zw), 2.0 * (xz + yw), 0.0],
        [2.0 * (xy + zw), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - xw), 0.0],
        [2.0 * (xz - yw), 2.0 * (yz + xw), 1.0 - 2.0 * (xx + yy), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _matrix_quaternion(matrix: Sequence[Sequence[float]]) -> List[float]:
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        x = (matrix[2][1] - matrix[1][2]) / scale
        y = (matrix[0][2] - matrix[2][0]) / scale
        z = (matrix[1][0] - matrix[0][1]) / scale
        w = 0.25 * scale
    elif matrix[0][0] > matrix[1][1] and matrix[0][0] > matrix[2][2]:
        scale = math.sqrt(
            1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2]
        ) * 2.0
        x = 0.25 * scale
        y = (matrix[0][1] + matrix[1][0]) / scale
        z = (matrix[0][2] + matrix[2][0]) / scale
        w = (matrix[2][1] - matrix[1][2]) / scale
    elif matrix[1][1] > matrix[2][2]:
        scale = math.sqrt(
            1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2]
        ) * 2.0
        x = (matrix[0][1] + matrix[1][0]) / scale
        y = 0.25 * scale
        z = (matrix[1][2] + matrix[2][1]) / scale
        w = (matrix[0][2] - matrix[2][0]) / scale
    else:
        scale = math.sqrt(
            1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1]
        ) * 2.0
        x = (matrix[0][2] + matrix[2][0]) / scale
        y = (matrix[1][2] + matrix[2][1]) / scale
        z = 0.25 * scale
        w = (matrix[1][0] - matrix[0][1]) / scale

    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= _INVERSE_TOLERANCE:
        raise InvalidNodeTransformError("could not derive a rotation quaternion")
    return [x / norm, y / norm, z / norm, w / norm]


def _local_from_trs(
    translation: Sequence[float],
    rotation: Sequence[float],
    scale: Sequence[float],
) -> List[List[float]]:
    translation_matrix = _identity_matrix()
    translation_matrix[0][3] = float(translation[0])
    translation_matrix[1][3] = float(translation[1])
    translation_matrix[2][3] = float(translation[2])

    scale_matrix = [
        [float(scale[0]), 0.0, 0.0, 0.0],
        [0.0, float(scale[1]), 0.0, 0.0],
        [0.0, 0.0, float(scale[2]), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    return _matmul(
        _matmul(translation_matrix, _quaternion_matrix(rotation)), scale_matrix
    )


def _validate_node_transform(node: Node) -> None:
    trs_present = any(
        value is not None for value in (node.translation, node.rotation, node.scale)
    )
    if node.matrix is not None and trs_present:
        raise InvalidNodeTransformError(
            "node has both matrix and translation/rotation/scale properties"
        )


def _node_local_matrix(node: Node) -> List[List[float]]:
    _validate_node_transform(node)
    if node.matrix is not None:
        return _matrix4(node.matrix, name="node.matrix")

    translation = _vector(
        node.translation, 3, (0.0, 0.0, 0.0), name="node.translation"
    )
    rotation = _vector(
        node.rotation, 4, (0.0, 0.0, 0.0, 1.0), name="node.rotation"
    )
    scale = _vector(node.scale, 3, (1.0, 1.0, 1.0), name="node.scale")
    return _local_from_trs(translation, rotation, scale)


def _validate_index(value: Any, size: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise InvalidNodeIndexError("{} must be an integer".format(name))
    value = int(value)
    if value < 0 or value >= size:
        raise InvalidNodeIndexError(
            "{} {} is out of range for {} nodes".format(name, value, size)
        )
    return value


def _children(node: Node, node_count: int) -> List[int]:
    children = node.children
    if children is None:
        return []
    if not isinstance(children, (list, tuple)):
        raise SceneGraphStructureError("node.children must be a list of indices")
    result = []
    for child in children:
        result.append(_validate_index(child, node_count, name="child index"))
    return result


def _scene_index(gltf: Any, scene_index: Optional[int]) -> int:
    if not gltf.scenes:
        raise InvalidSceneError("GLTF has no scenes")
    if scene_index is None:
        scene_index = gltf.scene
        if scene_index is None:
            scene_index = 0
    if isinstance(scene_index, bool) or not isinstance(scene_index, numbers.Integral):
        raise InvalidSceneError("scene index must be an integer")
    scene_index = int(scene_index)
    if scene_index < 0 or scene_index >= len(gltf.scenes):
        raise InvalidSceneError(
            "scene index {} is out of range for {} scenes".format(
                scene_index, len(gltf.scenes)
            )
        )
    return scene_index


def _scene_roots(gltf: Any, scene: Any) -> List[int]:
    roots = scene.nodes
    if roots is None:
        return []
    if not isinstance(roots, (list, tuple)):
        raise SceneGraphStructureError("scene.nodes must be a list of indices")

    result = []
    seen = set()
    for root in roots:
        root = _validate_index(root, len(gltf.nodes), name="scene root index")
        if root in seen:
            raise SceneGraphStructureError(
                "node {} appears more than once in scene roots".format(root)
            )
        seen.add(root)
        result.append(root)
    return result


def _walk_scene(
    gltf: Any, scene_index: Optional[int] = None
) -> Tuple[List[SceneNodeTransform], Dict[int, Optional[int]], Dict[int, List[int]]]:
    scene_index = _scene_index(gltf, scene_index)
    roots = _scene_roots(gltf, gltf.scenes[scene_index])

    entries: List[SceneNodeTransform] = []
    parent_of: Dict[int, Optional[int]] = {root: None for root in roots}
    children_of: Dict[int, List[int]] = {}
    stack: List[Tuple[int, Tuple[int, ...], List[List[float]]]] = []

    for root in reversed(roots):
        stack.append((root, (), _identity_matrix()))

    while stack:
        index, ancestors, parent_world = stack.pop()
        node = gltf.nodes[index]
        local = _node_local_matrix(node)
        world = _matmul(parent_world, local)
        children = _children(node, len(gltf.nodes))
        children_of[index] = children

        entries.append(
            SceneNodeTransform(
                index=index,
                ancestors=list(ancestors),
                local_matrix=local,
                world_matrix=world,
            )
        )

        child_ancestors = ancestors + (index,)
        for child in reversed(children):
            if child in child_ancestors:
                raise SceneGraphStructureError(
                    "cycle detected at node {}".format(child)
                )
            if child in parent_of:
                if parent_of[child] is None:
                    raise SceneGraphStructureError(
                        "node {} is both a scene root and a child".format(child)
                    )
                raise SceneGraphStructureError(
                    "node {} has multiple parents: {} and {}".format(
                        child, parent_of[child], index
                    )
                )
            parent_of[child] = index
            stack.append((child, child_ancestors, world))

    return entries, parent_of, children_of


def traverse_scene(
    gltf: Any, scene_index: Optional[int] = None
) -> List[SceneNodeTransform]:
    """Traverse the nodes reachable from a scene.

    Nodes are returned in depth-first pre-order, preserving the order of
    ``scene.nodes`` and each node's ``children`` list.  A node shared between
    different scenes is valid because only the requested scene is checked.

    Args:
        gltf: A :class:`pygltflib.GLTF2`-compatible object.
        scene_index: Scene to traverse.  Defaults to ``gltf.scene``, then 0.

    Returns:
        A list of :class:`SceneNodeTransform` entries.

    Raises:
        SceneGraphError: If the scene has invalid indices, a cycle, a node
            with multiple parents, or a node with both matrix and TRS values.
    """

    entries, _, _ = _walk_scene(gltf, scene_index)
    return entries


def node_local_matrix(node: Node) -> List[List[float]]:
    """Return a node's local matrix as a row-major nested list."""

    return _node_local_matrix(node)


def _affine_check(matrix: Sequence[Sequence[float]], *, name: str) -> None:
    for row in range(3):
        if not all(math.isfinite(matrix[row][column]) for column in range(4)):
            raise InvalidNodeTransformError("{} contains non-finite values".format(name))
    if not all(math.isfinite(matrix[3][column]) for column in range(4)):
        raise InvalidNodeTransformError("{} contains non-finite values".format(name))
    if (
        abs(matrix[3][0]) > _AFFINE_TOLERANCE
        or abs(matrix[3][1]) > _AFFINE_TOLERANCE
        or abs(matrix[3][2]) > _AFFINE_TOLERANCE
        or abs(matrix[3][3] - 1.0) > _AFFINE_TOLERANCE
    ):
        raise InvalidNodeTransformError(
            "{} is not an affine matrix".format(name)
        )


def _invert_matrix(matrix: Sequence[Sequence[float]]) -> List[List[float]]:
    _affine_check(matrix, name="world matrix")

    augmented = [
        [float(matrix[row][column]) for column in range(4)]
        + [1.0 if row == column else 0.0 for column in range(4)]
        for row in range(4)
    ]

    for column in range(4):
        pivot_row = max(
            range(column, 4), key=lambda row: abs(augmented[row][column])
        )
        pivot = augmented[pivot_row][column]
        if abs(pivot) <= _INVERSE_TOLERANCE:
            raise NonInvertibleMatrixError(
                "new parent world matrix is not invertible"
            )
        if pivot_row != column:
            augmented[column], augmented[pivot_row] = (
                augmented[pivot_row],
                augmented[column],
            )

        pivot = augmented[column][column]
        augmented[column] = [value / pivot for value in augmented[column]]
        for row in range(4):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0.0:
                continue
            augmented[row] = [
                augmented[row][index] - factor * augmented[column][index]
                for index in range(8)
            ]

    result = [row[4:] for row in augmented]
    _affine_check(result, name="inverted world matrix")
    return result


def _determinant3(matrix: Sequence[Sequence[float]]) -> float:
    return (
        matrix[0][0]
        * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1]
        * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2]
        * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )


def _cross(left: Sequence[float], right: Sequence[float]) -> List[float]:
    return [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]


def _normalize(vector: Sequence[float]) -> List[float]:
    length = math.sqrt(sum(component * component for component in vector))
    return [component / length for component in vector]


def _decompose_trs(
    matrix: Sequence[Sequence[float]],
) -> Tuple[List[float], List[float], List[float]]:
    _affine_check(matrix, name="new local matrix")

    columns = [
        [float(matrix[row][column]) for row in range(3)] for column in range(3)
    ]
    scales = []
    for column in columns:
        scales.append(
            math.sqrt(sum(component * component for component in column))
        )

    normalized_columns = [None, None, None]
    for index, (column, scale) in enumerate(zip(columns, scales)):
        if scale > _INVERSE_TOLERANCE:
            normalized_columns[index] = [
                component / scale for component in column
            ]

    for left in range(3):
        if normalized_columns[left] is None:
            continue
        for right in range(left + 1, 3):
            if normalized_columns[right] is None:
                continue
            dot = sum(
                normalized_columns[left][axis]
                * normalized_columns[right][axis]
                for axis in range(3)
            )
            if abs(dot) > _SHEAR_TOLERANCE:
                raise ShearNotAllowedError(
                    "new local transform contains shear, which glTF node "
                    "TRS properties cannot represent"
                )

    present = [
        index for index, column in enumerate(normalized_columns) if column is not None
    ]
    if not present:
        normalized_columns = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    elif len(present) == 1:
        first = present[0]
        axis = normalized_columns[first]
        helper = [1.0, 0.0, 0.0] if abs(axis[0]) < 0.9 else [0.0, 1.0, 0.0]
        second = _normalize(_cross(axis, helper))
        third = _cross(axis, second)
        other = [index for index in range(3) if index != first]
        normalized_columns[other[0]] = second
        normalized_columns[other[1]] = third
    elif len(present) == 2:
        first, second = present
        third = _cross(
            normalized_columns[first], normalized_columns[second]
        )
        normalized_columns[3 - first - second] = third

    rotation = [
        [
            normalized_columns[0][row],
            normalized_columns[1][row],
            normalized_columns[2][row],
            0.0,
        ]
        for row in range(3)
    ] + [[0.0, 0.0, 0.0, 1.0]]

    if _determinant3(rotation) < 0.0:
        flip = present[0] if present else 0
        scales[flip] = -scales[flip]
        for row in range(3):
            rotation[row][flip] = -rotation[row][flip]

    translation = [float(matrix[row][3]) for row in range(3)]
    quaternion = _matrix_quaternion(rotation)
    rebuilt = _local_from_trs(translation, quaternion, scales)

    tolerance = _SHEAR_TOLERANCE * max(
        1.0,
        max(abs(value) for row in matrix for value in row),
    )
    for row in range(4):
        for column in range(4):
            if abs(rebuilt[row][column] - matrix[row][column]) > tolerance:
                raise ShearNotAllowedError(
                    "new local transform cannot be represented by glTF TRS "
                    "properties without shear"
                )

    return translation, quaternion, scales


def _validate_adjustable_asset(gltf: Any) -> None:
    if len(gltf.scenes) != 1:
        raise UnsupportedAssetError(
            "hierarchy adjustment requires exactly one scene"
        )
    if gltf.animations:
        raise UnsupportedAssetError(
            "hierarchy adjustment does not support animations"
        )
    if gltf.skins:
        raise UnsupportedAssetError(
            "hierarchy adjustment does not support skins"
        )
    if any(node.skin is not None for node in gltf.nodes):
        raise UnsupportedAssetError(
            "hierarchy adjustment does not support skinned nodes"
        )


def _remove_reference(gltf: Any, node_index: int, parent: Optional[int]) -> None:
    container = gltf.scenes[0].nodes if parent is None else gltf.nodes[parent].children
    if container is None:
        container = []
        if parent is None:
            gltf.scenes[0].nodes = container
        else:
            gltf.nodes[parent].children = container
    if node_index not in container:
        raise SceneGraphStructureError(
            "node {} is not a child of the expected parent".format(node_index)
        )
    container.remove(node_index)


def _append_reference(gltf: Any, node_index: int, parent: Optional[int]) -> None:
    if parent is None:
        container = gltf.scenes[0].nodes
        if container is None:
            container = []
            gltf.scenes[0].nodes = container
    else:
        container = gltf.nodes[parent].children
        if container is None:
            container = []
            gltf.nodes[parent].children = container
    container.append(node_index)


def _write_trs(node: Node, matrix: Sequence[Sequence[float]]) -> None:
    translation, rotation, scale = _decompose_trs(matrix)
    node.matrix = None
    node.translation = translation
    node.rotation = rotation
    node.scale = scale


def reparent_node(
    gltf: Any,
    node_index: int,
    new_parent_index: Optional[int] = None,
    *,
    scene_index: Optional[int] = None,
    parent_index: Optional[int] = None,
) -> Any:
    """Return a copy with a node and its subtree moved to a new parent.

    The source node and every node in its subtree keep their original world
    transforms.  Nodes outside the subtree are not transformed.  Passing
    ``None`` as the new parent moves the node to the scene's root list.

    Only assets with one scene, no animations, and no skins are supported.
    The returned object is always a deep copy; the input is not modified.

    Args:
        gltf: A :class:`pygltflib.GLTF2`-compatible object.
        node_index: Node to move.
        new_parent_index: New parent node, or ``None`` for a scene root.
        scene_index: Must identify the asset's only scene when supplied.
        parent_index: Alias for ``new_parent_index``.

    Returns:
        A new GLTF2 object with the hierarchy change.

    Raises:
        SceneGraphError: If the asset is unsupported, an index is invalid,
            the graph contains a cycle or multiple parents, the new parent's
            world matrix is not invertible, or the new local transform
            requires shear.
    """

    if parent_index is not None:
        if new_parent_index is not None:
            raise SceneGraphError(
                "specify only one of new_parent_index and parent_index"
            )
        new_parent_index = parent_index

    _validate_adjustable_asset(gltf)
    resolved_scene = _scene_index(gltf, scene_index)
    if resolved_scene != 0:
        raise InvalidSceneError("the asset's only scene has index 0")

    node_index = _validate_index(node_index, len(gltf.nodes), name="node index")
    if new_parent_index is not None:
        new_parent_index = _validate_index(
            new_parent_index, len(gltf.nodes), name="new parent index"
        )

    entries, parent_of, children_of = _walk_scene(gltf, 0)
    by_index = {entry.index: entry for entry in entries}
    if node_index not in by_index:
        raise InvalidNodeIndexError(
            "node {} is not reachable from the asset's scene".format(node_index)
        )
    if new_parent_index is not None and new_parent_index not in by_index:
        raise InvalidNodeIndexError(
            "new parent {} is not reachable from the asset's scene".format(
                new_parent_index
            )
        )
    if new_parent_index == node_index:
        raise SceneGraphStructureError("a node cannot be its own parent")
    if new_parent_index is not None and node_index in by_index[
        new_parent_index
    ].ancestors:
        raise SceneGraphStructureError(
            "moving node {} below its descendant {} would create a cycle".format(
                node_index, new_parent_index
            )
        )

    result = copy.deepcopy(gltf)
    entries, parent_of, children_of = _walk_scene(result, 0)
    by_index = {entry.index: entry for entry in entries}
    old_parent = parent_of[node_index]

    source_world = by_index[node_index].world_matrix
    if new_parent_index is None:
        new_local = [row[:] for row in source_world]
    else:
        parent_world = by_index[new_parent_index].world_matrix
        try:
            inverse_parent = _invert_matrix(parent_world)
        except NonInvertibleMatrixError:
            raise NonInvertibleMatrixError(
                "cannot move node {} below node {} because its world matrix "
                "is not invertible".format(node_index, new_parent_index)
            )
        new_local = _matmul(inverse_parent, source_world)

    if new_parent_index == old_parent:
        return result

    _decompose_trs(new_local)
    _remove_reference(result, node_index, old_parent)
    _append_reference(result, node_index, new_parent_index)
    _write_trs(result.nodes[node_index], new_local)
    return result


move_node = reparent_node
reparent = reparent_node
reparent_subtree = reparent_node
set_node_parent = reparent_node
get_scene_node_transforms = traverse_scene
get_scene_transforms = traverse_scene
traverse_scene_nodes = traverse_scene

