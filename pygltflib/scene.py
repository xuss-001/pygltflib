"""Scene graph traversal and node reparenting helpers.

Matrices in this module are stored in row-major order for calculation.  Use
``Matrix4.to_gltf()`` (or compare a ``Matrix4`` with a flat 16 element list) to
get the column-major representation used by glTF.
"""

import copy
import math
import numbers
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


_EPSILON = 1e-9
_SHEAR_TOLERANCE = 1e-6


class SceneGraphError(ValueError):
    """Base class for invalid scene graph operations."""


class InvalidSceneIndexError(SceneGraphError, IndexError):
    """Raised when a scene index does not identify an existing scene."""


class InvalidNodeIndexError(SceneGraphError, IndexError):
    """Raised when a node index does not identify an existing node."""


class MatrixTRSConflictError(SceneGraphError):
    """Raised when ``matrix`` is combined with a TRS node property."""


class InvalidTransformError(SceneGraphError):
    """Raised when a node transform is malformed or cannot be interpreted."""


class MultipleParentsError(SceneGraphError):
    """Raised when a node is reachable more than once in one scene."""


class SceneGraphCycleError(SceneGraphError):
    """Raised when a scene graph contains a cycle."""


class ReparentUnsupportedError(SceneGraphError):
    """Raised when an asset does not support hierarchy editing."""


class NonInvertibleParentError(ReparentUnsupportedError):
    """Raised when the destination parent has a singular world transform."""


class ShearError(ReparentUnsupportedError):
    """Raised when preserving a world transform would require shear."""


class Matrix4(Sequence):
    """Immutable row-major 4x4 matrix.

    Iteration and integer indexing return rows.  ``to_gltf()`` returns the flat
    column-major array stored in a glTF ``node.matrix`` property.
    """

    __slots__ = ("_rows",)

    def __init__(self, rows: Iterable[Iterable[float]]):
        values = tuple(tuple(float(value) for value in row) for row in rows)
        if len(values) != 4 or any(len(row) != 4 for row in values):
            raise ValueError("a 4x4 matrix requires four rows of four values")
        self._rows = values

    @classmethod
    def identity(cls):
        return cls(
            (
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            )
        )

    @classmethod
    def from_gltf(cls, values: Sequence[float]):
        numbers = _validated_numbers(values, 16, "node.matrix")
        return cls(
            tuple(
                tuple(numbers[column * 4 + row] for column in range(4))
                for row in range(4)
            )
        )

    def to_gltf(self) -> Tuple[float, ...]:
        return tuple(
            self._rows[row][column]
            for column in range(4)
            for row in range(4)
        )

    @property
    def gltf(self) -> Tuple[float, ...]:
        return self.to_gltf()

    @property
    def flat(self) -> Tuple[float, ...]:
        return self.to_gltf()

    def tolist(self) -> List[List[float]]:
        return [list(row) for row in self._rows]

    def flatten(self) -> List[float]:
        return list(self.to_gltf())

    def __len__(self):
        return 4

    def __iter__(self):
        return iter(self._rows)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError("matrix indices must be int or (row, column)")
            return self._rows[key[0]][key[1]]
        return self._rows[key]

    def __eq__(self, other):
        if isinstance(other, Matrix4):
            return self._rows == other._rows
        if isinstance(other, (list, tuple)):
            if len(other) == 4:
                return self.tolist() == [
                    list(row) if isinstance(row, (list, tuple)) else row
                    for row in other
                ]
            if len(other) == 16:
                return self.to_gltf() == tuple(other)
        try:
            comparison = other.__eq__(self.to_gltf())
        except (AttributeError, TypeError):
            return NotImplemented
        if isinstance(comparison, bool):
            return comparison
        return NotImplemented

    def __hash__(self):
        return hash(self._rows)

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def __array__(self, dtype=None, copy=None):
        try:
            import numpy
        except ImportError as exc:  # pragma: no cover - numpy is optional
            raise TypeError("numpy is required to convert Matrix4 to an array") from exc
        array = numpy.asarray(self._rows, dtype=dtype)
        if copy:
            array = array.copy()
        return array

    def __repr__(self):
        return "Matrix4({!r})".format(self._rows)


class NodeIndexPath(tuple):
    """Immutable ancestor path that also compares equal to a list."""

    def __new__(cls, values=()):
        return tuple.__new__(cls, values)

    def __eq__(self, other):
        if isinstance(other, list):
            return list(self) == other
        return tuple.__eq__(self, other)

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    __hash__ = tuple.__hash__


@dataclass(frozen=True)
class NodeTransform:
    """The transform of one node reachable from a scene.

    ``path`` contains ancestor node indices from the scene root to the direct
    parent and therefore does not include ``index`` itself.
    """

    index: int
    path: Tuple[int, ...]
    local_matrix: Matrix4
    world_matrix: Matrix4

    def __post_init__(self):
        object.__setattr__(self, "path", NodeIndexPath(self.path))

    @property
    def node_index(self):
        return self.index

    @property
    def ancestor_indices(self):
        return self.path

    @property
    def ancestors(self):
        return self.path

    @property
    def ancestor_path(self):
        return self.path

    @property
    def ancestor_index_path(self):
        return self.path

    @property
    def full_path(self):
        return NodeIndexPath(self.path + (self.index,))

    @property
    def local(self):
        return self.local_matrix

    @property
    def local_transform(self):
        return self.local_matrix

    @property
    def world(self):
        return self.world_matrix

    @property
    def world_transform(self):
        return self.world_matrix

    @property
    def parent_index(self):
        return self.path[-1] if self.path else None

    def _as_tuple(self):
        return (self.index, self.path, self.local_matrix, self.world_matrix)

    def __iter__(self):
        return iter(self._as_tuple())

    def __len__(self):
        return 4

    def __getitem__(self, key):
        aliases = {
            "index": self.index,
            "node_index": self.index,
            "path": self.path,
            "ancestor_indices": self.path,
            "ancestors": self.path,
            "ancestor_path": self.path,
            "ancestor_index_path": self.path,
            "parent_index": self.parent_index,
            "local": self.local_matrix,
            "local_matrix": self.local_matrix,
            "local_transform": self.local_matrix,
            "world": self.world_matrix,
            "world_matrix": self.world_matrix,
            "world_transform": self.world_matrix,
        }
        if isinstance(key, str):
            try:
                return aliases[key]
            except KeyError:
                raise KeyError(key)
        return self._as_tuple()[key]

    def get(self, key, default=None):
        try:
            return self[key]
        except (KeyError, IndexError):
            return default

    def keys(self):
        return (
            "index",
            "path",
            "ancestor_indices",
            "local_matrix",
            "world_matrix",
        )

    def items(self):
        return tuple((key, self[key]) for key in self.keys())

    def as_dict(self):
        return dict(self.items())

    def __eq__(self, other):
        if isinstance(other, NodeTransform):
            return self._as_tuple() == other._as_tuple()
        if isinstance(other, dict):
            try:
                return all(self[key] == value for key, value in other.items())
            except KeyError:
                return False
        if isinstance(other, (list, tuple)) and len(other) == 4:
            return self._as_tuple() == tuple(other)
        return NotImplemented

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result


def _validated_numbers(values, length, label):
    if isinstance(values, (str, bytes)):
        raise InvalidTransformError("{} must be an array".format(label))
    try:
        values = list(values)
    except TypeError:
        raise InvalidTransformError("{} must be an array".format(label))
    if len(values) != length:
        raise InvalidTransformError(
            "{} must contain exactly {} numbers".format(label, length)
        )
    parsed = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise InvalidTransformError(
                "{} must contain only numbers".format(label)
            )
        number = float(value)
        if not math.isfinite(number):
            raise InvalidTransformError(
                "{} must contain only finite numbers".format(label)
            )
        parsed.append(number)
    return tuple(parsed)


def _matmul(left, right):
    return Matrix4(
        tuple(
            tuple(
                sum(left[row][k] * right[k][column] for k in range(4))
                for column in range(4)
            )
            for row in range(4)
        )
    )


def _identity_rows():
    return [[1.0 if row == column else 0.0 for column in range(4)] for row in range(4)]


def _inverse(matrix):
    rows = [list(row) for row in matrix]
    augmented = [
        row + identity_row
        for row, identity_row in zip(rows, _identity_rows())
    ]
    scale = max(1.0, max(abs(value) for row in rows for value in row))
    tolerance = _EPSILON * scale

    for column in range(4):
        pivot_row = max(
            range(column, 4), key=lambda row: abs(augmented[row][column])
        )
        pivot = augmented[pivot_row][column]
        if abs(pivot) <= tolerance:
            raise NonInvertibleParentError(
                "the destination parent world matrix is not invertible"
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
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[column])
            ]

    return Matrix4(tuple(tuple(row[4:]) for row in augmented))


def _quaternion_matrix(quaternion):
    x, y, z, w = quaternion
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return (
        (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)),
        (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)),
        (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)),
    )


def _trs_matrix(translation, rotation, scale):
    rotation_matrix = _quaternion_matrix(rotation)
    rows = _identity_rows()
    for row in range(3):
        for column in range(3):
            rows[row][column] = rotation_matrix[row][column] * scale[column]
        rows[row][3] = translation[row]
    return Matrix4(tuple(tuple(row) for row in rows))


def _normalised_quaternion(node, values):
    if values is None:
        return (0.0, 0.0, 0.0, 1.0)
    quaternion = _validated_numbers(values, 4, "node.rotation")
    length = math.sqrt(sum(component * component for component in quaternion))
    if length <= _EPSILON:
        raise InvalidTransformError("node.rotation must be a non-zero quaternion")
    return tuple(component / length for component in quaternion)


def _node_local_matrix(node):
    matrix = getattr(node, "matrix", None)
    translation = getattr(node, "translation", None)
    rotation = getattr(node, "rotation", None)
    scale = getattr(node, "scale", None)
    trs_properties = (translation, rotation, scale)

    if matrix is not None and any(value is not None for value in trs_properties):
        raise MatrixTRSConflictError(
            "node cannot define both matrix and translation, rotation, or scale"
        )

    if matrix is not None:
        return Matrix4.from_gltf(matrix)

    translation_values = (
        (0.0, 0.0, 0.0)
        if translation is None
        else _validated_numbers(translation, 3, "node.translation")
    )
    scale_values = (
        (1.0, 1.0, 1.0)
        if scale is None
        else _validated_numbers(scale, 3, "node.scale")
    )
    rotation_values = _normalised_quaternion(node, rotation)
    return _trs_matrix(translation_values, rotation_values, scale_values)


def _child_indices(node):
    children = getattr(node, "children", None)
    return _validated_indices(children, "node.children")


def _validated_indices(values, label):
    if values is None:
        return ()
    if isinstance(values, (str, bytes)):
        raise InvalidTransformError("{} must be an array of node indices".format(label))
    try:
        values = list(values)
    except TypeError:
        raise InvalidTransformError("{} must be an array of node indices".format(label))
    for value in values:
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise InvalidTransformError(
                "{} must contain only integer node indices".format(label)
            )
    return tuple(int(value) for value in values)


def _checked_node_index(node_index, node_count):
    if isinstance(node_index, bool) or not isinstance(node_index, numbers.Integral):
        raise InvalidNodeIndexError("node index must be an integer")
    node_index = int(node_index)
    if node_index < 0 or node_index >= node_count:
        raise InvalidNodeIndexError(
            "node index {} is outside the range [0, {})".format(
                node_index, node_count
            )
        )
    return node_index


def _resolve_scene_index(gltf, scene_index):
    scenes = getattr(gltf, "scenes", None) or []
    if not scenes:
        raise InvalidSceneIndexError("the asset does not contain a scene")
    if scene_index is None:
        scene_index = getattr(gltf, "scene", None)
        if scene_index is None:
            scene_index = 0
    if isinstance(scene_index, bool) or not isinstance(
        scene_index, numbers.Integral
    ):
        raise InvalidSceneIndexError("scene index must be an integer")
    scene_index = int(scene_index)
    if scene_index < 0 or scene_index >= len(scenes):
        raise InvalidSceneIndexError(
            "scene index {} is outside the range [0, {})".format(
                scene_index, len(scenes)
            )
        )
    return scene_index


def _consume_keyword_alias(kwargs, names, current, label):
    supplied = [name for name in names if name in kwargs]
    if len(supplied) > 1:
        raise TypeError(
            "{} was supplied more than once using aliases".format(label)
        )
    if supplied:
        if current is not None:
            raise TypeError("{} was supplied more than once".format(label))
        current = kwargs.pop(supplied[0])
    return current


def _node_object_index(gltf, value, label):
    if value is None or (
        isinstance(value, numbers.Integral) and not isinstance(value, bool)
    ):
        return value
    matches = [
        index for index, node in enumerate(gltf.nodes) if node is value
    ]
    if len(matches) == 1:
        return matches[0]
    raise InvalidNodeIndexError(
        "{} must be a node index or a node from this asset".format(label)
    )


def _scene_transforms(gltf, scene_index):
    scene_index = _resolve_scene_index(gltf, scene_index)
    scene = gltf.scenes[scene_index]
    roots = _validated_indices(getattr(scene, "nodes", None), "scene.nodes")

    nodes = getattr(gltf, "nodes", None) or []
    parent_by_node = {}
    visiting = set()
    transforms = []

    def visit(node_index, ancestors):
        node_index = _checked_node_index(node_index, len(nodes))
        if node_index in visiting:
            cycle = ancestors + (node_index,)
            raise SceneGraphCycleError(
                "scene graph contains a cycle: {}".format(
                    " -> ".join(str(index) for index in cycle)
                )
            )
        if node_index in parent_by_node:
            parent = parent_by_node[node_index]
            if parent is None:
                detail = "as a scene root or child"
            else:
                detail = "under node {}".format(parent)
            raise MultipleParentsError(
                "node {} is reachable more than once in the scene ({})".format(
                    node_index, detail
                )
            )

        parent_by_node[node_index] = ancestors[-1] if ancestors else None
        node = nodes[node_index]
        local_matrix = _node_local_matrix(node)
        if ancestors:
            parent_world = transforms_by_index[ancestors[-1]].world_matrix
            world_matrix = _matmul(parent_world, local_matrix)
        else:
            world_matrix = local_matrix

        transform = NodeTransform(
            index=node_index,
            path=tuple(ancestors),
            local_matrix=local_matrix,
            world_matrix=world_matrix,
        )
        transforms_by_index[node_index] = transform
        transforms.append(transform)

        visiting.add(node_index)
        for child_index in _child_indices(node):
            visit(child_index, ancestors + (node_index,))
        visiting.remove(node_index)

    transforms_by_index = {}
    for root_index in roots:
        visit(root_index, ())
    return transforms


def traverse_scene(gltf, scene_index=None, **kwargs):
    """Return transforms for every node reachable from a scene.

    The result is in pre-order and keeps the scene's root order and every
    node's child order.  Answering a traversal never modifies ``gltf``.
    """

    scene_index = _consume_keyword_alias(
        kwargs, ("scene", "scene_idx"), scene_index, "scene_index"
    )
    if kwargs:
        unexpected = next(iter(kwargs))
        raise TypeError("unexpected keyword argument {!r}".format(unexpected))
    return _scene_transforms(gltf, scene_index)


def _allclose_vectors(left, right, tolerance=1e-6):
    if len(left) != len(right):
        return False
    return all(
        abs(left_value - right_value)
        <= tolerance * max(1.0, abs(left_value), abs(right_value))
        for left_value, right_value in zip(left, right)
    )


def _allclose_matrices(left, right, tolerance=1e-6):
    return _allclose_vectors(left.to_gltf(), right.to_gltf(), tolerance)


def _decompose_trs(matrix):
    rows = [list(row) for row in matrix]
    if not _allclose_vectors(
        (rows[3][0], rows[3][1], rows[3][2], rows[3][3]),
        (0.0, 0.0, 0.0, 1.0),
        1e-7,
    ):
        raise ShearError(
            "the new local transform is not an affine glTF node transform"
        )

    columns = [
        tuple(rows[row][column] for row in range(3)) for column in range(3)
    ]
    norms = [
        math.sqrt(sum(component * component for component in column))
        for column in columns
    ]

    for left in range(3):
        for right in range(left + 1, 3):
            if norms[left] <= _EPSILON or norms[right] <= _EPSILON:
                continue
            dot = sum(
                columns[left][axis] * columns[right][axis] for axis in range(3)
            )
            if abs(dot) > _SHEAR_TOLERANCE * norms[left] * norms[right]:
                raise ShearError(
                    "preserving the world transform would introduce shear, "
                    "which glTF TRS properties cannot represent"
                )

    rotation_columns = [None, None, None]
    for column in range(3):
        if norms[column] > _EPSILON:
            rotation_columns[column] = tuple(
                component / norms[column] for component in columns[column]
            )

    present_columns = [
        column for column in range(3) if rotation_columns[column] is not None
    ]
    if not present_columns:
        rotation_columns = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    elif len(present_columns) == 1:
        present = present_columns[0]
        first = rotation_columns[present]
        reference = min(
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            key=lambda axis: abs(sum(first[i] * axis[i] for i in range(3))),
        )
        perpendicular = _normalise(
            _cross(reference, first),
            "the new local transform has no valid rotation",
        )
        next_column = (present + 1) % 3
        final_column = (present + 2) % 3
        rotation_columns[next_column] = perpendicular
        rotation_columns[final_column] = _cross(first, perpendicular)
    elif len(present_columns) == 2:
        missing = ({0, 1, 2} - set(present_columns)).pop()
        next_column = (missing + 1) % 3
        final_column = (missing + 2) % 3
        rotation_columns[missing] = _cross(
            rotation_columns[next_column], rotation_columns[final_column]
        )

    rotation_columns = [
        _normalise(column, "the new local transform has no valid rotation")
        for column in rotation_columns
    ]
    determinant = (
        rotation_columns[0][0]
        * (
            rotation_columns[1][1] * rotation_columns[2][2]
            - rotation_columns[1][2] * rotation_columns[2][1]
        )
        - rotation_columns[1][0]
        * (
            rotation_columns[0][1] * rotation_columns[2][2]
            - rotation_columns[0][2] * rotation_columns[2][1]
        )
        + rotation_columns[2][0]
        * (
            rotation_columns[0][1] * rotation_columns[1][2]
            - rotation_columns[0][2] * rotation_columns[1][1]
        )
    )

    scales = list(norms)
    if determinant < 0.0:
        negative_column = next(
            column for column in range(3) if scales[column] > _EPSILON
        )
        scales[negative_column] = -scales[negative_column]
        rotation_columns[negative_column] = tuple(
            -component for component in rotation_columns[negative_column]
        )
        determinant = -determinant
    if abs(determinant - 1.0) > _SHEAR_TOLERANCE:
        raise ShearError(
            "the new local transform cannot be represented by a glTF rotation"
        )

    rotation = (
        (rotation_columns[0][0], rotation_columns[1][0], rotation_columns[2][0]),
        (rotation_columns[0][1], rotation_columns[1][1], rotation_columns[2][1]),
        (rotation_columns[0][2], rotation_columns[1][2], rotation_columns[2][2]),
    )
    trace = rotation[0][0] + rotation[1][1] + rotation[2][2]
    if trace > 0.0:
        root = math.sqrt(trace + 1.0) * 2.0
        quaternion = (
            (rotation[2][1] - rotation[1][2]) / root,
            (rotation[0][2] - rotation[2][0]) / root,
            (rotation[1][0] - rotation[0][1]) / root,
            0.25 * root,
        )
    elif rotation[0][0] > rotation[1][1] and rotation[0][0] > rotation[2][2]:
        root = math.sqrt(
            1.0 + rotation[0][0] - rotation[1][1] - rotation[2][2]
        ) * 2.0
        quaternion = (
            0.25 * root,
            (rotation[0][1] + rotation[1][0]) / root,
            (rotation[0][2] + rotation[2][0]) / root,
            (rotation[2][1] - rotation[1][2]) / root,
        )
    elif rotation[1][1] > rotation[2][2]:
        root = math.sqrt(
            1.0 + rotation[1][1] - rotation[0][0] - rotation[2][2]
        ) * 2.0
        quaternion = (
            (rotation[0][1] + rotation[1][0]) / root,
            0.25 * root,
            (rotation[1][2] + rotation[2][1]) / root,
            (rotation[0][2] - rotation[2][0]) / root,
        )
    else:
        root = math.sqrt(
            1.0 + rotation[2][2] - rotation[0][0] - rotation[1][1]
        ) * 2.0
        quaternion = (
            (rotation[0][2] + rotation[2][0]) / root,
            (rotation[1][2] + rotation[2][1]) / root,
            0.25 * root,
            (rotation[1][0] - rotation[0][1]) / root,
        )

    quaternion_length = math.sqrt(
        sum(component * component for component in quaternion)
    )
    if quaternion_length <= _EPSILON:
        raise ShearError("the new local transform has no valid rotation")
    quaternion = tuple(
        component / quaternion_length for component in quaternion
    )
    translation = (rows[0][3], rows[1][3], rows[2][3])

    reconstructed = _trs_matrix(translation, quaternion, scales)
    if not _allclose_matrices(reconstructed, matrix, 1e-6):
        raise ShearError(
            "the new local transform cannot be represented by glTF TRS"
        )
    return translation, quaternion, tuple(scales)


def _cross(left, right):
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _normalise(vector, message):
    length = math.sqrt(sum(component * component for component in vector))
    if length <= _EPSILON:
        raise ShearError(message)
    return tuple(component / length for component in vector)


def _validate_reparent_asset(gltf):
    scenes = getattr(gltf, "scenes", None) or []
    if len(scenes) != 1:
        raise ReparentUnsupportedError(
            "hierarchy editing requires an asset with exactly one scene"
        )
    if getattr(gltf, "animations", None):
        raise ReparentUnsupportedError(
            "hierarchy editing does not support animated assets"
        )
    if getattr(gltf, "skins", None):
        raise ReparentUnsupportedError(
            "hierarchy editing does not support skinned assets"
        )
    if any(getattr(node, "skin", None) is not None for node in gltf.nodes):
        raise ReparentUnsupportedError(
            "hierarchy editing does not support nodes with a skin"
        )


def _remove_once(values, node_index):
    occurrences = [index for index, value in enumerate(values) if value == node_index]
    if len(occurrences) != 1:
        raise MultipleParentsError(
            "node {} is referenced more than once in one parent list".format(
                node_index
            )
        )
    values.pop(occurrences[0])


def reparent_node(
    gltf, node_index=None, new_parent_index=None, scene_index=None, **kwargs
):
    """Return a copied asset with one node and its subtree reattached.

    ``new_parent_index=None`` makes the node a scene root.  The node's world
    transform and the world transforms of its descendants are preserved.  The
    input asset is never modified.
    """

    node_index = _consume_keyword_alias(
        kwargs, ("node", "node_idx"), node_index, "node_index"
    )
    new_parent_index = _consume_keyword_alias(
        kwargs,
        ("parent_index", "parent_idx", "new_parent", "parent"),
        new_parent_index,
        "new_parent_index",
    )
    scene_index = _consume_keyword_alias(
        kwargs, ("scene", "scene_idx"), scene_index, "scene_index"
    )
    if kwargs:
        unexpected = next(iter(kwargs))
        raise TypeError("unexpected keyword argument {!r}".format(unexpected))
    node_index = _node_object_index(gltf, node_index, "node_index")
    new_parent_index = _node_object_index(
        gltf, new_parent_index, "new_parent_index"
    )

    _validate_reparent_asset(gltf)
    scene_index = _resolve_scene_index(gltf, scene_index)
    transforms = _scene_transforms(gltf, scene_index)
    transforms_by_index = {transform.index: transform for transform in transforms}

    _checked_node_index(node_index, len(gltf.nodes))
    if node_index not in transforms_by_index:
        raise InvalidNodeIndexError(
            "node {} is not reachable from the scene".format(node_index)
        )
    if new_parent_index is not None:
        _checked_node_index(new_parent_index, len(gltf.nodes))
        if new_parent_index not in transforms_by_index:
            raise InvalidNodeIndexError(
                "node {} is not reachable from the scene".format(
                    new_parent_index
                )
            )
        old_transform = transforms_by_index[node_index]
        if (
            new_parent_index == node_index
            or node_index in transforms_by_index[new_parent_index].path
        ):
            raise SceneGraphCycleError(
                "node {} cannot be parented to itself or one of its descendants".format(
                    node_index
                )
            )

    old_world = transforms_by_index[node_index].world_matrix
    if new_parent_index is None:
        parent_world = Matrix4.identity()
    else:
        parent_world = transforms_by_index[new_parent_index].world_matrix

    local_matrix = _matmul(_inverse(parent_world), old_world)
    translation, rotation, scale = _decompose_trs(local_matrix)

    result = copy.deepcopy(gltf)
    scene = result.scenes[scene_index]
    old_parent = transforms_by_index[node_index].parent_index
    if old_parent != new_parent_index:
        if old_parent is None:
            if scene.nodes is None:
                scene.nodes = []
            else:
                scene.nodes = list(scene.nodes)
            _remove_once(scene.nodes, node_index)
        else:
            old_parent_children = result.nodes[old_parent].children
            if old_parent_children is None:
                old_parent_children = []
                result.nodes[old_parent].children = old_parent_children
            else:
                old_parent_children = list(old_parent_children)
                result.nodes[old_parent].children = old_parent_children
            _remove_once(old_parent_children, node_index)

        if new_parent_index is None:
            if scene.nodes is None:
                scene.nodes = []
            elif not isinstance(scene.nodes, list):
                scene.nodes = list(scene.nodes)
            scene.nodes.append(node_index)
        else:
            new_parent_children = result.nodes[new_parent_index].children
            if new_parent_children is None:
                new_parent_children = []
                result.nodes[new_parent_index].children = new_parent_children
            elif not isinstance(new_parent_children, list):
                new_parent_children = list(new_parent_children)
                result.nodes[new_parent_index].children = new_parent_children
            new_parent_children.append(node_index)

    result_node = result.nodes[node_index]
    result_node.matrix = None
    result_node.translation = list(translation)
    result_node.rotation = list(rotation)
    result_node.scale = list(scale)

    updated_transforms = {
        transform.index: transform
        for transform in _scene_transforms(result, scene_index)
    }
    if set(updated_transforms) != set(transforms_by_index):
        raise SceneGraphError(
            "hierarchy editing unexpectedly changed scene reachability"
        )
    for index, original in transforms_by_index.items():
        if not _allclose_matrices(
            updated_transforms[index].world_matrix, original.world_matrix
        ):
            raise SceneGraphError(
                "hierarchy editing failed to preserve the world transform "
                "of node {}".format(index)
            )

    return result


def get_scene_node_transforms(gltf, scene_index=None, **kwargs):
    """Alias for :func:`traverse_scene`."""

    return traverse_scene(gltf, scene_index, **kwargs)


def get_scene_hierarchy(gltf, scene_index=None, **kwargs):
    """Alias for :func:`traverse_scene`."""

    return traverse_scene(gltf, scene_index, **kwargs)


def reparent_node_subtree(
    gltf, node_index=None, new_parent_index=None, scene_index=None, **kwargs
):
    """Alias for :func:`reparent_node`."""

    return reparent_node(
        gltf, node_index, new_parent_index, scene_index, **kwargs
    )


def move_node(
    gltf, node_index=None, new_parent_index=None, scene_index=None, **kwargs
):
    """Alias for :func:`reparent_node`."""

    return reparent_node(
        gltf, node_index, new_parent_index, scene_index, **kwargs
    )
