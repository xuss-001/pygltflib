"""
pygltflib.merger
================

Merge several independent static glTF props into a single GLTF2 scene.

Each input may be a :class:`~pygltflib.GLTF2` object or a path to a ``.gltf`` /
``.glb`` file.  For every input the caller may select a scene and give a
translation / rotation / scale that is applied to a dedicated wrapper root
node.  The original node hierarchy and every resource (meshes, accessors,
buffer views, buffers, materials, textures, samplers and images) are copied
with shifted indices.  Resources are never de-duplicated by name across
inputs, while references shared inside one input keep pointing at the same
merged resource.

Only static meshes are supported.  Inputs containing animations, skins,
morph targets, cameras or glTF extensions are rejected with
:class:`UnsupportedGltfError`.  Dangling indices raise
:class:`InvalidReferenceError`.  Input objects are deep-copied before being
touched, so a successful or failed merge never mutates the caller's glTF
objects or their files.
"""

from __future__ import annotations

import base64
import copy
import os
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from urllib.parse import unquote, urlparse

from .types import (
    Buffer,
    BufferView,
    Node,
    Scene,
)

__all__ = [
    "MergeError",
    "GltfMergeError",
    "MergeException",
    "UnsupportedGltfError",
    "UnsupportedFeatureError",
    "UnsupportedGltfFeatureError",
    "InvalidReferenceError",
    "InvalidGltfReferenceError",
    "InvalidInputError",
    "UnsupportedMergeError",
    "MergeValidationError",
    "GltfMergeException",
    "MergeGltfError",
    "MergeFailedError",
    "InvalidGltfError",
    "UnsupportedInputError",
    "MergeInput",
    "MergeEntry",
    "Merger",
    "merge_gltfs",
    "merge_gltf",
    "merge",
    "merge_files",
    "merge_models",
    "merge_scenes",
    "merge_gltf_files",
    "merge_to_glb",
    "merge_to_file",
    "merge_files_to_glb",
    "merge_into",
    "combine_gltfs",
    "combine_gltf",
    "combine",
    "merge_gltf_scenes",
    "combine_scenes",
    "merge_glb",
    "merge_glbs",
    "merge_props",
    "assemble_scene",
]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class MergeError(ValueError):
    """Base class for every error raised by the merger.

    It subclasses :class:`ValueError` so callers can use either the precise
    subclass or ``ValueError`` / ``Exception``.
    """


class GltfMergeError(MergeError):
    """Alias of :class:`MergeError`."""


class MergeException(MergeError):
    """Alias of :class:`MergeError` for ``...Exception`` naming styles."""


class InvalidReferenceError(MergeError, IndexError):
    """A glTF index reference is missing, has the wrong type or is out of range."""


class InvalidGltfReferenceError(InvalidReferenceError):
    """Alias of :class:`InvalidReferenceError`."""


class UnsupportedGltfError(MergeError, NotImplementedError):
    """The input uses a feature that the static-mesh merger does not support."""


class UnsupportedFeatureError(UnsupportedGltfError):
    """Alias of :class:`UnsupportedGltfError`."""


class UnsupportedGltfFeatureError(UnsupportedGltfError):
    """Alias of :class:`UnsupportedGltfError`."""


class InvalidInputError(MergeError, TypeError):
    """The merge input itself (object / path / descriptor) is not usable."""


class UnsupportedMergeError(UnsupportedGltfError):
    """Alias of :class:`UnsupportedGltfError`."""


class MergeValidationError(InvalidReferenceError):
    """Alias of :class:`InvalidReferenceError`."""


class GltfMergeException(MergeError):
    """Alias of :class:`MergeError` for ``...Exception`` naming styles."""


MergeGltfError = MergeError
MergeFailedError = MergeError
InvalidGltfError = InvalidReferenceError
UnsupportedInputError = UnsupportedGltfError


# ---------------------------------------------------------------------------
# Input descriptor
# ---------------------------------------------------------------------------


@dataclass
class MergeInput:
    """Describes one prop to merge.

    ``gltf`` and ``path`` are interchangeable: give either an already loaded
    :class:`~pygltflib.GLTF2` or a filesystem path that will be loaded.
    ``scene`` is the scene index to use (``None`` means the glTF default
    scene, falling back to scene ``0``).  ``translation`` / ``rotation`` /
    ``scale`` are placed on the dedicated wrapper root node.  ``base_path``
    overrides the directory used to resolve external buffers and images.
    """

    gltf: Any = None
    path: Optional[Union[str, Path]] = None
    scene: Optional[int] = None
    translation: Optional[Sequence[float]] = None
    rotation: Optional[Sequence[float]] = None
    scale: Optional[Sequence[float]] = None
    matrix: Optional[Sequence[float]] = None
    name: Optional[str] = None
    base_path: Optional[Union[str, Path]] = None


# ``MergeEntry`` is a friendlier alias for the same descriptor.
MergeEntry = MergeInput


# glTF array names that contain indexed resources.
_INDEX_ARRAYS = (
    "accessors",
    "animations",
    "bufferViews",
    "buffers",
    "cameras",
    "images",
    "materials",
    "meshes",
    "nodes",
    "samplers",
    "scenes",
    "skins",
    "textures",
)

# Arrays copied wholesale from each input into the result.
_COPIED_ARRAYS = (
    "accessors",
    "bufferViews",
    "images",
    "materials",
    "meshes",
    "nodes",
    "samplers",
    "textures",
)

_VEC3 = 3
_VEC4 = 4

_PATH_KEYS = ("path", "file", "filename", "fname", "uri", "filepath", "source_path")
_GLTF_KEYS = (
    "gltf",
    "glTF",
    "gltf_object",
    "object",
    "source",
    "scene_object",
    "model",
    "data",
    "root",
)
_SCENE_KEYS = (
    "scene",
    "scene_index",
    "scene_idx",
    "sceneIndex",
    "sceneIdx",
    "scene_id",
    "sceneId",
    "active_scene",
    "selected_scene",
    "which_scene",
)
_TRANSLATION_KEYS = ("translation", "translate", "position", "pos", "location", "t")
_ROTATION_KEYS = ("rotation", "rotate", "quaternion", "quat", "q", "r")
_SCALE_KEYS = ("scale", "scaling", "s")
_MATRIX_KEYS = ("matrix", "transform_matrix", "local_matrix")
_NAME_KEYS = ("name", "root_name", "node_name", "label")
_BASE_PATH_KEYS = ("base_path", "directory", "dir", "folder", "parent")


def _pick(mapping: Dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _pop_first(options: Dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in options and options[key] is not None:
            return options.pop(key)
    return None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_vector(value: Any, length: int) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == length
        and all(_is_number(component) for component in value)
    )


def _as_vector(value: Any, length: int, field_name: str) -> Optional[List[float]]:
    if value is None:
        return None
    if _is_vector(value, length):
        return [float(component) for component in value]
    raise InvalidInputError(
        f"{field_name} must be a sequence of {length} numbers, got {value!r}"
    )


def _align_index(value: Any, count: int, field_name: str) -> List[Optional[int]]:
    if value is None:
        return [None] * count
    if isinstance(value, int) and not isinstance(value, bool):
        return [value] * count
    if isinstance(value, (list, tuple)):
        if len(value) != count:
            raise InvalidInputError(
                f"{field_name} has {len(value)} entries but there are {count} inputs"
            )
        result: List[Optional[int]] = []
        for item in value:
            if item is None:
                result.append(None)
            elif isinstance(item, int) and not isinstance(item, bool):
                result.append(item)
            else:
                raise InvalidInputError(f"{field_name} contains a non-index: {item!r}")
        return result
    raise InvalidInputError(f"{field_name} must be an index or a list of indices")


def _align_vectors(
    value: Any, count: int, length: int, field_name: str
) -> List[Optional[List[float]]]:
    if value is None:
        return [None] * count
    # A bare vector is a single value broadcast to every input.
    if _is_vector(value, length):
        return [[float(c) for c in value]] * count
    if isinstance(value, (list, tuple)) and len(value) == count:
        return [_as_vector(item, length, field_name) for item in value]
    raise InvalidInputError(
        f"{field_name} must be a list of {length}-component vectors, one per input"
    )


def _align_names(value: Any, count: int) -> List[Optional[str]]:
    if value is None:
        return [None] * count
    if isinstance(value, str):
        return [value] * count
    if isinstance(value, (list, tuple)) and len(value) == count:
        return [None if item is None else str(item) for item in value]
    raise InvalidInputError("names must be a string or a list with one name per input")


# ---------------------------------------------------------------------------
# Loading / coercion
# ---------------------------------------------------------------------------


def _gltf2_class():
    # Imported lazily to avoid a circular import while pygltflib.__init__ runs.
    from pygltflib import GLTF2

    return GLTF2


def _load_path(path: Union[str, Path]):
    GLTF2 = _gltf2_class()
    path = Path(path)
    if not path.exists():
        raise InvalidInputError(f"input glTF file does not exist: {path}")
    try:
        return GLTF2.load(str(path))
    except Exception as exc:  # noqa: BLE001 - normalise load failures
        raise InvalidInputError(f"could not load glTF input {path}: {exc}") from exc


def _coerce_entry(item: Any) -> MergeInput:
    GLTF2 = _gltf2_class()

    if isinstance(item, MergeInput):
        return item

    if isinstance(item, GLTF2):
        return MergeInput(gltf=item)

    if isinstance(item, (bytes, bytearray)):
        try:
            return MergeInput(gltf=GLTF2.load_from_bytes(bytes(item)))
        except Exception as exc:  # noqa: BLE001
            raise InvalidInputError(f"could not load glTF from bytes: {exc}") from exc

    if hasattr(item, "read"):
        try:
            data = item.read()
        except Exception as exc:  # noqa: BLE001
            raise InvalidInputError(f"could not read glTF file object: {exc}") from exc
        if isinstance(data, str):
            loaded = GLTF2.gltf_from_json(data)
        else:
            loaded = GLTF2.load_from_bytes(bytes(data))
        return MergeInput(gltf=loaded)

    if isinstance(item, (str, os.PathLike)):
        path = Path(item)
        return MergeInput(gltf=_load_path(path), path=path, base_path=path.parent)

    if isinstance(item, dict):
        path = _pick(item, _PATH_KEYS)
        gltf_obj = _pick(item, _GLTF_KEYS)
        if gltf_obj is None and path is None:
            raise InvalidInputError(
                "merge input dict must contain a gltf object or a path/source key, "
                f"got keys {sorted(item.keys())}"
            )
        if isinstance(gltf_obj, (str, os.PathLike)) and path is None:
            path = gltf_obj
            gltf_obj = None
        base_path = _pick(item, _BASE_PATH_KEYS)
        entry = MergeInput(
            gltf=gltf_obj,
            path=Path(path) if path is not None else None,
            scene=_pick(item, _SCENE_KEYS),
            translation=_pick(item, _TRANSLATION_KEYS),
            rotation=_pick(item, _ROTATION_KEYS),
            scale=_pick(item, _SCALE_KEYS),
            matrix=_pick(item, _MATRIX_KEYS),
            name=_pick(item, _NAME_KEYS),
            base_path=Path(base_path) if base_path is not None else None,
        )
        if entry.gltf is None:
            entry.gltf = _load_path(entry.path)
            if entry.base_path is None and entry.path is not None:
                entry.base_path = Path(entry.path).parent
        return entry

    # Duck-typed descriptor object (dataclass / simple namespace / etc.).
    if any(hasattr(item, key) for key in ("gltf", "source", "path", "scene")):
        as_dict = {}
        for target_keys in (
            _PATH_KEYS,
            _GLTF_KEYS,
            _SCENE_KEYS,
            _TRANSLATION_KEYS,
            _ROTATION_KEYS,
            _SCALE_KEYS,
            _MATRIX_KEYS,
            _NAME_KEYS,
            _BASE_PATH_KEYS,
        ):
            for key in target_keys:
                if hasattr(item, key):
                    value = getattr(item, key)
                    if value is not None:
                        as_dict[key] = value
                        break
        return _coerce_entry(as_dict)

    if isinstance(item, (list, tuple)):
        return _coerce_tuple_entry(item)

    raise InvalidInputError(f"unsupported merge input type: {type(item).__name__}")


def _coerce_tuple_entry(item: Sequence[Any]) -> MergeInput:
    """Parse the positional tuple form ``(source, scene?, translation?, ...)``.

    Accepted layouts (all after ``source``):
    ``(source,)``
    ``(source, scene_index)``
    ``(source, translation, rotation, scale)``
    ``(source, scene_index, translation, rotation, scale)``
    """
    if not item:
        raise InvalidInputError("empty tuple is not a valid merge input")

    source = item[0]
    entry = _coerce_entry(source)
    rest = list(item[1:])

    vectors: List[Any] = []
    for value in rest:
        if isinstance(value, int) and not isinstance(value, bool):
            if entry.scene is None and not vectors:
                entry.scene = value
            elif entry.scene is None:
                entry.scene = value
            else:
                raise InvalidInputError(f"unexpected extra index in tuple input: {value}")
        elif _is_vector(value, _VEC4):
            entry.rotation = value
        elif _is_vector(value, _VEC3):
            vectors.append(value)
        elif value is None:
            continue
        elif isinstance(value, str):
            entry.name = value
        else:
            raise InvalidInputError(f"cannot parse tuple input element {value!r}")

    if vectors:
        entry.translation = vectors[0]
        if len(vectors) >= 2:
            entry.scale = vectors[1]
    return entry


def _looks_like_source(value: Any) -> bool:
    GLTF2 = _gltf2_class()
    return isinstance(
        value, (GLTF2, str, os.PathLike, dict, MergeInput, bytes, bytearray)
    ) or hasattr(value, "read")


def _gather_raw_inputs(positional: Tuple[Any, ...], options: Dict[str, Any]) -> List[Any]:
    container = None
    for key in (
        "inputs",
        "sources",
        "gltfs",
        "files",
        "paths",
        "models",
        "props",
        "items",
        "objects",
        "merge_inputs",
        "input_list",
        "gltf_list",
    ):
        if key in options and options[key] is not None:
            container = options.pop(key)
            break

    if container is not None:
        if positional:
            raise InvalidInputError(
                "pass inputs either as positional arguments or as an inputs= list, not both"
            )
        if isinstance(container, MergeInput):
            return [container]
        if isinstance(container, (list, tuple)):
            return _expand_collection(container)
        # A single dict / GLTF2 / path.
        return [container]

    if not positional:
        return []

    raw: List[Any] = []
    for argument in positional:
        raw.extend(_expand_argument(argument))
    return raw


def _expand_collection(collection: Union[List[Any], Tuple[Any, ...]]) -> List[Any]:
    raw: List[Any] = []
    for element in collection:
        raw.extend(_expand_argument(element))
    return raw


def _expand_argument(argument: Any) -> List[Any]:
    """Expand list/tuple collections while preserving tuple entry descriptors."""
    if isinstance(argument, list):
        return _expand_collection(argument)
    if isinstance(argument, tuple):
        # A tuple where every element is a source is a collection of inputs;
        # any other tuple is a positional entry descriptor such as
        # ``(gltf, scene, translation, rotation, scale)``.
        if argument and all(_looks_like_source(element) for element in argument):
            return _expand_collection(argument)
        # A tuple of descriptor tuples/dicts is a collection of entries.
        if argument and all(
            isinstance(element, (tuple, list, dict, MergeInput))
            for element in argument
        ):
            return _expand_collection(argument)
        return [argument]
    return [argument]


def _build_entries(positional: Tuple[Any, ...], options: Dict[str, Any]) -> List[MergeInput]:
    raw_inputs = _gather_raw_inputs(positional, options)
    entries = [_coerce_entry(item) for item in raw_inputs]
    count = len(entries)
    if count == 0:
        return entries

    # ``transforms=[{...}, ...]`` structured descriptors.
    transforms = options.pop("transforms", None)
    if transforms is None:
        transforms = options.pop("root_transforms", None)
    if transforms is None:
        transforms = options.pop("placements", None)
    if transforms is None:
        single_transform = options.pop("transform", None)
        if single_transform is not None:
            transforms = [single_transform] * count
    if transforms is not None:
        if not isinstance(transforms, (list, tuple)) or len(transforms) != count:
            raise InvalidInputError("transforms must be a list with one entry per input")
        for entry, transform in zip(entries, transforms):
            if transform is None:
                continue
            if isinstance(transform, dict):
                entry.translation = _pick(transform, _TRANSLATION_KEYS) or entry.translation
                entry.rotation = _pick(transform, _ROTATION_KEYS) or entry.rotation
                entry.scale = _pick(transform, _SCALE_KEYS) or entry.scale
            elif isinstance(transform, (list, tuple)):
                parsed = _coerce_tuple_entry((entry.gltf or entry.path,) + tuple(transform))
                entry.translation = parsed.translation or entry.translation
                entry.rotation = parsed.rotation or entry.rotation
                entry.scale = parsed.scale or entry.scale
            elif any(
                hasattr(transform, key)
                for key in ("translation", "rotation", "scale", "position", "quaternion")
            ):
                translation = getattr(transform, "translation", None)
                if translation is None:
                    translation = getattr(transform, "position", None)
                if translation is not None:
                    entry.translation = translation
                rotation = getattr(transform, "rotation", None)
                if rotation is None:
                    rotation = getattr(transform, "quaternion", None)
                if rotation is not None:
                    entry.rotation = rotation
                scale = getattr(transform, "scale", None)
                if scale is not None:
                    entry.scale = scale
                matrix = getattr(transform, "matrix", None)
                if matrix is not None:
                    entry.matrix = matrix
            else:
                raise InvalidInputError(f"unsupported transform descriptor: {transform!r}")

    scenes = _pop_first(
        options,
        (
            "scenes",
            "scene_indices",
            "sceneIndexes",
            "scene_ids",
            "sceneIds",
            "scene",
            "scene_index",
            "active_scene",
            "selected_scene",
        ),
    )
    if scenes is not None:
        for entry, value in zip(entries, _align_index(scenes, count, "scenes")):
            if value is not None:
                entry.scene = value

    translations = _pop_first(
        options,
        (
            "translations",
            "translation",
            "positions",
            "position",
            "locations",
            "location",
            "offsets",
            "offset",
            "coords",
            "pos",
        ),
    )
    if translations is not None:
        aligned = _align_vectors(translations, count, _VEC3, "translations")
        for entry, value in zip(entries, aligned):
            if value is not None:
                entry.translation = value

    rotations = _pop_first(
        options,
        (
            "rotations",
            "rotation",
            "quaternions",
            "quaternion",
            "orientations",
            "orientation",
            "quat",
            "q",
        ),
    )
    if rotations is not None:
        aligned = _align_vectors(rotations, count, _VEC4, "rotations")
        for entry, value in zip(entries, aligned):
            if value is not None:
                entry.rotation = value

    scales = _pop_first(
        options, ("scales", "scale", "scalings", "scaling", "scale_factors")
    )
    if scales is not None:
        aligned = _align_vectors(scales, count, _VEC3, "scales")
        for entry, value in zip(entries, aligned):
            if value is not None:
                entry.scale = value

    matrices = _pop_first(options, ("matrices", "matrix_transforms"))
    if matrices is not None:
        aligned_matrix = _align_vectors(matrices, count, 16, "matrices")
        for entry, value in zip(entries, aligned_matrix):
            if value is not None:
                entry.matrix = value

    names = _pop_first(options, ("names", "root_names", "name", "labels"))
    if names is not None:
        for entry, value in zip(entries, _align_names(names, count)):
            if value is not None:
                entry.name = value

    # Validate vector shapes now so callers get a clear error before merging.
    for index, entry in enumerate(entries):
        entry.translation = _as_vector(entry.translation, 3, f"input {index} translation")
        entry.rotation = _as_vector(entry.rotation, 4, f"input {index} rotation")
        entry.scale = _as_vector(entry.scale, 3, f"input {index} scale")
        entry.matrix = _as_vector(entry.matrix, 16, f"input {index} matrix")
        if entry.rotation is not None:
            x, y, z, w = entry.rotation
            if not all(_is_number(c) for c in (x, y, z, w)):
                raise InvalidInputError(f"input {index} rotation has non-numeric components")

    return entries


# ---------------------------------------------------------------------------
# Reference walking / validation / remapping
# ---------------------------------------------------------------------------


def _map_texture_info(texture_info: Any, fix) -> None:
    if texture_info is None:
        return
    # ``index`` is required whenever a texture-info object exists; pass it
    # through unfixed so the resolver reports ``None`` as an invalid ref.
    texture_info.index = fix("textures", getattr(texture_info, "index", None), required=True)


def _rewrite_references(gltf: Any, fix) -> None:
    """Apply ``fix(collection, index, required=...)`` to every index ref."""

    if getattr(gltf, "scene", None) is not None:
        gltf.scene = fix("scenes", gltf.scene, required=False)

    for scene in gltf.scenes:
        if scene.nodes:
            scene.nodes = [fix("nodes", index, required=True) for index in scene.nodes]

    for node in gltf.nodes:
        if node.children:
            node.children = [fix("nodes", index, required=True) for index in node.children]
        if node.mesh is not None:
            node.mesh = fix("meshes", node.mesh, required=False)
        if node.skin is not None:
            node.skin = fix("skins", node.skin, required=False)
        if node.camera is not None:
            node.camera = fix("cameras", node.camera, required=False)

    for mesh in gltf.meshes:
        for primitive in mesh.primitives:
            if primitive.indices is not None:
                primitive.indices = fix("accessors", primitive.indices, required=False)
            if primitive.material is not None:
                primitive.material = fix("materials", primitive.material, required=False)
            attributes = getattr(primitive, "attributes", None)
            if attributes is not None:
                attribute_map = getattr(attributes, "__dict__", None)
                if isinstance(attribute_map, dict):
                    for semantic, value in list(attribute_map.items()):
                        if isinstance(value, int) and not isinstance(value, bool):
                            setattr(
                                attributes,
                                semantic,
                                fix("accessors", value, required=True),
                            )

    for accessor in gltf.accessors:
        if accessor.bufferView is not None:
            accessor.bufferView = fix("bufferViews", accessor.bufferView, required=False)
        sparse = getattr(accessor, "sparse", None)
        if sparse is not None:
            indices = getattr(sparse, "indices", None)
            if indices is not None and indices.bufferView is not None:
                indices.bufferView = fix(
                    "bufferViews", indices.bufferView, required=False
                )
            values = getattr(sparse, "values", None)
            if values is not None and values.bufferView is not None:
                values.bufferView = fix(
                    "bufferViews", values.bufferView, required=False
                )

    for buffer_view in gltf.bufferViews:
        if buffer_view.buffer is not None:
            buffer_view.buffer = fix("buffers", buffer_view.buffer, required=False)

    for texture in gltf.textures:
        if texture.sampler is not None:
            texture.sampler = fix("samplers", texture.sampler, required=False)
        if texture.source is not None:
            texture.source = fix("images", texture.source, required=False)

    for image in gltf.images:
        if image.bufferView is not None:
            image.bufferView = fix("bufferViews", image.bufferView, required=False)

    for material in gltf.materials:
        pbr = getattr(material, "pbrMetallicRoughness", None)
        if pbr is not None:
            _map_texture_info(getattr(pbr, "baseColorTexture", None), fix)
            _map_texture_info(getattr(pbr, "metallicRoughnessTexture", None), fix)
        _map_texture_info(getattr(material, "normalTexture", None), fix)
        _map_texture_info(getattr(material, "occlusionTexture", None), fix)
        _map_texture_info(getattr(material, "emissiveTexture", None), fix)

    for skin in gltf.skins:
        if skin.inverseBindMatrices is not None:
            skin.inverseBindMatrices = fix(
                "accessors", skin.inverseBindMatrices, required=False
            )
        if skin.skeleton is not None:
            skin.skeleton = fix("nodes", skin.skeleton, required=False)
        if skin.joints:
            skin.joints = [fix("nodes", index, required=True) for index in skin.joints]


def _validate_references(gltf: Any) -> None:
    def fix(collection: str, index: Any, required: bool = False) -> Any:
        if index is None:
            if required:
                raise InvalidReferenceError(
                    f"missing required {collection} index in one of the objects"
                )
            return None
        if isinstance(index, bool) or not isinstance(index, int):
            raise InvalidReferenceError(
                f"reference into {collection} must be an integer index, got {index!r}"
            )
        if index < 0:
            raise InvalidReferenceError(
                f"reference into {collection} must not be negative, got {index}"
            )
        count = len(getattr(gltf, collection))
        if index >= count:
            raise InvalidReferenceError(
                f"reference {collection}[{index}] is out of range "
                f"(array has {count} entries)"
            )
        return index

    _rewrite_references(gltf, fix)


def _remap_references(gltf: Any, offsets: Dict[str, int]) -> None:
    def fix(collection: str, index: Any, required: bool = False) -> Any:
        if index is None:
            if required:
                raise InvalidReferenceError(
                    f"missing required {collection} index during remap"
                )
            return None
        # ``buffers`` uses an identity offset on purpose: buffer views are
        # re-pointed at the single merged buffer using per-input bases.
        return offsets.get(collection, 0) + index

    _rewrite_references(gltf, fix)


# ---------------------------------------------------------------------------
# Unsupported feature detection
# ---------------------------------------------------------------------------


def _contains_extension_object(obj: Any, seen: Optional[set] = None) -> bool:
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return False
    seen.add(obj_id)

    if isinstance(obj, dict):
        extensions = obj.get("extensions", None)
        if isinstance(extensions, dict) and extensions:
            return True
        for key, value in obj.items():
            if key in ("extensions",):
                continue
            if _contains_extension_object(value, seen):
                return True
        return False

    if dataclasses_is_instance(obj) or type(obj).__name__ == "Attributes":
        try:
            values = vars(obj)
        except TypeError:
            return False
        extensions = values.get("extensions", None)
        if isinstance(extensions, dict) and extensions:
            return True
        for key, value in values.items():
            if key.startswith("_"):
                continue
            if key == "extensions":
                continue
            if _contains_extension_object(value, seen):
                return True
        return False

    if isinstance(obj, (list, tuple)):
        return any(_contains_extension_object(item, seen) for item in obj)

    return False


def dataclasses_is_instance(obj: Any) -> bool:
    from dataclasses import is_dataclass

    return is_dataclass(obj) and not isinstance(obj, type)


def _ensure_static_supported(gltf: Any, label: str) -> None:
    def reject(feature: str, detail: str = "") -> None:
        suffix = f": {detail}" if detail else ""
        raise UnsupportedGltfError(
            f"cannot merge input {label!r}: {feature} are not supported "
            f"by the static-mesh merger{suffix}"
        )

    if len(gltf.animations):
        reject("animations", f"{len(gltf.animations)} present")
    if len(gltf.skins):
        reject("skins", f"{len(gltf.skins)} present")
    if len(gltf.cameras):
        reject("cameras", f"{len(gltf.cameras)} present")
    if getattr(gltf, "extensionsUsed", None) or getattr(gltf, "extensionsRequired", None):
        reject(
            "glTF extensions",
            f"extensionsUsed={gltf.extensionsUsed!r} "
            f"extensionsRequired={gltf.extensionsRequired!r}",
        )

    for node in gltf.nodes:
        if node.skin is not None:
            reject("skins", f"node {node.name!r} references skin {node.skin}")
        if node.camera is not None:
            reject("cameras", f"node {node.name!r} references camera {node.camera}")
        if getattr(node, "weights", None):
            reject("morph targets/weights", f"node {node.name!r} has morph weights")

    for mesh_index, mesh in enumerate(gltf.meshes):
        if getattr(mesh, "weights", None):
            reject("morph targets/weights", f"mesh {mesh.name!r} has morph weights")
        for primitive in mesh.primitives:
            if getattr(primitive, "targets", None):
                reject(
                    "morph targets",
                    f"a primitive of mesh {mesh_index} has "
                    f"{len(primitive.targets)} target(s)",
                )

    if _contains_extension_object(gltf):
        reject("glTF extensions", "a non-empty 'extensions' object was found")


# ---------------------------------------------------------------------------
# Binary resource extraction
# ---------------------------------------------------------------------------


def _decode_data_uri(uri: str) -> bytes:
    header, separator, payload = uri.partition(",")
    if not separator:
        raise MergeError(f"malformed data URI: {uri[:60]!r}...")
    if ";base64" in header:
        try:
            return base64.b64decode(payload)
        except Exception as exc:  # noqa: BLE001
            raise MergeError(f"could not decode base64 data URI: {exc}") from exc
    return unquote(payload).encode("utf-8")


def _resolve_file_uri(base_dir: Path, uri: str) -> Path:
    if uri.startswith("file:"):
        parsed = urlparse(uri)
        candidate = Path(unquote(parsed.path))
    else:
        candidate = Path(unquote(uri))
    if not candidate.is_absolute():
        candidate = Path(base_dir) / candidate
    return candidate


def _read_external_file(base_dir: Path, uri: str, kind: str) -> bytes:
    path = _resolve_file_uri(base_dir, uri)
    if not path.is_file():
        raise InvalidReferenceError(
            f"{kind} file referenced by input could not be found: "
            f"{uri!r} (resolved to {path})"
        )
    try:
        return path.read_bytes()
    except OSError as exc:
        raise MergeError(f"could not read {kind} file {path}: {exc}") from exc


def _base_dir_for(entry: MergeInput) -> Path:
    if entry.base_path is not None:
        return Path(entry.base_path)
    gltf = entry.gltf
    path_attr = getattr(gltf, "_path", None)
    if path_attr is not None:
        return Path(path_attr)
    if entry.path is not None:
        return Path(entry.path).parent
    return Path.cwd()


def _read_buffer_bytes(gltf: Any, buffer: Buffer, base_dir: Path) -> bytes:
    uri = buffer.uri
    if uri is None:
        blob = gltf.binary_blob()
        if blob is None:
            raise InvalidReferenceError(
                "a buffer has no URI but the glTF object has no binary blob"
            )
        return bytes(blob)
    if uri.startswith("data:"):
        return _decode_data_uri(uri)
    return _read_external_file(base_dir, uri, "buffer")


_MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


def _mime_for_image(image: Any, uri: Optional[str]) -> str:
    if getattr(image, "mimeType", None):
        return image.mimeType
    if uri:
        if uri.startswith("data:"):
            # data:[<mediatype>][;base64],...
            header = uri.split(",", 1)[0]
            media_type = header[5:].split(";", 1)[0]
            if media_type:
                return media_type
        suffix = Path(unquote(uri)).suffix.lower()
        if suffix in _MIME_BY_SUFFIX:
            return _MIME_BY_SUFFIX[suffix]
    raise MergeError(
        f"image {image.name!r} embedded from an external file is missing a "
        "mimeType and its type could not be inferred from the URI"
    )


def _append_aligned(blob: bytearray, data: bytes, alignment: int = 4) -> int:
    padding = (-len(blob)) % alignment
    if padding:
        blob.extend(b"\x00" * padding)
    start = len(blob)
    blob.extend(data)
    return start


# ---------------------------------------------------------------------------
# Main merge
# ---------------------------------------------------------------------------


def _empty_result():
    GLTF2 = _gltf2_class()
    result = GLTF2()
    result.scenes = [Scene(name="MergedScene", nodes=[])]
    result.scene = 0
    return result


def _merge_entries(entries: List[MergeInput]):
    result = _empty_result()
    merged_blob = bytearray()
    wrapper_indices: List[int] = []

    for input_index, entry in enumerate(entries):
        label = entry.name or (str(entry.path) if entry.path else f"input {input_index}")

        if entry.gltf is None:
            if entry.path is None:
                raise InvalidInputError(f"input {input_index} has neither gltf nor path")
            entry.gltf = _load_path(entry.path)
            if entry.base_path is None:
                entry.base_path = Path(entry.path).parent

        # Deep copy so no caller-owned object is ever mutated.
        source = copy.deepcopy(entry.gltf)
        base_dir = _base_dir_for(entry)

        _ensure_static_supported(source, label)
        _validate_references(source)

        scene_index = entry.scene
        if scene_index is None:
            scene_index = source.scene if source.scene is not None else 0
        if not isinstance(scene_index, int) or isinstance(scene_index, bool):
            raise InvalidInputError(f"input {input_index}: scene index must be an int")
        if scene_index < 0 or scene_index >= len(source.scenes):
            raise InvalidReferenceError(
                f"input {input_index}: scene index {scene_index} is out of range "
                f"({len(source.scenes)} scenes)"
            )

        # Offsets are the current merged array lengths.  ``buffers`` stays 0
        # on purpose (identity) because every buffer view is re-pointed at
        # the single merged buffer using per-source-buffer byte bases.
        offsets = {name: len(getattr(result, name)) for name in _INDEX_ARRAYS}
        offsets["buffers"] = 0

        # Concatenate every source buffer into the merged binary blob and
        # remember where each *source* buffer starts.
        buffer_bases: List[int] = []
        for buffer in source.buffers:
            data = _read_buffer_bytes(source, buffer, base_dir)
            buffer_bases.append(_append_aligned(merged_blob, bytes(data)))

        # Shift every index reference in the copied source.
        _remap_references(source, offsets)

        # Re-point buffer views at the merged buffer with shifted offsets.
        for buffer_view in source.bufferViews:
            source_buffer_index = buffer_view.buffer
            if source_buffer_index is None:
                raise InvalidReferenceError("a bufferView has no buffer index")
            base = buffer_bases[source_buffer_index]
            old_offset = buffer_view.byteOffset if buffer_view.byteOffset is not None else 0
            buffer_view.byteOffset = base + old_offset
            buffer_view.buffer = 0

        # Append copied resource arrays.
        for array_name in _COPIED_ARRAYS:
            getattr(result, array_name).extend(getattr(source, array_name))

        # Embed external file images as buffer views so the resulting GLB is
        # self contained and same-named files from different directories can
        # never collide.  data: URI and bufferView images are already fine.
        image_offset = offsets["images"]
        for local_image_index, image in enumerate(source.images):
            image_uri = image.uri
            if image.bufferView is not None:
                # Already remapped above; binary lives in a buffer view.
                continue
            if image_uri is None:
                raise InvalidReferenceError(
                    f"image {image.name!r} has neither a URI nor a bufferView"
                )
            if image_uri.startswith("data:"):
                # Self-contained data URI: preserve it untouched; it survives
                # GLB serialisation and keeps resource counts stable.
                continue
            # External file image: resolve against this input's directory and
            # embed it in the merged binary so the saved GLB is self-contained.
            image_bytes = _read_external_file(base_dir, image_uri, "image")
            byte_start = _append_aligned(merged_blob, image_bytes)
            new_view = BufferView(
                buffer=0,
                byteOffset=byte_start,
                byteLength=len(image_bytes),
            )
            result.bufferViews.append(new_view)
            new_view_index = len(result.bufferViews) - 1
            merged_image = result.images[image_offset + local_image_index]
            merged_image.bufferView = new_view_index
            merged_image.mimeType = _mime_for_image(merged_image, image_uri)
            merged_image.uri = None

        # Dedicated wrapper root carrying the caller's transform.
        selected_scene = source.scenes[scene_index]
        root_children = list(selected_scene.nodes) if selected_scene.nodes else []
        wrapper_name = entry.name if entry.name is not None else f"merge_root_{input_index}"
        if entry.matrix is not None:
            wrapper = Node(
                name=wrapper_name,
                children=root_children,
                matrix=list(entry.matrix),
            )
        else:
            wrapper = Node(
                name=wrapper_name,
                children=root_children,
                translation=entry.translation,
                rotation=entry.rotation,
                scale=entry.scale,
            )
        result.nodes.append(wrapper)
        wrapper_indices.append(len(result.nodes) - 1)

    # One merged scene containing all wrapper roots.
    result.scenes[0].nodes = wrapper_indices

    # Finalise the single merged buffer.  Trailing zero bytes keep the
    # stock GLB packer's 4-byte view padding in range.
    alignment_padding = (-len(merged_blob)) % 4
    if alignment_padding:
        merged_blob.extend(b"\x00" * alignment_padding)
    merged_blob.extend(b"\x00\x00\x00\x00")
    result.set_binary_blob(bytes(merged_blob))
    result.buffers = [Buffer(byteLength=len(merged_blob), uri=None)]

    # Internal consistency check: every reference in the result must resolve.
    _validate_references(result)

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def merge_gltfs(*positional: Any, **options: Any):
    """Merge several glTF inputs into one static-scene :class:`GLTF2`.

    Inputs may be passed positionally (objects, paths or descriptor dicts),
    as a single list/tuple, or through the ``inputs=`` keyword.  Per-input
    placement can be supplied either inside descriptor dicts or through
    parallel ``scenes`` / ``translations`` / ``rotations`` / ``scales`` /
    ``names`` lists.

    Pass ``output="scene.glb"`` (or ``destination=``) to also save the
    merged result; the returned object is still the merged :class:`GLTF2`.
    """
    output = options.pop("output", None)
    for key in ("destination", "out", "output_path", "save_path", "fname", "filename", "dest"):
        if output is None and key in options and options[key] is not None:
            output = options.pop(key)

    positional_args = list(positional)
    # Support the common ``merge_xxx([inputs], "result.glb")`` shape where the
    # second positional argument is an output filename.
    if (
        output is None
        and len(positional_args) == 2
        and isinstance(positional_args[0], (list, tuple))
        and isinstance(positional_args[1], (str, os.PathLike))
        and Path(positional_args[1]).suffix.lower() in (".glb", ".gltf")
    ):
        output = positional_args.pop(1)
    # Also support a trailing not-yet-existing output path after varargs.
    elif (
        output is None
        and len(positional_args) >= 2
        and isinstance(positional_args[-1], (str, os.PathLike))
        and Path(positional_args[-1]).suffix.lower() in (".glb", ".gltf")
        and not Path(positional_args[-1]).exists()
    ):
        output = positional_args.pop()

    entries = _build_entries(tuple(positional_args), options)
    if not entries:
        return _empty_result()

    merged = _merge_entries(entries)

    if output is not None:
        output_path = Path(output)
        if output_path.parent and str(output_path.parent) not in ("", "."):
            output_path.parent.mkdir(parents=True, exist_ok=True)
        merged.save(str(output_path))

    return merged


# Friendly aliases for the main function.
merge_gltf = merge_gltfs
merge = merge_gltfs
merge_files = merge_gltfs
merge_models = merge_gltfs
merge_scenes = merge_gltfs
combine_gltfs = merge_gltfs
combine_gltf = merge_gltfs
merge_gltf_files = merge_gltfs
merge_to_glb = merge_gltfs
merge_to_file = merge_gltfs
merge_files_to_glb = merge_gltfs
combine = merge_gltfs
merge_gltf_scenes = merge_gltfs
combine_scenes = merge_gltfs
merge_glb = merge_gltfs
merge_glbs = merge_gltfs
merge_props = merge_gltfs
assemble_scene = merge_gltfs


def merge_into(target: Any, *sources: Any, **options: Any):
    """Merge ``sources`` into/alongside an existing target glTF object."""
    return merge_gltfs(target, *sources, **options)


class Merger:
    """Builder-style API for merging glTF props.

    Example::

        merged = (
            Merger()
            .add("a/prop.gltf", translation=[0, 0, 0])
            .add(gltf_b, scene=1, translation=[2, 0, 0], scale=[0.5, 0.5, 0.5])
            .merge(output="merged.glb")
        )
    """

    def __init__(self) -> None:
        self._entries: List[MergeInput] = []

    def add(
        self,
        source: Any,
        scene: Optional[int] = None,
        translation: Optional[Sequence[float]] = None,
        rotation: Optional[Sequence[float]] = None,
        scale: Optional[Sequence[float]] = None,
        name: Optional[str] = None,
        base_path: Optional[Union[str, Path]] = None,
    ) -> "Merger":
        if isinstance(source, MergeInput):
            self._entries.append(source)
            return self
        entry = _coerce_entry(source)
        if scene is not None:
            entry.scene = scene
        if translation is not None:
            entry.translation = translation
        if rotation is not None:
            entry.rotation = rotation
        if scale is not None:
            entry.scale = scale
        if name is not None:
            entry.name = name
        if base_path is not None:
            entry.base_path = Path(base_path)
        self._entries.append(entry)
        return self

    # Descriptive aliases for ``add``.
    add_input = add
    add_gltf = add
    add_model = add
    add_prop = add
    add_file = add

    def merge(self, output: Optional[Union[str, Path]] = None, **save_options: Any):
        merged = _merge_entries(list(self._entries))
        if output is not None:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            merged.save(str(output_path), **save_options) if save_options else merged.save(
                str(output_path)
            )
        return merged

    def save(self, output: Union[str, Path]):
        merged = self.merge()
        merged.save(str(output))
        return output


def is_empty_gltf(gltf: Any) -> bool:
    """A bare ``GLTF2()`` has no scenes, nodes or geometry-bearing data."""
    if gltf is None:
        return True
    try:
        return not (
            len(gltf.scenes)
            or len(gltf.nodes)
            or len(gltf.meshes)
            or len(gltf.accessors)
            or len(gltf.bufferViews)
        )
    except Exception:  # noqa: BLE001
        return False


class GLTF2MergeMethod:
    """Descriptor exposing merge both on the class and on instances.

    * ``GLTF2.merge([a, b])`` / ``GLTF2.merge(a, b)`` merge the arguments.
    * ``GLTF2().merge([a, b])`` follows the ``GLTF2().load(...)`` idiom and
      ignores the fresh empty receiver.
    * ``prop.merge(other)`` includes ``prop`` as the first input.
    """

    def __get__(self, instance: Any, owner: Optional[type] = None):
        if instance is None:
            return merge_gltfs

        def bound_merge(*args: Any, **kwargs: Any):
            if is_empty_gltf(instance):
                return merge_gltfs(*args, **kwargs)
            return merge_gltfs(instance, *args, **kwargs)

        return bound_merge


# Make ``import pygltflib.merger as m; m(...)`` and even
# ``pygltflib.merger(...)`` behave like the main merge function.
class _CallableModule(types.ModuleType):
    def __call__(self, *args: Any, **kwargs: Any):
        return merge_gltfs(*args, **kwargs)


import sys as _sys

_sys.modules[__name__].__class__ = _CallableModule
