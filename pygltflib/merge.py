"""Combine selected scenes from multiple glTF files into one static scene."""

import base64
import copy
import math
import mimetypes
import operator
from dataclasses import dataclass, fields, is_dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Optional, Sequence
from urllib.parse import unquote

from . import GLTF2
from .types import Asset, Buffer, BufferView, Node, Scene


class MergeError(ValueError):
    """Raised when input scenes cannot be merged safely."""


@dataclass(init=False)
class MergeScene:
    """One input to :func:`merge_scenes`.

    ``scene`` selects the scene by index. ``translation``, ``rotation`` and
    ``scale`` are applied to a new root node that wraps that scene. ``rotation``
    is a glTF quaternion in ``[x, y, z, w]`` order.
    """

    gltf: Any
    scene: Optional[int] = None
    translation: Optional[Sequence[float]] = None
    rotation: Optional[Sequence[float]] = None
    scale: Optional[Sequence[float]] = None
    name: Optional[str] = None

    def __init__(
        self,
        gltf,
        scene=None,
        translation=None,
        rotation=None,
        scale=None,
        name=None,
        scene_index=None,
        node_name=None,
    ):
        if scene is not None and scene_index is not None and scene != scene_index:
            raise ValueError("scene and scene_index disagree")
        if name is not None and node_name is not None and name != node_name:
            raise ValueError("name and node_name disagree")
        self.gltf = gltf
        self.scene = scene if scene is not None else scene_index
        self.translation = translation
        self.rotation = rotation
        self.scale = scale
        self.name = name if name is not None else node_name

    @property
    def scene_index(self):
        return self.scene

    @scene_index.setter
    def scene_index(self, value):
        self.scene = value

    @property
    def node_name(self):
        return self.name

    @node_name.setter
    def node_name(self, value):
        self.name = value


SceneMergeInput = MergeScene
MergeInput = MergeScene


def _fail(path, message):
    raise MergeError("{}: {}".format(path, message))


def _index(value, size, path):
    if isinstance(value, bool):
        _fail(path, "expected an integer index, got a boolean")
    try:
        result = operator.index(value)
    except TypeError:
        _fail(path, "expected an integer index, got {!r}".format(value))
    if result < 0 or result >= size:
        _fail(path, "index {} is out of range for {} entries".format(result, size))
    return result


def _optional_index(value, size, path):
    if value is None:
        return None
    return _index(value, size, path)


def _vector(value, length, path):
    if value is None:
        return None
    if isinstance(value, (str, bytes, bytearray)):
        _fail(path, "expected {} numeric values".format(length))
    try:
        result = list(value)
    except TypeError:
        _fail(path, "expected {} numeric values".format(length))
    if len(result) != length:
        _fail(path, "expected {} numeric values, got {}".format(length, len(result)))
    for component in result:
        if isinstance(component, bool) or not isinstance(component, Real):
            _fail(path, "components must be numbers")
        if not math.isfinite(component):
            _fail(path, "components must be finite numbers")
    return [float(component) for component in result]


def _source_path(source):
    path = getattr(source, "_path", None)
    return Path(path) if path is not None else Path()


def _decode_data_uri(uri, path):
    if not isinstance(uri, str) or "," not in uri:
        _fail(path, "invalid data URI")
    header, encoded = uri.split(",", 1)
    if ";base64" not in header:
        _fail(path, "only base64 data URIs are supported")
    try:
        return base64.b64decode(encoded)
    except Exception as exc:
        _fail(path, "unable to decode data URI: {}".format(exc))


def _read_buffer(source, buffer_index):
    path = "buffers[{}]".format(buffer_index)
    if buffer_index < 0 or buffer_index >= len(source.buffers):
        _fail(path, "invalid buffer reference")
    buffer = source.buffers[buffer_index]

    if buffer.uri is None:
        blob = source.binary_blob()
        if blob is None:
            _fail(path, "buffer has no URI and no GLB binary chunk")
        data = bytes(blob)
    elif isinstance(buffer.uri, str) and buffer.uri.startswith("data:"):
        data = _decode_data_uri(buffer.uri, path)
    elif isinstance(buffer.uri, str):
        uri = unquote(buffer.uri)
        if "://" in uri:
            _fail(path, "remote buffer URIs are not supported")
        filename = Path(uri)
        if not filename.is_absolute():
            filename = _source_path(source) / filename
        if not filename.is_file():
            _fail(path, "buffer file not found: {}".format(filename))
        data = filename.read_bytes()
    else:
        _fail(path, "buffer URI must be a string")

    if buffer.byteLength is not None and len(data) < buffer.byteLength:
        _fail(
            path,
            "declares byteLength {} but only {} bytes are available".format(
                buffer.byteLength, len(data)
            ),
        )
    return data


def _sniff_mime(data):
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    return None


def _read_image(source, image, image_path):
    if image.uri is None:
        _fail(image_path, "image has neither a URI nor a bufferView")
    if not isinstance(image.uri, str):
        _fail(image_path, "image URI must be a string")
    if image.uri.startswith("data:"):
        header = image.uri.split(",", 1)[0]
        mime_type = header[5:].split(";", 1)[0] or image.mimeType
        data = _decode_data_uri(image.uri, image_path)
    else:
        uri = unquote(image.uri)
        if "://" in uri:
            _fail(image_path, "remote image URIs are not supported")
        filename = Path(uri)
        if not filename.is_absolute():
            filename = _source_path(source) / filename
        if not filename.is_file():
            _fail(image_path, "image file not found: {}".format(filename))
        data = filename.read_bytes()
        mime_type = image.mimeType or mimetypes.guess_type(str(filename))[0]

    mime_type = mime_type or _sniff_mime(data)
    if not mime_type:
        _fail(image_path, "unable to determine image MIME type")
    return data, mime_type


def _check_extensions(value, path):
    if isinstance(value, list):
        for index, item in enumerate(value):
            _check_extensions(item, "{}[{}]".format(path, index))
        return
    if isinstance(value, tuple):
        for index, item in enumerate(value):
            _check_extensions(item, "{}[{}]".format(path, index))
        return
    if not is_dataclass(value):
        return

    for item in fields(value):
        if item.name in ("extensions", "extras"):
            continue
        _check_extensions(getattr(value, item.name), "{}.{}".format(path, item.name))

    extensions = getattr(value, "extensions", None)
    if extensions:
        _fail(path, "extensions are not supported")


def _check_texture_info(info, texture_count, path):
    if info is None:
        return
    if info.index is None:
        _fail(path, "texture index is required")
    _index(info.index, texture_count, "{}.index".format(path))


def _validate_references(source):
    scene_count = len(source.scenes)
    node_count = len(source.nodes)
    mesh_count = len(source.meshes)
    material_count = len(source.materials)
    texture_count = len(source.textures)
    image_count = len(source.images)
    sampler_count = len(source.samplers)
    accessor_count = len(source.accessors)
    buffer_view_count = len(source.bufferViews)
    buffer_count = len(source.buffers)

    if source.scene is not None:
        _index(source.scene, scene_count, "scene")

    for scene_index, scene in enumerate(source.scenes):
        path = "scenes[{}]".format(scene_index)
        for node_index, value in enumerate(scene.nodes or []):
            _index(value, node_count, "{}.nodes[{}]".format(path, node_index))

    for node_index, node in enumerate(source.nodes):
        path = "nodes[{}]".format(node_index)
        _optional_index(node.mesh, mesh_count, "{}.mesh".format(path))
        _optional_index(node.skin, len(source.skins), "{}.skin".format(path))
        _optional_index(node.camera, len(source.cameras), "{}.camera".format(path))
        for child_index, value in enumerate(node.children or []):
            _index(value, node_count, "{}.children[{}]".format(path, child_index))

    for buffer_index, buffer in enumerate(source.buffers):
        if buffer.byteLength is None:
            _fail("buffers[{}]".format(buffer_index), "byteLength is required")
        if buffer.byteLength < 0:
            _fail("buffers[{}]".format(buffer_index), "byteLength must be non-negative")

    for buffer_view_index, buffer_view in enumerate(source.bufferViews):
        path = "bufferViews[{}]".format(buffer_view_index)
        _index(buffer_view.buffer, buffer_count, "{}.buffer".format(path))
        if buffer_view.byteLength is None or buffer_view.byteLength < 0:
            _fail(path, "byteLength must be a non-negative integer")
        if buffer_view.byteOffset is not None and buffer_view.byteOffset < 0:
            _fail(path, "byteOffset must be non-negative")

    for accessor_index, accessor in enumerate(source.accessors):
        path = "accessors[{}]".format(accessor_index)
        if accessor.bufferView is not None:
            _index(accessor.bufferView, buffer_view_count, "{}.bufferView".format(path))
        if accessor.sparse is not None:
            sparse = accessor.sparse
            if sparse.indices is None or sparse.values is None:
                _fail(path, "sparse accessors require indices and values")
            _index(
                sparse.indices.bufferView,
                buffer_view_count,
                "{}.sparse.indices.bufferView".format(path),
            )
            _index(
                sparse.values.bufferView,
                buffer_view_count,
                "{}.sparse.values.bufferView".format(path),
            )

    for mesh_index, mesh in enumerate(source.meshes):
        path = "meshes[{}]".format(mesh_index)
        for primitive_index, primitive in enumerate(mesh.primitives):
            primitive_path = "{}.primitives[{}]".format(path, primitive_index)
            if primitive.attributes is None:
                _fail(primitive_path, "attributes are required")
            for attribute_name, accessor_index in vars(primitive.attributes).items():
                if accessor_index is not None:
                    _index(
                        accessor_index,
                        accessor_count,
                        "{}.attributes.{}".format(primitive_path, attribute_name),
                    )
            _optional_index(
                primitive.indices,
                accessor_count,
                "{}.indices".format(primitive_path),
            )
            _optional_index(
                primitive.material,
                material_count,
                "{}.material".format(primitive_path),
            )

    for material_index, material in enumerate(source.materials):
        path = "materials[{}]".format(material_index)
        if material.pbrMetallicRoughness is not None:
            pbr = material.pbrMetallicRoughness
            _check_texture_info(
                pbr.baseColorTexture,
                texture_count,
                "{}.pbrMetallicRoughness.baseColorTexture".format(path),
            )
            _check_texture_info(
                pbr.metallicRoughnessTexture,
                texture_count,
                "{}.pbrMetallicRoughness.metallicRoughnessTexture".format(path),
            )
        _check_texture_info(
            material.normalTexture,
            texture_count,
            "{}.normalTexture".format(path),
        )
        _check_texture_info(
            material.occlusionTexture,
            texture_count,
            "{}.occlusionTexture".format(path),
        )
        _check_texture_info(
            material.emissiveTexture,
            texture_count,
            "{}.emissiveTexture".format(path),
        )

    for texture_index, texture in enumerate(source.textures):
        path = "textures[{}]".format(texture_index)
        _optional_index(texture.source, image_count, "{}.source".format(path))
        _optional_index(texture.sampler, sampler_count, "{}.sampler".format(path))

    for image_index, image in enumerate(source.images):
        path = "images[{}]".format(image_index)
        if image.bufferView is not None:
            _index(image.bufferView, buffer_view_count, "{}.bufferView".format(path))
        if image.uri is not None and image.bufferView is not None:
            _fail(path, "image cannot define both URI and bufferView")
        if image.uri is None and image.bufferView is None:
            _fail(path, "image must define a URI or bufferView")


def _validate_source(source, input_index):
    path = "inputs[{}]".format(input_index)
    if not isinstance(source, GLTF2):
        _fail(path, "gltf must be a GLTF2 object")
    if source.extensionsUsed or source.extensionsRequired:
        _fail(path, "extensions are not supported")
    _check_extensions(source, path)
    if source.animations:
        _fail(path, "animations are not supported")
    if source.skins:
        _fail(path, "skins are not supported")
    for node_index, node in enumerate(source.nodes):
        if node.skin is not None:
            _fail(path, "node {} uses a skin".format(node_index))
        if node.camera is not None:
            _fail(path, "node {} uses a camera".format(node_index))
        if node.weights:
            _fail(path, "node {} uses morph target weights".format(node_index))
    if source.cameras:
        _fail(path, "cameras are not supported")
    for mesh_index, mesh in enumerate(source.meshes):
        if mesh.weights:
            _fail(path, "mesh {} uses morph target weights".format(mesh_index))
        for primitive_index, primitive in enumerate(mesh.primitives):
            if primitive.targets:
                _fail(
                    path,
                    "mesh {} primitive {} uses morph targets".format(
                        mesh_index, primitive_index
                    ),
                )
    _validate_references(source)


def _resolve_scene(source, spec, input_index):
    path = "inputs[{}]".format(input_index)
    if not source.scenes:
        _fail(path, "contains no scenes")

    scene_index = spec.scene
    if scene_index is None:
        scene_index = source.scene
    if scene_index is None:
        if len(source.scenes) == 1:
            scene_index = 0
        else:
            _fail(path, "multiple scenes are present; select one explicitly")
    return _index(scene_index, len(source.scenes), "{}.scene".format(path))


def _mapping_from_dict(value):
    if not isinstance(value, dict):
        return None
    gltf = value.get("gltf", value.get("model", value.get("source")))
    if gltf is None:
        raise MergeError("input mapping requires a 'gltf' entry")
    transform = value.get("transform") or {}
    if not isinstance(transform, dict):
        raise MergeError("'transform' must be a mapping")
    return MergeScene(
        gltf=gltf,
        scene=value.get("scene", value.get("scene_index", transform.get("scene"))),
        translation=value.get("translation", transform.get("translation")),
        rotation=value.get("rotation", transform.get("rotation")),
        scale=value.get("scale", transform.get("scale")),
        name=value.get("name", value.get("node_name", transform.get("name"))),
    )


def _spec_from_entry(entry):
    if isinstance(entry, MergeScene):
        return MergeScene(
            gltf=entry.gltf,
            scene=entry.scene,
            translation=entry.translation,
            rotation=entry.rotation,
            scale=entry.scale,
            name=entry.name,
        )
    mapping = _mapping_from_dict(entry)
    if mapping is not None:
        return mapping
    if isinstance(entry, (str, Path)):
        return MergeScene(gltf=entry)
    if isinstance(entry, (tuple, list)):
        if not entry:
            raise MergeError("empty input tuple")
        gltf = entry[0]
        if len(entry) == 2 and isinstance(entry[1], dict):
            values = dict(entry[1])
            values["gltf"] = gltf
            return _mapping_from_dict(values)
        if len(entry) == 3 and isinstance(entry[2], dict):
            values = dict(entry[2])
            values["gltf"] = gltf
            values.setdefault("scene", entry[1])
            return _mapping_from_dict(values)
        if len(entry) > 6:
            raise MergeError("input tuples accept at most six values")
        values = list(entry[1:]) + [None] * (5 - len(entry[1:]))
        return MergeScene(
            gltf=gltf,
            scene=values[0],
            translation=values[1],
            rotation=values[2],
            scale=values[3],
            name=values[4],
        )
    return MergeScene(gltf=entry)


def _parallel_value(values, index, path, count):
    if values is None:
        return None
    try:
        length = len(values)
        if count == 1 and length != 1:
            return values
        if length != count:
            raise MergeError(
                "{} must contain one value per input ({} != {})".format(
                    path, length, count
                )
            )
        return values[index]
    except TypeError:
        if count == 1:
            return values
        raise MergeError("{} must be a sequence".format(path))


def _transform_values(transforms, count):
    if transforms is None:
        return [None] * count
    if isinstance(transforms, (str, bytes, bytearray)):
        raise MergeError("transforms must be a sequence")
    try:
        values = list(transforms)
    except TypeError:
        raise MergeError("transforms must be a sequence")
    if len(values) != count:
        raise MergeError(
            "transforms must contain one value per input ({} != {})".format(
                len(values), count
            )
        )

    result = []
    for index, value in enumerate(values):
        path = "transforms[{}]".format(index)
        if value is None:
            result.append({})
        elif isinstance(value, dict):
            result.append(value)
        elif isinstance(value, (tuple, list)) and len(value) in (3, 4):
            result.append(
                {
                    "translation": value[0],
                    "rotation": value[1],
                    "scale": value[2],
                    "name": value[3] if len(value) == 4 else None,
                }
            )
        else:
            _fail(path, "expected a mapping or (translation, rotation, scale)")
    return result


def _looks_like_transforms(values):
    if values is None or isinstance(values, (str, bytes, bytearray)):
        return False
    try:
        values = list(values)
    except TypeError:
        return False
    for value in values:
        if value is None:
            continue
        if isinstance(value, dict):
            continue
        if isinstance(value, (tuple, list)) and len(value) in (3, 4):
            if all(
                not isinstance(item, (str, bytes, bytearray))
                and hasattr(item, "__iter__")
                for item in value[:3]
            ):
                continue
        return False
    return bool(values)


def _normalize_inputs(
    inputs,
    scene_indices,
    translations,
    rotations,
    scales,
    names,
    transforms,
):
    if isinstance(inputs, (GLTF2, str, Path, MergeScene)):
        inputs = [inputs]
    elif isinstance(inputs, (tuple, list)) and len(inputs) > 1:
        first = inputs[0]
        if isinstance(first, (GLTF2, str, Path, MergeScene)):
            second = inputs[1]
            second_is_entry = isinstance(second, (GLTF2, str, Path, MergeScene))
            if isinstance(second, dict):
                second_is_entry = any(
                    key in second for key in ("gltf", "model", "source")
                )
            if not second_is_entry:
                inputs = [inputs]
    try:
        entries = list(inputs)
    except TypeError:
        raise MergeError("inputs must be a sequence")
    if not entries:
        raise MergeError("at least one input is required")

    transform_values = _transform_values(transforms, len(entries))
    specs = []
    for index, entry in enumerate(entries):
        spec = _spec_from_entry(entry)
        if isinstance(spec.gltf, (str, Path)):
            spec.gltf = GLTF2.load(spec.gltf)

        transform = transform_values[index] or {}
        scene_index = _parallel_value(
            scene_indices, index, "scene_indices", len(entries)
        )
        translation = _parallel_value(
            translations, index, "translations", len(entries)
        )
        rotation = _parallel_value(rotations, index, "rotations", len(entries))
        scale = _parallel_value(scales, index, "scales", len(entries))
        name = _parallel_value(names, index, "names", len(entries))

        if scene_index is None:
            scene_index = transform.get("scene", transform.get("scene_index"))
        if translation is None:
            translation = transform.get("translation")
        if rotation is None:
            rotation = transform.get("rotation")
        if scale is None:
            scale = transform.get("scale")
        if name is None:
            name = transform.get("name", transform.get("node_name"))

        if scene_index is not None:
            spec.scene = scene_index
        if translation is not None:
            spec.translation = translation
        if rotation is not None:
            spec.rotation = rotation
        if scale is not None:
            spec.scale = scale
        if name is not None:
            spec.name = name

        spec.translation = _vector(
            spec.translation, 3, "inputs[{}].translation".format(index)
        )
        spec.rotation = _vector(
            spec.rotation, 4, "inputs[{}].rotation".format(index)
        )
        spec.scale = _vector(spec.scale, 3, "inputs[{}].scale".format(index))
        if spec.name is not None and not isinstance(spec.name, str):
            _fail("inputs[{}].name".format(index), "name must be a string")
        specs.append(spec)
    return specs


def _remap_references(source, merged, base):
    accessor_base = base["accessors"]
    buffer_view_base = base["bufferViews"]
    mesh_base = base["meshes"]
    material_base = base["materials"]
    texture_base = base["textures"]
    image_base = base["images"]
    sampler_base = base["samplers"]
    node_base = base["nodes"]

    for index, accessor in enumerate(source.accessors):
        result = merged.accessors[accessor_base + index]
        if accessor.bufferView is not None:
            result.bufferView = buffer_view_base + accessor.bufferView
        if accessor.sparse is not None:
            result.sparse.indices.bufferView = (
                buffer_view_base + accessor.sparse.indices.bufferView
            )
            result.sparse.values.bufferView = (
                buffer_view_base + accessor.sparse.values.bufferView
            )

    for index, buffer_view in enumerate(source.bufferViews):
        result = merged.bufferViews[buffer_view_base + index]
        result.buffer = 0

    for index, node in enumerate(source.nodes):
        result = merged.nodes[node_base + index]
        if node.mesh is not None:
            result.mesh = mesh_base + node.mesh
        if node.children:
            result.children = [node_base + child for child in node.children]
        if node.skin is not None:
            result.skin = None
        if node.camera is not None:
            result.camera = None

    for index, mesh in enumerate(source.meshes):
        result = merged.meshes[mesh_base + index]
        for primitive_index, primitive in enumerate(mesh.primitives):
            result_primitive = result.primitives[primitive_index]
            for attribute_name, accessor_index in vars(primitive.attributes).items():
                if accessor_index is not None:
                    setattr(
                        result_primitive.attributes,
                        attribute_name,
                        accessor_base + accessor_index,
                    )
            if primitive.indices is not None:
                result_primitive.indices = accessor_base + primitive.indices
            if primitive.material is not None:
                result_primitive.material = material_base + primitive.material

    for index, material in enumerate(source.materials):
        result = merged.materials[material_base + index]
        texture_infos = []
        if material.pbrMetallicRoughness is not None:
            pbr = material.pbrMetallicRoughness
            result_pbr = result.pbrMetallicRoughness
            texture_infos.extend(
                [
                    (pbr.baseColorTexture, result_pbr.baseColorTexture),
                    (
                        pbr.metallicRoughnessTexture,
                        result_pbr.metallicRoughnessTexture,
                    ),
                ]
            )
        texture_infos.extend(
            [
                (material.normalTexture, result.normalTexture),
                (material.occlusionTexture, result.occlusionTexture),
                (material.emissiveTexture, result.emissiveTexture),
            ]
        )
        for source_info, result_info in texture_infos:
            if source_info is not None:
                result_info.index = texture_base + source_info.index

    for index, texture in enumerate(source.textures):
        result = merged.textures[texture_base + index]
        if texture.source is not None:
            result.source = image_base + texture.source
        if texture.sampler is not None:
            result.sampler = sampler_base + texture.sampler

    for index, image in enumerate(source.images):
        result = merged.images[image_base + index]
        if image.bufferView is not None:
            result.bufferView = buffer_view_base + image.bufferView


def _append_aligned(blob, data):
    offset = len(blob)
    blob.extend(data)
    padding = (4 - len(blob) % 4) % 4
    if padding:
        blob.extend(b"\x00" * padding)
    return offset


def _copy_source(source, merged, base, blob):
    buffer_data = [_read_buffer(source, index) for index in range(len(source.buffers))]

    for index, buffer_view in enumerate(source.bufferViews):
        result = merged.bufferViews[base["bufferViews"] + index]
        data = buffer_data[buffer_view.buffer]
        offset = buffer_view.byteOffset or 0
        end = offset + buffer_view.byteLength
        if end > len(data):
            _fail(
                "bufferViews[{}]".format(index),
                "byte range {}..{} exceeds buffer size {}".format(
                    offset, end, len(data)
                ),
            )
        new_offset = _append_aligned(blob, data[offset:end])
        result.buffer = 0
        result.byteOffset = new_offset

    for index, image in enumerate(source.images):
        result = merged.images[base["images"] + index]
        if image.uri is not None:
            data, mime_type = _read_image(
                source, image, "images[{}]".format(index)
            )
            new_offset = _append_aligned(blob, data)
            merged.bufferViews.append(
                BufferView(
                    buffer=0,
                    byteOffset=new_offset,
                    byteLength=len(data),
                )
            )
            result.uri = None
            result.bufferView = len(merged.bufferViews) - 1
            result.mimeType = mime_type
        else:
            result.uri = None
            if not result.mimeType:
                buffer_view = merged.bufferViews[
                    base["bufferViews"] + image.bufferView
                ]
                start = buffer_view.byteOffset or 0
                end = start + buffer_view.byteLength
                result.mimeType = _sniff_mime(bytes(blob[start:end]))
            if not result.mimeType:
                _fail(
                    "images[{}]".format(index),
                    "MIME type is required for images stored in a bufferView",
                )


def merge_scenes(
    inputs,
    scene_indices=None,
    translations=None,
    rotations=None,
    scales=None,
    names=None,
    transforms=None,
    scene_name=None,
    name=None,
    **kwargs
):
    """Merge selected scenes from several glTF objects into a new static GLB.

    Each input may be a :class:`MergeScene`, a ``GLTF2`` object, a file path, a
    mapping, or a tuple ``(gltf, scene, translation, rotation, scale, name)``.
    The optional parallel sequences offer an alternative way to specify scene
    selection and root transforms.

    The result contains one scene with one new root node per input. The chosen
    source scene's roots become children of that node. Buffers are flattened
    into the GLB binary chunk and URI images are embedded. Source objects are
    never modified.
    """
    if "scenes" in kwargs and scene_indices is None:
        scene_indices = kwargs.pop("scenes")
    if "transforms" in kwargs:
        if transforms is not None:
            raise TypeError("transforms was provided twice")
        transforms = kwargs.pop("transforms")
    if kwargs:
        raise TypeError("unexpected keyword arguments: {}".format(", ".join(kwargs)))
    if transforms is None and _looks_like_transforms(translations):
        transforms = translations
        translations = None
    if scene_name is not None and name is not None:
        raise TypeError("use either scene_name or name, not both")
    output_scene_name = scene_name if scene_name is not None else name
    if output_scene_name is not None and not isinstance(output_scene_name, str):
        raise TypeError("scene_name must be a string")

    specs = _normalize_inputs(
        inputs,
        scene_indices,
        translations,
        rotations,
        scales,
        names,
        transforms,
    )
    selected_scenes = []
    for index, spec in enumerate(specs):
        _validate_source(spec.gltf, index)
        selected_scenes.append(_resolve_scene(spec.gltf, spec, index))

    merged = GLTF2()
    merged.asset = Asset(version="2.0")
    merged.scene = 0
    if output_scene_name is None:
        output_scene_name = "Merged Scene"
    merged.scenes = [Scene(name=output_scene_name, nodes=[])]

    bases = []
    for spec in specs:
        source = spec.gltf
        base = {
            "accessors": len(merged.accessors),
            "bufferViews": len(merged.bufferViews),
            "images": len(merged.images),
            "materials": len(merged.materials),
            "meshes": len(merged.meshes),
            "nodes": len(merged.nodes),
            "samplers": len(merged.samplers),
            "textures": len(merged.textures),
        }
        bases.append(base)
        merged.accessors.extend(copy.deepcopy(source.accessors))
        merged.bufferViews.extend(copy.deepcopy(source.bufferViews))
        merged.images.extend(copy.deepcopy(source.images))
        merged.materials.extend(copy.deepcopy(source.materials))
        merged.meshes.extend(copy.deepcopy(source.meshes))
        merged.nodes.extend(copy.deepcopy(source.nodes))
        merged.samplers.extend(copy.deepcopy(source.samplers))
        merged.textures.extend(copy.deepcopy(source.textures))

    blob = bytearray()
    for spec, base in zip(specs, bases):
        _copy_source(spec.gltf, merged, base, blob)
    for spec, base in zip(specs, bases):
        _remap_references(spec.gltf, merged, base)

    for spec, scene_index, base in zip(specs, selected_scenes, bases):
        source_scene = spec.gltf.scenes[scene_index]
        root = Node(
            children=[base["nodes"] + node for node in (source_scene.nodes or [])],
            translation=spec.translation,
            rotation=spec.rotation,
            scale=spec.scale,
            name=spec.name if spec.name is not None else source_scene.name,
        )
        merged.nodes.append(root)
        merged.scenes[0].nodes.append(len(merged.nodes) - 1)

    merged.buffers = [Buffer(uri=None, byteLength=len(blob))]
    merged.set_binary_blob(bytes(blob))
    return merged


merge_gltf_scenes = merge_scenes
