"""
Animation sampling helpers for pygltflib.

The functions in this module evaluate glTF node animations at a point in time.
They only compute local node transforms; they do not apply skinning or morph
targets.
"""

import base64
import math
import struct
from bisect import bisect_right
from pathlib import Path
from urllib.parse import unquote, unquote_to_bytes

from .types import (
    ANIM_CUBICSPLINE,
    ANIM_LINEAR,
    ANIM_STEP,
    BYTE,
    FLOAT,
    MAT2,
    MAT3,
    MAT4,
    ROTATION,
    SCALAR,
    SCALE,
    SHORT,
    TRANSLATION,
    UNSIGNED_BYTE,
    UNSIGNED_INT,
    UNSIGNED_SHORT,
    VEC2,
    VEC3,
    VEC4,
    WEIGHTS,
)


class AnimationSamplingError(ValueError):
    """Base error raised when an animation cannot be sampled."""


class InvalidAnimationError(AnimationSamplingError):
    """The animation definition is structurally invalid."""


class InvalidKeyframeError(AnimationSamplingError):
    """Animation keyframe data is missing, malformed, or not finite."""


class UnsupportedAnimationChannelError(AnimationSamplingError):
    """The animation uses a target path or extension that is not supported."""


class DuplicateAnimationChannelError(AnimationSamplingError):
    """Two channels target the same node property in one animation."""


class AnimationDataError(AnimationSamplingError):
    """Buffer data referenced by an animation cannot be read."""


class NodeTransform(dict):
    """
    Sampled local transform for one node.

    The object is a regular mapping with ``translation``, ``rotation`` and
    ``scale`` keys, and also exposes the values as attributes for convenience.
    ``rotation`` uses glTF's ``[x, y, z, w]`` quaternion order.
    """

    def __init__(self, translation, rotation, scale, node_index=None, node=None):
        super().__init__(
            translation=list(translation),
            rotation=list(rotation),
            scale=list(scale),
        )
        self.node_index = node_index
        self.node = node

    @property
    def translation(self):
        return self["translation"]

    @translation.setter
    def translation(self, value):
        self["translation"] = list(value)

    @property
    def rotation(self):
        return self["rotation"]

    @rotation.setter
    def rotation(self, value):
        self["rotation"] = list(value)

    @property
    def scale(self):
        return self["scale"]

    @scale.setter
    def scale(self, value):
        self["scale"] = list(value)


class AnimationSample(dict):
    """
    Mapping of target node index to :class:`NodeTransform`.

    ``transforms`` is an alias for the mapping itself.  The animation index and
    sample time are available as attributes.
    """

    def __init__(self, *args, animation_index=None, time=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.animation_index = animation_index
        self.time = time

    @property
    def transforms(self):
        return self

    @property
    def node_transforms(self):
        return self


_COMPONENT_FORMATS = {
    BYTE: ("b", 1),
    UNSIGNED_BYTE: ("B", 1),
    SHORT: ("h", 2),
    UNSIGNED_SHORT: ("H", 2),
    UNSIGNED_INT: ("I", 4),
    FLOAT: ("f", 4),
}

_TYPE_COMPONENT_COUNTS = {
    SCALAR: 1,
    VEC2: 2,
    VEC3: 3,
    VEC4: 4,
    MAT2: 4,
    MAT3: 9,
    MAT4: 16,
}

_NORMALIZE_FACTORS = {
    BYTE: 127.0,
    UNSIGNED_BYTE: 255.0,
    SHORT: 32767.0,
    UNSIGNED_SHORT: 65535.0,
    UNSIGNED_INT: 4294967295.0,
}

_SUPPORTED_INTERPOLATIONS = {ANIM_STEP, ANIM_LINEAR, ANIM_CUBICSPLINE}
_ANIMATED_PATHS = {TRANSLATION, ROTATION, SCALE}


def _component_info(component_type):
    try:
        return _COMPONENT_FORMATS[component_type]
    except KeyError:
        raise InvalidKeyframeError(
            f"Unsupported accessor componentType {component_type!r}"
        )


def _type_component_count(accessor_type):
    try:
        return _TYPE_COMPONENT_COUNTS[accessor_type]
    except KeyError:
        raise InvalidKeyframeError(
            f"Unsupported accessor type {accessor_type!r}"
        )


def _decode_data_uri(uri):
    header, separator, payload = uri.partition(",")
    if not separator:
        raise AnimationDataError("Malformed data URI in animation buffer")
    raw = unquote_to_bytes(payload)
    if ";base64" in header.lower():
        try:
            return base64.b64decode(raw)
        except Exception as exc:
            raise AnimationDataError(
                f"Unable to decode animation buffer data URI: {exc}"
            ) from exc
    return raw


def _load_buffer_bytes(gltf, buffer_index):
    if (
        not isinstance(buffer_index, int)
        or isinstance(buffer_index, bool)
        or buffer_index < 0
    ):
        raise InvalidKeyframeError("bufferView is missing a buffer index")
    try:
        buffer = gltf.buffers[buffer_index]
    except (IndexError, TypeError):
        raise InvalidKeyframeError(
            f"bufferView references missing buffer {buffer_index}"
        )

    uri = getattr(buffer, "uri", None)
    if uri is None:
        data = gltf.binary_blob()
        if data is None:
            raise AnimationDataError(
                "Animation references a GLB buffer but no binary blob is loaded"
            )
        return data

    if isinstance(uri, str) and uri.startswith("data:"):
        return _decode_data_uri(uri)

    uri = unquote(str(uri))
    candidates = []
    gltf_path = getattr(gltf, "_path", None)
    if gltf_path:
        candidates.append(Path(gltf_path) / uri)
    candidates.append(Path(uri))

    for candidate in candidates:
        if candidate.is_file():
            try:
                return candidate.read_bytes()
            except OSError as exc:
                raise AnimationDataError(
                    f"Unable to read animation buffer {candidate}: {exc}"
                ) from exc

    raise AnimationDataError(
        f"Unable to locate animation buffer file {uri!r}"
    )


def _normalized_value(value, component_type):
    if component_type == FLOAT:
        return float(value)
    factor = _NORMALIZE_FACTORS[component_type]
    if value < 0:
        return max(float(value) / factor, -1.0)
    return float(value) / factor


def _unpack_elements(
    data,
    byte_offset,
    count,
    component_type,
    component_count,
    byte_stride,
    normalized,
):
    fmt, component_size = _component_info(component_type)
    element_size = component_size * component_count
    if byte_stride is None:
        byte_stride = element_size
    if byte_stride < element_size:
        raise InvalidKeyframeError(
            f"bufferView byteStride {byte_stride} is smaller than element size "
            f"{element_size}"
        )

    if count < 0:
        raise InvalidKeyframeError("Accessor count must not be negative")

    values = []
    unpack = struct.Struct("<" + fmt * component_count).unpack_from
    for index in range(count):
        offset = byte_offset + index * byte_stride
        try:
            components = unpack(data, offset)
        except struct.error as exc:
            raise AnimationDataError(
                "Animation buffer is too short for accessor data"
            ) from exc
        if normalized:
            components = tuple(
                _normalized_value(value, component_type) for value in components
            )
        else:
            components = tuple(float(value) for value in components)
        values.append(list(components))
    return values


def _read_bufferview_elements(
    gltf,
    buffer_view_index,
    byte_offset,
    count,
    component_type,
    component_count,
    normalized,
    buffer_cache,
):
    if (
        not isinstance(buffer_view_index, int)
        or isinstance(buffer_view_index, bool)
        or buffer_view_index < 0
    ):
        raise InvalidKeyframeError(
            f"Accessor references invalid bufferView {buffer_view_index!r}"
        )
    try:
        buffer_view = gltf.bufferViews[buffer_view_index]
    except (IndexError, TypeError):
        raise InvalidKeyframeError(
            f"Accessor references missing bufferView {buffer_view_index}"
        )

    buffer_index = getattr(buffer_view, "buffer", None)
    if buffer_index not in buffer_cache:
        buffer_cache[buffer_index] = _load_buffer_bytes(gltf, buffer_index)
    data = buffer_cache[buffer_index]

    buffer_view_offset = getattr(buffer_view, "byteOffset", 0) or 0
    byte_offset = (byte_offset or 0) + buffer_view_offset
    byte_stride = getattr(buffer_view, "byteStride", None)

    fmt, component_size = _component_info(component_type)
    element_size = component_size * component_count
    effective_stride = byte_stride if byte_stride is not None else element_size
    required_end = byte_offset
    if count:
        required_end = byte_offset + (count - 1) * effective_stride + element_size

    buffer_length = getattr(buffer_view, "byteLength", None)
    if buffer_length is not None:
        view_end = buffer_view_offset + buffer_length
        if required_end > view_end:
            raise AnimationDataError(
                "Accessor data extends past the end of its bufferView"
            )
    if required_end > len(data):
        raise AnimationDataError(
            "Accessor data extends past the end of the animation buffer"
        )

    return _unpack_elements(
        data,
        byte_offset,
        count,
        component_type,
        component_count,
        byte_stride,
        normalized,
    )


def _read_accessor(gltf, accessor_index, buffer_cache):
    if (
        not isinstance(accessor_index, int)
        or isinstance(accessor_index, bool)
        or accessor_index < 0
    ):
        raise InvalidKeyframeError(
            f"Animation references invalid accessor {accessor_index!r}"
        )
    try:
        accessor = gltf.accessors[accessor_index]
    except (IndexError, TypeError):
        raise InvalidKeyframeError(
            f"Animation references missing accessor {accessor_index}"
        )

    accessor_type = getattr(accessor, "type", None)
    component_type = getattr(accessor, "componentType", None)
    count = getattr(accessor, "count", None)
    if count is None:
        raise InvalidKeyframeError("Animation accessor is missing count")

    component_count = _type_component_count(accessor_type)
    _component_info(component_type)
    accessor_offset = getattr(accessor, "byteOffset", 0) or 0
    normalized = bool(getattr(accessor, "normalized", False))

    buffer_view_index = getattr(accessor, "bufferView", None)
    if buffer_view_index is None:
        if getattr(accessor, "sparse", None):
            values = [[0.0] * component_count for _ in range(count)]
        else:
            raise InvalidKeyframeError(
                "Animation accessor has neither a bufferView nor sparse data"
            )
    else:
        values = _read_bufferview_elements(
            gltf,
            buffer_view_index,
            accessor_offset,
            count,
            component_type,
            component_count,
            normalized,
            buffer_cache,
        )

    sparse = getattr(accessor, "sparse", None)
    if sparse:
        sparse_count = getattr(sparse, "count", None)
        if sparse_count is None:
            raise InvalidKeyframeError("Sparse accessor is missing count")
        indices = getattr(sparse, "indices", None)
        sparse_values = getattr(sparse, "values", None)
        if indices is None or sparse_values is None:
            raise InvalidKeyframeError(
                "Sparse accessor is missing indices or values"
            )

        index_component_type = getattr(indices, "componentType", None)
        index_values = _read_bufferview_elements(
            gltf,
            getattr(indices, "bufferView", None),
            getattr(indices, "byteOffset", 0) or 0,
            sparse_count,
            index_component_type,
            1,
            False,
            buffer_cache,
        )
        replacement_values = _read_bufferview_elements(
            gltf,
            getattr(sparse_values, "bufferView", None),
            getattr(sparse_values, "byteOffset", 0) or 0,
            sparse_count,
            component_type,
            component_count,
            normalized,
            buffer_cache,
        )
        for index_row, replacement in zip(index_values, replacement_values):
            target_index = int(index_row[0])
            if target_index < 0 or target_index >= count:
                raise InvalidKeyframeError(
                    f"Sparse accessor index {target_index} is out of range"
                )
            values[target_index] = replacement

    return values


def _require_float_accessor(gltf, accessor_index, role):
    if (
        not isinstance(accessor_index, int)
        or isinstance(accessor_index, bool)
        or accessor_index < 0
    ):
        raise InvalidKeyframeError(
            f"{role} references invalid accessor {accessor_index!r}"
        )
    try:
        accessor = gltf.accessors[accessor_index]
    except (IndexError, TypeError):
        raise InvalidKeyframeError(
            f"{role} references missing accessor {accessor_index}"
        )
    if getattr(accessor, "componentType", None) != FLOAT:
        raise InvalidKeyframeError(
            f"{role} accessor must use FLOAT componentType"
        )
    return accessor


def _coerce_animation_index(gltf, animation_index):
    if not isinstance(animation_index, int) or isinstance(animation_index, bool):
        raise InvalidAnimationError(
            f"animation_index must be an integer, got {animation_index!r}"
        )
    if animation_index < 0 or animation_index >= len(gltf.animations):
        raise InvalidAnimationError(
            f"animation_index {animation_index} is out of range"
        )
    return gltf.animations[animation_index]


def _coerce_time(time):
    try:
        value = float(time)
    except (TypeError, ValueError):
        raise AnimationSamplingError(
            f"time must be a number of seconds, got {time!r}"
        )
    if not math.isfinite(value):
        raise AnimationSamplingError("time must be finite")
    return value


def _validate_times(times, role):
    if not times:
        raise InvalidKeyframeError(f"{role} has no keyframes")
    previous = None
    for index, row in enumerate(times):
        value = row[0]
        if not math.isfinite(value):
            raise InvalidKeyframeError(
                f"{role} keyframe time at index {index} is not finite"
            )
        if value < 0:
            raise InvalidKeyframeError(
                f"{role} keyframe time at index {index} is negative"
            )
        if previous is not None and value <= previous:
            raise InvalidKeyframeError(
                f"{role} keyframe times must be strictly increasing"
            )
        previous = value


def _validate_channel_values(path, interpolation, values, keyframe_count):
    expected_count = keyframe_count * 3 if interpolation == ANIM_CUBICSPLINE else keyframe_count
    if len(values) != expected_count:
        raise InvalidKeyframeError(
            f"{path} channel has {len(values)} output values for "
            f"{keyframe_count} keyframes; expected {expected_count}"
        )

    expected_components = 4 if path == ROTATION else 3
    for index, value in enumerate(values):
        if len(value) != expected_components:
            raise InvalidKeyframeError(
                f"{path} output at index {index} has {len(value)} components; "
                f"expected {expected_components}"
            )
        if any(not math.isfinite(component) for component in value):
            raise InvalidKeyframeError(
                f"{path} output at index {index} contains a non-finite value"
            )

    if path == ROTATION:
        step = 3 if interpolation == ANIM_CUBICSPLINE else 1
        offset = 1 if interpolation == ANIM_CUBICSPLINE else 0
        for keyframe in range(keyframe_count):
            value = values[keyframe * step + offset]
            if _quaternion_norm(value) <= 1e-12:
                raise InvalidKeyframeError(
                    f"rotation keyframe {keyframe} is a zero quaternion"
                )


def _prepare_channels(gltf, animation, buffer_cache):
    channels = []
    seen = {}

    for channel_index, channel in enumerate(animation.channels):
        if getattr(channel, "extensions", None):
            raise UnsupportedAnimationChannelError(
                f"animation channel {channel_index} uses extensions, which are "
                "not supported"
            )

        target = getattr(channel, "target", None)
        if target is None:
            raise InvalidAnimationError(
                f"animation channel {channel_index} has no target"
            )
        if getattr(target, "extensions", None):
            raise UnsupportedAnimationChannelError(
                f"animation channel {channel_index} targets an extension path, "
                "which is not supported"
            )

        path = getattr(target, "path", None)
        if path == WEIGHTS:
            raise UnsupportedAnimationChannelError(
                f"animation channel {channel_index} targets morph target "
                "weights, which are not supported"
            )
        if path not in _ANIMATED_PATHS:
            raise UnsupportedAnimationChannelError(
                f"animation channel {channel_index} uses unsupported target "
                f"path {path!r}"
            )

        node_index = getattr(target, "node", None)
        if node_index is None:
            raise InvalidAnimationError(
                f"animation channel {channel_index} has no target node"
            )
        if (
            not isinstance(node_index, int)
            or isinstance(node_index, bool)
            or node_index < 0
            or node_index >= len(gltf.nodes)
        ):
            raise InvalidAnimationError(
                f"animation channel {channel_index} references invalid node "
                f"{node_index!r}"
            )

        channel_key = (node_index, path)
        if channel_key in seen:
            raise DuplicateAnimationChannelError(
                f"animation channels {seen[channel_key]} and {channel_index} "
                f"both target node {node_index} property {path!r}"
            )
        seen[channel_key] = channel_index

        sampler_index = getattr(channel, "sampler", None)
        if (
            not isinstance(sampler_index, int)
            or isinstance(sampler_index, bool)
            or sampler_index < 0
        ):
            raise InvalidAnimationError(
                f"animation channel {channel_index} has invalid sampler index "
                f"{sampler_index!r}"
            )
        try:
            sampler = animation.samplers[sampler_index]
        except (IndexError, TypeError):
            raise InvalidAnimationError(
                f"animation channel {channel_index} references missing sampler "
                f"{sampler_index}"
            )
        if getattr(sampler, "extensions", None):
            raise UnsupportedAnimationChannelError(
                f"animation sampler {sampler_index} uses extensions, which are "
                "not supported"
            )

        interpolation = getattr(sampler, "interpolation", None) or ANIM_LINEAR
        interpolation = str(interpolation).upper()
        if interpolation not in _SUPPORTED_INTERPOLATIONS:
            raise UnsupportedAnimationChannelError(
                f"animation channel {channel_index} uses unsupported "
                f"interpolation {interpolation!r}"
            )

        input_index = getattr(sampler, "input", None)
        output_index = getattr(sampler, "output", None)
        if input_index is None or output_index is None:
            raise InvalidAnimationError(
                f"animation sampler {sampler_index} is missing input or output"
            )

        input_accessor = _require_float_accessor(
            gltf, input_index, "animation input"
        )
        output_accessor = _require_float_accessor(
            gltf, output_index, "animation output"
        )
        if getattr(input_accessor, "type", None) != SCALAR:
            raise InvalidKeyframeError(
                "animation input accessor must have SCALAR type"
            )
        expected_output_type = VEC4 if path == ROTATION else VEC3
        if getattr(output_accessor, "type", None) != expected_output_type:
            raise InvalidKeyframeError(
                f"{path} animation output accessor must have "
                f"{expected_output_type} type"
            )

        time_rows = _read_accessor(gltf, input_index, buffer_cache)
        _validate_times(time_rows, "animation input")
        times = [row[0] for row in time_rows]

        values = _read_accessor(gltf, output_index, buffer_cache)
        _validate_channel_values(path, interpolation, values, len(times))

        channels.append(
            {
                "node": node_index,
                "path": path,
                "interpolation": interpolation,
                "times": times,
                "values": values,
            }
        )

    return channels


def _quaternion_norm(quaternion):
    return math.sqrt(sum(component * component for component in quaternion))


def _normalize_quaternion(quaternion, context="rotation"):
    norm = _quaternion_norm(quaternion)
    if not math.isfinite(norm) or norm <= 1e-12:
        raise InvalidKeyframeError(f"{context} is a zero or invalid quaternion")
    return [component / norm for component in quaternion]


def _lerp_values(start, end, factor):
    return [
        start[index] + (end[index] - start[index]) * factor
        for index in range(len(start))
    ]


def _slerp_quaternions(start, end, factor):
    start = _normalize_quaternion(start, "rotation keyframe")
    end = _normalize_quaternion(end, "rotation keyframe")
    dot = sum(a * b for a, b in zip(start, end))
    if dot < 0.0:
        end = [-component for component in end]
        dot = -dot

    if dot > 0.9995:
        result = _lerp_values(start, end, factor)
        return _normalize_quaternion(result, "interpolated rotation")

    dot = max(-1.0, min(1.0, dot))
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    start_weight = math.sin((1.0 - factor) * theta) / sin_theta
    end_weight = math.sin(factor * theta) / sin_theta
    result = [
        start[index] * start_weight + end[index] * end_weight
        for index in range(4)
    ]
    return _normalize_quaternion(result, "interpolated rotation")


def _sample_channel(channel, time):
    times = channel["times"]
    values = channel["values"]
    path = channel["path"]
    interpolation = channel["interpolation"]

    if time <= times[0]:
        keyframe = 0
        index = 1 if interpolation == ANIM_CUBICSPLINE else 0
        value = values[keyframe * (3 if interpolation == ANIM_CUBICSPLINE else 1) + index]
        return _normalize_quaternion(value) if path == ROTATION else list(value)

    if time >= times[-1]:
        keyframe = len(times) - 1
        index = 1 if interpolation == ANIM_CUBICSPLINE else 0
        value = values[keyframe * (3 if interpolation == ANIM_CUBICSPLINE else 1) + index]
        return _normalize_quaternion(value) if path == ROTATION else list(value)

    if len(times) == 1:
        value = values[0]
        return _normalize_quaternion(value) if path == ROTATION else list(value)

    keyframe = bisect_right(times, time) - 1
    next_keyframe = keyframe + 1
    interval = times[next_keyframe] - times[keyframe]
    factor = (time - times[keyframe]) / interval

    if interpolation == ANIM_STEP:
        value = values[keyframe]
        return _normalize_quaternion(value) if path == ROTATION else list(value)

    if interpolation == ANIM_LINEAR:
        start = values[keyframe]
        end = values[next_keyframe]
        if path == ROTATION:
            return _slerp_quaternions(start, end, factor)
        return _lerp_values(start, end, factor)

    start_value = values[keyframe * 3 + 1]
    start_tangent = [
        component * interval for component in values[keyframe * 3 + 2]
    ]
    end_value = values[next_keyframe * 3 + 1]
    end_tangent = [
        component * interval for component in values[next_keyframe * 3]
    ]

    factor2 = factor * factor
    factor3 = factor2 * factor
    start_basis = 2.0 * factor3 - 3.0 * factor2 + 1.0
    start_tangent_basis = factor3 - 2.0 * factor2 + factor
    end_basis = -2.0 * factor3 + 3.0 * factor2
    end_tangent_basis = factor3 - factor2

    result = [
        start_basis * start_value[index]
        + start_tangent_basis * start_tangent[index]
        + end_basis * end_value[index]
        + end_tangent_basis * end_tangent[index]
        for index in range(len(start_value))
    ]
    if path == ROTATION:
        return _normalize_quaternion(result, "cubic spline rotation")
    return result


def _copy_transform_value(value, default):
    if value is None or len(value) == 0:
        return list(default)
    return list(value)


def sample_animation(gltf, animation_index, time):
    """
    Sample local node transforms from an animation at ``time`` seconds.

    Args:
        gltf (GLTF2): glTF object whose buffer data is available in memory, as
            a data URI, or as a local file relative to the glTF file.
        animation_index (int): index into ``gltf.animations``.
        time (float): sample time in seconds.

    Returns:
        AnimationSample: mapping of target node index to ``NodeTransform``.
        Each transform contains ``translation``, ``rotation`` (``[x, y, z, w]``)
        and ``scale``.  Only nodes targeted by this animation are included.
        Properties without a channel keep the node's original value, or the
        glTF default when the node value is unset.

    Raises:
        AnimationSamplingError: if the animation is invalid or unsupported.
    """
    animation = _coerce_animation_index(gltf, animation_index)
    sample_time = _coerce_time(time)
    buffer_cache = {}
    channels = _prepare_channels(gltf, animation, buffer_cache)

    sampled = {}
    target_nodes = []
    for channel in channels:
        if channel["node"] not in sampled:
            sampled[channel["node"]] = {}
            target_nodes.append(channel["node"])
        sampled[channel["node"]][channel["path"]] = _sample_channel(
            channel, sample_time
        )

    result = AnimationSample(
        animation_index=animation_index, time=sample_time
    )
    for node_index in target_nodes:
        node = gltf.nodes[node_index]
        transforms = sampled[node_index]
        translation = transforms.get(
            TRANSLATION,
            _copy_transform_value(getattr(node, "translation", None), (0.0, 0.0, 0.0)),
        )
        rotation = transforms.get(
            ROTATION,
            _copy_transform_value(
                getattr(node, "rotation", None), (0.0, 0.0, 0.0, 1.0)
            ),
        )
        scale = transforms.get(
            SCALE,
            _copy_transform_value(getattr(node, "scale", None), (1.0, 1.0, 1.0)),
        )
        result[node_index] = NodeTransform(
            translation=translation,
            rotation=rotation,
            scale=scale,
            node_index=node_index,
            node=node,
        )

    return result


sample_animation_at = sample_animation
get_animation_sample = sample_animation


__all__ = [
    "AnimationDataError",
    "AnimationSample",
    "AnimationSamplingError",
    "DuplicateAnimationChannelError",
    "InvalidAnimationError",
    "InvalidKeyframeError",
    "NodeTransform",
    "UnsupportedAnimationChannelError",
    "get_animation_sample",
    "sample_animation",
    "sample_animation_at",
]
