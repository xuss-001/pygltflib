"""
Read the values described by glTF accessors.

The functions in this module only read from the GLTF2 object and its
referenced binary data.  They never mutate the object, its buffers or the
files the buffers point to.
"""

import base64
import operator
import struct
from pathlib import Path
from urllib.parse import unquote

from .types import (
    BYTE,
    FLOAT,
    MAT2,
    MAT3,
    MAT4,
    SHORT,
    UNSIGNED_BYTE,
    UNSIGNED_INT,
    UNSIGNED_SHORT,
)


class AccessorError(ValueError):
    """Base class for errors raised while reading accessor data."""


class AccessorIndexError(IndexError, AccessorError):
    """The requested accessor or buffer index does not exist."""


class AccessorDataError(AccessorError):
    """The accessor description is invalid or unsupported."""


class AccessorBoundsError(AccessorDataError):
    """An accessor, buffer view or sparse range is out of bounds."""


class AccessorStrideError(AccessorDataError):
    """A buffer view byteStride is invalid for the accessor."""


class AccessorSparseError(AccessorDataError):
    """The accessor sparse description is invalid."""


class AccessorSparseIndexError(IndexError, AccessorSparseError):
    """A sparse index points outside the accessor's element range."""


_COMPONENT_FORMATS = {
    BYTE: "b",
    UNSIGNED_BYTE: "B",
    SHORT: "h",
    UNSIGNED_SHORT: "H",
    UNSIGNED_INT: "I",
    FLOAT: "f",
}

_COMPONENT_SIZES = {
    BYTE: 1,
    UNSIGNED_BYTE: 1,
    SHORT: 2,
    UNSIGNED_SHORT: 2,
    UNSIGNED_INT: 4,
    FLOAT: 4,
}

_TYPE_COMPONENT_COUNTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}

_MATRIX_SHAPES = {
    MAT2: (2, 2),
    MAT3: (3, 3),
    MAT4: (4, 4),
}

_SPARSE_INDEX_COMPONENT_TYPES = {
    UNSIGNED_BYTE,
    UNSIGNED_SHORT,
    UNSIGNED_INT,
}

_NORMALIZED_DIVISORS = {
    BYTE: 127.0,
    UNSIGNED_BYTE: 255.0,
    SHORT: 32767.0,
    UNSIGNED_SHORT: 65535.0,
}

_SIGNED_COMPONENT_TYPES = {BYTE, SHORT}

_ACCESSOR_INDEX_KWARGS = ("index", "accessor_id")


class AccessorElement(tuple):
    """
    One accessor element.

    The element is tuple-like (and therefore works with code that expects
    ``struct.unpack`` tuples) but its equality also accepts the equivalent
    list or, for scalars, the single component itself.
    """

    __slots__ = ()

    def __new__(cls, values):
        return super().__new__(cls, values)

    def __eq__(self, other):
        if isinstance(other, (list, tuple)):
            return tuple(self) == tuple(other)
        if len(self) == 1:
            return self[0] == other
        return NotImplemented

    def __ne__(self, other):
        equal = self.__eq__(other)
        if equal is NotImplemented:
            return NotImplemented
        return not equal

    __hash__ = tuple.__hash__


class AccessorData(list):
    """
    List of :class:`AccessorElement` values returned by accessor reads.

    It is a normal list, but equality additionally accepts a flat sequence of
    all components.  This makes scalar accessors convenient to consume while
    keeping elements grouped by accessor element.
    """

    def _flattened(self):
        return [component for element in self for component in element]

    def __eq__(self, other):
        if isinstance(other, (list, tuple)):
            if len(self) == len(other):
                if all(element == expected for element, expected in zip(self, other)):
                    return True
            if self._flattened() == list(other):
                return True
            return False
        return list.__eq__(self, other)

    def __ne__(self, other):
        return not self.__eq__(other)

    __hash__ = None


def _resolve_accessor_index(gltf, accessor_index, kwargs):
    if accessor_index is None:
        for name in _ACCESSOR_INDEX_KWARGS:
            if name in kwargs:
                accessor_index = kwargs.pop(name)
                break
    if accessor_index is None and "accessor" in kwargs:
        accessor_index = kwargs.pop("accessor")
    if accessor_index is None:
        raise TypeError("accessor_index is required")

    if hasattr(accessor_index, "componentType") and hasattr(accessor_index, "count"):
        return accessor_index, None

    try:
        index = operator.index(accessor_index)
    except TypeError:
        raise TypeError("accessor_index must be an integer or an Accessor") from None

    accessors = getattr(gltf, "accessors", None)
    if accessors is None:
        raise AccessorDataError("GLTF object has no accessors")
    if index < 0 or index >= len(accessors):
        raise AccessorIndexError(
            f"accessor index {index} is out of range (have {len(accessors)} accessors)"
        )
    return accessors[index], index


def _resolve_normalized(normalized, kwargs):
    for name in (
        "normalize",
        "use_normalized",
        "convert_normalized",
        "apply_normalized",
        "apply_normalization",
    ):
        if name in kwargs:
            normalized = kwargs.pop(name)
            break
    if "raw" in kwargs:
        normalized = not kwargs.pop("raw")
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"unexpected keyword argument(s): {unknown}")
    return bool(normalized)


def _component_info(component_type):
    try:
        return _COMPONENT_FORMATS[component_type], _COMPONENT_SIZES[component_type]
    except KeyError:
        raise AccessorDataError(
            f"unsupported accessor componentType {component_type!r}"
        ) from None


def _element_layout(accessor_type, component_size):
    try:
        component_count = _TYPE_COMPONENT_COUNTS[accessor_type]
    except KeyError:
        raise AccessorDataError(
            f"unsupported accessor type {accessor_type!r}"
        ) from None

    if accessor_type in _MATRIX_SHAPES:
        columns, rows = _MATRIX_SHAPES[accessor_type]
        # glTF stores matrices column-major and pads every column so that its
        # start is aligned to four bytes.  This is a no-op for 4-byte
        # components but is essential for BYTE/SHORT matrices.
        column_stride = ((rows * component_size + 3) // 4) * 4
        offsets = [
            column * column_stride + row * component_size
            for column in range(columns)
            for row in range(rows)
        ]
        element_size = columns * column_stride
    else:
        offsets = [index * component_size for index in range(component_count)]
        element_size = component_count * component_size

    return offsets, element_size


def _buffer_bytes(gltf, buffer_index, cache):
    if buffer_index in cache:
        return cache[buffer_index]

    buffers = getattr(gltf, "buffers", None)
    if buffers is None:
        raise AccessorDataError("GLTF object has no buffers")
    if buffer_index < 0 or buffer_index >= len(buffers):
        raise AccessorIndexError(
            f"buffer index {buffer_index} is out of range (have {len(buffers)} buffers)"
        )

    buffer = buffers[buffer_index]
    uri = getattr(buffer, "uri", None)

    if uri is None:
        binary_blob = getattr(gltf, "binary_blob", None)
        data = binary_blob() if callable(binary_blob) else getattr(gltf, "_glb_data", None)
        if data is None:
            raise AccessorDataError(
                f"buffer {buffer_index} has no uri and no GLB binary data is available"
            )
        if len(buffers) > 1 and buffer_index != 0:
            raise AccessorDataError(
                "multiple buffers use the implicit GLB binary blob; "
                f"cannot select buffer {buffer_index}"
            )
    elif isinstance(uri, str) and uri.startswith("data:"):
        try:
            header, payload = uri.split(",", 1)
        except ValueError:
            raise AccessorDataError("malformed data URI buffer") from None
        if ";base64" not in header:
            raise AccessorDataError("only base64 data URI buffers are supported")
        try:
            data = base64.b64decode(payload)
        except Exception as exc:
            raise AccessorDataError(f"unable to decode data URI buffer: {exc}") from exc
    elif isinstance(uri, str):
        base_path = getattr(gltf, "_path", None) or Path()
        path = Path(base_path, unquote(uri))
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise AccessorDataError(
                f"unable to read buffer file {path!s}: {exc}"
            ) from exc
    else:
        raise AccessorDataError(f"unsupported buffer uri {uri!r}")

    cache[buffer_index] = data
    return data


def _buffer_view(gltf, buffer_view_index, cache):
    buffer_views = getattr(gltf, "bufferViews", None)
    if buffer_views is None:
        raise AccessorDataError("GLTF object has no bufferViews")
    if (
        not isinstance(buffer_view_index, int)
        or buffer_view_index < 0
        or buffer_view_index >= len(buffer_views)
    ):
        raise AccessorIndexError(
            f"bufferView index {buffer_view_index!r} is out of range "
            f"(have {len(buffer_views)} bufferViews)"
        )

    buffer_view = buffer_views[buffer_view_index]
    buffer_index = getattr(buffer_view, "buffer", None)
    if not isinstance(buffer_index, int):
        raise AccessorDataError(
            f"bufferView {buffer_view_index} has no valid buffer index"
        )
    data = _buffer_bytes(gltf, buffer_index, cache)

    byte_offset = getattr(buffer_view, "byteOffset", 0)
    if byte_offset is None:
        byte_offset = 0
    byte_length = getattr(buffer_view, "byteLength", None)
    if byte_length is None:
        raise AccessorDataError(f"bufferView {buffer_view_index} has no byteLength")
    if not isinstance(byte_offset, int) or byte_offset < 0:
        raise AccessorBoundsError(
            f"bufferView {buffer_view_index} byteOffset must be >= 0"
        )
    if not isinstance(byte_length, int) or byte_length < 0:
        raise AccessorBoundsError(
            f"bufferView {buffer_view_index} byteLength must be >= 0"
        )

    end = byte_offset + byte_length
    if end > len(data):
        raise AccessorBoundsError(
            f"bufferView {buffer_view_index} range "
            f"[{byte_offset}, {end}) exceeds buffer {buffer_index} data "
            f"length {len(data)}"
        )

    buffer = gltf.buffers[buffer_index]
    declared_length = getattr(buffer, "byteLength", None)
    if declared_length is not None and end > declared_length:
        raise AccessorBoundsError(
            f"bufferView {buffer_view_index} range "
            f"[{byte_offset}, {end}) exceeds declared buffer {buffer_index} "
            f"byteLength {declared_length}"
        )

    return buffer_view, data, byte_offset


def _element_stride(buffer_view, element_size, buffer_view_index):
    byte_stride = getattr(buffer_view, "byteStride", None)
    if byte_stride is None:
        return element_size
    if not isinstance(byte_stride, int) or isinstance(byte_stride, bool):
        raise AccessorStrideError(
            f"bufferView {buffer_view_index} byteStride must be an integer"
        )
    if byte_stride < 4 or byte_stride > 252 or byte_stride % 4 != 0:
        raise AccessorStrideError(
            f"bufferView {buffer_view_index} byteStride {byte_stride} is invalid; "
            "it must be a multiple of 4 between 4 and 252"
        )
    if byte_stride < element_size:
        raise AccessorStrideError(
            f"bufferView {buffer_view_index} byteStride {byte_stride} is smaller "
            f"than the accessor element size {element_size}"
        )
    return byte_stride


def _read_element(data, absolute_offset, fmt, component_offsets):
    return AccessorElement(
        struct.unpack_from("<" + fmt, data, absolute_offset + offset)[0]
        for offset in component_offsets
    )


def _read_base_elements(
    gltf,
    accessor,
    fmt,
    component_offsets,
    element_size,
    cache,
    zero_value,
):
    count = getattr(accessor, "count", None)
    if count is None or not isinstance(count, int) or count < 0:
        raise AccessorDataError("accessor count must be a non-negative integer")

    buffer_view_index = getattr(accessor, "bufferView", None)
    if buffer_view_index is None:
        return [
            AccessorElement([zero_value] * len(component_offsets))
            for _ in range(count)
        ]

    buffer_view, data, buffer_view_start = _buffer_view(
        gltf, buffer_view_index, cache
    )
    byte_offset = getattr(accessor, "byteOffset", 0)
    if byte_offset is None:
        byte_offset = 0
    if not isinstance(byte_offset, int) or byte_offset < 0:
        raise AccessorBoundsError("accessor byteOffset must be >= 0")

    stride = _element_stride(buffer_view, element_size, buffer_view_index)
    byte_length = buffer_view.byteLength
    if count:
        end = byte_offset + (count - 1) * stride + element_size
        if end > byte_length:
            raise AccessorBoundsError(
                f"accessor data range [{byte_offset}, {end}) exceeds "
                f"bufferView {buffer_view_index} byteLength {byte_length}"
            )

    start = buffer_view_start + byte_offset
    return [
        _read_element(data, start + index * stride, fmt, component_offsets)
        for index in range(count)
    ]


def _read_sparse_indices(gltf, accessor, sparse, cache):
    sparse_indices = getattr(sparse, "indices", None)
    if sparse_indices is None:
        raise AccessorSparseError("sparse accessor has no indices")
    component_type = getattr(sparse_indices, "componentType", None)
    if component_type not in _SPARSE_INDEX_COMPONENT_TYPES:
        raise AccessorSparseError(
            f"unsupported sparse index componentType {component_type!r}"
        )

    buffer_view_index = getattr(sparse_indices, "bufferView", None)
    if buffer_view_index is None:
        raise AccessorSparseError("sparse indices have no bufferView")
    buffer_view, data, buffer_view_start = _buffer_view(
        gltf, buffer_view_index, cache
    )
    if getattr(buffer_view, "byteStride", None) is not None:
        raise AccessorStrideError(
            f"sparse indices bufferView {buffer_view_index} must not define byteStride"
        )

    fmt, component_size = _component_info(component_type)
    byte_offset = getattr(sparse_indices, "byteOffset", 0)
    if byte_offset is None:
        byte_offset = 0
    count = getattr(sparse, "count", None)
    if count is None or not isinstance(count, int) or count < 0:
        raise AccessorSparseError("sparse count must be a non-negative integer")
    accessor_count = getattr(accessor, "count", None)
    if (
        not isinstance(accessor_count, int)
        or accessor_count < 0
        or count > accessor_count
    ):
        raise AccessorSparseError(
            f"sparse count {count} exceeds accessor count {accessor_count!r}"
        )
    if not isinstance(byte_offset, int) or byte_offset < 0:
        raise AccessorBoundsError("sparse indices byteOffset must be >= 0")

    end = byte_offset + count * component_size
    if count and end > buffer_view.byteLength:
        raise AccessorBoundsError(
            f"sparse indices range [{byte_offset}, {end}) exceeds "
            f"bufferView {buffer_view_index} byteLength {buffer_view.byteLength}"
        )

    start = buffer_view_start + byte_offset
    indices = [
        struct.unpack_from("<" + fmt, data, start + index * component_size)[0]
        for index in range(count)
    ]
    for index in indices:
        if index >= accessor_count:
            raise AccessorSparseIndexError(
                f"sparse index {index} is outside accessor element range "
                f"[0, {accessor_count})"
            )
    return indices


def _read_sparse_values(
    gltf,
    sparse,
    count,
    fmt,
    component_offsets,
    element_size,
    cache,
):
    sparse_values = getattr(sparse, "values", None)
    if sparse_values is None:
        raise AccessorSparseError("sparse accessor has no values")
    buffer_view_index = getattr(sparse_values, "bufferView", None)
    if buffer_view_index is None:
        raise AccessorSparseError("sparse values have no bufferView")
    buffer_view, data, buffer_view_start = _buffer_view(
        gltf, buffer_view_index, cache
    )
    if getattr(buffer_view, "byteStride", None) is not None:
        raise AccessorStrideError(
            f"sparse values bufferView {buffer_view_index} must not define byteStride"
        )

    byte_offset = getattr(sparse_values, "byteOffset", 0)
    if byte_offset is None:
        byte_offset = 0
    if not isinstance(byte_offset, int) or byte_offset < 0:
        raise AccessorBoundsError("sparse values byteOffset must be >= 0")

    end = byte_offset + count * element_size
    if count and end > buffer_view.byteLength:
        raise AccessorBoundsError(
            f"sparse values range [{byte_offset}, {end}) exceeds "
            f"bufferView {buffer_view_index} byteLength {buffer_view.byteLength}"
        )

    start = buffer_view_start + byte_offset
    return [
        _read_element(data, start + index * element_size, fmt, component_offsets)
        for index in range(count)
    ]


def _normalize_value(value, component_type):
    divisor = _NORMALIZED_DIVISORS.get(component_type)
    if divisor is None:
        return value
    normalized = value / divisor
    if component_type in _SIGNED_COMPONENT_TYPES:
        normalized = max(normalized, -1.0)
    return normalized


def _apply_normalization(elements, component_type):
    return AccessorData(
        AccessorElement(
            _normalize_value(component, component_type) for component in element
        )
        for element in elements
    )


def get_accessor_data(gltf, accessor_index=None, normalized=False, **kwargs):
    """
    Return the values of an accessor.

    Parameters
    ----------
    gltf:
        A :class:`pygltflib.GLTF2` instance (or a compatible object).
    accessor_index:
        Index into ``gltf.accessors``.  An ``Accessor`` instance is also
        accepted.
    normalized:
        If true and the accessor has ``normalized=True``, convert integer
        components to floats using the glTF rules.  The default is false, so
        the stored integer values are returned unchanged.

    Returns
    -------
    AccessorData
        A list containing one tuple-like element per accessor element.  Matrix
        components are returned in glTF's column-major order.
    """
    # Also accept get_accessor_data(accessor_index, gltf) for convenience.
    if not hasattr(gltf, "accessors") and hasattr(accessor_index, "accessors"):
        gltf, accessor_index = accessor_index, gltf

    accessor_index, _ = _resolve_accessor_index(gltf, accessor_index, kwargs)
    normalized = _resolve_normalized(normalized, kwargs)

    accessor = accessor_index
    accessor_type = getattr(accessor, "type", None)
    component_type = getattr(accessor, "componentType", None)
    fmt, component_size = _component_info(component_type)
    component_offsets, element_size = _element_layout(accessor_type, component_size)
    count = getattr(accessor, "count", None)
    if count is None or not isinstance(count, int) or count < 0:
        raise AccessorDataError("accessor count must be a non-negative integer")

    cache = {}
    elements = _read_base_elements(
        gltf,
        accessor,
        fmt,
        component_offsets,
        element_size,
        cache,
        0.0 if component_type == FLOAT else 0,
    )

    sparse = getattr(accessor, "sparse", None)
    if sparse is not None:
        sparse_count = getattr(sparse, "count", None)
        if sparse_count is None or not isinstance(sparse_count, int) or sparse_count < 0:
            raise AccessorSparseError("sparse count must be a non-negative integer")
        if sparse_count > count:
            raise AccessorSparseError(
                f"sparse count {sparse_count} exceeds accessor count {count}"
            )
        sparse_indices = _read_sparse_indices(gltf, accessor, sparse, cache)
        sparse_values = _read_sparse_values(
            gltf,
            sparse,
            sparse_count,
            fmt,
            component_offsets,
            element_size,
            cache,
        )
        for index, value in zip(sparse_indices, sparse_values):
            elements[index] = value

    result = AccessorData(elements)
    if normalized and getattr(accessor, "normalized", False):
        result = _apply_normalization(result, component_type)
    return result


def read_accessor_data(gltf, accessor_index=None, normalized=False, **kwargs):
    """Alias for :func:`get_accessor_data`."""

    return get_accessor_data(
        gltf, accessor_index=accessor_index, normalized=normalized, **kwargs
    )


def read_accessor(gltf, accessor_index=None, normalized=False, **kwargs):
    """Alias for :func:`get_accessor_data`."""

    return get_accessor_data(
        gltf, accessor_index=accessor_index, normalized=normalized, **kwargs
    )


def get_accessor_values(gltf, accessor_index=None, normalized=False, **kwargs):
    """Alias for :func:`get_accessor_data`."""

    return get_accessor_data(
        gltf, accessor_index=accessor_index, normalized=normalized, **kwargs
    )
