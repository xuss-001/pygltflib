"""Vertex reordering helpers for :class:`pygltflib.GLTF2`.

The public entry point is ``GLTF2.reorder_mesh_vertices``.  The implementation
plans and validates the complete operation before mutating the GLTF object, so
failures leave the original asset untouched.
"""

import base64
import copy
import struct
from operator import index as _operator_index
from pathlib import Path
from urllib.parse import unquote

from .types import (
    BYTE,
    ELEMENT_ARRAY_BUFFER,
    FLOAT,
    MAT2,
    MAT3,
    MAT4,
    SCALAR,
    SHORT,
    TRIANGLES,
    UNSIGNED_BYTE,
    UNSIGNED_INT,
    UNSIGNED_SHORT,
    VEC2,
    VEC3,
    VEC4,
    Accessor,
    BufferView,
)


_COMPONENT_FORMATS = {
    BYTE: ("b", 1),
    UNSIGNED_BYTE: ("B", 1),
    SHORT: ("h", 2),
    UNSIGNED_SHORT: ("H", 2),
    UNSIGNED_INT: ("I", 4),
    FLOAT: ("f", 4),
}

_TYPE_COMPONENTS = {
    SCALAR: 1,
    VEC2: 2,
    VEC3: 3,
    VEC4: 4,
    MAT2: 4,
    MAT3: 9,
    MAT4: 16,
}

_MATRIX_LAYOUTS = {
    MAT2: (2, 2),
    MAT3: (3, 3),
    MAT4: (4, 4),
}

_INDEX_COMPONENT_TYPES = (UNSIGNED_BYTE, UNSIGNED_SHORT, UNSIGNED_INT)


def _as_index(value, description):
    if isinstance(value, bool):
        raise ValueError("{} must be an integer, not bool".format(description))
    try:
        return _operator_index(value)
    except TypeError:
        raise ValueError("{} must be an integer".format(description))


def _normalise_mesh_indices(gltf, mesh_indices):
    try:
        scalar = _as_index(mesh_indices, "mesh_indices")
    except ValueError:
        try:
            values = list(mesh_indices)
        except TypeError:
            raise ValueError("mesh_indices must be an integer or an iterable of integers")
    else:
        values = [scalar]

    normalised = []
    seen = set()
    for position, value in enumerate(values):
        mesh_index = _as_index(value, "mesh_indices[{}]".format(position))
        if mesh_index < 0 or mesh_index >= len(gltf.meshes):
            raise ValueError(
                "Mesh index {} is out of range (0..{})".format(
                    mesh_index, len(gltf.meshes) - 1
                )
            )
        if mesh_index not in seen:
            seen.add(mesh_index)
            normalised.append(mesh_index)

    if not normalised:
        raise ValueError("At least one mesh must be selected")
    return normalised


def _normalise_permutation(permutation, vertex_count):
    try:
        values = list(permutation)
    except TypeError:
        raise ValueError("permutation must be an iterable of vertex indices")

    if len(values) != vertex_count:
        raise ValueError(
            "permutation has {} entries but the selected meshes have {} vertices".format(
                len(values), vertex_count
            )
        )

    normalised = []
    seen = set()
    for position, value in enumerate(values):
        source_index = _as_index(value, "permutation[{}]".format(position))
        if source_index < 0 or source_index >= vertex_count:
            raise ValueError(
                "permutation[{}]={} is outside the vertex range 0..{}".format(
                    position, source_index, vertex_count - 1
                )
            )
        if source_index in seen:
            raise ValueError(
                "permutation contains duplicate source vertex {}".format(source_index)
            )
        seen.add(source_index)
        normalised.append(source_index)

    return normalised


def _accessor(gltf, accessor_index, description):
    accessor_index = _as_index(accessor_index, description)
    if accessor_index < 0 or accessor_index >= len(gltf.accessors):
        raise ValueError(
            "{} accessor {} is out of range (0..{})".format(
                description, accessor_index, len(gltf.accessors) - 1
            )
        )
    return gltf.accessors[accessor_index]


def _attribute_items(attributes):
    if attributes is None:
        return []
    if isinstance(attributes, dict):
        return list(attributes.items())
    return list(getattr(attributes, "__dict__", {}).items())


def _set_attribute(attributes, name, value):
    if isinstance(attributes, dict):
        attributes[name] = value
    else:
        setattr(attributes, name, value)


def _primitive_attribute_entries(primitive):
    entries = []
    for name, accessor_index in _attribute_items(primitive.attributes):
        if accessor_index is not None:
            entries.append(("attributes", primitive.attributes, name, accessor_index))

    for target_index, target in enumerate(primitive.targets or []):
        for name, accessor_index in _attribute_items(target):
            if accessor_index is not None:
                entries.append(
                    (
                        "targets[{}]".format(target_index),
                        target,
                        name,
                        accessor_index,
                    )
                )
    return entries


def _primitive_accessor_indices(primitive):
    indices = []
    for _kind, _container, _name, accessor_index in _primitive_attribute_entries(
        primitive
    ):
        try:
            indices.append(_as_index(accessor_index, "primitive attribute"))
        except ValueError:
            continue
    if primitive.indices is not None:
        try:
            indices.append(_as_index(primitive.indices, "primitive.indices"))
        except ValueError:
            pass
    return set(indices)


def _component_size(component_type):
    try:
        return _COMPONENT_FORMATS[component_type][1]
    except KeyError:
        raise ValueError(
            "Unsupported accessor componentType {}".format(component_type)
        )


def _component_format(component_type):
    try:
        return _COMPONENT_FORMATS[component_type][0]
    except KeyError:
        raise ValueError(
            "Unsupported accessor componentType {}".format(component_type)
        )


def _accessor_element_size(accessor):
    try:
        component_count = _TYPE_COMPONENTS[accessor.type]
    except KeyError:
        raise ValueError("Unsupported accessor type {!r}".format(accessor.type))

    component_size = _component_size(accessor.componentType)
    if accessor.type in _MATRIX_LAYOUTS:
        columns, rows = _MATRIX_LAYOUTS[accessor.type]
        column_size = rows * component_size
        column_size = (column_size + 3) // 4 * 4
        return columns * column_size
    return component_count * component_size


def _accessor_layout(gltf, accessor_index, description):
    accessor = _accessor(gltf, accessor_index, description)
    if accessor.sparse is not None:
        raise ValueError(
            "{} accessor {} is sparse; sparse accessors are not supported".format(
                description, accessor_index
            )
        )
    if accessor.bufferView is None:
        raise ValueError(
            "{} accessor {} has no bufferView".format(description, accessor_index)
        )
    if accessor.count is None:
        raise ValueError(
            "{} accessor {} has no count".format(description, accessor_index)
        )
    count = _as_index(accessor.count, "{} accessor count".format(description))
    if count <= 0:
        raise ValueError(
            "{} accessor {} has a non-positive count".format(
                description, accessor_index
            )
        )

    buffer_view_index = _as_index(
        accessor.bufferView, "{} accessor bufferView".format(description)
    )
    if buffer_view_index < 0 or buffer_view_index >= len(gltf.bufferViews):
        raise ValueError(
            "{} accessor {} has an out-of-range bufferView {}".format(
                description, accessor_index, buffer_view_index
            )
        )
    buffer_view = gltf.bufferViews[buffer_view_index]
    buffer_index = _as_index(
        buffer_view.buffer, "{} bufferView.buffer".format(description)
    )
    if buffer_index < 0 or buffer_index >= len(gltf.buffers):
        raise ValueError(
            "{} accessor {} has an out-of-range buffer {}".format(
                description, accessor_index, buffer_index
            )
        )

    element_size = _accessor_element_size(accessor)
    view_offset = buffer_view.byteOffset or 0
    accessor_offset = accessor.byteOffset or 0
    stride = buffer_view.byteStride
    if stride is None:
        stride = element_size
    stride = _as_index(stride, "{} bufferView.byteStride".format(description))
    if stride < element_size:
        raise ValueError(
            "{} accessor {} has byteStride {} smaller than its element size {}".format(
                description, accessor_index, stride, element_size
            )
        )

    view_length = buffer_view.byteLength
    if view_length is None:
        raise ValueError(
            "{} accessor {} has a bufferView without byteLength".format(
                description, accessor_index
            )
        )
    view_length = _as_index(view_length, "{} bufferView.byteLength".format(description))
    view_start = view_offset + accessor_offset
    view_end = view_offset + view_length
    data_end = view_start + (count - 1) * stride + element_size
    if view_start < view_offset or data_end > view_end:
        raise ValueError(
            "{} accessor {} reads outside bufferView {}".format(
                description, accessor_index, buffer_view_index
            )
        )

    return {
        "accessor": accessor,
        "accessor_index": accessor_index,
        "buffer_view": buffer_view,
        "buffer_view_index": buffer_view_index,
        "buffer_index": buffer_index,
        "start": view_start,
        "stride": stride,
        "element_size": element_size,
        "count": count,
        "end": data_end,
    }


def _read_scalar(data, offset, component_type):
    fmt = "<" + _component_format(component_type)
    size = _component_size(component_type)
    return struct.unpack_from(fmt, data, offset)[0], size


def _pack_scalar(value, component_type):
    return struct.pack("<" + _component_format(component_type), value)


class _BufferStage:
    def __init__(self, gltf, buffer_index):
        self.buffer_index = buffer_index
        self.buffer = gltf.buffers[buffer_index]
        self.uri_prefix = None
        self.path = None

        if self.buffer.uri is None:
            data = gltf.binary_blob()
            if data is None:
                raise ValueError(
                    "Buffer {} has no URI and the GLTF has no binary blob".format(
                        buffer_index
                    )
                )
            self.mode = "binary"
        elif self.buffer.uri.startswith("data:"):
            if "," not in self.buffer.uri:
                raise ValueError("Buffer {} has an invalid data URI".format(buffer_index))
            self.uri_prefix, encoded = self.buffer.uri.split(",", 1)
            self.uri_prefix += ","
            try:
                data = base64.b64decode(encoded)
            except Exception as error:
                raise ValueError(
                    "Unable to decode buffer {} data URI: {}".format(buffer_index, error)
                )
            self.mode = "datauri"
        else:
            path = Path(getattr(gltf, "_path", Path()), unquote(self.buffer.uri))
            try:
                with open(path, "rb") as buffer_file:
                    data = buffer_file.read()
            except OSError as error:
                raise ValueError(
                    "Unable to read buffer {} from {}: {}".format(
                        buffer_index, path, error
                    )
                )
            self.mode = "file"
            self.path = path

        self.original = bytes(data)
        self.data = bytearray(data)


class _Plan:
    def __init__(self, gltf):
        self.gltf = gltf
        self.stages = {}
        self.new_accessors = []
        self.new_buffer_views = []
        self.attribute_updates = []
        self.index_updates = []
        self.accessor_metadata_updates = []

    def stage(self, buffer_index):
        if buffer_index not in self.stages:
            self.stages[buffer_index] = _BufferStage(self.gltf, buffer_index)
        return self.stages[buffer_index]

    def append_bytes(self, buffer_index, payload, alignment=4):
        stage = self.stage(buffer_index)
        padding = (-len(stage.data)) % alignment
        if padding:
            stage.data.extend(b"\x00" * padding)
        offset = len(stage.data)
        stage.data.extend(payload)
        return offset, len(payload)

    def add_buffer_view(self, buffer_view):
        index = len(self.gltf.bufferViews) + len(self.new_buffer_views)
        self.new_buffer_views.append(buffer_view)
        return index

    def add_accessor(self, accessor):
        index = len(self.gltf.accessors) + len(self.new_accessors)
        self.new_accessors.append(accessor)
        return index


def _ranges_overlap(first, second):
    return first["start"] < second["end"] and second["start"] < first["end"]


def _layouts_overlap(first, second):
    if first["buffer_index"] != second["buffer_index"]:
        return False
    if first["stride"] == second["stride"]:
        relative_offset = (first["start"] - second["start"]) % first["stride"]
        if relative_offset >= second["element_size"]:
            return False
    return _ranges_overlap(first, second)


def _accessors_sharing_storage(gltf, selected_layouts, unselected_accessor_indices):
    shared = set()
    for candidate in selected_layouts:
        for unselected_index in unselected_accessor_indices:
            try:
                other = _accessor_layout(gltf, unselected_index, "unselected")
            except ValueError:
                continue
            if _layouts_overlap(candidate, other):
                shared.add(candidate["accessor_index"])
                break
    return shared


def _clone_accessor_with_payload(plan, layout, payload):
    buffer_view = layout["buffer_view"]
    offset, length = plan.append_bytes(layout["buffer_index"], payload)
    new_view = copy.deepcopy(buffer_view)
    new_view.buffer = layout["buffer_index"]
    new_view.byteOffset = offset
    new_view.byteLength = length
    new_view.byteStride = None
    new_view.target = (
        buffer_view.target
        if buffer_view.target is not None
        else (ELEMENT_ARRAY_BUFFER if layout.get("is_index") else None)
    )
    new_view_index = plan.add_buffer_view(new_view)

    new_accessor = copy.deepcopy(layout["accessor"])
    new_accessor.bufferView = new_view_index
    new_accessor.byteOffset = 0
    new_accessor.sparse = None
    new_accessor_index = plan.add_accessor(new_accessor)
    return new_accessor, new_accessor_index


def _validate_buffer_length(plan, layout):
    stage = plan.stage(layout["buffer_index"])
    buffer_view = layout["buffer_view"]
    view_end = (buffer_view.byteOffset or 0) + buffer_view.byteLength
    if len(stage.original) < view_end:
        raise ValueError(
            "Buffer {} is shorter than bufferView {} requires".format(
                layout["buffer_index"], layout["buffer_view_index"]
            )
        )
    return stage


def _reorder_vertex_accessor(plan, layout, permutation, replacement):
    stage = _validate_buffer_length(plan, layout)
    data = stage.original
    start = layout["start"]
    stride = layout["stride"]
    element_size = layout["element_size"]

    if replacement is not None:
        payload = bytearray()
        for source_index in permutation:
            source = start + source_index * stride
            payload.extend(data[source : source + element_size])
        new_accessor, new_accessor_index = _clone_accessor_with_payload(
            plan, layout, bytes(payload)
        )
        replacement[layout["accessor_index"]] = new_accessor_index
        return

    for destination_index, source_index in enumerate(permutation):
        source = start + source_index * stride
        destination = start + destination_index * stride
        stage.data[destination : destination + element_size] = data[
            source : source + element_size
        ]


def _rewrite_index_accessor(plan, layout, inverse_permutation, replacement):
    stage = _validate_buffer_length(plan, layout)
    data = stage.original
    accessor = layout["accessor"]
    component_type = accessor.componentType
    start = layout["start"]
    stride = layout["stride"]

    mapped_values = []
    for index in range(layout["count"]):
        value, _size = _read_scalar(data, start + index * stride, component_type)
        if value < 0 or value >= len(inverse_permutation):
            raise ValueError(
                "Index {} in accessor {} is outside the vertex range 0..{}".format(
                    value,
                    layout["accessor_index"],
                    len(inverse_permutation) - 1,
                )
            )
        mapped_values.append(inverse_permutation[value])

    minimum = min(mapped_values)
    maximum = max(mapped_values)
    if replacement is not None:
        payload = b"".join(
            _pack_scalar(value, component_type) for value in mapped_values
        )
        new_accessor, new_accessor_index = _clone_accessor_with_payload(
            plan, layout, payload
        )
        new_accessor.min = [minimum]
        new_accessor.max = [maximum]
        replacement[layout["accessor_index"]] = new_accessor_index
        return

    for index, value in enumerate(mapped_values):
        offset = start + index * stride
        stage.data[offset : offset + _component_size(component_type)] = _pack_scalar(
            value, component_type
        )
    plan.accessor_metadata_updates.append((accessor, [minimum], [maximum]))


def _generated_index_accessor(plan, primitive, base_layout, inverse_permutation):
    count = len(inverse_permutation)
    maximum = count - 1
    if maximum <= 0xFF:
        component_type = UNSIGNED_BYTE
    elif maximum <= 0xFFFF:
        component_type = UNSIGNED_SHORT
    else:
        component_type = UNSIGNED_INT

    payload = b"".join(
        _pack_scalar(value, component_type) for value in inverse_permutation
    )
    offset, length = plan.append_bytes(base_layout["buffer_index"], payload)
    new_view = BufferView(
        buffer=base_layout["buffer_index"],
        byteOffset=offset,
        byteLength=length,
        target=ELEMENT_ARRAY_BUFFER,
    )
    new_view_index = plan.add_buffer_view(new_view)
    new_accessor = Accessor(
        bufferView=new_view_index,
        byteOffset=0,
        componentType=component_type,
        normalized=False,
        count=count,
        type=SCALAR,
        sparse=None,
        min=[0],
        max=[maximum],
    )
    new_accessor_index = plan.add_accessor(new_accessor)
    plan.index_updates.append((primitive, new_accessor_index))
    return new_accessor_index


def _commit(plan):
    gltf = plan.gltf

    for buffer_index, stage in plan.stages.items():
        stage.buffer.byteLength = len(stage.data)
        if stage.mode == "binary":
            gltf.set_binary_blob(bytes(stage.data))
        elif stage.mode == "datauri":
            encoded = base64.b64encode(bytes(stage.data)).decode("ascii")
            stage.buffer.uri = stage.uri_prefix + encoded
        else:
            with open(stage.path, "wb") as buffer_file:
                buffer_file.write(stage.data)

    for accessor, minimum, maximum in plan.accessor_metadata_updates:
        accessor.min = minimum
        accessor.max = maximum

    gltf.bufferViews.extend(plan.new_buffer_views)
    gltf.accessors.extend(plan.new_accessors)

    for attributes, name, accessor_index in plan.attribute_updates:
        _set_attribute(attributes, name, accessor_index)
    for primitive, accessor_index in plan.index_updates:
        primitive.indices = accessor_index


def reorder_mesh_vertices(
    gltf, mesh_indices=None, permutation=None, **kwargs
):
    """Reorder selected mesh vertices without changing triangle appearance.

    ``permutation[i]`` is the old vertex index stored at new vertex position
    ``i``.  All selected primitives must be triangles and must have the same
    vertex/accessor count.  The operation is atomic with respect to validation
    errors.
    """

    if mesh_indices is None and "mesh_index" in kwargs:
        mesh_indices = kwargs.pop("mesh_index")
    if kwargs:
        raise TypeError(
            "Unexpected keyword arguments: {}".format(", ".join(sorted(kwargs)))
        )
    if mesh_indices is None or permutation is None:
        raise ValueError("mesh_indices and permutation are required")

    selected_meshes = _normalise_mesh_indices(gltf, mesh_indices)
    selected_locations = set()
    selected_primitives = []
    for mesh_index in selected_meshes:
        mesh = gltf.meshes[mesh_index]
        for primitive_index, primitive in enumerate(mesh.primitives):
            selected_locations.add((mesh_index, primitive_index))
            selected_primitives.append((mesh_index, primitive_index, primitive))

    if not selected_primitives:
        raise ValueError("Selected meshes do not contain any primitives")

    vertex_count = None
    vertex_accessor_indices = []
    index_accessor_indices = []
    primitive_entries = []

    for mesh_index, primitive_index, primitive in selected_primitives:
        mode = TRIANGLES if primitive.mode is None else primitive.mode
        if mode != TRIANGLES:
            raise ValueError(
                "Mesh {} primitive {} uses mode {}; only TRIANGLES is supported".format(
                    mesh_index, primitive_index, mode
                )
            )

        entries = _primitive_attribute_entries(primitive)
        base_entries = [
            entry for entry in entries if entry[0] == "attributes"
        ]
        if not base_entries:
            raise ValueError(
                "Mesh {} primitive {} has no vertex attributes".format(
                    mesh_index, primitive_index
                )
            )

        primitive_counts = set()
        for _kind, _container, _name, accessor_index in base_entries:
            accessor = _accessor(
                gltf, accessor_index, "primitive attribute"
            )
            if accessor.count is None:
                raise ValueError(
                    "Mesh {} primitive {} has an attribute accessor without a count".format(
                        mesh_index, primitive_index
                    )
                )
            primitive_counts.add(
                _as_index(accessor.count, "primitive attribute count")
            )
        if len(primitive_counts) != 1:
            raise ValueError(
                "Mesh {} primitive {} has inconsistent attribute counts {}".format(
                    mesh_index, primitive_index, sorted(primitive_counts)
                )
            )

        current_count = next(iter(primitive_counts))
        if vertex_count is None:
            vertex_count = current_count
        elif current_count != vertex_count:
            raise ValueError(
                "Selected primitives have inconsistent vertex counts {} and {}".format(
                    vertex_count, current_count
                )
            )

        for entry in entries:
            accessor_index = _as_index(entry[3], "primitive attribute")
            accessor = _accessor(gltf, accessor_index, "primitive attribute")
            if accessor.count != current_count:
                raise ValueError(
                    "Mesh {} primitive {} attribute {} has count {} but the primitive has {} vertices".format(
                        mesh_index,
                        primitive_index,
                        entry[2],
                        accessor.count,
                        current_count,
                    )
                )
            vertex_accessor_indices.append(accessor_index)

        primitive_entries.append((primitive, entries, base_entries))

        if primitive.indices is None:
            if current_count % 3 != 0:
                raise ValueError(
                    "Non-indexed triangle primitive in mesh {} has vertex count {} not divisible by 3".format(
                        mesh_index, current_count
                    )
                )
        else:
            index_accessor_index = _as_index(
                primitive.indices, "primitive.indices"
            )
            index_accessor = _accessor(
                gltf, index_accessor_index, "primitive.indices"
            )
            if index_accessor.type != SCALAR:
                raise ValueError(
                    "Index accessor {} must have type SCALAR".format(primitive.indices)
                )
            if index_accessor.componentType not in _INDEX_COMPONENT_TYPES:
                raise ValueError(
                    "Index accessor {} uses unsupported componentType {}".format(
                        primitive.indices, index_accessor.componentType
                    )
                )
            if (
                index_accessor.count is None
                or index_accessor.count <= 0
                or index_accessor.count % 3 != 0
            ):
                raise ValueError(
                    "Index accessor {} count must be a positive multiple of 3".format(
                        primitive.indices
                    )
                )
            index_accessor_indices.append(index_accessor_index)

    permutation = _normalise_permutation(permutation, vertex_count)
    inverse_permutation = [0] * vertex_count
    for new_index, old_index in enumerate(permutation):
        inverse_permutation[old_index] = new_index

    seen_vertex_accessors = set()
    vertex_layouts = []
    for accessor_index in vertex_accessor_indices:
        if accessor_index in seen_vertex_accessors:
            continue
        seen_vertex_accessors.add(accessor_index)
        vertex_layouts.append(
            _accessor_layout(gltf, accessor_index, "vertex attribute")
        )

    seen_index_accessors = set()
    index_layouts = []
    for accessor_index in index_accessor_indices:
        if accessor_index in seen_index_accessors:
            continue
        seen_index_accessors.add(accessor_index)
        layout = _accessor_layout(gltf, accessor_index, "index")
        layout["is_index"] = True
        index_layouts.append(layout)

    overlap = seen_vertex_accessors.intersection(seen_index_accessors)
    if overlap:
        raise ValueError(
            "Accessor(s) {} are used as both vertex attributes and indices".format(
                sorted(overlap)
            )
        )

    unselected_accessor_indices = set()
    for mesh_index, mesh in enumerate(gltf.meshes):
        for primitive_index, primitive in enumerate(mesh.primitives):
            if (mesh_index, primitive_index) not in selected_locations:
                unselected_accessor_indices.update(
                    _primitive_accessor_indices(primitive)
                )

    all_selected_layouts = vertex_layouts + index_layouts
    replacements = {}
    shared_accessor_indices = _accessors_sharing_storage(
        gltf, all_selected_layouts, unselected_accessor_indices
    )

    plan = _Plan(gltf)
    for layout in vertex_layouts:
        _reorder_vertex_accessor(
            plan,
            layout,
            permutation,
            replacements if layout["accessor_index"] in shared_accessor_indices else None,
        )
    for layout in index_layouts:
        _rewrite_index_accessor(
            plan,
            layout,
            inverse_permutation,
            replacements if layout["accessor_index"] in shared_accessor_indices else None,
        )

    for primitive, entries, base_entries in primitive_entries:
        for _kind, container, name, accessor_index in entries:
            if accessor_index in replacements:
                plan.attribute_updates.append(
                    (container, name, replacements[accessor_index])
                )
        if primitive.indices is not None:
            if primitive.indices in replacements:
                plan.index_updates.append(
                    (primitive, replacements[primitive.indices])
                )
        else:
            base_layout = _accessor_layout(
                gltf, base_entries[0][3], "non-indexed triangle"
            )
            _generated_index_accessor(
                plan, primitive, base_layout, inverse_permutation
            )

    _commit(plan)
    return gltf
