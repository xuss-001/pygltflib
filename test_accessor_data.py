import base64
import copy
import struct
from pathlib import Path

import pytest

import pygltflib
from pygltflib import (
    BYTE,
    FLOAT,
    MAT2,
    MAT3,
    SCALAR,
    SHORT,
    UNSIGNED_BYTE,
    UNSIGNED_INT,
    UNSIGNED_SHORT,
    VEC2,
    VEC3,
    Accessor,
    AccessorSparseIndices,
    AccessorSparseValues,
    Buffer,
    BufferView,
    GLTF2,
    Sparse,
)


def _data_uri(data):
    return "data:application/octet-stream;base64," + base64.b64encode(data).decode()


def _single_accessor(data, accessor, **buffer_view_kwargs):
    buffer_view_kwargs.setdefault("buffer", 0)
    buffer_view_kwargs.setdefault("byteOffset", 0)
    buffer_view_kwargs.setdefault("byteLength", len(data))
    return GLTF2(
        buffers=[Buffer(byteLength=len(data), uri=_data_uri(data))],
        bufferViews=[BufferView(**buffer_view_kwargs)],
        accessors=[accessor],
    )


@pytest.mark.parametrize(
    "component_type, values, fmt",
    [
        (BYTE, [-128, -1, 0, 127], "b"),
        (UNSIGNED_BYTE, [0, 1, 255], "B"),
        (SHORT, [-32768, -1, 32767], "h"),
        (UNSIGNED_SHORT, [0, 1, 65535], "H"),
        (UNSIGNED_INT, [0, 1, 4_000_000_000], "I"),
        (FLOAT, [-1.5, 0.0, 3.25], "f"),
    ],
)
def test_all_component_types(component_type, values, fmt):
    data = struct.pack("<" + fmt * len(values), *values)
    accessor = Accessor(
        bufferView=0,
        componentType=component_type,
        count=len(values),
        type=SCALAR,
    )
    gltf = _single_accessor(data, accessor)
    assert gltf.get_accessor_data(0) == [(value,) for value in values]


def test_accessor_and_buffer_view_offsets():
    data = b"\xaa" * 4 + struct.pack("<ffffff", 1, 2, 3, 4, 5, 6) + b"\xbb"
    accessor = Accessor(
        bufferView=0,
        byteOffset=0,
        componentType=FLOAT,
        count=2,
        type=VEC3,
    )
    gltf = _single_accessor(
        data,
        accessor,
        byteOffset=4,
        byteLength=24,
    )
    assert gltf.get_accessor_data(0) == [(1, 2, 3), (4, 5, 6)]


def test_accessor_byte_offset():
    data = b"\xaa" * 4 + struct.pack("<ffffff", 1, 2, 3, 4, 5, 6)
    accessor = Accessor(
        bufferView=0,
        byteOffset=4,
        componentType=FLOAT,
        count=2,
        type=VEC3,
    )
    gltf = _single_accessor(data, accessor)
    assert gltf.get_accessor_data(0) == [(1, 2, 3), (4, 5, 6)]


@pytest.mark.parametrize(
    "accessor_type, component_type, fmt, columns, rows",
    [
        (MAT2, UNSIGNED_BYTE, "B", 2, 2),
        (MAT3, UNSIGNED_BYTE, "B", 3, 3),
        (MAT3, UNSIGNED_SHORT, "H", 3, 3),
    ],
)
def test_matrix_column_alignment_padding(
    accessor_type, component_type, fmt, columns, rows
):
    size = struct.calcsize(fmt)
    column_stride = ((rows * size + 3) // 4) * 4
    expected = list(range(1, columns * rows + 1))
    chunks = []
    value = 1
    for _column in range(columns):
        column = b""
        for _row in range(rows):
            column += struct.pack("<" + fmt, value)
            value += 1
        column += b"\xff" * (column_stride - len(column))
        chunks.append(column)
    data = b"".join(chunks)

    accessor = Accessor(
        bufferView=0,
        componentType=component_type,
        count=1,
        type=accessor_type,
    )
    gltf = _single_accessor(data, accessor)
    assert gltf.get_accessor_data(0) == [expected]


def test_interleaved_stride_with_accessor_offset():
    # Two VEC2 unsigned shorts at stride 12, each stored at offset 4.
    data = (
        b"\x00\x00\x00\x00"
        + struct.pack("<HH", 1, 2)
        + b"\x00\x00\x00\x00"
        + b"\x00\x00\x00\x00"
        + struct.pack("<HH", 3, 4)
        + b"\x00\x00\x00\x00"
    )
    accessor = Accessor(
        bufferView=0,
        byteOffset=4,
        componentType=UNSIGNED_SHORT,
        count=2,
        type=VEC2,
    )
    gltf = _single_accessor(
        data,
        accessor,
        byteStride=12,
    )
    assert gltf.get_accessor_data(0) == [(1, 2), (3, 4)]


def _sparse_gltf(base_count=4, base_values=None, sparse_indices=(1, 3), sparse_values=None):
    components = []
    if base_values is not None:
        for value in base_values:
            components.extend(value)
    base_data = struct.pack("<" + "f" * len(components), *components)

    index_data = struct.pack("<HH", *sparse_indices)
    sparse_value_data = struct.pack(
        "<" + "f" * (len(sparse_values) * 3),
        *(component for value in sparse_values for component in value),
    )

    data = base_data + index_data + sparse_value_data
    buffer_views = [
        BufferView(
            buffer=0,
            byteOffset=0,
            byteLength=len(base_data),
        ),
        BufferView(
            buffer=0,
            byteOffset=len(base_data),
            byteLength=len(index_data),
        ),
        BufferView(
            buffer=0,
            byteOffset=len(base_data) + len(index_data),
            byteLength=len(sparse_value_data),
        ),
    ]
    sparse = Sparse(
        count=len(sparse_indices),
        indices=AccessorSparseIndices(
            bufferView=1,
            byteOffset=0,
            componentType=UNSIGNED_SHORT,
        ),
        values=AccessorSparseValues(bufferView=2, byteOffset=0),
    )
    accessor = Accessor(
        bufferView=0 if base_values is not None else None,
        componentType=FLOAT,
        count=base_count,
        type=VEC3,
        sparse=sparse,
    )
    return GLTF2(
        buffers=[Buffer(byteLength=len(data), uri=_data_uri(data))],
        bufferViews=buffer_views,
        accessors=[accessor],
    )


def test_sparse_overrides_base_values():
    gltf = _sparse_gltf(
        base_values=[
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            (2.0, 2.0, 2.0),
            (3.0, 3.0, 3.0),
        ],
        sparse_values=[(10.0, 11.0, 12.0), (30.0, 31.0, 32.0)],
    )
    assert gltf.get_accessor_data(0) == [
        (0.0, 0.0, 0.0),
        (10.0, 11.0, 12.0),
        (2.0, 2.0, 2.0),
        (30.0, 31.0, 32.0),
    ]


def test_sparse_without_base_buffer_view_uses_zero():
    gltf = _sparse_gltf(
        base_values=None,
        sparse_values=[(10.0, 11.0, 12.0), (30.0, 31.0, 32.0)],
    )
    assert gltf.get_accessor_data(0) == [
        (0.0, 0.0, 0.0),
        (10.0, 11.0, 12.0),
        (0.0, 0.0, 0.0),
        (30.0, 31.0, 32.0),
    ]


def test_signed_normalization():
    values = [-128, -127, -1, 0, 1, 127]
    data = struct.pack("<" + "b" * len(values), *values)
    accessor = Accessor(
        bufferView=0,
        componentType=BYTE,
        normalized=True,
        count=len(values),
        type=SCALAR,
    )
    gltf = _single_accessor(data, accessor)

    assert gltf.get_accessor_data(0) == [(value,) for value in values]
    assert gltf.get_accessor_data(0, normalized=True) == [
        (-1.0,),
        (-1.0,),
        (-1.0 / 127.0,),
        (0.0,),
        (1.0 / 127.0,),
        (1.0,),
    ]


def test_short_normalization_clamps_minimum():
    values = [-32768, -32767, 0, 32767]
    data = struct.pack("<" + "h" * len(values), *values)
    accessor = Accessor(
        bufferView=0,
        componentType=SHORT,
        normalized=True,
        count=len(values),
        type=SCALAR,
    )
    gltf = _single_accessor(data, accessor)
    result = gltf.get_accessor_data(0, normalize=True)
    assert result[0] == -1.0
    assert result[1] == pytest.approx(-1.0)
    assert result[2] == 0.0
    assert result[3] == 1.0


def test_glb_source_and_immutability():
    values = struct.pack("<ffffff", 1, 2, 3, 4, 5, 6)
    gltf = GLTF2(
        buffers=[Buffer(byteLength=len(values))],
        bufferViews=[BufferView(buffer=0, byteOffset=0, byteLength=len(values))],
        accessors=[
            Accessor(
                bufferView=0,
                componentType=FLOAT,
                count=2,
                type=VEC3,
            )
        ],
    )
    gltf.set_binary_blob(values)
    glb = b"".join(gltf.save_to_bytes())

    loaded = GLTF2.load_from_bytes(glb)
    before_object = copy.deepcopy(loaded)
    before_blob = bytes(loaded.binary_blob())
    assert loaded.get_accessor_data(0) == [(1, 2, 3), (4, 5, 6)]
    assert loaded == before_object
    assert loaded.binary_blob() == before_blob


def test_local_buffer_file_source(tmp_path):
    values = struct.pack("<HH", 7, 9)
    buffer_path = Path(tmp_path) / "values.bin"
    buffer_path.write_bytes(values)
    gltf_path = Path(tmp_path) / "asset.gltf"
    gltf_path.write_text("{}")

    gltf = GLTF2(
        buffers=[Buffer(byteLength=len(values), uri="values.bin")],
        bufferViews=[BufferView(buffer=0, byteOffset=0, byteLength=len(values))],
        accessors=[
            Accessor(
                bufferView=0,
                componentType=UNSIGNED_SHORT,
                count=2,
                type=SCALAR,
            )
        ],
    )
    loaded = GLTF2.load(gltf_path)
    loaded.buffers = gltf.buffers
    loaded.bufferViews = gltf.bufferViews
    loaded.accessors = gltf.accessors

    before_file = buffer_path.read_bytes()
    assert loaded.get_accessor_data(0) == [(7,), (9,)]
    assert buffer_path.read_bytes() == before_file


def test_explicit_errors():
    data = struct.pack("<HH", 1, 2)
    accessor = Accessor(
        bufferView=0,
        componentType=UNSIGNED_SHORT,
        count=2,
        type=SCALAR,
    )
    gltf = _single_accessor(data, accessor)

    with pytest.raises(IndexError):
        gltf.get_accessor_data(99)

    gltf.accessors[0].count = 3
    with pytest.raises(ValueError):
        gltf.get_accessor_data(0)

    gltf.accessors[0].count = 2
    gltf.bufferViews[0].byteLength = len(data) + 1
    with pytest.raises(ValueError):
        gltf.get_accessor_data(0)


@pytest.mark.parametrize("stride", [2, 3, 6, 256])
def test_invalid_stride(stride):
    data = b"\x00" * 64
    accessor = Accessor(
        bufferView=0,
        componentType=UNSIGNED_SHORT,
        count=2,
        type=SCALAR,
    )
    gltf = _single_accessor(data, accessor, byteStride=stride)
    with pytest.raises(ValueError):
        gltf.get_accessor_data(0)


def test_stride_smaller_than_element():
    data = b"\x00" * 64
    accessor = Accessor(
        bufferView=0,
        componentType=FLOAT,
        count=2,
        type=VEC3,
    )
    gltf = _single_accessor(data, accessor, byteStride=4)
    with pytest.raises(ValueError):
        gltf.get_accessor_data(0)


def test_invalid_sparse_index():
    gltf = _sparse_gltf(
        base_values=[
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            (2.0, 2.0, 2.0),
            (3.0, 3.0, 3.0),
        ],
        sparse_indices=(1, 4),
        sparse_values=[(10.0, 11.0, 12.0), (40.0, 41.0, 42.0)],
    )
    with pytest.raises(ValueError):
        gltf.get_accessor_data(0)


def test_package_level_function_and_aliases():
    data = struct.pack("<HH", 5, 6)
    accessor = Accessor(
        bufferView=0,
        componentType=UNSIGNED_SHORT,
        count=2,
        type=SCALAR,
    )
    gltf = _single_accessor(data, accessor)
    assert pygltflib.get_accessor_data(gltf, 0) == [(5,), (6,)]
    assert pygltflib.get_accessor_data(0, gltf) == [(5,), (6,)]
    assert pygltflib.get_accessor_data(gltf, accessor) == [(5,), (6,)]
    assert gltf.read_accessor(0) == [(5,), (6,)]
    assert gltf.get_accessor_values(0) == [(5,), (6,)]
