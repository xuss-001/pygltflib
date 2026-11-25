"""Reorder mesh vertices while preserving triangle data and winding.

Run from the repository root:

    python examples/reorder_mesh_vertices.py
"""

import base64
from pathlib import Path
import struct
import sys
import tempfile


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pygltflib


POSITIONS = [
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (2.0, 0.0, 0.0),
    (3.0, 0.0, 0.0),
    (2.0, 1.0, 0.0),
]
COLORS = [
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 1.0, 0.0),
    (1.0, 0.0, 1.0),
    (0.0, 1.0, 1.0),
]
UVS = [
    (0.0, 0.0),
    (1.0, 0.0),
    (0.0, 1.0),
    (0.5, 0.25),
    (0.75, 0.5),
    (0.25, 0.75),
]
MORPH_DELTAS = [
    (0.0, 0.0, 0.0),
    (0.1, 0.0, 0.0),
    (0.0, 0.1, 0.0),
    (0.0, 0.0, 0.1),
    (0.1, 0.1, 0.0),
    (0.0, 0.1, 0.1),
]
INDICES = [0, 1, 2, 3, 4, 5]
PERMUTATION = [5, 2, 0, 4, 1, 3]


def build_gltf():
    vertex_data = bytearray()
    for position, color, uv in zip(POSITIONS, COLORS, UVS):
        vertex_data.extend(struct.pack("<3f3f2f", *(position + color + uv)))

    morph_data = b"".join(struct.pack("<3f", *delta) for delta in MORPH_DELTAS)
    index_data = struct.pack("<6H", *INDICES)
    index_offset = len(vertex_data) + len(morph_data)
    blob = bytes(vertex_data) + morph_data + index_data

    gltf = pygltflib.GLTF2(
        buffers=[
            pygltflib.Buffer(
                uri="data:application/octet-stream;base64,"
                + base64.b64encode(blob).decode("ascii"),
                byteLength=len(blob),
            )
        ],
        bufferViews=[
            pygltflib.BufferView(
                buffer=0,
                byteOffset=0,
                byteLength=len(vertex_data),
                byteStride=32,
                target=pygltflib.ARRAY_BUFFER,
            ),
            pygltflib.BufferView(
                buffer=0,
                byteOffset=len(vertex_data),
                byteLength=len(morph_data),
                target=pygltflib.ARRAY_BUFFER,
            ),
            pygltflib.BufferView(
                buffer=0,
                byteOffset=index_offset,
                byteLength=len(index_data),
                target=pygltflib.ELEMENT_ARRAY_BUFFER,
            ),
        ],
        accessors=[
            pygltflib.Accessor(
                bufferView=0,
                byteOffset=0,
                componentType=pygltflib.FLOAT,
                count=len(POSITIONS),
                type=pygltflib.VEC3,
            ),
            pygltflib.Accessor(
                bufferView=0,
                byteOffset=12,
                componentType=pygltflib.FLOAT,
                count=len(COLORS),
                type=pygltflib.VEC3,
            ),
            pygltflib.Accessor(
                bufferView=0,
                byteOffset=24,
                componentType=pygltflib.FLOAT,
                count=len(UVS),
                type=pygltflib.VEC2,
            ),
            pygltflib.Accessor(
                bufferView=1,
                componentType=pygltflib.FLOAT,
                count=len(MORPH_DELTAS),
                type=pygltflib.VEC3,
            ),
            pygltflib.Accessor(
                bufferView=2,
                componentType=pygltflib.UNSIGNED_SHORT,
                count=len(INDICES),
                type=pygltflib.SCALAR,
                min=[min(INDICES)],
                max=[max(INDICES)],
            ),
        ],
        meshes=[
            pygltflib.Mesh(
                primitives=[
                    pygltflib.Primitive(
                        attributes=pygltflib.Attributes(
                            POSITION=0,
                            COLOR_0=1,
                            TEXCOORD_0=2,
                        ),
                        indices=4,
                        targets=[pygltflib.Attributes(POSITION=3)],
                    )
                ]
            )
        ],
    )
    return gltf


def read_accessor(gltf, accessor_index):
    accessor = gltf.accessors[accessor_index]
    buffer_view = gltf.bufferViews[accessor.bufferView]
    data = gltf.get_data_from_buffer_uri(gltf.buffers[buffer_view.buffer].uri)
    component_count = {
        pygltflib.SCALAR: 1,
        pygltflib.VEC2: 2,
        pygltflib.VEC3: 3,
    }[accessor.type]
    element_size = 4 * component_count
    stride = buffer_view.byteStride or element_size
    start = (buffer_view.byteOffset or 0) + (accessor.byteOffset or 0)
    unpack_format = "<" + "f" * component_count
    return [
        struct.unpack_from(unpack_format, data, start + index * stride)
        for index in range(accessor.count)
    ]


def read_indices(gltf, primitive):
    accessor = gltf.accessors[primitive.indices]
    buffer_view = gltf.bufferViews[accessor.bufferView]
    data = gltf.get_data_from_buffer_uri(gltf.buffers[buffer_view.buffer].uri)
    component_format = {
        pygltflib.UNSIGNED_BYTE: "B",
        pygltflib.UNSIGNED_SHORT: "H",
        pygltflib.UNSIGNED_INT: "I",
    }[accessor.componentType]
    stride = buffer_view.byteStride or struct.calcsize("<" + component_format)
    start = (buffer_view.byteOffset or 0) + (accessor.byteOffset or 0)
    return [
        struct.unpack_from("<" + component_format, data, start + index * stride)[0]
        for index in range(accessor.count)
    ]


def cross(first, second, third):
    edge1 = tuple(second[i] - first[i] for i in range(3))
    edge2 = tuple(third[i] - first[i] for i in range(3))
    return (
        edge1[1] * edge2[2] - edge1[2] * edge2[1],
        edge1[2] * edge2[0] - edge1[0] * edge2[2],
        edge1[0] * edge2[1] - edge1[1] * edge2[0],
    )


def triangle_snapshot(gltf):
    primitive = gltf.meshes[0].primitives[0]
    positions = read_accessor(gltf, primitive.attributes.POSITION)
    colors = read_accessor(gltf, primitive.attributes.COLOR_0)
    uvs = read_accessor(gltf, primitive.attributes.TEXCOORD_0)
    indices = read_indices(gltf, primitive)
    triangles = []
    for start in range(0, len(indices), 3):
        triangle_indices = indices[start : start + 3]
        triangle_positions = [positions[index] for index in triangle_indices]
        triangles.append(
            {
                "positions": triangle_positions,
                "colors": [colors[index] for index in triangle_indices],
                "uvs": [uvs[index] for index in triangle_indices],
                "normal": cross(*triangle_positions),
            }
        )
    return triangles


def main():
    gltf = build_gltf()
    before = triangle_snapshot(gltf)

    gltf.reorder_mesh_vertices(mesh_indices=0, permutation=PERMUTATION)

    output = Path("reordered_vertices.glb")
    gltf.save(output)
    reloaded = pygltflib.GLTF2().load(output)
    after = triangle_snapshot(reloaded)

    if after != before:
        raise SystemExit("Triangle data or winding changed after reorder")

    for triangle_index, triangle in enumerate(after):
        print(
            "triangle {}: positions={} colors={} uvs={} normal={}".format(
                triangle_index,
                triangle["positions"],
                triangle["colors"],
                triangle["uvs"],
                triangle["normal"],
            )
        )
    print("Saved and verified {}".format(output.resolve()))


if __name__ == "__main__":
    main()
