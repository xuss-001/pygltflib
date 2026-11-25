"""
pygltflib : A Python library for reading, writing and handling GLTF files.


Copyright (c) 2018,2021 Luke Miller

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import base64
import copy
from dataclasses import (
    _is_dataclass_instance,
    dataclass,
    field,
    fields,
)
from datetime import date, datetime
from enum import Enum
import json
import math
import mimetypes
import numbers
import os
from pathlib import Path
from shutil import copyfile
import tempfile
from typing import Any, Dict, List
from typing import Callable, Optional, Tuple, TypeVar, Union
from urllib.parse import unquote
import struct
import warnings

from dataclasses_json import dataclass_json as dataclass_json
from dataclasses_json.core import _decode_dataclass
from dataclasses_json.core import _ExtendedEncoder as JsonEncoder

from .types import *

__version__ = "1.15.3"

"""
About the GLTF2 file format:

glTF uses a right-handed coordinate system, that is, the cross product of +X and +Y yields +Z. glTF defines +Y as up.
The front of a glTF asset faces +Z.
The units for all linear distances are meters.
All angles are in radians.
Positive rotation is counterclockwise.
"""


@dataclass
class GLTF2(Property):
    accessors: List[Accessor] = field(default_factory=list)
    animations: List[Animation] = field(default_factory=list)
    asset: Asset = field(default_factory=Asset)  # required
    bufferViews: List[BufferView] = field(default_factory=list)
    buffers: List[Buffer] = field(default_factory=list)
    cameras: List[Camera] = field(default_factory=list)
    extensionsUsed: List[str] = field(default_factory=list)
    extensionsRequired: List[str] = field(default_factory=list)
    images: List[Image] = field(default_factory=list)
    materials: List[Material] = field(default_factory=list)
    meshes: List[Mesh] = field(default_factory=list)
    nodes: List[Node] = field(default_factory=list)
    samplers: List[Sampler] = field(default_factory=list)
    scene: int = None
    scenes: List[Scene] = field(default_factory=list)
    skins: List[Skin] = field(default_factory=list)
    textures: List[Texture] = field(default_factory=list)

    #    _glb_data: Any = None
    #    _path: Any = None

    def binary_blob(self):
        """Get the binary blob associated with glb files if available

        Returns
            (bytes): binary data
        """
        return getattr(self, "_glb_data", None)

    def set_binary_blob(self, blob):
        setattr(self, "_glb_data", blob)

    def destroy_binary_blob(self):
        if hasattr(self, "_glb_data"):
            setattr(self, "_glb_data", None)

    def load_file_uri(self, uri):
        """
        Loads a file pointed to by a uri
        """
        path = getattr(self, "_path", Path())
        with open(Path(path, uri), "rb") as fb:
            data = fb.read()
        return data

    @staticmethod
    def decode_data_uri(uri, header=None):
        """
        Decodes the binary portion of a data uri.
        """
        data = uri.split(header or DATA_URI_HEADER)[1]
        data = base64.decodebytes(bytes(data, "utf8"))
        return data

    def identify_uri(self, uri):
        """
        Identify the format of the requested buffer. File, data or binary blob.

        Returns
            buffer_type (str)
        """
        path = getattr(self, "_path", Path())
        uri_format = None

        if uri is None:  # assume loaded from glb binary file
            uri_format = BufferFormat.BINARYBLOB
            if len(self.buffers) > 1:
                warnings.warn(
                    "GLTF has multiple buffers but only one buffer binary blob, pygltflib might corrupt data."
                    "Please open an issue at https://gitlab.com/dodgyville/pygltflib/issues"
                )
        elif uri.startswith("data"):
            uri_format = BufferFormat.DATAURI
        elif Path(path, uri).is_file():
            uri_format = BufferFormat.BINFILE
        else:
            warnings.warn(
                "pygltf.GLTF.identify_buffer_format can not identify buffer."
                "Please open an issue at https://gitlab.com/dodgyville/pygltflib/issues"
            )
        return uri_format

    def get_data_from_buffer_uri(self, uri):
        """
        No matter how the buffer data is stored (the uri may be a long string, a file name or imply
        a binary blob), strip off any headers and do any conversions are return a universal binary
        blob for manipulation.
        """
        current_buffer_format = self.identify_uri(uri)

        if current_buffer_format == BufferFormat.BINFILE:
            data = self.load_file_uri(uri)
        elif current_buffer_format == BufferFormat.DATAURI:
            data = self.decode_data_uri(uri)
        elif current_buffer_format == BufferFormat.BINARYBLOB:
            data = self.binary_blob()
        else:
            return None

        return data

    def append_to_buffer(self, array_bytes):
        buf_bytes = bytearray(self.binary_blob())
        old_buf_len = len(buf_bytes)

        # alignment
        itemsize = 4
        pad_num = (itemsize - len(array_bytes) % itemsize) % itemsize
        buf_bytes += array_bytes + ("0" * pad_num).encode()
        self.set_binary_blob(bytes(buf_bytes))

        return old_buf_len, len(array_bytes)

    def create_bufferView(self, new_attr_buffer):
        byte_offset, byte_len = self.append_to_buffer(
            new_attr_buffer
        )
        new_buffer_view = BufferView(
            buffer=0, byteOffset=byte_offset, byteLength=byte_len
        )
        self.bufferViews.append(new_buffer_view)
        return len(self.bufferViews) - 1

    def replace_accessor_data(self, accessor_index, data):
        """Replace all values in an accessor while preserving the rest of the GLTF.

        The accessor's byte offset, bufferView byte stride, normalized integer
        conversion and sparse overrides are taken into account.  ``data`` must
        have shape ``(accessor.count, component_count)``.  For SCALAR accessors
        a flat sequence of values is also accepted.

        Normalized integer accessors take their input as the normalized value,
        for example ``0.5`` or ``1.0``, rather than as the raw integer payload.
        Values outside the component's representable range raise
        :class:`AccessorDataError`.

        ``accessor.min`` and ``accessor.max`` are updated to the values stored
        in the buffer, as required by glTF; for normalized integers these are
        therefore the raw component values (for example 255, not 1.0).

        The replacement is atomic with respect to validation failures: if the
        shape, a value or a source data reference is invalid, neither the GLTF
        object nor its binary buffer is changed.
        """
        _replace_accessor_data(self, accessor_index, data)

    set_accessor_data = replace_accessor_data
    update_accessor_data = replace_accessor_data

    # noinspection PyPep8Naming
    def remove_bufferView(self, buffer_view_id):
        """
        Remove a bufferView and update all the bufferView pointers in the GLTF object
        """
        bufferView = self.bufferViews.pop(buffer_view_id)

        def update_obj(title, obj):

            if not obj:
                print(f"{obj} empty")

            if not obj.bufferView:
                return

            if obj.bufferView == buffer_view_id:
                warnings.warn(
                    f"Removing bufferView {buffer_view_id} but "
                    f"{title}.bufferView still points to it. This may corrupt the GLTF."
                )
            if obj.bufferView >= buffer_view_id:
                obj.bufferView -= 1

        for i, accessor in enumerate(self.accessors):
            update_obj(f"gltf.accessors[{i}]", accessor)
            if accessor and accessor.sparse:
                update_obj(
                    f"gltf.accessors[{i}].sparse.indices", accessor.sparse.indices
                )
                update_obj(f"gltf.accessors[{i}].sparse.values", accessor.sparse.values)

        for i, obj in enumerate(self.images):
            update_obj(f"gltf.images[{i}]", obj)

        def min_tuple(iterable):
            return min(enumerate(iterable), key=lambda x: x[1][1])[1]

        other_buffers = [(x, x.byteOffset) for x in self.bufferViews if x.byteOffset > bufferView.byteOffset]
        if (len(other_buffers) == 0):
            # last buffer
            # shift = bufferView.byteLength
            alignment = 4
            shift = ((bufferView.byteLength + alignment - 1) // alignment) * alignment
        else: 
            nextBuffer = min_tuple(other_buffers)[0]
            shift = nextBuffer.byteOffset - bufferView.byteOffset

        for i, cur_buffer_view in enumerate(self.bufferViews):
            if cur_buffer_view.byteOffset > bufferView.byteOffset:
                cur_buffer_view.byteOffset -= shift

                if(cur_buffer_view.byteOffset % 4) != 0:
                    warnings.warn(f"byteOffset is not dividable by 4")
                    raise Exception("Something is wrong with the library.")

            if cur_buffer_view.byteStride is not None:
                warnings.warn(f"ByteStride is not None, may cause issues in GLTF")

        # self.convert_buffers(BufferFormat.BINARYBLOB)
        data: bytes = self.binary_blob()

        # Shrink the buffer
        data = (
            data[: bufferView.byteOffset]
            + data[bufferView.byteOffset + shift :]
        )
        self.set_binary_blob(data)

        # Update meta
        self.buffers[0].byteLength = len(data)
        # self.convert_buffers(BufferFormat.DATAURI)

        return bufferView

    def export_datauri_as_image_file(
        self, data_uri, name, destination, override=False, index=0
    ):
        """convert data uri to image file
        If destination is full path and file name, use that.
        If destination is just a directory, use the name of the data_uri
        """
        header, encoded = data_uri.split(",", 1)
        mime = header.split(":")[1].split(";")[0]
        if name:  # use image.name
            file_name = name
        else:
            extension = mimetypes.guess_extension(mime)
            file_name = f"{index}{extension}"
        destination = Path(destination)
        if destination.is_dir():
            image_path = destination / unquote(file_name)
        else:  # assume filepath
            image_path = destination
        if image_path.is_file() and not override:
            warnings.warn(
                f"Unable to write image file, a file already exists at {image_path}"
            )
            return None
        data = base64.b64decode(encoded)

        with open(image_path, "wb") as image_file:
            image_file.write(data)
        return file_name

    def export_fileuri_as_image_file(self, file_uri, destination, override=False):
        """Export file uri as another image file (ie copy out of GLTF into own location)"""
        path = getattr(self, "_path", Path())
        image_path = Path(path / unquote(file_uri))
        if not image_path.exists():
            warnings.warn(f"Unable to find image {image_path} for export.")
            return None
        if image_path.is_file() and not override:
            warnings.warn(
                f"Unable to write image file, a file already exists at {image_path}"
            )
            return None

        copyfile(image_path, destination)
        return file_uri

    def convert_images(self, image_format, path=None):
        """
        GLTF files can store the image data in three different formats: In the buffers, as a data
        uri string and as external images files. This converts the images between the formats.

        image_format (ImageFormat.ENUM): Destination format to convert images
        path (str|Path): Path to the directory to use for loading or saving images
        override (bool): Override an image file if it already exists and is about to be replaced

        """
        if path is None:
            path = getattr(self, "_path", Path())
        else:
            path = Path(path)

        target_format = image_format

        for image_index, image in enumerate(self.images):

            if image.current_format == target_format:
                continue

            image_bytes = image_get_bytes(image, self, path)

            #############################
            # Clean up old representation
            #############################
            if image.current_format == ImageFormat.BUFFERVIEW:
                # self.convert_buffers(BufferFormat.BINARYBLOB)

                # data: bytes = self.binary_blob()
                # bufferView = self.bufferViews[image.bufferView]

                # # Shrink the buffer
                # data = (
                #     data[: bufferView.byteOffset]
                #     + data[bufferView.byteOffset + bufferView.byteLength :]
                # )
                # self.set_binary_blob(data)

                # # Update meta
                # self.buffers[0].byteLength = len(data)

                # Remove the bufferView
                idx = image.bufferView
                image.bufferView = None

                self.remove_bufferView(idx)

                # self.convert_buffers(BufferFormat.DATAURI)

            image.uri = None

            ########################
            # Convert to new format
            ########################
            if target_format == ImageFormat.DATAURI:  # convert to data uri
                encoded_string = str(base64.b64encode(image_bytes).decode("utf-8"))
                image.name = copy.copy(image.uri) if not image.name else image.name
                image.uri = f"data:{image.mimeType};base64,{encoded_string}"
            elif target_format == ImageFormat.BUFFERVIEW:
                # self.convert_buffers(BufferFormat.BINARYBLOB)

                # Add to buffer
                data = self.binary_blob()
                byte_offset, byte_len = len(data), len(image_bytes)
                pad_num = 4 - len(image_bytes) % 4
                data = data + image_bytes + ("0" * pad_num).encode()
                self.set_binary_blob(data)

                # Update meta
                self.buffers[0].byteLength = byte_offset + byte_len + pad_num

                # Create buffer view
                new_buffer_view = BufferView(
                    buffer=0, byteOffset=byte_offset, byteLength=byte_len
                )
                self.bufferViews.append(new_buffer_view)

                image.bufferView = len(self.bufferViews) - 1

                # Back to DATAURI format
                # self.convert_buffers(BufferFormat.DATAURI)

            elif target_format == ImageFormat.FILE:  # convert to images

                extension = mimetypes.guess_extension(image.mimeType)
                file_name = f"{image.name}{extension}"
                image_path = path / file_name

                with open(image_path, "wb") as f:
                    f.write(image_bytes)

                if file_name:  # replace data uri with pointer to file
                    image.uri = file_name

            else:
                raise Exception(f"{image_format=} incorrect argument")

    def convert_buffers(self, buffer_format, override=True, path=None):
        """
        GLTF files can store the buffer data in three different formats: As a binary blob ready for glb, as a data
        uri string and as external bin files. This converts the buffers between the formats.

        buffer_format (BufferFormat.ENUM)
        override (bool): Override a bin file if it already exists and is about to be replaced
        """

        if path is not None:
            path = Path(path)
        else:
            path: Path = getattr(self, "_path", Path())

        for i, buffer in enumerate(self.buffers):
            current_buffer_format = self.identify_uri(buffer.uri)
            if current_buffer_format == buffer_format:  # already in the format
                continue
            if current_buffer_format == BufferFormat.BINFILE:
                warnings.warn(
                    f"Conversion will leave {buffer.uri} file orphaned since data is now in the GLTF object."
                )
            data = self.get_data_from_buffer_uri(buffer.uri)

            if not data:
                return

            self.destroy_binary_blob()  # free up any binary blob floating around

            if buffer_format == BufferFormat.BINARYBLOB:
                if len(self.buffers) > 1:
                    warnings.warn(
                        "pygltflib currently unable to convert multiple buffers to a single binary blob."
                        "Please open an issue at https://gitlab.com/dodgyville/pygltflib/issues"
                    )
                    return
                self.set_binary_blob(data)
                buffer.uri = None
            elif buffer_format == BufferFormat.DATAURI:
                # convert buffer to a data uri
                data = base64.b64encode(data).decode("utf-8")
                buffer.uri = f"{DATA_URI_HEADER}{data}"
            elif buffer_format == BufferFormat.BINFILE:
                filename = Path(f"{i}").with_suffix(".bin")
                binfile_path = path / filename
                if binfile_path.is_file() and not override:
                    warnings.warn(
                        f"Unable to write buffer file, a file already exists at {binfile_path}"
                    )
                    continue
                with open(binfile_path, "wb") as f:  # save bin file with the gltf file
                    f.write(data)
                buffer.uri = str(filename)

    def to_json(
        self,
        *,
        skipkeys: bool = False,
        ensure_ascii: bool = True,
        check_circular: bool = True,
        allow_nan: bool = True,
        indent: Optional[Union[int, str]] = None,
        separators: Tuple[str, str] = None,
        default: Callable = None,
        sort_keys: bool = False,
        **kw,
    ) -> str:
        """
        to_json and from_json from dataclasses_json
        courtesy https://github.com/lidatong/dataclasses-json
        """

        data = gltf_asdict(self)
        data = delete_empty_keys(data)
        return json.dumps(
            data,
            cls=JsonEncoder,
            skipkeys=skipkeys,
            ensure_ascii=ensure_ascii,
            check_circular=check_circular,
            allow_nan=allow_nan,
            indent=indent,
            separators=separators,
            default=default,
            sort_keys=sort_keys,
            **kw,
        )

    @classmethod
    def from_json(
        cls: A,
        s: str,
        *,
        parse_float=None,
        parse_int=None,
        parse_constant=None,
        infer_missing=False,
        **kw,
    ) -> A:
        init_kwargs = json.loads(
            s,
            parse_float=parse_float,
            parse_int=parse_int,
            parse_constant=parse_constant,
            **kw,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = _decode_dataclass(cls, init_kwargs, infer_missing)  # type: GLTF2
        for mesh in result.meshes:
            for primitive in mesh.primitives:
                raw_attributes = primitive.attributes
                if raw_attributes:
                    attributes = Attributes(**raw_attributes)
                    primitive.attributes = attributes
        return result

    def gltf_to_json(self, separators=None, indent="  ") -> str:
        return self.to_json(
            default=json_serial,
            indent=indent,
            allow_nan=False,
            skipkeys=True,
            separators=separators,
        )

    def save_json(self, fname):
        path = Path(fname)
        original_buffers = copy.deepcopy(self.buffers)
        for i, buffer in enumerate(self.buffers):
            if buffer.uri is None:  # save glb_data as bin file
                # update the buffer uri to point to our new local bin file
                glb_data = self.binary_blob()
                if glb_data:
                    buffer.uri = str(Path(path.stem).with_suffix(".bin"))
                    with open(
                        path.with_suffix(".bin"), "wb"
                    ) as f:  # save bin file with the gltf file
                        f.write(glb_data)
                else:
                    warnings.warn(f"buffer {i} is empty: {buffer}")

        with open(path, "w") as f:
            f.write(self.gltf_to_json())

        self.buffers = original_buffers  # restore buffers
        return True

    def buffers_to_binary_blob(self):
        """Flatten all buffers into a single buffer"""
        buffer_blob = bytearray()

        offset = 0
        path = getattr(self, "_path", Path())
        binary_blob: Optional[bytearray] = None

        for i, bufferView in enumerate(self.bufferViews):
            buffer = self.buffers[bufferView.buffer]
            if buffer.uri is None:  # assume loaded from glb binary file
                if binary_blob is None:
                    binary_blob = self.binary_blob()
                data = binary_blob
            elif buffer.uri.startswith("data"):
                data = self.decode_data_uri(buffer.uri)
            elif Path(path, buffer.uri).is_file():
                with open(Path(path, buffer.uri), "rb") as fb:
                    data = fb.read()
            else:
                warnings.warn(
                    f"Unable to save bufferView {buffer.uri[:20]} to glb, skipping. "
                    "Please open an issue at https://gitlab.com/dodgyville/pygltflib/issues"
                )
                continue
            byte_offset = (
                bufferView.byteOffset if bufferView.byteOffset is not None else 0
            )
            byte_length = bufferView.byteLength
            if byte_length % 4 != 0:  # pad each segment of binary blob
                byte_length += 4 - byte_length % 4

            buffer_blob += data[byte_offset : byte_offset + byte_length]

            bufferView.byteOffset = offset
            bufferView.byteLength = byte_length
            bufferView.buffer = 0
            offset += byte_length

        return buffer_blob

    def save_to_bytes(self):
        # setup
        original_buffer_views = copy.deepcopy(self.bufferViews)
        original_buffers = copy.deepcopy(self.buffers)
        new_buffer = Buffer()

        buffer_blob = self.buffers_to_binary_blob()
        new_buffer.byteLength = len(buffer_blob)

        self.buffers = [new_buffer]
        json_blob = self.gltf_to_json(separators=(",", ":"), indent=None).encode(
            "utf-8"
        )

        # pad each blob if needed
        if len(json_blob) % 4 != 0:
            json_blob += b"   "[0 : 4 - len(json_blob) % 4]

        version = struct.pack("<I", GLTF_VERSION)
        chunk_header_len = 8
        length = (
            len(MAGIC)
            + len(version)
            + 4
            + chunk_header_len * 2
            + len(json_blob)
            + len(buffer_blob)
        )

        self.bufferViews = original_buffer_views  # restore unpacked bufferViews
        self.buffers = original_buffers  # restore unpacked buffers

        # header is MAGIC, version, length
        # json chunk is json_blob length, JSON, json_blob
        # buffer chunk is length of buffer_blob, utf-8, buffer_blob
        return [
            MAGIC,
            version,
            struct.pack("<I", length),
            struct.pack("<I", len(json_blob)),
            bytes(JSON, "utf-8"),
            json_blob,
            struct.pack("<I", len(buffer_blob)),
            bytes(BIN, "utf-8"),
            buffer_blob,
        ]

    def save_binary(self, fname):
        with open(fname, "wb") as f:
            glb_structure = self.save_to_bytes()
            if not glb_structure:
                return False
            for data in glb_structure:
                f.write(data)
        return True

    def save(self, fname, asset=Asset()):
        self.asset = asset
        self._path = Path(fname).parent
        self._name = Path(fname).name
        ext = Path(fname).suffix
        if ext.lower() in [".glb"]:
            return self.save_binary(fname)
        else:
            if getattr(self, "_glb_data", None):
                warnings.warn(
                    f"This file ({fname}) contains a binary blob loaded from a .glb file, "
                    "and this will be saved to a .bin file next to the json file."
                )
            return self.save_json(fname)

    @classmethod
    def gltf_from_json(cls, json_data):
        return cls.from_json(json_data, infer_missing=True)

    @classmethod
    def load_json(cls, fname):
        with open(fname, "r") as f:
            obj = cls.gltf_from_json(f.read())
        return obj

    @classmethod
    def load_from_bytes(cls, data):
        magic = struct.unpack("<BBBB", data[:4])
        version, length = struct.unpack("<II", data[4:12])
        if bytearray(magic) != MAGIC:
            raise IOError(
                "Unable to load binary gltf file. Header does not appear to be valid glb format."
            )
        if version > GLTF_MAX_VERSION:
            warnings.warn(
                f"pygltflib supports v{GLTF_MAX_VERSION} of the binary gltf format, "
                "this file is version {version}, "
                "it may not import correctly. "
                "Please open an issue at https://gitlab.com/dodgyville/pygltflib/issues"
            )
        index = 12
        i = 0
        obj = None
        while index < length:
            chunk_length = struct.unpack("<I", data[index : index + 4])[0]
            index += 4
            chunk_type = bytearray(
                struct.unpack("<BBBB", data[index : index + 4])
            ).decode()
            index += 4
            if chunk_type not in [JSON, BIN]:
                warnings.warn(
                    f"Ignoring chunk {i} with unknown type '{chunk_type}', probably glTF extensions. "
                    "Please open an issue at https://gitlab.com/dodgyville/pygltflib/issues"
                )
            elif chunk_type == JSON:
                raw_json = data[index : index + chunk_length].decode("utf-8")
                obj = cls.from_json(raw_json, infer_missing=True)
            else:
                obj.set_binary_blob(data[index : index + chunk_length])
            index += chunk_length
            i += 1
        return obj

    @classmethod
    def load_binary(cls, fname):
        with open(fname, "rb") as f:
            data = f.read()
        return cls.load_from_bytes(data)

    @classmethod
    def load_binary_from_file_object(cls, f):
        data = f.read()
        return cls.load_from_bytes(data)

    @classmethod
    def load(cls, fname):
        path = Path(fname)
        ext = path.suffix
        if ext.lower() in [".bin", ".glb"]:
            obj = cls.load_binary(fname)
        else:
            obj = cls.load_json(fname)
        obj._path = path.parent
        obj._name = path.name
        return obj


class AccessorDataError(ValueError):
    """Raised when accessor data cannot be replaced safely."""


_ACCESSOR_COMPONENT_COUNTS = {
    SCALAR: 1,
    VEC2: 2,
    VEC3: 3,
    VEC4: 4,
    MAT2: 4,
    MAT3: 9,
    MAT4: 16,
}

_ACCESSOR_COMPONENT_INFO = {
    BYTE: ("b", 1, -128, 127, True),
    UNSIGNED_BYTE: ("B", 1, 0, 255, False),
    SHORT: ("h", 2, -32768, 32767, True),
    UNSIGNED_SHORT: ("H", 2, 0, 65535, False),
    UNSIGNED_INT: ("I", 4, 0, 4294967295, False),
    FLOAT: ("f", 4, None, None, None),
}

_SPARSE_INDEX_COMPONENT_INFO = {
    UNSIGNED_BYTE: ("B", 1),
    UNSIGNED_SHORT: ("H", 2),
    UNSIGNED_INT: ("I", 4),
}


def _accessor_error(message):
    raise AccessorDataError(message)


def _resolve_accessor(gltf, accessor):
    if isinstance(accessor, Accessor):
        for index, candidate in enumerate(gltf.accessors):
            if candidate is accessor:
                return index, candidate
        _accessor_error("accessor object does not belong to this GLTF2 object")
    if isinstance(accessor, bool) or not isinstance(accessor, numbers.Integral):
        _accessor_error("accessor_index must be an integer or an Accessor object")
    accessor = int(accessor)
    if accessor < 0 or accessor >= len(gltf.accessors):
        _accessor_error(
            f"accessor index {accessor} is out of range; "
            f"the GLTF2 object has {len(gltf.accessors)} accessor(s)"
        )
    return accessor, gltf.accessors[accessor]


def _is_sequence(value):
    if isinstance(value, (str, bytes, bytearray)):
        return False
    try:
        iter(value)
    except TypeError:
        return False
    return True


def _as_python(value):
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        return value.tolist()
    return value


def _accessor_rows(data, count, component_count, accessor_index):
    data = _as_python(data)
    if isinstance(data, (str, bytes, bytearray)) or not _is_sequence(data):
        _accessor_error(
            f"accessor {accessor_index} data must be a sequence with shape "
            f"({count}, {component_count})"
        )
    try:
        rows = list(data)
    except TypeError:
        _accessor_error(
            f"accessor {accessor_index} data must be a sequence with shape "
            f"({count}, {component_count})"
        )
    if len(rows) != count:
        _accessor_error(
            f"accessor {accessor_index} expects {count} element(s), "
            f"got {len(rows)}"
        )

    result = []
    for element_index, row in enumerate(rows):
        if component_count == 1 and not _is_sequence(row):
            components = [row]
        elif _is_sequence(row):
            components = list(row)
        else:
            _accessor_error(
                f"accessor {accessor_index} element {element_index} must contain "
                f"{component_count} component(s), got a scalar"
            )
        if len(components) != component_count:
            _accessor_error(
                f"accessor {accessor_index} element {element_index} must contain "
                f"{component_count} component(s), got {len(components)}"
            )
        result.append(components)
    return result


def _component_storage_value(value, component_type, normalized, location):
    if isinstance(value, (str, bytes, bytearray)):
        _accessor_error(f"{location} must be a number, got {value!r}")
    if _is_sequence(value):
        _accessor_error(f"{location} must be a number, got a sequence")
    try:
        numeric_value = float(value)
    except (TypeError, ValueError, OverflowError):
        _accessor_error(f"{location} must be a finite number, got {value!r}")
    if not math.isfinite(numeric_value):
        _accessor_error(f"{location} must be a finite number, got {value!r}")

    fmt, _, minimum, maximum, signed = _ACCESSOR_COMPONENT_INFO[component_type]
    if component_type == FLOAT:
        if normalized:
            _accessor_error(f"{location} cannot be normalized because its componentType is FLOAT")
        try:
            return struct.pack("<" + fmt, numeric_value)
        except (OverflowError, struct.error):
            _accessor_error(f"{location} value {value!r} is outside the FLOAT range")

    if normalized:
        normalized_minimum = -1.0 if signed else 0.0
        normalized_maximum = 1.0
        if numeric_value < normalized_minimum or numeric_value > normalized_maximum:
            _accessor_error(
                f"{location} normalized value {value!r} is outside "
                f"[{normalized_minimum}, {normalized_maximum}]"
            )
        scale = maximum
        integer_value = int(round(numeric_value * scale))
        if signed and integer_value < minimum:
            integer_value = minimum
        if integer_value > maximum:
            integer_value = maximum
    else:
        try:
            integer_value = int(value)
        except (TypeError, ValueError, OverflowError):
            _accessor_error(f"{location} must be an integer, got {value!r}")
        if numeric_value != integer_value:
            _accessor_error(f"{location} must be an integer, got {value!r}")
        if integer_value < minimum or integer_value > maximum:
            _accessor_error(
                f"{location} value {value!r} is outside the representable "
                f"range [{minimum}, {maximum}] for componentType {component_type}"
            )

    try:
        return struct.pack("<" + fmt, integer_value)
    except struct.error:
        _accessor_error(
            f"{location} value {value!r} cannot be encoded as componentType "
            f"{component_type}"
        )


def _decode_storage_value(packed, component_type, normalized):
    fmt, _, _, _, signed = _ACCESSOR_COMPONENT_INFO[component_type]
    value = struct.unpack("<" + fmt, packed)[0]
    if normalized:
        if signed:
            return max(value / ((1 << (struct.calcsize(fmt) * 8 - 1)) - 1), -1.0)
        return value / ((1 << (struct.calcsize(fmt) * 8)) - 1)
    return value


def _encode_accessor_rows(rows, component_type, normalized, accessor_index):
    encoded_rows = []
    actual_rows = []
    for element_index, row in enumerate(rows):
        encoded_components = []
        actual_components = []
        for component_index, value in enumerate(row):
            location = (
                f"accessor {accessor_index} element {element_index} "
                f"component {component_index}"
            )
            packed = _component_storage_value(
                value, component_type, normalized, location
            )
            encoded_components.append(packed)
            actual_components.append(
                _decode_storage_value(packed, component_type, False)
            )
        encoded_rows.append(b"".join(encoded_components))
        actual_rows.append(actual_components)
    return encoded_rows, actual_rows


def _accessor_min_max(actual_rows, component_count):
    minimum = [None] * component_count
    maximum = [None] * component_count
    for row in actual_rows:
        for component_index, value in enumerate(row):
            if minimum[component_index] is None or value < minimum[component_index]:
                minimum[component_index] = value
            if maximum[component_index] is None or value > maximum[component_index]:
                maximum[component_index] = value
    return minimum, maximum


class _BufferStorage:
    def __init__(self, gltf, buffer_index):
        if isinstance(buffer_index, bool) or not isinstance(buffer_index, int):
            _accessor_error(f"buffer index {buffer_index!r} is not an integer")
        if buffer_index < 0 or buffer_index >= len(gltf.buffers):
            _accessor_error(
                f"buffer index {buffer_index} is out of range; "
                f"the GLTF2 object has {len(gltf.buffers)} buffer(s)"
            )
        self.gltf = gltf
        self.index = buffer_index
        self.buffer = gltf.buffers[buffer_index]
        self.dirty = False

        if self.buffer.uri is None:
            if buffer_index != 0:
                _accessor_error(
                    f"buffer {buffer_index} has no URI and is not the GLB binary buffer"
                )
            data = gltf.binary_blob()
            if data is None:
                _accessor_error(
                    "buffer 0 has no URI and the GLTF2 object has no binary blob"
                )
        else:
            try:
                data = gltf.get_data_from_buffer_uri(self.buffer.uri)
            except Exception as exc:
                _accessor_error(
                    f"unable to load source data for buffer {buffer_index} "
                    f"(uri={self.buffer.uri!r}): {exc}"
                )
            if data is None:
                _accessor_error(
                    f"unable to load source data for buffer {buffer_index} "
                    f"(uri={self.buffer.uri!r})"
                )

        try:
            self.data = bytearray(data)
        except TypeError:
            _accessor_error(f"source data for buffer {buffer_index} is not binary data")
        self.byte_length = self.buffer.byteLength
        if self.byte_length is None:
            self.byte_length = len(self.data)
        if not isinstance(self.byte_length, int) or self.byte_length < 0:
            _accessor_error(f"buffer {buffer_index} byteLength must be a non-negative integer")
        if self.byte_length > len(self.data):
            _accessor_error(
                f"buffer {buffer_index} declares byteLength {self.byte_length}, "
                f"but only {len(self.data)} byte(s) are available"
            )

    def append_aligned(self, chunk, alignment=4):
        padding = (-len(self.data)) % alignment
        if padding:
            self.data.extend(b"\x00" * padding)
        byte_offset = len(self.data)
        self.data.extend(chunk)
        self.dirty = True
        return byte_offset

    def write(self, byte_offset, chunk):
        if byte_offset < 0 or byte_offset + len(chunk) > self.byte_length:
            _accessor_error(
                f"write range [{byte_offset}, {byte_offset + len(chunk)}) "
                f"is outside buffer {self.index} byteLength {self.byte_length}"
            )
        self.data[byte_offset : byte_offset + len(chunk)] = chunk
        self.dirty = True

    def commit(self):
        if not self.dirty:
            return
        data = bytes(self.data)
        if self.buffer.uri is None:
            self.buffer.byteLength = len(data)
            self.gltf.set_binary_blob(data)
        elif self.buffer.uri.startswith("data"):
            encoded = DATA_URI_HEADER + base64.b64encode(data).decode("ascii")
            self.buffer.byteLength = len(data)
            self.buffer.uri = encoded
        else:
            path = getattr(self.gltf, "_path", Path())
            source_path = Path(path, unquote(self.buffer.uri))
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=source_path.parent, delete=False
            ) as temp_file:
                temp_file.write(data)
                temp_path = temp_file.name
            try:
                os.replace(temp_path, source_path)
            except Exception:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
                raise
            self.buffer.byteLength = len(data)


class _AccessorReplacement:
    def __init__(self, gltf, accessor_index, accessor):
        self.gltf = gltf
        self.accessor_index = accessor_index
        self.accessor = accessor
        self.storages = {}
        self.new_buffer_views = []
        self.new_base_buffer_view = None
        self.new_sparse_values_buffer_view = None

    def storage(self, buffer_index):
        if buffer_index not in self.storages:
            self.storages[buffer_index] = _BufferStorage(self.gltf, buffer_index)
        return self.storages[buffer_index]

    def buffer_view(self, buffer_view_index):
        if isinstance(buffer_view_index, bool) or not isinstance(buffer_view_index, int):
            _accessor_error(f"bufferView index {buffer_view_index!r} is not an integer")
        if buffer_view_index < 0 or buffer_view_index >= len(self.gltf.bufferViews):
            _accessor_error(
                f"bufferView index {buffer_view_index} is out of range; "
                f"the GLTF2 object has {len(self.gltf.bufferViews)} bufferView(s)"
            )
        return self.gltf.bufferViews[buffer_view_index]

    def element_layout(
        self, buffer_view_index, accessor_offset, count, component_size, component_count
    ):
        buffer_view = self.buffer_view(buffer_view_index)
        if buffer_view.byteLength is None:
            _accessor_error(f"bufferView {buffer_view_index} has no byteLength")
        if not isinstance(buffer_view.byteLength, int) or buffer_view.byteLength < 0:
            _accessor_error(
                f"bufferView {buffer_view_index} byteLength must be a non-negative integer"
            )
        view_offset = 0 if buffer_view.byteOffset is None else buffer_view.byteOffset
        if not isinstance(view_offset, int) or view_offset < 0:
            _accessor_error(
                f"bufferView {buffer_view_index} byteOffset must be a non-negative integer"
            )
        if not isinstance(accessor_offset, int) or accessor_offset < 0:
            _accessor_error("accessor byteOffset must be a non-negative integer")
        element_size = component_size * component_count
        if buffer_view.byteStride is None:
            stride = element_size
        else:
            stride = buffer_view.byteStride
            if not isinstance(stride, int) or stride < element_size:
                _accessor_error(
                    f"bufferView {buffer_view_index} byteStride {stride!r} "
                    f"must be at least the element size {element_size}"
                )
        start = view_offset + accessor_offset
        end = start + (count - 1) * stride + element_size if count else start
        view_end = view_offset + buffer_view.byteLength
        if end > view_end:
            _accessor_error(
                f"accessor data range [{start}, {end}) exceeds bufferView "
                f"{buffer_view_index} range [{view_offset}, {view_end})"
            )
        if buffer_view.buffer is None:
            _accessor_error(f"bufferView {buffer_view_index} has no buffer")
        storage = self.storage(buffer_view.buffer)
        if end > storage.byte_length:
            _accessor_error(
                f"accessor data range [{start}, {end}) exceeds buffer "
                f"{buffer_view.buffer} byteLength {storage.byte_length}"
            )
        return storage, start, stride, element_size

    def ranges(self, layout, count, element_size=None):
        storage, start, stride, default_element_size = layout
        element_size = element_size or default_element_size
        return [
            (storage.index, start + element_index * stride, start + element_index * stride + element_size)
            for element_index in range(count)
        ]

    def add_buffer_view(self, buffer_index, chunk, template=None):
        storage = self.storage(buffer_index)
        byte_offset = storage.append_aligned(chunk)
        new_buffer_view = BufferView(
            buffer=buffer_index,
            byteOffset=byte_offset,
            byteLength=len(chunk),
            target=template.target if template is not None else None,
            name=template.name if template is not None else None,
        )
        if template is not None:
            new_buffer_view.extensions = copy.deepcopy(template.extensions)
            new_buffer_view.extras = copy.deepcopy(template.extras)
        new_index = len(self.gltf.bufferViews) + len(self.new_buffer_views)
        self.new_buffer_views.append(new_buffer_view)
        return new_index

    def sparse_indices(self):
        sparse = self.accessor.sparse
        if sparse.count is None or not isinstance(sparse.count, int) or sparse.count < 1:
            _accessor_error("accessor.sparse.count must be a positive integer")
        if sparse.count > self.accessor.count:
            _accessor_error(
                f"accessor.sparse.count {sparse.count} exceeds accessor.count "
                f"{self.accessor.count}"
            )
        if sparse.indices is None:
            _accessor_error("accessor.sparse.indices is required")
        indices = sparse.indices
        if indices.componentType not in _SPARSE_INDEX_COMPONENT_INFO:
            _accessor_error(
                f"accessor.sparse.indices.componentType {indices.componentType!r} "
                "must be UNSIGNED_BYTE, UNSIGNED_SHORT or UNSIGNED_INT"
            )
        if indices.bufferView is None:
            _accessor_error("accessor.sparse.indices.bufferView is required")
        fmt, component_size = _SPARSE_INDEX_COMPONENT_INFO[indices.componentType]
        layout = self.element_layout(
            indices.bufferView, indices.byteOffset or 0, sparse.count, component_size, 1
        )
        storage, start, stride, _ = layout
        values = []
        for index in range(sparse.count):
            byte_offset = start + index * stride
            raw = storage.data[byte_offset : byte_offset + component_size]
            if len(raw) != component_size:
                _accessor_error("accessor.sparse.indices data is truncated")
            sparse_index = struct.unpack("<" + fmt, raw)[0]
            if sparse_index >= self.accessor.count:
                _accessor_error(
                    f"accessor.sparse.indices[{index}] value {sparse_index} "
                    f"is outside accessor count {self.accessor.count}"
                )
            if values and sparse_index <= values[-1]:
                _accessor_error("accessor.sparse.indices must be strictly increasing")
            values.append(sparse_index)
        return values, layout

    def protected_ranges(self, target_sparse_indices_layout=None):
        ranges = []
        for index, accessor in enumerate(self.gltf.accessors):
            if index == self.accessor_index:
                continue
            try:
                ranges.extend(_value_ranges_for_accessor(self, accessor))
                if accessor.sparse is not None:
                    ranges.extend(_sparse_ranges_for_accessor(self, accessor))
            except (AccessorDataError, KeyError, TypeError, ValueError, struct.error):
                continue
        for image in self.gltf.images:
            if image.bufferView is None:
                continue
            try:
                buffer_view = self.buffer_view(image.bufferView)
                if buffer_view.buffer is not None:
                    start = buffer_view.byteOffset or 0
                    ranges.append(
                        (buffer_view.buffer, start, start + (buffer_view.byteLength or 0))
                    )
            except AccessorDataError:
                continue
        if target_sparse_indices_layout is not None:
            ranges.extend(self.ranges(target_sparse_indices_layout, self.accessor.sparse.count))
        return ranges

    def relocate_base(self, encoded_rows):
        old_buffer_view = (
            self.buffer_view(self.accessor.bufferView)
            if self.accessor.bufferView is not None
            else None
        )
        if old_buffer_view is None:
            if not self.gltf.buffers:
                _accessor_error("cannot create accessor storage because the GLTF2 object has no buffers")
            buffer_index = 0
        else:
            if old_buffer_view.buffer is None:
                _accessor_error("accessor bufferView has no buffer")
            buffer_index = old_buffer_view.buffer
        chunk = b"".join(encoded_rows)
        self.new_base_buffer_view = self.add_buffer_view(
            buffer_index, chunk, template=old_buffer_view
        )

    def replace_base(self, encoded_rows, protected_ranges):
        if self.accessor.bufferView is None:
            self.relocate_base(encoded_rows)
            return
        component_count = _ACCESSOR_COMPONENT_COUNTS[self.accessor.type]
        _, component_size, _, _, _ = _ACCESSOR_COMPONENT_INFO[self.accessor.componentType]
        layout = self.element_layout(
            self.accessor.bufferView,
            self.accessor.byteOffset or 0,
            self.accessor.count,
            component_size,
            component_count,
        )
        target_ranges = self.ranges(layout, self.accessor.count)
        if _ranges_intersect(target_ranges, protected_ranges):
            self.relocate_base(encoded_rows)
            return
        storage = layout[0]
        for byte_range, encoded_row in zip(target_ranges, encoded_rows):
            storage.write(byte_range[1], encoded_row)

    def replace_sparse_values(self, encoded_rows, sparse_indices, protected_ranges):
        sparse = self.accessor.sparse
        if sparse.values is None:
            _accessor_error("accessor.sparse.values is required")
        if sparse.values.bufferView is None:
            _accessor_error("accessor.sparse.values.bufferView is required")
        component_count = _ACCESSOR_COMPONENT_COUNTS[self.accessor.type]
        _, component_size, _, _, _ = _ACCESSOR_COMPONENT_INFO[self.accessor.componentType]
        layout = self.element_layout(
            sparse.values.bufferView,
            sparse.values.byteOffset or 0,
            sparse.count,
            component_size,
            component_count,
        )
        selected_rows = [encoded_rows[index] for index in sparse_indices]
        target_ranges = self.ranges(layout, sparse.count)
        if _ranges_intersect(target_ranges, protected_ranges):
            buffer_view = self.buffer_view(sparse.values.bufferView)
            if buffer_view.buffer is None:
                _accessor_error("accessor.sparse.values bufferView has no buffer")
            self.new_sparse_values_buffer_view = self.add_buffer_view(
                buffer_view.buffer, b"".join(selected_rows), template=buffer_view
            )
            return
        storage = layout[0]
        for byte_range, encoded_row in zip(target_ranges, selected_rows):
            storage.write(byte_range[1], encoded_row)

    def commit(self, minimum, maximum):
        for storage in self.storages.values():
            storage.commit()
        self.gltf.bufferViews.extend(self.new_buffer_views)
        if self.new_base_buffer_view is not None:
            self.accessor.bufferView = self.new_base_buffer_view
            self.accessor.byteOffset = 0
        if self.new_sparse_values_buffer_view is not None:
            self.accessor.sparse.values.bufferView = self.new_sparse_values_buffer_view
            self.accessor.sparse.values.byteOffset = 0
        self.accessor.min = minimum
        self.accessor.max = maximum


def _ranges_intersect(left, right):
    for left_buffer, left_start, left_end in left:
        for right_buffer, right_start, right_end in right:
            if (
                left_buffer == right_buffer
                and left_start < right_end
                and right_start < left_end
            ):
                return True
    return False


def _value_ranges_for_accessor(replacement, accessor):
    if accessor.bufferView is None or accessor.count is None or accessor.count < 1:
        return []
    if accessor.type not in _ACCESSOR_COMPONENT_COUNTS:
        raise AccessorDataError(f"accessor type {accessor.type!r} is invalid")
    if accessor.componentType not in _ACCESSOR_COMPONENT_INFO:
        raise AccessorDataError(
            f"accessor componentType {accessor.componentType!r} is invalid"
        )
    component_count = _ACCESSOR_COMPONENT_COUNTS[accessor.type]
    component_size = _ACCESSOR_COMPONENT_INFO[accessor.componentType][1]
    layout = replacement.element_layout(
        accessor.bufferView,
        accessor.byteOffset or 0,
        accessor.count,
        component_size,
        component_count,
    )
    return replacement.ranges(layout, accessor.count)


def _sparse_ranges_for_accessor(replacement, accessor):
    sparse = accessor.sparse
    if sparse.indices is None or sparse.indices.bufferView is None:
        raise AccessorDataError("sparse indices are missing")
    if sparse.values is None or sparse.values.bufferView is None:
        raise AccessorDataError("sparse values are missing")
    if sparse.count is None or sparse.count < 1:
        raise AccessorDataError("sparse count is invalid")
    if sparse.indices.componentType not in _SPARSE_INDEX_COMPONENT_INFO:
        raise AccessorDataError("sparse index componentType is invalid")
    if accessor.type not in _ACCESSOR_COMPONENT_COUNTS:
        raise AccessorDataError("accessor type is invalid")
    if accessor.componentType not in _ACCESSOR_COMPONENT_INFO:
        raise AccessorDataError("accessor componentType is invalid")
    index_size = _SPARSE_INDEX_COMPONENT_INFO[sparse.indices.componentType][1]
    index_layout = replacement.element_layout(
        sparse.indices.bufferView,
        sparse.indices.byteOffset or 0,
        sparse.count,
        index_size,
        1,
    )
    component_count = _ACCESSOR_COMPONENT_COUNTS[accessor.type]
    component_size = _ACCESSOR_COMPONENT_INFO[accessor.componentType][1]
    values_layout = replacement.element_layout(
        sparse.values.bufferView,
        sparse.values.byteOffset or 0,
        sparse.count,
        component_size,
        component_count,
    )
    return (
        replacement.ranges(index_layout, sparse.count)
        + replacement.ranges(values_layout, sparse.count)
    )


def _replace_accessor_data(gltf, accessor_index, data):
    accessor_index, accessor = _resolve_accessor(gltf, accessor_index)
    if accessor.componentType not in _ACCESSOR_COMPONENT_INFO:
        _accessor_error(
            f"accessor {accessor_index} has invalid componentType "
            f"{accessor.componentType!r}"
        )
    if accessor.type not in _ACCESSOR_COMPONENT_COUNTS:
        _accessor_error(
            f"accessor {accessor_index} has invalid type {accessor.type!r}"
        )
    if accessor.count is None or not isinstance(accessor.count, int) or accessor.count < 1:
        _accessor_error(f"accessor {accessor_index} count must be a positive integer")
    if accessor.normalized and accessor.componentType == FLOAT:
        _accessor_error(
            f"accessor {accessor_index} cannot be normalized with componentType FLOAT"
        )

    component_count = _ACCESSOR_COMPONENT_COUNTS[accessor.type]
    rows = _accessor_rows(data, accessor.count, component_count, accessor_index)
    encoded_rows, actual_rows = _encode_accessor_rows(
        rows, accessor.componentType, bool(accessor.normalized), accessor_index
    )
    minimum, maximum = _accessor_min_max(actual_rows, component_count)

    replacement = _AccessorReplacement(gltf, accessor_index, accessor)
    sparse_indices = None
    sparse_indices_layout = None
    if accessor.sparse is not None:
        sparse_indices, sparse_indices_layout = replacement.sparse_indices()

    protected_ranges = replacement.protected_ranges(sparse_indices_layout)
    replacement.replace_base(encoded_rows, protected_ranges)
    if accessor.sparse is not None:
        replacement.replace_sparse_values(
            encoded_rows, sparse_indices, protected_ranges
        )

    replacement.commit(minimum, maximum)


def replace_accessor_data(gltf, accessor_index, data):
    """Replace an accessor's values using :meth:`GLTF2.replace_accessor_data`."""
    gltf.replace_accessor_data(accessor_index, data)


set_accessor_data = replace_accessor_data
update_accessor_data = replace_accessor_data


def main():
    import doctest

    doctest.testfile("../README.md")


if __name__ == "__main__":
    main()


def image_get_bytes(image: Image, gltf: GLTF2, path):

    if image.current_format == ImageFormat.FILE:
        image_path = Path(path / unquote(image.uri))
        if not image_path.exists():
            warnings.warn(f"Expected image file at {image_path} not found.")

        with open(image_path, "rb") as image_file:
            return image_file.read()

    elif image.current_format == ImageFormat.BUFFERVIEW:
        # gltf.convert_buffers(BufferFormat.BINARYBLOB)
        data = gltf.binary_blob()
        # gltf.convert_buffers(BufferFormat.DATAURI)

        bufferView = gltf.bufferViews[image.bufferView]
        image_data: bytes = data[
            bufferView.byteOffset : bufferView.byteOffset + bufferView.byteLength
        ]
        return image_data
    elif image.current_format == ImageFormat.DATAURI:
        return gltf.decode_data_uri(
            image.uri, image.uri[: image.uri.find("base64,") + 7]
        )
