"""
Helpers for consolidating equivalent materials in a GLTF2 document.
"""

import copy
from collections.abc import Mapping
from enum import Enum
from typing import Any, Dict, Optional

from .types import OPAQUE


# Material extensions whose semantics are understood well enough to compare.
# Materials carrying any other extension are kept independent so that a custom
# extension is never silently discarded by an unsafe merge.
KNOWN_MATERIAL_EXTENSIONS = frozenset(
    {
        "KHR_materials_anisotropy",
        "KHR_materials_clearcoat",
        "KHR_materials_diffuse_transmission",
        "KHR_materials_dispersion",
        "KHR_materials_emissive_strength",
        "KHR_materials_ior",
        "KHR_materials_iridescence",
        "KHR_materials_pbrSpecularGlossiness",
        "KHR_materials_sheen",
        "KHR_materials_specular",
        "KHR_materials_transmission",
        "KHR_materials_unlit",
        "KHR_materials_variants",
        "KHR_materials_volume",
    }
)

_TEXTURE_TRANSFORM = "KHR_texture_transform"


def _get(value, name, default=None):
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _freeze(value):
    """Convert a value to a stable, hashable representation."""
    if isinstance(value, Mapping):
        items = [(_freeze(key), _freeze(item)) for key, item in value.items()]
        return tuple(sorted(items, key=repr))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze(item) for item in value), key=repr))
    if isinstance(value, Enum):
        return value.value
    return value


def _freeze_mapping(value):
    return _freeze(value) if value else ()


def _vector(value, default):
    if value is None:
        value = default
    return tuple(value)


def _number(value, default):
    return default if value is None else value


def _canonical_texture_transform(value, inherited_tex_coord):
    if not isinstance(value, Mapping):
        return ("invalid", _freeze(value))

    known_keys = {"offset", "rotation", "scale", "texCoord", "extensions", "extras"}
    unknown = {
        key: item for key, item in value.items() if key not in known_keys
    }
    tex_coord = value.get("texCoord")
    if tex_coord is None:
        tex_coord = inherited_tex_coord

    return (
        _vector(value.get("offset"), (0.0, 0.0)),
        _number(value.get("rotation"), 0.0),
        _vector(value.get("scale"), (1.0, 1.0)),
        tex_coord,
        _freeze_mapping(value.get("extensions")),
        _freeze_mapping(value.get("extras")),
        _freeze_mapping(unknown),
    )


def _canonical_texture_extensions(extensions, inherited_tex_coord):
    if not extensions:
        return ()
    if not isinstance(extensions, Mapping):
        return (("invalid", _freeze(extensions)),)

    items = []
    for name, value in extensions.items():
        if name == _TEXTURE_TRANSFORM:
            transform = _canonical_texture_transform(value, inherited_tex_coord)
            default_transform = (
                (0.0, 0.0),
                0.0,
                (1.0, 1.0),
                inherited_tex_coord,
                (),
                (),
                (),
            )
            if transform == default_transform:
                continue
            value = transform
        items.append((name, _freeze(value)))
    return tuple(sorted(items, key=repr))


def _canonical_texture_info(value, kind):
    if value is None:
        return None

    index = _get(value, "index")
    tex_coord = _number(_get(value, "texCoord"), 0)

    fields = [("index", index), ("texCoord", tex_coord)]
    if kind == "normal":
        fields.append(("scale", _number(_get(value, "scale"), 1.0)))
    elif kind == "occlusion":
        fields.append(("strength", _number(_get(value, "strength"), 1.0)))

    fields.append(
        (
            "extensions",
            _canonical_texture_extensions(
                _get(value, "extensions", {}), tex_coord
            ),
        )
    )
    fields.append(("extras", _freeze_mapping(_get(value, "extras", {}))))
    return tuple(fields)


def _canonical_pbr(value):
    # An omitted pbrMetallicRoughness object is equivalent to one containing
    # only the glTF default values.
    fields = (
        (
            "baseColorFactor",
            _vector(
                _get(value, "baseColorFactor") if value is not None else None,
                (1.0, 1.0, 1.0, 1.0),
            ),
        ),
        (
            "metallicFactor",
            _number(
                _get(value, "metallicFactor") if value is not None else None,
                1.0,
            ),
        ),
        (
            "roughnessFactor",
            _number(
                _get(value, "roughnessFactor") if value is not None else None,
                1.0,
            ),
        ),
        (
            "baseColorTexture",
            _canonical_texture_info(
                _get(value, "baseColorTexture") if value is not None else None,
                "color",
            ),
        ),
        (
            "metallicRoughnessTexture",
            _canonical_texture_info(
                _get(value, "metallicRoughnessTexture")
                if value is not None
                else None,
                "metallic_roughness",
            ),
        ),
        (
            "extensions",
            _freeze_mapping(_get(value, "extensions", {}) if value is not None else {}),
        ),
        (
            "extras",
            _freeze_mapping(_get(value, "extras", {}) if value is not None else {}),
        ),
    )
    return fields


def _canonical_material_extensions(value):
    extensions = _get(value, "extensions", {})
    if not extensions:
        return ()
    if not isinstance(extensions, Mapping):
        return None

    if any(name not in KNOWN_MATERIAL_EXTENSIONS for name in extensions):
        return None

    return tuple(
        sorted(
            ((name, _freeze(item)) for name, item in extensions.items()),
            key=repr,
        )
    )


def _canonical_material(value):
    """Return a hashable semantic key, or None for extensions we cannot merge."""
    if value is None:
        return ("material", None)

    extensions = _canonical_material_extensions(value)
    if extensions is None:
        return None

    alpha_mode = _get(value, "alphaMode")
    if alpha_mode is None:
        alpha_mode = OPAQUE

    return (
        "material",
        _canonical_pbr(_get(value, "pbrMetallicRoughness")),
        _canonical_texture_info(_get(value, "normalTexture"), "normal"),
        _canonical_texture_info(_get(value, "occlusionTexture"), "occlusion"),
        _vector(
            _get(value, "emissiveFactor"), (0.0, 0.0, 0.0)
        ),
        _canonical_texture_info(_get(value, "emissiveTexture"), "emissive"),
        alpha_mode,
        _number(_get(value, "alphaCutoff"), 0.5),
        _number(_get(value, "doubleSided"), False),
        extensions,
        _freeze_mapping(_get(value, "extras", {})),
    )


def deduplicate_materials(
    gltf, remove_unused=False, *, remove_unused_materials=None
):
    """
    Merge equivalent materials without modifying ``gltf``.

    Materials are compared by their rendering semantics, not by name. Omitted
    values are normalized to the glTF defaults, texture references are compared
    by texture index, and texture transforms are included in the comparison.
    Materials with extensions not understood by this function are never merged.

    Args:
        gltf (GLTF2): Source document.
        remove_unused (bool): If true, materials not referenced by any mesh
            primitive are omitted from the returned document. The keyword
            alias ``remove_unused_materials`` is also accepted.

    Returns:
        tuple: ``(deduplicated_gltf, material_mapping)`` where
        ``material_mapping`` maps each original material index to its index in
        the returned document. A material removed as unused maps to ``None``
        unless it is equivalent to a material that is still used.
    """
    if remove_unused_materials is not None:
        remove_unused = remove_unused_materials

    result = copy.deepcopy(gltf)

    original_materials = list(result.materials)
    representative_by_key: Dict[Any, int] = {}
    group_for_material = {}
    groups = []

    for original_index, material in enumerate(original_materials):
        key = _canonical_material(material)
        if key is None:
            # Unknown extensions are intentionally isolated from one another.
            key = ("unknown-material-extension", original_index)

        if key not in representative_by_key:
            representative_by_key[key] = original_index
            groups.append((key, original_index))
        group_for_material[original_index] = representative_by_key[key]

    used_materials = set()
    for mesh in result.meshes:
        for primitive in _get(mesh, "primitives", []):
            material_index = _get(primitive, "material")
            if material_index is not None:
                used_materials.add(material_index)

    if remove_unused:
        retained_representatives = {
            group_for_material[material_index]
            for material_index in used_materials
            if material_index in group_for_material
        }
        retained_groups = [
            (key, representative)
            for key, representative in groups
            if representative in retained_representatives
        ]
    else:
        retained_groups = groups

    new_index_by_representative = {
        representative: new_index
        for new_index, (_, representative) in enumerate(retained_groups)
    }

    material_mapping: Dict[int, Optional[int]] = {}
    for original_index in range(len(original_materials)):
        representative = group_for_material[original_index]
        material_mapping[original_index] = new_index_by_representative.get(
            representative
        )

    result.materials = [
        copy.deepcopy(original_materials[representative])
        for _, representative in retained_groups
    ]

    for mesh in result.meshes:
        for primitive in _get(mesh, "primitives", []):
            material_index = _get(primitive, "material")
            if material_index is None:
                continue
            if material_index in material_mapping:
                mapped_index = material_mapping[material_index]
                if mapped_index is not None:
                    primitive.material = mapped_index

    return result, material_mapping
