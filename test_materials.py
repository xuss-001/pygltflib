import tempfile
from pathlib import Path

from pygltflib import (
    BLEND,
    OPAQUE,
    Attributes,
    GLTF2,
    Image,
    Material,
    Mesh,
    NormalMaterialTexture,
    OcclusionTextureInfo,
    PbrMetallicRoughness,
    Primitive,
    Texture,
    TextureInfo,
    deduplicate_materials,
)
from pygltflib.materials import _canonical_material


def _material(name, **kwargs):
    return Material(name=name, **kwargs)


def _gltf_with_materials(materials, material_refs):
    mesh = Mesh(
        primitives=[
            Primitive(attributes=Attributes(), material=material_index)
            for material_index in material_refs
        ]
    )
    return GLTF2(materials=materials, meshes=[mesh])


def test_merges_equivalent_materials_and_remaps_zero_and_shared_references():
    explicit_defaults = Material(
        name="explicit",
        pbrMetallicRoughness=PbrMetallicRoughness(
            baseColorFactor=[1.0, 1.0, 1.0, 1.0],
            metallicFactor=1.0,
            roughnessFactor=1.0,
        ),
        emissiveFactor=[0.0, 0.0, 0.0],
        alphaMode=OPAQUE,
        alphaCutoff=0.5,
        doubleSided=False,
    )
    red = _material(
        "red-a",
        pbrMetallicRoughness=PbrMetallicRoughness(
            baseColorFactor=[1.0, 0.0, 0.0, 1.0]
        ),
    )
    red_copy = _material(
        "red-b",
        pbrMetallicRoughness=PbrMetallicRoughness(
            baseColorFactor=[1.0, 0.0, 0.0, 1.0]
        ),
    )
    gltf = _gltf_with_materials(
        [_material("default"), explicit_defaults, red, red_copy],
        [0, 1, 2, 3, 1],
    )

    result, mapping = deduplicate_materials(gltf)

    assert mapping == {0: 0, 1: 0, 2: 1, 3: 1}
    assert [material.name for material in result.materials] == [
        "default",
        "red-a",
    ]
    assert [primitive.material for primitive in result.meshes[0].primitives] == [
        0,
        0,
        1,
        1,
        0,
    ]

    assert len(gltf.materials) == 4
    assert [primitive.material for primitive in gltf.meshes[0].primitives] == [
        0,
        1,
        2,
        3,
        1,
    ]
    assert result.materials[0] is not gltf.materials[0]


def test_omitted_and_explicit_texture_defaults_are_equivalent():
    omitted_defaults = _material(
        "omitted",
        pbrMetallicRoughness=PbrMetallicRoughness(
            baseColorFactor=None,
            metallicFactor=None,
            roughnessFactor=None,
            baseColorTexture=TextureInfo(index=0, texCoord=None),
        ),
        normalTexture=NormalMaterialTexture(
            index=1, texCoord=None, scale=None
        ),
        occlusionTexture=OcclusionTextureInfo(
            index=2, texCoord=None, strength=None
        ),
    )
    explicit_defaults = _material(
        "explicit",
        pbrMetallicRoughness=PbrMetallicRoughness(
            baseColorFactor=[1.0, 1.0, 1.0, 1.0],
            metallicFactor=1.0,
            roughnessFactor=1.0,
            baseColorTexture=TextureInfo(index=0, texCoord=0),
        ),
        normalTexture=NormalMaterialTexture(index=1, texCoord=0, scale=1.0),
        occlusionTexture=OcclusionTextureInfo(
            index=2, texCoord=0, strength=1.0
        ),
    )
    gltf = _gltf_with_materials(
        [omitted_defaults, explicit_defaults], [0, 1]
    )

    result, mapping = deduplicate_materials(gltf)

    assert mapping == {0: 0, 1: 0}
    assert len(result.materials) == 1
    assert [primitive.material for primitive in result.meshes[0].primitives] == [
        0,
        0,
    ]


def test_json_omitted_values_match_explicit_defaults():
    gltf = GLTF2.gltf_from_json(
        """
        {
          "materials": [
            {
              "name": "omitted",
              "normalTexture": {"index": 1},
              "occlusionTexture": {"index": 2}
            },
            {
              "name": "explicit",
              "pbrMetallicRoughness": {
                "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                "metallicFactor": 1.0,
                "roughnessFactor": 1.0
              },
              "normalTexture": {"index": 1, "texCoord": 0, "scale": 1.0},
              "occlusionTexture": {
                "index": 2,
                "texCoord": 0,
                "strength": 1.0
              },
              "emissiveFactor": [0.0, 0.0, 0.0],
              "alphaMode": "OPAQUE",
              "alphaCutoff": 0.5,
              "doubleSided": false
            }
          ],
          "meshes": [
            {
              "primitives": [
                {"attributes": {}, "material": 0},
                {"attributes": {}, "material": 1}
              ]
            }
          ]
        }
        """
    )

    result, mapping = gltf.deduplicate_materials(
        remove_unused_materials=True
    )

    assert mapping == {0: 0, 1: 0}
    assert len(result.materials) == 1
    assert [primitive.material for primitive in result.meshes[0].primitives] == [
        0,
        0,
    ]


def test_render_semantics_and_texture_references_prevent_unsafe_merges():
    base = _material(
        "base",
        pbrMetallicRoughness=PbrMetallicRoughness(
            baseColorTexture=TextureInfo(index=0)
        ),
    )
    blend = _material(
        "blend",
        pbrMetallicRoughness=PbrMetallicRoughness(
            baseColorTexture=TextureInfo(index=0)
        ),
        alphaMode=BLEND,
    )
    double_sided = _material(
        "double-sided",
        pbrMetallicRoughness=PbrMetallicRoughness(
            baseColorTexture=TextureInfo(index=0)
        ),
        doubleSided=True,
    )
    other_slot = _material(
        "other-slot",
        pbrMetallicRoughness=PbrMetallicRoughness(
            metallicRoughnessTexture=TextureInfo(index=0)
        ),
    )
    other_tex_coord = _material(
        "other-tex-coord",
        pbrMetallicRoughness=PbrMetallicRoughness(
            baseColorTexture=TextureInfo(index=0, texCoord=1)
        ),
    )
    other_texture = _material(
        "other-texture",
        pbrMetallicRoughness=PbrMetallicRoughness(
            baseColorTexture=TextureInfo(index=1)
        ),
    )
    transformed = _material(
        "transformed",
        pbrMetallicRoughness=PbrMetallicRoughness(
            baseColorTexture=TextureInfo(
                index=0,
                extensions={
                    "KHR_texture_transform": {
                        "offset": [0.25, 0.5],
                        "rotation": 0.0,
                        "scale": [1.0, 1.0],
                    }
                },
            )
        ),
    )
    default_transform = _material(
        "default-transform",
        pbrMetallicRoughness=PbrMetallicRoughness(
            baseColorTexture=TextureInfo(
                index=0,
                extensions={
                    "KHR_texture_transform": {
                        "offset": [0.0, 0.0],
                        "rotation": 0.0,
                        "scale": [1.0, 1.0],
                        "texCoord": 0,
                    }
                },
            )
        ),
    )
    gltf = _gltf_with_materials(
        [
            base,
            blend,
            double_sided,
            other_slot,
            other_tex_coord,
            other_texture,
            transformed,
            default_transform,
        ],
        list(range(8)),
    )
    gltf.textures = [
        Texture(name="same-name", source=0),
        Texture(name="same-name", source=1),
    ]
    gltf.images = [
        Image(name="same-name", uri="same-name.png"),
        Image(name="same-name", uri="same-name.png"),
    ]

    result, mapping = deduplicate_materials(gltf)

    assert mapping == {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 0,
    }
    assert len(result.materials) == 7


def test_unknown_material_extensions_stay_independent():
    unknown_extension = {"VENDOR_material_settings": {"value": 1}}
    materials = [
        _material("custom-a", extensions=unknown_extension),
        _material("custom-b", extensions={"VENDOR_material_settings": {"value": 1}}),
        _material("unlit-a", extensions={"KHR_materials_unlit": {}}),
        _material("unlit-b", extensions={"KHR_materials_unlit": {}}),
    ]
    gltf = _gltf_with_materials(materials, [0, 1, 2, 3])

    result, mapping = deduplicate_materials(gltf)

    assert mapping == {0: 0, 1: 1, 2: 2, 3: 2}
    assert [material.name for material in result.materials] == [
        "custom-a",
        "custom-b",
        "unlit-a",
    ]


def test_remove_unused_materials_and_zero_reference():
    used = _material("used")
    used_alias = _material("used-alias")
    unused = _material("unused", alphaMode=BLEND)
    unused_alias = _material("unused-alias")
    gltf = _gltf_with_materials(
        [used, used_alias, unused, unused_alias], [0, 1, 0]
    )

    result, mapping = deduplicate_materials(gltf, remove_unused=True)

    assert mapping == {0: 0, 1: 0, 2: None, 3: 0}
    assert [material.name for material in result.materials] == ["used"]
    assert [primitive.material for primitive in result.meshes[0].primitives] == [
        0,
        0,
        0,
    ]

    all_materials, _ = deduplicate_materials(gltf, remove_unused=False)
    assert [material.name for material in all_materials.materials] == [
        "used",
        "unused",
    ]


def test_saved_and_reloaded_primitive_materials_remain_equivalent():
    gltf = _gltf_with_materials(
        [
            _material(
                "red-a",
                pbrMetallicRoughness=PbrMetallicRoughness(
                    baseColorFactor=[1.0, 0.0, 0.0, 1.0]
                ),
            ),
            _material(
                "red-b",
                pbrMetallicRoughness=PbrMetallicRoughness(
                    baseColorFactor=[1.0, 0.0, 0.0, 1.0]
                ),
            ),
            _material(
                "blue",
                pbrMetallicRoughness=PbrMetallicRoughness(
                    baseColorFactor=[0.0, 0.0, 1.0, 1.0]
                ),
            ),
        ],
        [0, 1, 2, 0],
    )
    original_keys = [
        _canonical_material(gltf.materials[primitive.material])
        for primitive in gltf.meshes[0].primitives
    ]

    result, mapping = gltf.deduplicate_materials()
    expected_refs = [mapping[index] for index in [0, 1, 2, 0]]

    with tempfile.TemporaryDirectory() as tmpdirname:
        output = Path(tmpdirname) / "materials.gltf"
        result.save(output)
        reloaded = GLTF2.load(output)

    reloaded_refs = [
        primitive.material for primitive in reloaded.meshes[0].primitives
    ]
    assert reloaded_refs == expected_refs
    assert [
        _canonical_material(reloaded.materials[index])
        for index in reloaded_refs
    ] == original_keys
    assert [
        primitive.material for primitive in gltf.meshes[0].primitives
    ] == [0, 1, 2, 0]
