import base64
import copy
import math
import struct

import pytest

from pygltflib import (
    Animation,
    AnimationChannel,
    AnimationChannelTarget,
    AnimationSampler,
    Buffer,
    BufferView,
    DuplicateAnimationChannelError,
    FLOAT,
    GLTF2,
    InvalidKeyframeError,
    Node,
    ROTATION,
    SCALAR,
    TRANSLATION,
    UnsupportedAnimationChannelError,
    VEC3,
    VEC4,
    WEIGHTS,
    Accessor,
)


def _pack_rows(rows, component_count):
    return b"".join(
        struct.pack("<%df" % component_count, *row) for row in rows
    )


class _Builder:
    def __init__(self):
        self.gltf = GLTF2()
        self.data = bytearray()

    def add_accessor(
        self,
        rows,
        component_count,
        accessor_type,
        prefix=b"",
        byte_stride=None,
    ):
        view_offset = len(self.data)
        accessor_offset = len(prefix)
        self.data.extend(prefix)

        element_size = component_count * 4
        if byte_stride is None:
            byte_stride = element_size
        for row in rows:
            assert len(row) == component_count
            self.data.extend(struct.pack("<%df" % component_count, *row))
            self.data.extend(b"\x00" * (byte_stride - element_size))

        view_index = len(self.gltf.bufferViews)
        self.gltf.bufferViews.append(
            BufferView(
                buffer=0,
                byteOffset=view_offset,
                byteLength=len(self.data) - view_offset,
                byteStride=byte_stride if byte_stride != element_size else None,
            )
        )
        accessor_index = len(self.gltf.accessors)
        self.gltf.accessors.append(
            Accessor(
                bufferView=view_index,
                byteOffset=accessor_offset,
                componentType=FLOAT,
                count=len(rows),
                type=accessor_type,
            )
        )
        return accessor_index

    def finish(self, mode="glb"):
        self.gltf.buffers = [Buffer(byteLength=len(self.data))]
        if mode == "glb":
            self.gltf.set_binary_blob(bytes(self.data))
        elif mode == "datauri":
            encoded = base64.b64encode(bytes(self.data)).decode("ascii")
            self.gltf.buffers[0].uri = (
                "data:application/octet-stream;base64," + encoded
            )
        else:
            self.gltf.buffers[0].uri = "animation.bin"
        return self.gltf


def _add_sampler(
    gltf, path, input_accessor, output_accessor, interpolation="LINEAR"
):
    sampler_index = len(gltf.animations[0].samplers)
    gltf.animations[0].samplers.append(
        AnimationSampler(
            input=input_accessor,
            output=output_accessor,
            interpolation=interpolation,
        )
    )
    gltf.animations[0].channels.append(
        AnimationChannel(
            sampler=sampler_index,
            target=AnimationChannelTarget(node=0, path=path),
        )
    )


def _basic_animation():
    builder = _Builder()
    times = builder.add_accessor(
        [[0.0], [1.0], [3.0]], 1, SCALAR
    )
    translations = builder.add_accessor(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 20.0, 0.0]],
        3,
        VEC3,
    )
    rotations = builder.add_accessor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        4,
        VEC4,
    )
    gltf = builder.finish()
    gltf.nodes = [
        Node(
            name="animated",
            translation=[99.0, 99.0, 99.0],
            rotation=[0.0, 0.0, 0.0, 1.0],
            scale=[2.0, 3.0, 4.0],
        ),
        Node(name="static", translation=[7.0, 8.0, 9.0]),
    ]
    gltf.animations = [Animation()]
    _add_sampler(gltf, TRANSLATION, times, translations)
    _add_sampler(gltf, ROTATION, times, rotations)
    return gltf


def test_linear_sampling_continuous_unequal_intervals_and_shortest_path():
    gltf = _basic_animation()

    samples = {}
    for index in range(33):
        time = -0.5 + index * (4.0 / 32.0)
        samples[time] = gltf.sample_animation(0, time)

    assert list(samples[0.0][0].keys()) == [
        "translation",
        "rotation",
        "scale",
    ]

    assert samples[-0.5][0].translation == pytest.approx([0.0, 0.0, 0.0])
    assert samples[0.0][0].translation == pytest.approx([0.0, 0.0, 0.0])
    assert samples[0.5][0].translation == pytest.approx([5.0, 0.0, 0.0])
    assert samples[1.0][0].translation == pytest.approx([10.0, 0.0, 0.0])
    assert samples[2.0][0].translation == pytest.approx([10.0, 10.0, 0.0])
    assert samples[3.0][0].translation == pytest.approx([10.0, 20.0, 0.0])
    assert samples[3.5][0].translation == pytest.approx([10.0, 20.0, 0.0])

    for transform in samples.values():
        rotation = transform[0].rotation
        assert math.sqrt(sum(value * value for value in rotation)) == pytest.approx(
            1.0
        )

    # The two key quaternions have opposite signs but represent identity.  A
    # shortest-path slerp keeps the rotation at identity between them.
    assert samples[0.5][0].rotation == pytest.approx(
        [0.0, 0.0, 0.0, 1.0], abs=1e-6
    )

    # Scale is not animated, so the node's original scale is preserved.
    for transform in samples.values():
        assert transform[0].scale == pytest.approx([2.0, 3.0, 4.0])

    # Non-target nodes are not included in the result.
    assert 1 not in samples[2.0]


def test_step_sampling_uses_left_keyframe_and_clamps_outside_range():
    builder = _Builder()
    times = builder.add_accessor([[0.0], [2.0], [5.0]], 1, SCALAR)
    translations = builder.add_accessor(
        [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        3,
        VEC3,
    )
    gltf = builder.finish()
    gltf.nodes = [Node()]
    gltf.animations = [Animation()]
    _add_sampler(gltf, TRANSLATION, times, translations, interpolation="STEP")

    assert gltf.sample_animation(0, -1.0)[0].translation == pytest.approx(
        [1.0, 0.0, 0.0]
    )
    assert gltf.sample_animation(0, 0.0)[0].translation == pytest.approx(
        [1.0, 0.0, 0.0]
    )
    assert gltf.sample_animation(0, 1.999)[0].translation == pytest.approx(
        [1.0, 0.0, 0.0]
    )
    assert gltf.sample_animation(0, 2.0)[0].translation == pytest.approx(
        [2.0, 0.0, 0.0]
    )
    assert gltf.sample_animation(0, 4.999)[0].translation == pytest.approx(
        [2.0, 0.0, 0.0]
    )
    assert gltf.sample_animation(0, 5.0)[0].translation == pytest.approx(
        [3.0, 0.0, 0.0]
    )
    assert gltf.sample_animation(0, 99.0)[0].translation == pytest.approx(
        [3.0, 0.0, 0.0]
    )


def _cubic_reference(times, keyframes, time):
    if time <= times[0]:
        return list(keyframes[0][1])
    if time >= times[-1]:
        return list(keyframes[-1][1])
    index = max(i for i in range(len(times) - 1) if times[i] <= time)
    interval = times[index + 1] - times[index]
    factor = (time - times[index]) / interval
    start_tangent = keyframes[index][2]
    start_value = keyframes[index][1]
    end_value = keyframes[index + 1][1]
    end_tangent = keyframes[index + 1][0]

    factor2 = factor * factor
    factor3 = factor2 * factor
    return [
        (2 * factor3 - 3 * factor2 + 1) * start_value[i]
        + (factor3 - 2 * factor2 + factor) * interval * start_tangent[i]
        + (-2 * factor3 + 3 * factor2) * end_value[i]
        + (factor3 - factor2) * interval * end_tangent[i]
        for i in range(len(start_value))
    ]


def test_cubic_spline_accounts_for_unequal_keyframe_intervals():
    builder = _Builder()
    times = [0.0, 1.0, 4.0]
    keyframes = [
        ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [4.0, 0.0, 0.0]),
        ([0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]),
        ([1.0, 0.0, 0.0], [-1.0, 3.0, 0.0], [0.0, 0.0, 0.0]),
    ]
    input_accessor = builder.add_accessor(
        [[value] for value in times], 1, SCALAR
    )
    output_rows = []
    for incoming, value, outgoing in keyframes:
        output_rows.extend([incoming, value, outgoing])
    output_accessor = builder.add_accessor(output_rows, 3, VEC3)
    gltf = builder.finish()
    gltf.nodes = [Node()]
    gltf.animations = [Animation()]
    _add_sampler(
        gltf,
        TRANSLATION,
        input_accessor,
        output_accessor,
        interpolation="CUBICSPLINE",
    )

    for time in [-1.0, 0.0, 0.25, 0.5, 0.9, 1.0, 1.5, 2.5, 3.9, 4.0, 8.0]:
        expected = _cubic_reference(times, keyframes, time)
        sampled = gltf.sample_animation(0, time)[0].translation
        assert sampled == pytest.approx(expected, abs=1e-6)


def test_cubic_spline_rotation_is_normalized():
    builder = _Builder()
    times = builder.add_accessor([[0.0], [1.0], [3.0]], 1, SCALAR)
    rotations = builder.add_accessor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.1, 0.0, 0.9],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        4,
        VEC4,
    )
    gltf = builder.finish()
    gltf.nodes = [Node()]
    gltf.animations = [Animation()]
    _add_sampler(
        gltf,
        ROTATION,
        times,
        rotations,
        interpolation="CUBICSPLINE",
    )

    for time in [-1.0, 0.0, 0.2, 0.7, 1.0, 2.0, 3.0, 5.0]:
        rotation = gltf.sample_animation(0, time)[0].rotation
        assert math.sqrt(sum(value * value for value in rotation)) == pytest.approx(
            1.0
        )


def test_accessor_and_bufferview_offsets_and_bytestride():
    builder = _Builder()
    # Padding before the bufferView and between accessor start and data verifies
    # that both offsets are applied.  A stride larger than the element verifies
    # that interleaving is handled.
    times = builder.add_accessor(
        [[0.0], [2.0]], 1, SCALAR, prefix=b"\x00" * 16
    )
    translations = builder.add_accessor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        3,
        VEC3,
        prefix=b"\x00" * 20,
        byte_stride=20,
    )
    gltf = builder.finish()
    gltf.nodes = [Node()]
    gltf.animations = [Animation()]
    _add_sampler(gltf, TRANSLATION, times, translations)

    assert gltf.sample_animation(0, 1.0)[0].translation == pytest.approx(
        [2.5, 3.5, 4.5]
    )


def test_glb_datauri_and_local_buffer_sources(tmp_path):
    reference = _basic_animation()
    expected = reference.sample_animation(0, 2.0)[0]

    glb = GLTF2.load_from_bytes(b"".join(reference.save_to_bytes()))
    assert glb.sample_animation(0, 2.0)[0] == expected

    data_uri_gltf = _basic_animation()
    data_uri_gltf.buffers[0].uri = (
        "data:application/octet-stream;base64,"
        + base64.b64encode(data_uri_gltf.binary_blob()).decode("ascii")
    )
    data_uri_gltf.destroy_binary_blob()
    assert data_uri_gltf.sample_animation(0, 2.0)[0] == expected

    local_gltf = _basic_animation()
    local_blob = local_gltf.binary_blob()
    (tmp_path / "animation.bin").write_bytes(local_blob)
    local_gltf.destroy_binary_blob()
    local_gltf.buffers[0].uri = "animation.bin"
    local_gltf._path = tmp_path
    assert local_gltf.sample_animation(0, 2.0)[0] == expected


def test_defaults_and_unanimated_properties_are_preserved():
    builder = _Builder()
    times = builder.add_accessor([[0.0], [1.0]], 1, SCALAR)
    translations = builder.add_accessor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], 3, VEC3
    )
    gltf = builder.finish()
    gltf.nodes = [
        Node(translation=[99.0, 99.0, 99.0]),
        Node(rotation=[0.0, 0.0, 0.0, 1.0], scale=[]),
    ]
    gltf.animations = [Animation()]
    _add_sampler(gltf, TRANSLATION, times, translations)

    transform = gltf.sample_animation(0, 0.0)[0]
    assert transform.rotation == pytest.approx([0.0, 0.0, 0.0, 1.0])
    assert transform.scale == pytest.approx([1.0, 1.0, 1.0])


def test_sampling_does_not_modify_gltf_or_buffer_data():
    gltf = _basic_animation()
    json_before = gltf.to_json()
    blob_before = bytes(gltf.binary_blob())
    node_before = copy.deepcopy(gltf.nodes[0])
    accessors_before = copy.deepcopy(gltf.accessors)
    buffer_views_before = copy.deepcopy(gltf.bufferViews)

    sample = gltf.sample_animation(0, 0.75)
    sample[0].translation[0] = 12345.0
    sample[0].rotation[0] = 12345.0
    sample[0].scale[0] = 12345.0

    assert gltf.to_json() == json_before
    assert bytes(gltf.binary_blob()) == blob_before
    assert gltf.nodes[0] == node_before
    assert gltf.accessors == accessors_before
    assert gltf.bufferViews == buffer_views_before
    assert gltf.sample_animation(0, 0.75)[0].translation == pytest.approx(
        [7.5, 0.0, 0.0]
    )


def _single_channel_gltf(path, interpolation="LINEAR"):
    builder = _Builder()
    times = builder.add_accessor([[0.0], [1.0]], 1, SCALAR)
    values = builder.add_accessor(
        [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], 3, VEC3
    )
    gltf = builder.finish()
    gltf.nodes = [Node()]
    gltf.animations = [Animation()]
    _add_sampler(gltf, path, times, values, interpolation)
    return gltf


def test_morph_weight_channels_are_rejected():
    gltf = _single_channel_gltf(WEIGHTS)
    with pytest.raises(UnsupportedAnimationChannelError, match="weights"):
        gltf.sample_animation(0, 0.5)


def test_extension_channels_are_rejected():
    gltf = _single_channel_gltf(TRANSLATION)
    gltf.animations[0].channels[0].target.extensions = {
        "KHR_animation_pointer": {"pointer": "/nodes/0/translation"}
    }
    with pytest.raises(UnsupportedAnimationChannelError, match="extension"):
        gltf.sample_animation(0, 0.5)


def test_duplicate_node_property_channels_are_rejected():
    gltf = _single_channel_gltf(TRANSLATION)
    gltf.animations[0].channels.append(
        AnimationChannel(
            sampler=0,
            target=AnimationChannelTarget(node=0, path=TRANSLATION),
        )
    )
    with pytest.raises(DuplicateAnimationChannelError, match="both target"):
        gltf.sample_animation(0, 0.5)


def test_unknown_interpolation_is_rejected():
    gltf = _single_channel_gltf(TRANSLATION)
    gltf.animations[0].samplers[0].interpolation = "LINEAR_SPLINE"
    with pytest.raises(UnsupportedAnimationChannelError, match="interpolation"):
        gltf.sample_animation(0, 0.5)


def test_non_monotonic_keyframes_are_rejected():
    builder = _Builder()
    times = builder.add_accessor([[0.0], [2.0], [1.0]], 1, SCALAR)
    translations = builder.add_accessor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], 3, VEC3
    )
    gltf = builder.finish()
    gltf.nodes = [Node()]
    gltf.animations = [Animation()]
    _add_sampler(gltf, TRANSLATION, times, translations)

    with pytest.raises(InvalidKeyframeError, match="strictly increasing"):
        gltf.sample_animation(0, 0.5)


def test_mismatched_keyframe_count_is_rejected():
    builder = _Builder()
    times = builder.add_accessor([[0.0], [1.0]], 1, SCALAR)
    translations = builder.add_accessor([[0.0, 0.0, 0.0]], 3, VEC3)
    gltf = builder.finish()
    gltf.nodes = [Node()]
    gltf.animations = [Animation()]
    _add_sampler(gltf, TRANSLATION, times, translations)

    with pytest.raises(InvalidKeyframeError, match="output values"):
        gltf.sample_animation(0, 0.5)
