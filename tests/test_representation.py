from xml.etree import ElementTree

import numpy as np
import pytest

from sketchlab.representation import (
    encode_strokes, decode_tokens, encode_deltas, decode_deltas,
    pad_sequences, in_stroke_mask, prefix_token_count,
    encode_sample, decode_sample,
)
from sketchlab.rendering import render_svg, render_grid


def strokes():
    return [np.array([[100, 200], [110.125, 210.5]], dtype=np.float32),
            np.array([[450, 490]], dtype=np.float32),
            np.array([[-1.9, 3.8224968329503284e-11], [400, 300]], dtype=np.float32)]


def test_exact_absolute_roundtrip_including_near_zero():
    original = strokes()
    tokens = encode_strokes(original)
    restored = decode_tokens(tokens)
    assert tokens.dtype == np.float64
    assert len(tokens) == 6
    np.testing.assert_array_equal(tokens[-1], [0, 0, 0, 0, 1])
    assert len(restored) == len(original)
    for a, b in zip(original, restored):
        np.testing.assert_array_equal(a, b)


def test_complete_raw_sample_roundtrip_and_metadata_copy():
    original = {"id": 4322, "strokes": strokes(), "parts": ["body", "eye", "detail"],
                "step_ids": [0, 1, 1], "description": "Near-zero coordinate"}
    encoded = encode_sample(original)
    restored = decode_sample(encoded)
    assert set(original) == set(restored)
    for key in ("id", "parts", "step_ids", "description"):
        assert original[key] == restored[key]
    for a, b in zip(original["strokes"], restored["strokes"]):
        assert a.dtype == b.dtype
        assert a.tobytes() == b.tobytes()
    encoded["metadata"]["parts"][0] = "modified"
    assert original["parts"][0] == "body"
    assert restored["parts"][0] == "body"


def test_relative_view_has_same_boundaries_but_finite_precision():
    original = strokes()
    canonical = encode_strokes(original)
    relative = encode_deltas(original)
    np.testing.assert_array_equal(relative[:, 2:], canonical[:, 2:])
    for a, b in zip(original, decode_deltas(relative)):
        np.testing.assert_allclose(a, b, atol=1e-12, rtol=0)
    # The near-zero example proves that delta float64 is not bit-exact.
    assert not np.array_equal(original[2], decode_deltas(relative)[2])


def test_prefix_encoding_does_not_depend_on_future():
    original = strokes()
    prefix = original[:2]
    for encoder in (encode_strokes, encode_deltas):
        np.testing.assert_array_equal(encoder(prefix)[:-1], encoder(original)[:prefix_token_count(original, 2)])


def test_inter_stroke_moves_excluded_from_ink_mask():
    tokens = encode_deltas(strokes())
    np.testing.assert_array_equal(in_stroke_mask(tokens), [False, True, False, False, True, False])
    assert np.linalg.norm(tokens[2, :2]) > 1  # large relocation is valid


def test_padding_preserves_every_token_and_includes_eos():
    sequences = [encode_strokes(strokes()), encode_strokes(strokes()[:1])]
    batch = pad_sequences(sequences)
    assert batch["tokens"].shape == (2, 6, 5)
    assert batch["mask"].sum() == 9
    assert batch["mask"][1, 2]  # EOS has a target
    assert not batch["mask"][1, 3]
    assert not batch["tokens"][1, 3:].any()


def test_empty_sketch_is_eos_and_bad_shapes_fail():
    assert decode_tokens(encode_strokes([])) == []
    with pytest.raises(ValueError):
        encode_strokes([np.empty((0, 2))])
    with pytest.raises(ValueError):
        encode_strokes(strokes(), scale=0)


def test_many_strokes_long_stroke_and_boundaries():
    many = [np.array([[-1.9937844, 504]], dtype=np.float32) for _ in range(211)]
    long = np.stack([np.linspace(-1.4159602, 504, 1148), np.linspace(504, 0, 1148)], axis=1).astype(np.float32)
    original = [*many, long]
    restored = decode_tokens(encode_strokes(original))
    assert len(restored) == 212
    for a, b in zip(original, restored):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_representation_rejected(value):
    with pytest.raises(ValueError, match="finite"):
        encode_strokes([np.array([[value, 1]])])
    with pytest.raises(ValueError, match="finite"):
        decode_tokens(np.array([[value, 0, 0, 1, 0]]))


def test_svg_preserves_stroke_separation_and_prefix_color(tmp_path):
    svg = render_svg(strokes(), prefix_count=1, title="<safe>", show_order=True)
    root = ElementTree.fromstring(svg)
    namespace = {"s": "http://www.w3.org/2000/svg"}
    polylines = root.findall("s:polyline", namespace)
    assert len(polylines) == 3
    assert polylines[0].attrib["stroke"] != polylines[1].attrib["stroke"]
    assert root.find("s:title", namespace).text == "<safe>"
    serialized = np.array([[float(value) for value in point.split(",")]
                           for point in polylines[2].attrib["points"].split()])
    np.testing.assert_array_equal(serialized, strokes()[2])
    target = render_grid([strokes()], tmp_path / "render.png", prefix_counts=[1])
    assert target.stat().st_size > 500
