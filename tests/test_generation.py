import json

import numpy as np
import pytest
import torch

from sketchlab.generation import (complete_sketch, encode, generate_random, load_model,
                                  sample, sample_latent)
from sketchlab.models import create_model


@pytest.fixture(autouse=True)
def small_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def model_for_test(name):
    torch.manual_seed(9)
    model = create_model({"model": name, "hidden_dim": 12, "latent_dim": 4,
                          "stroke_latent_dim": 3, "mixtures": 2}).eval()
    # Force finite non-empty examples for exact preservation/limit tests.
    with torch.no_grad():
        if name in "ABC":
            model.pen_head.weight.zero_()
            model.pen_head.bias.copy_(torch.tensor([-30., 30., -30.]))
        else:
            model.sketch_end_head.weight.zero_()
            model.sketch_end_head.bias.copy_(torch.tensor([30., -30.]))
            model.point_end_head.weight.zero_()
            model.point_end_head.bias.copy_(torch.tensor([-30., 30.]))
    return model


def prefix():
    return [np.array([[-2., 504.], [200., 120.]], np.float32),
            np.array([[120.3, 3.82e-11]], np.float64)]


@pytest.mark.parametrize("name", list("ABCDE"))
def test_completion_preserves_prefix_exactly_and_sampling_is_reproducible(name):
    model = model_for_test(name)
    original = prefix()
    first, info = sample(model, original, seed=8, decoder_seed=19, max_points=8, max_strokes=3, return_info=True)
    second = sample(model, original, seed=8, decoder_seed=19, max_points=8, max_strokes=3)
    assert len(first) == len(original) + 3
    for actual, expected, repeated in zip(first, original + first[len(original):], second):
        assert np.array_equal(actual, expected)
        assert np.array_equal(actual, repeated)
        assert actual.dtype == expected.dtype
        assert np.isfinite(actual).all()
    for copy, source in zip(first, original):
        assert not np.shares_memory(copy, source)
    assert info["generated_strokes"] == 3 and info["generated_points"] == 3
    assert info["capped"] and not info["ended_by_eos"]
    assert info["termination"] == "max_strokes"
    json.dumps(info, allow_nan=False)


@pytest.mark.parametrize("name", list("ABCDE"))
def test_fixed_decoder_noise_changes_geometry_when_global_latent_changes(name):
    model = model_for_test(name)
    a = sample(model, prefix(), z=np.zeros(4), seed=42, decoder_seed=3,
               temperature=0, max_points=3, max_strokes=2)
    b = sample(model, prefix(), z=np.ones(4), seed=42, decoder_seed=3,
               temperature=0, max_points=3, max_strokes=2)
    assert not np.allclose(a[-1], b[-1])


@pytest.mark.parametrize("name", list("ABCDE"))
def test_checkpoint_encode_prior_and_random_generation(name, tmp_path):
    model = model_for_test(name)
    checkpoint = tmp_path / f"{name}.pt"
    torch.save({"model_config": model.config, "model_state": model.state_dict()}, checkpoint)
    restored = load_model(checkpoint)
    encoded = encode(restored, prefix())
    assert encoded["mu"].shape == (4,) and encoded["logvar"].shape == (4,)
    assert np.isfinite(encoded["mu"]).all()
    np.testing.assert_array_equal(sample_latent(model, prefix(), seed=2), sample_latent(restored, prefix(), seed=2))
    assert len(generate_random(restored, n_samples=2, max_points=4, max_strokes=2)) == 2
    completed = complete_sketch(restored, prefix(), n_samples=1, z=encoded["mu"], max_points=4, max_strokes=2)
    for original, actual in zip(prefix(), completed[0]):
        np.testing.assert_array_equal(original, actual)


@pytest.mark.parametrize("name", list("ABCDE"))
def test_natural_eos_is_distinguished_from_caps(name):
    model = model_for_test(name)
    with torch.no_grad():
        if name in "ABC":
            model.pen_head.bias.copy_(torch.tensor([-30., -30., 30.]))
        else:
            model.sketch_end_head.bias.copy_(torch.tensor([-30., 30.]))
    result, info = sample(model, prefix(), return_info=True)
    assert len(result) == len(prefix())
    assert info["ended_by_eos"] and not info["capped"] and not info["partial_stroke"]


def test_sampling_does_not_silently_clip_or_normalize_coordinates():
    model = model_for_test("A")
    with torch.no_grad():
        model.coordinate_head.projection.weight.zero_()
        bias = model.coordinate_head.projection.bias.view(2, 6)
        bias[:, 1:3] = 50
    generated, info = sample(model, temperature=0, max_strokes=1, return_info=True)
    assert (generated[0][0] > 504).all()
    assert info["finite"] and info["capped"]


def test_nonfinite_prediction_stops_without_returning_corrupted_points():
    model = model_for_test("A")
    with torch.no_grad():
        model.coordinate_head.projection.bias.fill_(float("nan"))
    generated, info = sample(model, prefix(), return_info=True)
    assert len(generated) == len(prefix())
    assert info["termination"] == "nonfinite_prediction" and not info["ended_by_eos"]
    assert info["finite"]


def test_bad_generation_arguments_are_rejected():
    model = model_for_test("C")
    for kwargs in [{"temperature": float("nan")}, {"temperature": -1}, {"max_points": 0},
                   {"max_strokes": -1}, {"z": [1, 2]}, {"z": [1, 2, 3, float("inf")]}]:
        with pytest.raises(ValueError):
            sample(model, **kwargs)
    with pytest.raises(ValueError):
        sample(model, [np.array([[float("nan"), 0]])])


@pytest.mark.parametrize("name", list("ABCDE"))
def test_explicit_sampled_global_latent_matches_implicit_prior_sample(name):
    model = model_for_test(name)
    latent = sample_latent(model, prefix(), seed=71)
    implicit = sample(model, prefix(), seed=71, decoder_seed=9, max_strokes=2, max_points=6)
    explicit = sample(model, prefix(), seed=71, decoder_seed=9, z=latent, max_strokes=2, max_points=6)
    for a, b in zip(implicit, explicit):
        np.testing.assert_array_equal(a, b)
