import numpy as np
import pytest
import torch

from sketchlab.models import create_model
from sketchlab.training import validate


@pytest.mark.parametrize("variant", list("ABCDE"))
def test_validation_is_independent_of_batch_partition(variant):
    torch.manual_seed(3)
    torch.set_num_threads(1)
    model = create_model({"model": variant, "hidden_dim": 8, "latent_dim": 4, "mixtures": 2, "stroke_latent_dim": 3}).eval()
    samples = []
    for n in (1, 80, 15):
        stroke = np.stack([np.linspace(100, 120, n), np.linspace(90, 150, n)], axis=-1).astype(np.float32)
        samples.append({"strokes": [np.array([[20., 30.]], dtype=np.float32), stroke]})
    single = validate(model, samples, 1, .05, .02)
    batch = validate(model, samples, 3, .05, .02)
    for key in ("coordinate_nll", "pen_ce", "reconstruction", "global_kl", "stroke_kl", "loss", "negative_elbo"):
        assert single[key] == pytest.approx(batch[key], rel=2e-5, abs=2e-6), (variant, key)
    for key in ("coordinate_events", "pen_events", "stroke_events"):
        assert single[key] == batch[key]
