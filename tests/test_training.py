import numpy as np
import pytest
import torch

from sketchlab.models import create_model
from sketchlab.training import teacher_forcing_at, validate


def test_teacher_forcing_schedule_has_warmup_and_bounded_linear_ramp():
    config = {"architecture": {"teacher_forcing": 1.0},
              "teacher_forcing_schedule": {"type": "linear_self_feed",
                                             "warmup_fraction": .2, "final_self_feed": .25}}
    assert teacher_forcing_at(0, 600, config) == 1.0
    assert teacher_forcing_at(119, 600, config) == 1.0
    assert teacher_forcing_at(599, 600, config) == pytest.approx(.75)
    assert .75 < teacher_forcing_at(360, 600, config) < 1.0

    with pytest.raises(ValueError, match="final_self_feed"):
        teacher_forcing_at(0, 600, {"teacher_forcing_schedule": {
            "type": "linear_self_feed", "warmup_fraction": .2, "final_self_feed": .31}})


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
