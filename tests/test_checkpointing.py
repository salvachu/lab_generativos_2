import numpy as np
import pytest
import torch

from sketchlab.checkpointing import save_checkpoint, load_checkpoint
from sketchlab.models import create_model
from sketchlab.training import run_experiment, BatchStream
from sketchlab.diagnostics import kl_schedule, collapse_report


@pytest.mark.parametrize("variant", list("ABCDE"))
def test_all_models_checkpoint_and_overwrite_guard(tmp_path, variant):
    model = create_model({"model": variant, "hidden_dim": 8, "latent_dim": 4, "mixtures": 2, "stroke_latent_dim": 3})
    path = tmp_path / "weights.pt"
    save_checkpoint(path, model)
    loaded, metadata = load_checkpoint(path)
    for key, value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key])
    assert metadata["normalization"]["scale"] == 256
    assert not loaded.training
    with pytest.raises(FileExistsError):
        save_checkpoint(path, model)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_exact_resume_matches_uninterrupted_training(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    samples = [{"id": i, "strokes": [np.array([[10.+i, 10.], [11.+i, 11.]], dtype=np.float32),
                                      np.array([[20., 20.], [21., 21.]], dtype=np.float32)]} for i in range(3)]
    config = {"seed": 15, "device": device, "cpu_threads": 1, "batch_size": 2, "steps": 4,
              "max_train_seconds": 60, "eval_every": 2, "learning_rate": .001,
              "generation_eval": False, "warmup_steps": 10, "beta": .05,
              "architecture": {"hidden_dim": 8, "latent_dim": 4, "mixtures": 2,
                               "layers": 2, "dropout": .1, "stroke_latent_dim": 3}}
    run_experiment(config, "B", samples, samples[:1], tmp_path / "continuous")
    run_experiment({**config, "steps": 2}, "B", samples, samples[:1], tmp_path / "first")
    run_experiment({**config, "steps": 2}, "B", samples, samples[:1], tmp_path / "resumed",
                   resume=tmp_path / "first" / "last.pt")
    a = torch.load(tmp_path / "continuous" / "last.pt", weights_only=True)
    b = torch.load(tmp_path / "resumed" / "last.pt", weights_only=True)
    assert a["step"] == b["step"] == 4
    assert a["training_state"]["sampler"] == b["training_state"]["sampler"]
    for key in a["model_state"]:
        assert torch.equal(a["model_state"][key], b["model_state"][key]), key
    assert torch.equal(a["rng_state"]["torch_cpu"], b["rng_state"]["torch_cpu"])


def test_resume_requires_optimizer_and_same_representation(tmp_path):
    model = create_model({"hidden_dim": 8, "latent_dim": 4, "mixtures": 2})
    path = tmp_path / "state.pt"
    save_checkpoint(path, model)
    with pytest.raises(ValueError, match="optimizer"):
        load_checkpoint(path, mode="resume")
    optimizer = torch.optim.Adam(model.parameters())
    load_checkpoint(path, model=model, optimizer=optimizer, mode="warm_start")
    assert len(optimizer.state) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_resume_cuda_alias_matches_cuda_zero(tmp_path):
    model = create_model({"hidden_dim": 8, "latent_dim": 4, "mixtures": 2}).to("cuda")
    optimizer = torch.optim.Adam(model.parameters())
    path = tmp_path / "cuda.pt"
    save_checkpoint(path, model, optimizer=optimizer)
    loaded, _ = load_checkpoint(path, model=model, optimizer=optimizer, mode="resume", device="cuda")
    assert str(loaded.device) == "cuda:0"


def test_sampler_keeps_partial_batch_and_serializes_prefix_rng():
    train = [{"strokes": [np.zeros((2,2)), np.ones((2,2))]} for _ in range(3)]
    stream = BatchStream(3, 2, 1)
    assert len(stream.next(train)[0]) == 2
    restored = BatchStream(3, 2, 1, stream.state_dict())
    a, b = stream.next(train), restored.next(train)
    assert len(a[0]) == len(b[0]) == 1
    assert a[1] == b[1]


def test_schedule_and_collapse_alerts_require_history():
    c = {"beta": .1, "warmup_steps": 10, "kl_schedule": "linear"}
    assert kl_schedule(0, c) == pytest.approx(.01)
    assert kl_schedule(100, c) == .1
    assert not collapse_report([{"kl": 0.}])["persistent_near_zero_kl"]
    assert collapse_report([{"kl": 0.}] * 20)["persistent_near_zero_kl"]
