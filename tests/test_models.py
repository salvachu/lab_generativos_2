import numpy as np
import pytest
import torch

from sketchlab.losses import bivariate_mixture_nll, gaussian_kl, masked_mean, reparameterize
from sketchlab.models import create_model
from sketchlab.models.common import batch_tokens


@pytest.fixture(autouse=True)
def small_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def examples():
    return [[np.array([[0, 0], [1, 2], [5, 5]], np.float32),
             np.array([[200, 210]], np.float32),
             np.array([[180, 120], [180, 122]], np.float32)],
            [np.array([[10, 10], [11, 12], [12, 15], [14, 13]], np.float32),
             np.array([[110, 70], [111, 68]], np.float32)]]


def tiny(name, **kwargs):
    torch.manual_seed(16)
    return create_model({"model": name, "hidden_dim": 12, "latent_dim": 4,
                         "stroke_latent_dim": 3, "mixtures": 2, **kwargs})


@pytest.mark.parametrize("name", list("ABCDE"))
@pytest.mark.parametrize("layers", [1, 2])
def test_all_models_variable_length_forward_backward(name, layers):
    model = tiny(name, layers=layers, dropout=0.1)
    result = model.batch_loss(examples(), [1, 1], beta=0.2)
    expected = {"loss", "reconstruction", "coordinate_nll", "pen_ce", "kl", "pen_accuracy",
                "auxiliary_loss", "total_loss", "negative_elbo", "elbo", "beta_effective"}
    assert expected <= result.keys()
    assert all(value.ndim == 0 and torch.isfinite(value) for value in result.values())
    assert 0 <= result["pen_accuracy"] <= 1
    assert result["beta_effective"].item() == pytest.approx(1 if name == "A" else 0.2)
    assert result["elbo"].item() == -result["negative_elbo"].item()
    result["loss"].backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert model.posterior_head.weight.grad.abs().sum() > 0
    assert model.encoder.bidirectional if name in "ABC" else model.sketch_encoder.bidirectional


@pytest.mark.parametrize("name", list("ABCDE"))
def test_scheduled_sampling_is_configurable_and_differentiable(name):
    model = tiny(name, teacher_forcing=0.0)
    result = model.batch_loss(examples(), [1, 1], beta=0.1)
    result["loss"].backward()
    assert torch.isfinite(result["loss"])
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    with pytest.raises(ValueError):
        model.batch_loss(examples(), [1, 1], beta=0.1, teacher_forcing=1.1)


@pytest.mark.parametrize("name", list("ABCDE"))
def test_encoder_ignores_batch_padding(name):
    model = tiny(name).eval()
    data = examples()
    context = model.context([[], []])
    together, _ = model.posterior(data, context)
    alone, _ = model.posterior(data[:1], context[:1])
    torch.testing.assert_close(together[0], alone[0], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("name", ["C", "E"])
def test_conditional_loss_excludes_prefix_and_eos_has_no_coordinates(name):
    model = tiny(name).eval()
    data = examples()
    result = model.batch_loss(data, [1, 1], beta=0.1, deterministic=True)
    suffix_points = sum(sum(len(s) for s in sample[1:]) for sample in data)
    suffix_strokes = sum(len(sample) - 1 for sample in data)
    assert result["coordinate_events"].item() == suffix_points
    assert result["pen_events"].item() == suffix_points + len(data) + (suffix_strokes if name == "E" else 0)
    complete_prefix = model.batch_loss(data, [len(s) for s in data], beta=0.1, deterministic=True)
    assert complete_prefix["coordinate_events"].item() == 0
    assert complete_prefix["pen_events"].item() == len(data)
    assert complete_prefix["coordinate_nll"].item() == 0


@pytest.mark.parametrize("name", ["A", "B", "D"])
def test_unconditional_training_ignores_completion_prefix_count(name):
    model = tiny(name).eval()
    a = model.batch_loss(examples(), [0, 0], 0.2, deterministic=True)
    b = model.batch_loss(examples(), [1, 1], 0.2, deterministic=True)
    torch.testing.assert_close(a["loss"], b["loss"])


@pytest.mark.parametrize("name", ["C", "E"])
def test_conditional_prior_only_observes_prefix(name):
    model = tiny(name).eval()
    original = examples()[0]
    changed = [s.copy() for s in original]
    changed[-1] += 400
    context = model.context([original[:1]])
    context_changed = model.context([changed[:1]])
    torch.testing.assert_close(context, context_changed, atol=0, rtol=0)
    first_prior, _ = model.prior(context)
    second_prior, _ = model.prior(context_changed)
    torch.testing.assert_close(first_prior, second_prior, atol=0, rtol=0)
    full_q, _ = model.posterior([original], context)
    changed_q, _ = model.posterior([changed], context)
    assert not torch.allclose(full_q, changed_q)
    other_prior, _ = model.prior(model.context([original[:2]]))
    assert not torch.allclose(first_prior, other_prior)
    if name == "E":
        embeddings, _ = model.sketch_embeddings([original, changed])
        contexts = model.context_from_embeddings(embeddings, torch.tensor([1, 1]))
        torch.testing.assert_close(contexts[0], contexts[1], atol=0, rtol=0)


@pytest.mark.parametrize("name", list("ABCDE"))
def test_decoder_causality_with_fixed_latent(name):
    model = tiny(name, layers=2).eval()
    z = torch.randn(1, model.latent_dim)
    context = model.context([examples()[0][:1]])
    if name in "ABC":
        past = torch.randn(1, 7, 5)
        future_changed = past.clone()
        future_changed[:, 4:] += 100
        a, ap, _ = model.decode(past, z, context)
        b, bp, _ = model.decode(future_changed, z, context)
        torch.testing.assert_close(a[:, :4], b[:, :4], atol=0, rtol=0)
        torch.testing.assert_close(ap[:, :4], bp[:, :4], atol=0, rtol=0)
        assert not model.decoder.bidirectional
    else:
        past = torch.randn(1, 7, model.hidden_dim)
        future_changed = past.clone()
        future_changed[:, 4:] += 100
        a, _ = model.decode_strokes(past, z, context)
        b, _ = model.decode_strokes(future_changed, z, context)
        torch.testing.assert_close(a[:, :4], b[:, :4], atol=0, rtol=0)
        condition = model.point_condition(a[:, 0], torch.randn(1, model.stroke_latent_dim), torch.zeros(1, 2))
        points = torch.randn(1, 8, 3)
        changed = points.clone()
        changed[:, 5:] -= 100
        x, xp, _ = model.decode_points(points, condition)
        y, yp, _ = model.decode_points(changed, condition)
        torch.testing.assert_close(x[:, :5], y[:, :5], atol=0, rtol=0)
        torch.testing.assert_close(xp[:, :5], yp[:, :5], atol=0, rtol=0)
        assert not model.point_decoder.bidirectional and not model.stroke_decoder.bidirectional


def test_mdn_kl_and_masks_are_numerically_stable():
    raw = torch.tensor([[[0., 0., 0., -100., 100., 100.], [100., 1., 1., 5., -5., -100.]]],
                       requires_grad=True)
    nll = bivariate_mixture_nll(raw, torch.tensor([[0.1, 0.2]]))
    assert torch.isfinite(nll).all()
    nll.sum().backward()
    assert torch.isfinite(raw.grad).all()
    zeros = torch.zeros(2, 4)
    torch.testing.assert_close(gaussian_kl(zeros, zeros), torch.zeros(2))
    torch.testing.assert_close(gaussian_kl(zeros, zeros, free_bits=0.1), torch.full((2,), 0.4))
    masked = masked_mean(torch.tensor([2., 4., float("nan")]), torch.tensor([True, True, False]))
    assert masked.item() == 3
    mu, logvar = torch.ones(2, 4, requires_grad=True), torch.zeros(2, 4, requires_grad=True)
    torch.testing.assert_close(reparameterize(mu, logvar, deterministic=True), mu)
    torch.manual_seed(42)
    random_z = reparameterize(mu, logvar)
    assert not torch.equal(random_z, mu)
    random_z.sum().backward()
    assert mu.grad is not None and logvar.grad is not None


def test_internal_delta_token_mask_keeps_every_point_and_eos():
    data = examples()
    padded, lengths, valid = batch_tokens(data, torch.device("cpu"))
    assert lengths.tolist() == [sum(len(s) for s in sketch) + 1 for sketch in data]
    assert valid.sum().item() == sum(lengths.tolist())
    assert padded.shape[-1] == 5


@pytest.mark.parametrize("name", list("ABCDE"))
def test_more_than_two_hundred_strokes_and_long_stroke_are_not_truncated(name):
    model = tiny(name).eval()
    singleton = np.array([[-2., 504.]], np.float32)
    long_stroke = np.stack([np.linspace(0, 500, 257), np.linspace(2, 400, 257)], -1).astype(np.float32)
    data = [[singleton.copy() for _ in range(210)] + [long_stroke]]
    result = model.batch_loss(data, [0], beta=0.1, deterministic=True)
    assert result["coordinate_events"].item() == 210 + 257
    assert torch.isfinite(result["loss"])
