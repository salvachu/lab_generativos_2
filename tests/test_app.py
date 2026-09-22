"""HTTP contract checks without loading or training a neural network."""

from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.server import create_app


@pytest.fixture
def demo(tmp_path):
    checkpoint = tmp_path / "A-smoke" / "checkpoint.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"test checkpoint")
    calls = []

    def load_model(path, device):
        calls.append(("load", path, device))
        return object()

    def sample_multiple(handle, prefix=None, **kwargs):
        calls.append(("multiple", prefix, kwargs))
        candidates = []
        for index in range(kwargs["n_candidates"]):
            strokes = [*(prefix or []), np.array([[60, 70], [80 + index, 90]], dtype=float)]
            candidates.append({"id": index, "seed": kwargs["seed"] + index, "raw_output": strokes, "postprocessed_output": strokes, "validation": {"valid": True, "score": 1., "warnings": [], "penalties": {}, "corrections": []}, "termination": {"eos": True}})
        return {"selected": candidates[:kwargs["top_k"]], "candidates": candidates, "report": {"n_candidates": len(candidates)}}

    backend = SimpleNamespace(
        load_model=load_model,
        sample_multiple=sample_multiple,
        render=lambda strokes, prefix_count=0: f'<svg xmlns="http://www.w3.org/2000/svg" data-prefix="{prefix_count}"/>',
    )
    return TestClient(create_app(tmp_path, backend)), calls, backend, checkpoint


def test_static_demo_and_models(demo):
    client, _, _, _ = demo
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    models = client.get("/api/models").json()
    assert models["canvas_size"] == 512
    assert models["models"][0]["id"] == "A-smoke/checkpoint.pt"


def test_random_generation_uses_backend_and_cached_checkpoint(demo):
    client, calls, _, _ = demo
    request = {"model": "A-smoke/checkpoint.pt", "n_candidates": 8, "top_k": 3, "seed": 27}
    first = client.post("/api/generate", json=request)
    second = client.post("/api/generate", json=request)
    assert first.status_code == 200
    assert first.json() == second.json()
    assert len(first.json()["samples"]) == 3
    assert first.json()["prefix_count"] == 0
    assert sum(call[0] == "load" for call in calls) == 1
    assert calls[1][2]["seed"] == 27
    assert calls[1][1] is None
    assert len(first.json()["candidates"]) == 8
    assert "raw_output" not in first.json()["candidates"][0]


def test_completion_keeps_every_prefix_coordinate(demo):
    client, calls, _, _ = demo
    prefix = [[[1.12, 2.3456], [2.2, 3.3]], [[40.0, 10.0]]]
    response = client.post("/api/generate", json={"model": "A-smoke/checkpoint.pt", "prefix": prefix})
    assert response.status_code == 200
    result = response.json()
    assert result["prefix_count"] == 2
    assert len(result["samples"]) == 6
    for sample in result["samples"]:
        assert sample["raw_strokes"][:2] == prefix
        assert sample["postprocessed_strokes"][:2] == prefix
        assert 'data-prefix="2"' in sample["raw_svg"]
    assert calls[1][0] == "multiple"
    assert calls[1][1][0].tolist() == prefix[0]


@pytest.mark.parametrize("prefix", [[[]], [[[0.0, 1.0]]] * 97, [[[0.0, 1.0]] * 4097], [[[1e9, 0.0]]]])
def test_input_limits(demo, prefix):
    client, calls, _, _ = demo
    response = client.post("/api/generate", json={"model": "A-smoke/checkpoint.pt", "prefix": prefix})
    assert response.status_code == 422
    assert not calls


@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_input_returns_validation_error(demo, number):
    client, calls, _, _ = demo
    response = client.post(
        "/api/generate",
        content='{"model":"A-smoke/checkpoint.pt","prefix":[[[' + number + ',0]]]}',
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    assert not calls


@pytest.mark.parametrize("model", ["../checkpoint.pt", "/tmp/checkpoint.pt"])
def test_checkpoint_cannot_escape_runs(demo, model):
    client, calls, _, _ = demo
    assert client.post("/api/generate", json={"model": model}).status_code == 400
    assert not calls


def test_missing_checkpoint_has_actionable_error(demo):
    client, calls, _, _ = demo
    result = client.post("/api/generate", json={"model": "missing/checkpoint.pt"})
    assert result.status_code == 404
    assert "checkpoint" in result.json()["detail"]
    assert not calls


def test_no_fake_results_when_runs_is_empty(tmp_path):
    client = TestClient(create_app(tmp_path))
    assert client.get("/api/models").json()["models"] == []
    assert client.post("/api/generate", json={"model": "A/checkpoint.pt"}).status_code == 404


def test_backend_cannot_silently_modify_prefix(demo):
    client, _, backend, _ = demo
    backend.sample_multiple = lambda handle, **kwargs: {"selected": [{"raw_output": [np.array([[999, 999]])], "postprocessed_output": [np.array([[999, 999]])]}]}
    result = client.post("/api/generate", json={"model": "A-smoke/checkpoint.pt", "prefix": [[[1, 2]]]})
    assert result.status_code == 500


def test_backend_nonfinite_output_rejected(demo):
    client, _, backend, _ = demo
    backend.sample_multiple = lambda handle, **kwargs: {"selected": [{"id": 4, "raw_output": [np.array([[np.nan, 1]])], "postprocessed_output": [np.array([[np.nan, 1]])]}]}
    response = client.post("/api/generate", json={"model": "A-smoke/checkpoint.pt", "validate": False})
    assert response.status_code == 200
    assert response.json()["samples"] == []
    assert response.json()["report"]["unrenderable_selected"] == [4]


def test_best_checkpoint_preferred_without_duplicate_aliases(demo):
    client, _, _, checkpoint = demo
    for name in ("best.pt", "last.pt"):
        checkpoint.with_name(name).write_bytes(b"checkpoint alias")
    models = client.get("/api/models").json()["models"]
    assert len(models) == 1
    assert models[0]["id"] == "A-smoke/best.pt"
    assert client.post("/api/generate", json={"model": "A-smoke/last.pt"}).status_code == 200


def test_all_invalid_is_a_successful_empty_result(demo):
    client, _, backend, _ = demo
    backend.sample_multiple = lambda handle, **kwargs: {"selected": [], "candidates": [], "report": {"rejected_invalid": 20}}
    response = client.post("/api/generate", json={"model": "A-smoke/checkpoint.pt"})
    assert response.status_code == 200
    assert response.json()["samples"] == []
    assert response.json()["report"]["rejected_invalid"] == 20


def test_raw_mode_preserves_outputs_and_passes_explicit_flags(demo):
    client, calls, _, _ = demo
    response = client.post("/api/generate", json={"model": "A-smoke/checkpoint.pt", "validate": False, "postprocess": False, "rerank": False})
    assert response.status_code == 200
    assert response.json()["validated"] is False
    assert calls[1][2]["validate"] is False
    assert calls[1][2]["postprocess"] is False
    assert calls[1][2]["rerank"] is False
    sample = response.json()["samples"][0]
    assert sample["raw_strokes"] == sample["postprocessed_strokes"]


@pytest.mark.parametrize("options", [{"n_candidates": 2, "top_k": 6}, {"n_candidates": 65}, {"top_k": 10}])
def test_sampling_resource_limits(demo, options):
    client, calls, _, _ = demo
    assert client.post("/api/generate", json={"model": "A-smoke/checkpoint.pt", **options}).status_code == 422
    assert not calls
