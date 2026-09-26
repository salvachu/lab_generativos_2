"""Smoke checks for the exact models distributed with the local demo."""

import pytest
from fastapi.testclient import TestClient

from app.server import create_app


@pytest.mark.parametrize("name", ["H9", "H13", "H20"])
def test_packaged_model_completes_on_cpu(name, monkeypatch):
    monkeypatch.setenv("SKETCHLAB_DEMO_DEVICE", "cpu")
    client = TestClient(create_app())
    model_id = f"artifacts/models/{name}/best.pt"
    available = client.get("/api/models").json()
    assert available["demo_device"] == "cpu"
    assert model_id in {model["id"] for model in available["models"]}

    prefix = [[[100.0, 100.0], [180.0, 160.0]]]
    response = client.post("/api/demo/generate", json={
        "model": model_id, "prefix": prefix, "seed": 41,
        "n_candidates": 1, "top_k": 1, "validate": False,
    })
    assert response.status_code == 200
    result = response.json()
    assert len(result["samples"]) == 1
    assert result["samples"][0]["postprocessed_strokes"][0] == prefix[0]
