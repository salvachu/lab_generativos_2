import numpy as np
import pytest

from sketchlab.geometry import fit_geometry, postprocess_candidate
from sketchlab.ranking import rank_candidates, suffix_chamfer, diversity_metrics


def fixture():
    sketches = []
    for offset in range(10):
        sketch = [np.array([[10+offset, 10], [20+offset, 20], [30+offset, 15]], dtype=np.float64),
                  np.array([[60+offset, 60], [70+offset, 70], [80+offset, 65]], dtype=np.float64)]
        sketches.append({"strokes": sketch})
    return sketches, fit_geometry(sketches)


def test_invalid_never_selected_and_duplicates_filtered():
    sketches, stats = fixture()
    first = postprocess_candidate(sketches[0]["strokes"], stats, termination="eos")
    second = postprocess_candidate(sketches[8]["strokes"], stats, termination="eos")
    corrupt = postprocess_candidate([np.array([[1e8, 1e8]])], stats, termination="eos")
    result = rank_candidates([first, dict(first), corrupt, second], 4, stats)
    assert result["report"]["invalid_count"] == 1
    assert result["report"]["selected_count"] == 2
    assert 2 not in result["selected_indices"]
    assert len(result["report"]["diversity_filtered"]) == 1
    assert all(r["validation"]["valid"] for r in result["selected"])


def test_suffix_only_chamfer_ignores_shared_prefix():
    prefix = [np.zeros((1000, 2))]
    a = [np.array([[0, 0], [1, 1]], dtype=float)]
    b = [np.array([[10, 0], [11, 1]], dtype=float)]
    assert suffix_chamfer(prefix+a, prefix+b, 1) == suffix_chamfer(a, b)
    assert suffix_chamfer(a, a) == 0


def test_same_polyline_with_different_point_density_is_duplicate():
    sparse = [np.array([[10, 10], [30, 30]], dtype=float)]
    dense = [np.array([[10, 10], [20, 20], [30, 30]], dtype=float)]
    assert suffix_chamfer(sparse, dense) == pytest.approx(0.0, abs=1e-12)
    _, stats = fixture()
    records = [{"postprocessed_output": stroke, "validation": {"valid": True, "score": 1.0}}
               for stroke in (sparse, dense)]
    assert rank_candidates(records, 2, stats)["report"]["selected_count"] == 1


def test_lexicographic_configuration_and_analysis_report():
    sketches, stats = fixture()
    records = [postprocess_candidate(s["strokes"], stats, termination="eos") for s in (sketches[0], sketches[8])]
    records[0]["model_score"] = -10
    records[1]["model_score"] = -1
    ranked = rank_candidates(records, 2, stats, {"ordering": "model_first"})
    assert ranked["selected_indices"][0] == 1
    assert ranked["report"]["diversity"]["suffix_chamfer_min"] > 0
    metrics = diversity_metrics(ranked["selected"])
    assert metrics["pair_count"] == 1
    assert metrics["generated_stroke_count_std"] == 0
    assert rank_candidates(records, 0, stats)["selected"] == []
