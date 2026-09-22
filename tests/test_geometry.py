import json
from copy import deepcopy

import numpy as np
import pytest

from sketchlab.geometry import fit_geometry, validate_candidate, postprocess_candidate, save_geometry, load_geometry


@pytest.fixture
def train_fixture():
    samples = []
    for index in range(30):
        angles = np.linspace(0, 2*np.pi, 20 + index % 5)
        radius = 25 + index
        strokes = []
        for part in range(5 + index % 3):
            center = np.array([80 + part*50, 120 + index*5])
            strokes.append(center + radius*np.stack([np.cos(angles), np.sin(angles)], axis=1))
        samples.append({"strokes": strokes})
    return samples


@pytest.fixture
def stats(train_fixture):
    return fit_geometry(train_fixture)


def test_calibration_train_only_and_persistence(stats, tmp_path, train_fixture):
    assert stats["fit_split"] == "train"
    assert stats["intra_segment_p995"] == stats["distributions"]["segment_length"]["soft_upper"]
    assert stats["distributions"]["stroke_count"]["median"] == 6
    with pytest.raises(ValueError, match="TRAIN only"):
        fit_geometry(train_fixture, source_path=tmp_path / "val.pkl")
    path = save_geometry(stats, tmp_path/"geometry.json")
    assert load_geometry(path) == stats


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_prefix_or_suffix_rejected(stats, train_fixture, bad):
    sketch = deepcopy(train_fixture[0]["strokes"])
    sketch[0][0, 0] = bad
    result = validate_candidate(sketch, stats, prefix_count=1, termination="eos")
    assert not result.valid
    assert "nonfinite_stroke:0" in result.severe_errors


def test_extreme_coordinates_rejected_without_silent_clipping(stats, train_fixture):
    sketch = deepcopy(train_fixture[0]["strokes"])
    sketch[0][0] = [1e308, -1e308]
    record = postprocess_candidate(sketch, stats, termination="eos")
    assert not record["validation"].valid
    assert "extreme_coordinates" in record["validation"].severe_errors
    np.testing.assert_array_equal(record["postprocessed_output"][0], sketch[0])
    json.dumps(record["validation"].to_dict(), allow_nan=False)


def test_intra_stroke_anomaly_not_inter_stroke_relocation(stats):
    low = np.array(stats["coordinates"]["min"]) + 5
    high = np.array(stats["coordinates"]["max"]) - 5
    connected = validate_candidate([np.stack([low, high])], stats, termination="eos")
    relocated = validate_candidate([low[None, :], high[None, :]], stats, termination="eos")
    assert "segment_length" in connected.penalties
    assert "segment_length" not in relocated.penalties
    assert "extreme_segment_length" in connected.severe_errors


def test_prefix_unmodified_and_not_scored_against_train_bounds(stats, train_fixture):
    prefix = np.array([[10000.0, 20000.0], [11000.0, 25000.0]], dtype=np.float32)
    generated = train_fixture[0]["strokes"]
    with_prefix = postprocess_candidate([prefix, *generated], stats, prefix_count=1, termination="eos")
    ordinary_prefix = np.array([[100, 100], [110, 110]], dtype=np.float32)
    baseline = validate_candidate([ordinary_prefix, *generated], stats, prefix_count=1, termination="eos")
    assert with_prefix["validation"].valid == baseline.valid
    assert with_prefix["validation"].penalties == baseline.penalties
    assert with_prefix["postprocessed_output"][0].dtype == prefix.dtype
    assert with_prefix["postprocessed_output"][0].tobytes() == prefix.tobytes()


def test_only_suffix_duplicates_removed_and_raw_retained(stats, train_fixture):
    base = train_fixture[0]["strokes"][0]
    duplicated = np.repeat(base[:4], 2, axis=0)
    record = postprocess_candidate([duplicated.copy(), duplicated.copy()], stats, prefix_count=1, termination="eos")
    assert record["validation"].valid
    assert len(record["postprocessed_output"][0]) == 8
    assert len(record["postprocessed_output"][1]) == 4
    assert len(record["raw_output"][1]) == 8
    assert record["validation"].corrections[0]["stroke_index"] == 1
    assert stats["config"]["smoothing_alpha"] == 0


def test_caps_missing_eos_and_exploding_counts(stats, train_fixture):
    strokes = train_fixture[0]["strokes"]
    for termination in ({"capped": True, "ended_by_eos": False}, "max_points"):
        result = validate_candidate(strokes, stats, termination=termination)
        assert not result.valid and "missing_eos" in result.severe_errors
    assert "termination_unreported" in validate_candidate(strokes, stats).warnings
    too_many = [strokes[0]] * (int(stats["distributions"]["stroke_count"]["severe_upper"]) + 1)
    result = validate_candidate(too_many, stats, termination="eos")
    assert "extreme_stroke_count" in result.severe_errors


def test_degenerate_stroke_is_a_warning_and_empty_stroke_is_corrupt(stats):
    point = np.mean([stats["coordinates"]["min"], stats["coordinates"]["max"]], axis=0)
    singleton = validate_candidate([point[None, :]], stats, termination="eos")
    assert singleton.valid
    assert "degenerate_generated_strokes" in singleton.warnings
    malformed = validate_candidate([np.empty((0, 2))], stats, termination="eos")
    assert not malformed.valid
