import io
import pickle

import numpy as np
import pytest

from sketchlab.data import (
    _RestrictedSketchUnpickler, _numeric_frombuffer, load_raw, validate_sample,
)


def sample():
    return {"id": 1, "strokes": [np.array([[1, 2], [3, 4]], dtype=np.float32)],
            "parts": ["body"], "step_ids": [0], "description": "test"}


def test_safe_numpy_roundtrip(tmp_path):
    path = tmp_path / "sample.pkl"
    path.write_bytes(pickle.dumps([sample()], protocol=5))
    decoded = load_raw(path)
    assert validate_sample(decoded[0]) == []
    np.testing.assert_array_equal(decoded[0]["strokes"][0], sample()["strokes"][0])


def test_block_executable_global():
    class Attack:
        def __reduce__(self):
            return eval, ("40 + 2",)
    data = pickle.dumps(Attack())
    with pytest.raises(pickle.UnpicklingError, match="Forbidden pickle global"):
        _RestrictedSketchUnpickler(io.BytesIO(data)).load()


@pytest.mark.parametrize("dtype", [np.dtype(object), np.dtype([("x", "f4")]), np.dtype("U4")])
def test_block_non_numeric_buffers(dtype):
    with pytest.raises(pickle.UnpicklingError, match="plain numeric"):
        _numeric_frombuffer(bytes(dtype.itemsize), dtype, (1,), "C")


def test_block_wrong_shape_and_trailing_bytes(tmp_path):
    with pytest.raises(pickle.UnpicklingError):
        _numeric_frombuffer(bytes(4), np.dtype("f4"), (10_000_000,), "C")
    path = tmp_path / "trailing.pkl"
    path.write_bytes(pickle.dumps([sample()], protocol=5) + b"trailing")
    with pytest.raises(ValueError, match="Unexpected bytes"):
        load_raw(path)


def test_validate_nonfinite_and_metadata_alignment():
    value = sample()
    value["strokes"][0][0, 0] = np.nan
    value["parts"] = []
    value["step_ids"] = []
    errors = validate_sample(value)
    assert any("non-finite" in error for error in errors)
    assert any("parts must align" in error for error in errors)
    assert any("step_ids must align" in error for error in errors)
