"""Restricted ingestion and validation of the supplied vector sketch dataset.

Pickle is not a general interchange format. This reader intentionally supports only
the two NumPy constructors observed in the distributed files; no arbitrary imports
or Python callables are accepted. A size limit bounds input and array allocations.
"""
from __future__ import annotations

import hashlib
import io
import pickle
from pathlib import Path
from typing import Any

import numpy as np

MAX_FILE_BYTES = 512 * 1024 * 1024
MAX_ARRAY_BYTES = 64 * 1024 * 1024
MAX_SAMPLES = 100_000
EXPECTED_KEYS = {"id", "strokes", "parts", "step_ids", "description"}


def _numeric_frombuffer(buffer: bytes, dtype: np.dtype, shape: tuple[int, ...], order: str) -> np.ndarray:
    dtype = np.dtype(dtype)
    if dtype.hasobject or dtype.fields is not None or dtype.kind not in "biuf":
        raise pickle.UnpicklingError("Only plain numeric NumPy arrays are permitted")
    if not isinstance(buffer, (bytes, bytearray)) or len(buffer) > MAX_ARRAY_BYTES:
        raise pickle.UnpicklingError("Invalid or oversized NumPy buffer")
    if not isinstance(shape, tuple) or not 1 <= len(shape) <= 3:
        raise pickle.UnpicklingError("Invalid NumPy array shape")
    if any(type(n) is not int or n < 0 or n > 1_000_000 for n in shape):
        raise pickle.UnpicklingError("Invalid or oversized NumPy array shape")
    count = 1
    for size in shape:
        count *= size
    if count * dtype.itemsize != len(buffer) or order not in ("C", "F"):
        raise pickle.UnpicklingError("Array shape does not match its buffer")
    return np.frombuffer(buffer, dtype=dtype).reshape(shape, order=order).copy()


class _RestrictedSketchUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if (module, name) in {
            ("numpy._core.numeric", "_frombuffer"),
            ("numpy.core.numeric", "_frombuffer"),
        }:
            return _numeric_frombuffer
        if (module, name) == ("numpy", "dtype"):
            return np.dtype
        raise pickle.UnpicklingError(f"Forbidden pickle global: {module}.{name}")

    def persistent_load(self, pid: Any) -> Any:
        raise pickle.UnpicklingError("Persistent pickle references are forbidden")


def validate_sample(sample: Any) -> list[str]:
    """Return concrete schema/geometry errors without modifying the input."""
    if type(sample) is not dict:
        return [f"sample must be dict, got {type(sample).__name__}"]
    errors: list[str] = []
    if set(sample) != EXPECTED_KEYS:
        errors.append(f"keys differ: {sorted(str(key) for key in sample)}")
    if type(sample.get("id")) is not int:
        errors.append("id must be int")
    strokes = sample.get("strokes")
    if type(strokes) is not list or not strokes:
        errors.append("strokes must be a non-empty list")
    else:
        if len(strokes) > 10_000:
            errors.append("too many strokes")
        for index, stroke in enumerate(strokes):
            if not isinstance(stroke, np.ndarray):
                errors.append(f"stroke {index} must be ndarray")
            elif stroke.ndim != 2 or stroke.shape[1] != 2 or len(stroke) == 0:
                errors.append(f"stroke {index} must have non-empty shape (N, 2)")
            elif stroke.dtype != np.float32:
                errors.append(f"stroke {index} dtype is {stroke.dtype}, expected float32")
            elif not np.isfinite(stroke).all():
                errors.append(f"stroke {index} contains non-finite coordinates")
    parts = sample.get("parts")
    if type(parts) is not list or any(type(part) is not str for part in parts):
        errors.append("parts must be list[str]")
    elif type(strokes) is list and len(parts) != len(strokes):
        errors.append("parts must align one-to-one with strokes")
    steps = sample.get("step_ids")
    if type(steps) is not list or any(type(step) is not int for step in steps):
        errors.append("step_ids must be list[int]")
    elif type(strokes) is list and len(steps) != len(strokes):
        errors.append("step_ids must align one-to-one with strokes")
    if type(sample.get("description")) is not str:
        errors.append("description must be str")
    return errors


def load_raw(path: str | Path, *, validate: bool = True) -> list[dict]:
    """Load supplied .pkl data with an explicit constructor allowlist.

    Never use this API as a guarantee against resource exhaustion for arbitrary
    internet files. It rejects executable globals and limits supported arrays;
    pickle container nesting itself should still be treated as untrusted input.
    """
    path = Path(path)
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("Dataset exceeds the 512 MiB input limit")
    payload = path.read_bytes()
    stream = io.BytesIO(payload)
    samples = _RestrictedSketchUnpickler(stream).load()
    if stream.read(1):
        raise ValueError("Unexpected bytes after pickle object")
    if type(samples) is not list or len(samples) > MAX_SAMPLES:
        raise ValueError("Dataset must be a list with at most 100,000 samples")
    if validate:
        for index, sample in enumerate(samples):
            errors = validate_sample(sample)
            if errors:
                raise ValueError(f"Invalid sample {index}: {'; '.join(errors)}")
    return samples


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def geometry_hash(strokes: list[np.ndarray]) -> str:
    """Hash exact point values and stroke boundaries, independently of metadata."""
    digest = hashlib.sha256()
    for stroke in strokes:
        array = np.asarray(stroke, dtype="<f4")
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()
