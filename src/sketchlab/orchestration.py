"""Candidate generation, conservative processing and diversity-aware selection.

This service is model-independent and contains no HTTP or UI code. Every sampled
candidate, including rejected ones, remains in the returned audit record.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import time

import numpy as np

from sketchlab.evaluation import prefix_preserved


def json_ready(value):
    if is_dataclass(value):
        return json_ready(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def sample_multiple(handle, prefix=None, n_candidates=20, top_k=6, seed=42,
                    temperature=.6, max_points=384, max_strokes=64, validate=True,
                    postprocess=True, rerank=True, geometry_path=None, ranking_config=None):
    from sketchlab.generation import sample
    from sketchlab.models.common import validate_sketch
    from sketchlab.geometry import validate_candidate, postprocess_candidate, load_geometry
    from sketchlab.ranking import rank_candidates

    if not isinstance(n_candidates, int) or not 1 <= n_candidates <= 64:
        raise ValueError("n_candidates must be an integer from 1 to 64")
    if not isinstance(top_k, int) or not 1 <= top_k <= n_candidates:
        raise ValueError("top_k must be between 1 and n_candidates")
    prefix = [s.copy() for s in validate_sketch(prefix if prefix is not None else [])]
    stats = None
    if validate:
        path = Path(geometry_path or "data/processed/geometry.json")
        if not path.exists():
            raise FileNotFoundError("TRAIN geometry statistics are missing; run python -m scripts.fit_geometry once")
        stats = load_geometry(path)
    candidates = []
    tick = time.perf_counter()
    for index in range(n_candidates):
        raw, info = sample(handle, prefix=prefix, seed=seed+index,
                           decoder_seed=seed+1_000_003, temperature=temperature,
                           max_points=max_points, max_strokes=max_strokes, return_info=True)
        if not prefix_preserved(prefix, raw):
            raise AssertionError("Model modified the observed prefix")
        if validate and postprocess:
            candidate = postprocess_candidate(raw, stats, prefix_count=len(prefix), termination=info)
            candidate["validation"] = json_ready(candidate["validation"])
        else:
            validation = (json_ready(validate_candidate(raw, stats, prefix_count=len(prefix), termination=info))
                          if validate else {"valid": None, "score": None, "warnings": ["RAW: geometry not validated"],
                                            "penalties": {}, "corrections": []})
            candidate = {"raw_output": [s.copy() for s in raw],
                         "postprocessed_output": [s.copy() for s in raw], "validation": validation}
        if not prefix_preserved(prefix, candidate["postprocessed_output"]):
            raise AssertionError("Postprocessing modified the observed prefix")
        candidate.update(id=index, seed=seed+index, termination=info, prefix_count=len(prefix))
        candidates.append(candidate)
    if validate and rerank:
        ranking = rank_candidates(candidates, top_k, stats, config=ranking_config)
        selected = ranking["selected"]
        ranking_report = ranking["report"]
    elif validate:
        selected = [c for c in candidates if c["validation"]["valid"]][:top_k]
        ranking_report = {"method": "sampling order among geometrically valid candidates"}
    else:
        selected = candidates[:top_k]
        ranking_report = {"method": "RAW diagnostic mode; validation, corrections and ranking bypassed"}
    report = {
        "n_candidates": len(candidates), "top_k_requested": top_k, "n_selected": len(selected),
        "n_valid": sum(c["validation"]["valid"] is True for c in candidates) if validate else None,
        "rejected": sum(c["validation"]["valid"] is False for c in candidates) if validate else None,
        "validation_enabled": validate, "postprocessing_enabled": validate and postprocess,
        "reranking_enabled": validate and rerank, "prefix_exact": True,
        "insufficient_valid_diverse_candidates": validate and len(selected) < top_k,
        "seconds": time.perf_counter()-tick, "ranking": ranking_report,
        "limitations": "Geometry is a filter, not proof that a creature is semantically plausible. A finite sampling budget can return fewer than top_k.",
    }
    return {"selected": selected, "candidates": candidates, "report": report}
