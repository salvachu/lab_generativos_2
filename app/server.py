"""Local drawing demo. All model and rendering logic lives in sketchlab.

Run from the repository root with ``python -m uvicorn app.server:app --reload``.
"""

from __future__ import annotations

import importlib
import math
import os
import pickle
import threading
from collections import OrderedDict
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator, model_validator
from starlette.concurrency import run_in_threadpool


ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).resolve().parent


class GenerateRequest(BaseModel):
    model: str = Field(min_length=1, max_length=1024)
    prefix: list[list[tuple[float, float]]] = Field(default_factory=list)
    n_candidates: int = Field(default=20, ge=1, le=64)
    top_k: int = Field(default=6, ge=1, le=9)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    temperature: float = Field(default=0.6, ge=0.05, le=1.5)
    max_points: int = Field(default=384, ge=16, le=1024)
    max_strokes: int = Field(default=64, ge=1, le=96)
    validate_candidates: bool = Field(default=True, alias="validate")
    postprocess: bool = True
    rerank: bool = True

    @model_validator(mode="after")
    def validate_budget(self):
        if self.top_k > self.n_candidates:
            raise ValueError("El número a mostrar no puede superar el presupuesto de candidatos.")
        return self

    @field_validator("prefix")
    @classmethod
    def validate_prefix(cls, strokes: list[list[tuple[float, float]]]):
        if len(strokes) > 96:
            raise ValueError("El prefijo admite como máximo 96 strokes.")
        if sum(len(stroke) for stroke in strokes) > 4096:
            raise ValueError("El prefijo admite como máximo 4096 puntos.")
        for stroke in strokes:
            if not stroke:
                raise ValueError("Cada stroke debe contener al menos un punto.")
            for point in stroke:
                if any(not math.isfinite(value) or abs(value) > 100_000 for value in point):
                    raise ValueError("Las coordenadas deben ser finitas y estar en un rango válido.")
        return strokes


def _as_strokes(sketch: Any) -> list[list[list[float]]]:
    """Make inference output JSON compatible and refuse non-finite output."""
    result = []
    for stroke in sketch:
        points = stroke.tolist() if hasattr(stroke, "tolist") else stroke
        output_stroke = []
        for point in points:
            if len(point) != 2 or not all(math.isfinite(float(value)) for value in point):
                raise ValueError("El modelo produjo coordenadas inválidas.")
            output_stroke.append([float(point[0]), float(point[1])])
        result.append(output_stroke)
    return result


def _json_safe(value: Any):
    """Serialize validation dataclasses/NumPy without illegal NaN JSON values."""
    if is_dataclass(value):
        value = asdict(value)
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def create_app(runs_dir: str | Path | None = None, backend: Any = None) -> FastAPI:
    """Create an app with injectable inference backend for small HTTP tests."""
    application = FastAPI(title="Sketch Lab · Laboratorio 2")
    run_root = Path(runs_dir or ROOT / "runs").resolve()
    inference_lock = threading.Lock()
    model_cache: OrderedDict[tuple[str, int, int], Any] = OrderedDict()
    device = os.environ.get("SKETCHLAB_DEVICE", "cpu")

    @application.exception_handler(RequestValidationError)
    async def invalid_request(_request, exc):
        # Validation errors can contain the offending NaN/Inf input itself.
        # Return only safe error metadata so that invalid JSON numbers produce
        # a proper 422 response instead of failing during JSON serialization.
        return JSONResponse(status_code=422, content={"detail": [
            {"loc": list(error["loc"]), "msg": error["msg"], "type": error["type"]}
            for error in exc.errors()
        ]})

    def resolve_checkpoint(model_id: str) -> Path:
        relative = Path(model_id)
        checkpoint = (run_root / relative).resolve()
        if relative.is_absolute() or not checkpoint.is_relative_to(run_root):
            raise HTTPException(status_code=400, detail="El checkpoint debe estar dentro de runs/.")
        if checkpoint.name not in {"checkpoint.pt", "best.pt", "last.pt"} or not checkpoint.is_file():
            raise HTTPException(status_code=404, detail="No se encontró el checkpoint. Actualiza la lista de modelos.")
        return checkpoint

    @application.get("/api/models")
    def models():
        available = []
        if run_root.is_dir():
            priority = {"best.pt": 0, "checkpoint.pt": 1, "last.pt": 2}
            by_directory = {}
            for candidate in run_root.rglob("*.pt"):
                if candidate.name not in priority:
                    continue
                old = by_directory.get(candidate.parent)
                if old is None or priority[candidate.name] < priority[old.name]:
                    by_directory[candidate.parent] = candidate
            for checkpoint in sorted(by_directory.values()):
                resolved = checkpoint.resolve()
                if not resolved.is_relative_to(run_root) or not resolved.is_file():
                    continue
                # F representation pretraining does not yield a generative model.
                if checkpoint.parent.name == "F":
                    try:
                        generation = importlib.import_module("sketchlab.generation")
                        generation.load_model(resolved)
                    except (ValueError, RuntimeError, KeyError, EOFError, OSError, pickle.UnpicklingError):
                        continue
                info = resolved.stat()
                model_id = checkpoint.relative_to(run_root).as_posix()
                available.append({
                    "id": model_id,
                    "label": f"{checkpoint.parent.relative_to(run_root).as_posix()} · {checkpoint.name}",
                    "modified_at": datetime.fromtimestamp(info.st_mtime, timezone.utc).isoformat(),
                    "size_bytes": info.st_size,
                })
        return {"models": available, "canvas_size": 512, "device": device}

    def infer(request: GenerateRequest, checkpoint: Path):
        # Sampling can use global Torch/NumPy RNG state. Serialize inference,
        # including checkpoint loading, to make seeded calls reproducible.
        with inference_lock:
            generation = backend if backend is not None else importlib.import_module("sketchlab.generation")
            info = checkpoint.stat()
            key = (str(checkpoint), info.st_mtime_ns, info.st_size)
            if key not in model_cache:
                model_cache[key] = generation.load_model(checkpoint, device=device)
                while len(model_cache) > 2:
                    model_cache.popitem(last=False)
            model_cache.move_to_end(key)
            handle = model_cache[key]
            kwargs = {
                "n_candidates": request.n_candidates,
                "top_k": request.top_k,
                "seed": request.seed,
                "temperature": request.temperature,
                "max_points": request.max_points,
                "max_strokes": request.max_strokes,
                "validate": request.validate_candidates,
                "postprocess": request.postprocess,
                "rerank": request.rerank,
            }
            import numpy as np

            prefix = [np.asarray(stroke, dtype=np.float64) for stroke in request.prefix] if request.prefix else None
            result = generation.sample_multiple(handle, prefix=prefix, **kwargs)
            samples = []
            expected_prefix = [[list(point) for point in stroke] for stroke in request.prefix]
            unrenderable = []
            for candidate in result["selected"]:
                raw = candidate["raw_output"]
                processed = candidate["postprocessed_output"]
                try:
                    raw_strokes, processed_strokes = _as_strokes(raw), _as_strokes(processed)
                except ValueError:
                    # An explicitly unvalidated candidate may have non-finite
                    # geometry. Report it; never fabricate display coordinates.
                    unrenderable.append(candidate.get("id"))
                    continue
                for strokes in (raw_strokes, processed_strokes):
                    if request.prefix and strokes[:len(expected_prefix)] != expected_prefix:
                        raise ValueError("La generación no conservó exactamente el prefijo.")
                raw_svg = generation.render(raw, prefix_count=len(request.prefix))
                processed_svg = generation.render(processed, prefix_count=len(request.prefix))
                samples.append({
                    "id": candidate.get("id"),
                    "seed": candidate.get("seed"),
                    "raw_strokes": raw_strokes,
                    "postprocessed_strokes": processed_strokes,
                    "raw_svg": raw_svg,
                    "postprocessed_svg": processed_svg,
                    "validation": _json_safe(candidate.get("validation")),
                    "termination": _json_safe(candidate.get("termination")),
                })
            report = _json_safe(result.get("report", {}))
            report["unrenderable_selected"] = unrenderable
            return {
                "samples": samples,
                "selected": samples,
                "candidates": [_json_safe({key: value for key, value in candidate.items() if key not in {"raw_output", "postprocessed_output"}}) for candidate in result.get("candidates", [])],
                "report": report,
                "model": request.model,
                "seed": request.seed,
                "temperature": request.temperature,
                "prefix_count": len(request.prefix),
                "n_candidates": request.n_candidates,
                "top_k": request.top_k,
                "validated": request.validate_candidates,
                "postprocess": request.postprocess,
                "rerank": request.rerank,
            }

    @application.post("/api/generate")
    async def generate(request: GenerateRequest):
        checkpoint = resolve_checkpoint(request.model)
        try:
            return await run_in_threadpool(infer, request, checkpoint)
        except HTTPException:
            raise
        except (ImportError, ModuleNotFoundError) as exc:
            raise HTTPException(status_code=503, detail="El backend no está instalado. Sigue la instalación del README.") from exc
        except Exception as exc:
            # Keep local paths and Python tracebacks out of the browser response.
            import logging

            logging.getLogger(__name__).exception("Sketch generation failed")
            raise HTTPException(status_code=500, detail="No se pudo generar. Revisa el checkpoint y el registro del servidor.") from exc

    @application.get("/")
    def index():
        return FileResponse(ASSETS / "index.html")

    application.mount("/static", StaticFiles(directory=ASSETS), name="static")
    return application


app = create_app()
