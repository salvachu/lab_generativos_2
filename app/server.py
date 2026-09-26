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
    artifact_root = (ROOT / "artifacts/models" if runs_dir is None else run_root / "artifacts/models").resolve()
    inference_lock = threading.Lock()
    model_cache: OrderedDict[tuple[str, int, int, str], Any] = OrderedDict()
    demo_model_cache: OrderedDict[tuple[str, int, int, str], Any] = OrderedDict()
    device = os.environ.get("SKETCHLAB_DEVICE", "cpu")

    def demo_device() -> str:
        import torch

        return os.environ.get("SKETCHLAB_DEMO_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

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
        packaged = relative.parts[:2] == ("artifacts", "models")
        allowed_root = artifact_root if packaged else run_root
        checkpoint = (allowed_root / Path(*relative.parts[2:]) if packaged else allowed_root / relative).resolve()
        if relative.is_absolute() or not checkpoint.is_relative_to(allowed_root):
            raise HTTPException(status_code=400, detail="El checkpoint debe estar dentro de runs/ o artifacts/models/.")
        if checkpoint.name not in {"checkpoint.pt", "best.pt", "last.pt"} or not checkpoint.is_file():
            raise HTTPException(status_code=404, detail="No se encontró el checkpoint. Actualiza la lista de modelos.")
        return checkpoint

    @application.get("/api/models")
    def models():
        available = []
        if run_root.is_dir() or artifact_root.is_dir():
            priority = {"best.pt": 0, "checkpoint.pt": 1, "last.pt": 2}
            by_directory = {}
            for source in (run_root, artifact_root):
                if not source.is_dir():
                    continue
                for candidate in source.rglob("*.pt"):
                    if candidate.name not in priority:
                        continue
                    old = by_directory.get(candidate.parent)
                    if old is None or priority[candidate.name] < priority[old.name]:
                        by_directory[candidate.parent] = candidate
            for checkpoint in sorted(by_directory.values()):
                resolved = checkpoint.resolve()
                source = artifact_root if resolved.is_relative_to(artifact_root) else run_root
                if not resolved.is_relative_to(source) or not resolved.is_file():
                    continue
                # F representation pretraining does not yield a generative model.
                if checkpoint.parent.name == "F":
                    try:
                        generation = importlib.import_module("sketchlab.generation")
                        generation.load_model(resolved)
                    except (ValueError, RuntimeError, KeyError, EOFError, OSError, pickle.UnpicklingError):
                        continue
                info = resolved.stat()
                model_id = (("artifacts/models/" if source == artifact_root else "")
                            + checkpoint.relative_to(source).as_posix())
                available.append({
                    "id": model_id,
                    "label": f"{'Final / ' if source == artifact_root else ''}{checkpoint.parent.relative_to(source).as_posix()} · {checkpoint.name}",
                    "modified_at": datetime.fromtimestamp(info.st_mtime, timezone.utc).isoformat(),
                    "size_bytes": info.st_size,
                })
        return {"models": available, "canvas_size": 512, "device": device,
                "demo_device": demo_device()}

    def infer(request: GenerateRequest, checkpoint: Path, demo: bool = False):
        # Sampling can use global Torch/NumPy RNG state. Serialize inference,
        # including checkpoint loading, to make seeded calls reproducible.
        with inference_lock:
            packaged = checkpoint.is_relative_to(artifact_root)
            relative_checkpoint = checkpoint.relative_to(artifact_root if packaged else run_root).as_posix()
            generation = backend if backend is not None else importlib.import_module(
                "app.h20_backend" if (packaged and relative_checkpoint in {"H13/best.pt", "H20/best.pt"}) or
                relative_checkpoint == "H20_training/short/best.pt" or
                (demo and relative_checkpoint == "H13_training/short/best.pt")
                else "sketchlab.generation")
            info = checkpoint.stat()
            inference_device = device
            if demo:
                inference_device = demo_device()
            cache = demo_model_cache if demo else model_cache
            key = (str(checkpoint), info.st_mtime_ns, info.st_size, inference_device)
            if key not in cache:
                cache[key] = generation.load_model(checkpoint, device=inference_device)
                while len(cache) > (3 if demo else 2):
                    cache.popitem(last=False)
            cache.move_to_end(key)
            handle = cache[key]
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
            selected = list(result["selected"])
            if demo and len(selected) < request.top_k:
                chosen_ids = {candidate.get("id") for candidate in selected}
                for candidate in result.get("candidates", []):
                    if candidate.get("id") not in chosen_ids:
                        selected.append(candidate)
                        chosen_ids.add(candidate.get("id"))
                    if len(selected) >= request.top_k:
                        break
            for candidate in selected:
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
            if demo:
                report["presentation_fallbacks"] = max(0, len(selected) - len(result["selected"]))
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

    @application.post("/api/demo/generate")
    async def demo_generate(request: GenerateRequest):
        if request.model not in {
            "artifacts/models/H9/best.pt", "artifacts/models/H13/best.pt", "artifacts/models/H20/best.pt",
            "H_training/H9/short/H/best.pt", "H13_training/short/best.pt", "H20_training/short/best.pt"
        }:
            raise HTTPException(status_code=400, detail="Elige H9, H13 o H20.")
        checkpoint = resolve_checkpoint(request.model)
        try:
            return await run_in_threadpool(infer, request, checkpoint, True)
        except HTTPException:
            raise
        except Exception as exc:
            import logging
            logging.getLogger(__name__).exception("Presentation demo generation failed")
            raise HTTPException(status_code=500, detail="No se pudo generar esta variante. Prueba de nuevo.") from exc

    @application.get("/demo")
    def presentation_demo():
        return FileResponse(ASSETS / "demo.html")

    @application.get("/")
    def index():
        return FileResponse(ASSETS / "index.html")

    application.mount("/static", StaticFiles(directory=ASSETS), name="static")
    return application


app = create_app()
