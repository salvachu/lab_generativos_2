"""Reproducible full-data audit, with no writes to data/raw."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sketchlab.data import geometry_hash, load_raw, sha256_file, validate_sample
from sketchlab.representation import decode_tokens, encode_strokes, decode_deltas, encode_deltas
from sketchlab.rendering import render_grid


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {"count": 0}
    quantiles = (0, .01, .05, .25, .5, .75, .95, .99, .999, 1)
    return {
        "count": int(values.size), "mean": float(values.mean()), "std": float(values.std()),
        "percentiles": {f"{100*q:g}": float(v) for q, v in zip(quantiles, np.quantile(values, quantiles))},
    }


def duplicate_groups(items):
    groups = {}
    for index, value in enumerate(items):
        groups.setdefault(value, []).append(index)
    return [indices for indices in groups.values() if len(indices) > 1]


def audit_split(samples, source):
    invalid = [{"index": i, "errors": validate_sample(sample)} for i, sample in enumerate(samples)
               if validate_sample(sample)]
    if invalid:
        raise ValueError(f"Invalid samples require explicit handling: {invalid[:5]}")
    strokes = [stroke for sample in samples for stroke in sample["strokes"]]
    points = np.concatenate(strokes).astype(np.float64)
    lengths = np.array([len(stroke) for stroke in strokes])
    nstrokes = np.array([len(sample["strokes"]) for sample in samples])
    npoints = np.array([sum(map(len, sample["strokes"])) for sample in samples])
    intra = np.concatenate([np.linalg.norm(np.diff(stroke.astype(np.float64), axis=0), axis=1)
                            for stroke in strokes if len(stroke) > 1])
    inter = np.array([float(np.linalg.norm(b[0].astype(np.float64) - a[-1]))
                      for sample in samples for a, b in zip(sample["strokes"], sample["strokes"][1:])])
    arc = np.array([np.linalg.norm(np.diff(stroke.astype(np.float64), axis=0), axis=1).sum()
                    for stroke in strokes])
    ids = [sample["id"] for sample in samples]
    hashes = [geometry_hash(sample["strokes"]) for sample in samples]
    bbox_extents = np.array([np.ptp(np.concatenate(sample["strokes"]), axis=0) for sample in samples])
    exact = 0
    delta_error = {"float64_max_abs_coordinate_error": 0.0, "float32_max_abs_coordinate_error": 0.0,
                   "float64_nonexact_sketches": 0, "float32_nonexact_sketches": 0, "examples": []}
    for sample in samples:
        decoded = decode_tokens(encode_strokes(sample["strokes"]))
        if len(decoded) != len(sample["strokes"]) or any(not np.array_equal(a, b) for a, b in zip(sample["strokes"], decoded)):
            raise AssertionError(f"Non-exact canonical round trip: id={sample['id']}")
        exact += 1
        delta = encode_deltas(sample["strokes"])
        for dtype in ("float64", "float32"):
            restored = decode_deltas(delta.astype(dtype))
            error = max(float(np.max(np.abs(a.astype(np.float64) - b))) for a, b in zip(sample["strokes"], restored))
            delta_error[f"{dtype}_max_abs_coordinate_error"] = max(delta_error[f"{dtype}_max_abs_coordinate_error"], error)
            delta_error[f"{dtype}_nonexact_sketches"] += int(error > 0)
            if dtype == "float64" and error > 0 and len(delta_error["examples"]) < 5:
                delta_error["examples"].append({"id": sample["id"], "max_abs_error": error})
    threshold = float(np.quantile(npoints, .99))
    return {
        "source": str(source), "sha256": sha256_file(source), "bytes": source.stat().st_size,
        "samples": len(samples), "total_strokes": len(strokes), "total_points": len(points),
        "schema": {
            "container": "builtins.list", "sample": "builtins.dict",
            "key_sets": [list(keys) for keys in sorted({tuple(sorted(sample)) for sample in samples})],
            "fields": {key: sorted({f"{type(sample[key]).__module__}.{type(sample[key]).__name__}" for sample in samples})
                       for key in sorted(samples[0])},
            "stroke_types": sorted({f"{type(stroke).__module__}.{type(stroke).__name__}" for stroke in strokes}),
            "stroke_dtypes": sorted({str(stroke.dtype) for stroke in strokes}),
            "stroke_shapes": "(N, 2), N variable", "coordinates": "absolute x,y",
            "explicit_pen_state": False, "stroke_boundary": "separate ndarray in ordered list",
        },
        "coordinates": {"min": points.min(axis=0).tolist(), "max": points.max(axis=0).tolist(),
                        "x": summary(points[:, 0]), "y": summary(points[:, 1]),
                        "negative_points": int((points < 0).any(axis=1).sum()),
                        "above_504_points": int((points > 504).any(axis=1).sum()),
                        "outside_nominal_0_504_points": int(((points < 0) | (points > 504)).any(axis=1).sum()),
                        "outside_display_minus4_508_points": int(((points < -4) | (points > 508)).any(axis=1).sum()),
                        "cumulative_max_if_misread_as_relative": np.concatenate([np.cumsum(stroke.astype(np.float64), axis=0) for stroke in strokes]).max(axis=0).tolist()},
        "strokes_per_sketch": summary(nstrokes), "points_per_sketch": summary(npoints),
        "points_per_stroke": summary(lengths), "stroke_arc_length": summary(arc),
        "intra_stroke_step_distance": summary(intra), "inter_stroke_move_distance": summary(inter),
        "sketch_bbox_width": summary(bbox_extents[:, 0]), "sketch_bbox_height": summary(bbox_extents[:, 1]),
        "degenerate": {"single_point_strokes": int((lengths == 1).sum()),
                       "zero_arc_strokes": int((arc == 0).sum()), "zero_intra_steps": int((intra == 0).sum())},
        "metadata": {
            "parts_element_types": sorted({type(part).__name__ for sample in samples for part in sample["parts"]}),
            "step_ids_element_types": sorted({type(step).__name__ for sample in samples for step in sample["step_ids"]}),
            "parts_counts": dict(Counter(part for sample in samples for part in sample["parts"])),
            "step_ids_counts": dict(sorted(Counter(step for sample in samples for step in sample["step_ids"]).items())),
            "parts_aligned_to_strokes": all(len(s["parts"]) == len(s["strokes"]) for s in samples),
            "step_ids_aligned_to_strokes": all(len(s["step_ids"]) == len(s["strokes"]) for s in samples),
            "nonmonotonic_step_id_sketches": sum(any(a > b for a, b in zip(s["step_ids"], s["step_ids"][1:])) for s in samples),
            "empty_descriptions": sum(not s["description"].strip() for s in samples),
            "description_characters": summary([len(s["description"]) for s in samples]),
            "unique_descriptions": len({s["description"] for s in samples}),
            "sample_examples": [{"index": i, "id": samples[i]["id"], "parts": samples[i]["parts"],
                                 "step_ids": samples[i]["step_ids"], "description": samples[i]["description"]}
                                for i in (0, len(samples)//2, len(samples)-1)],
        },
        "outliers": {"policy": "flag only, no clipping, removal, simplification or resampling",
                     "point_count_gt_p99": [{"index": int(i), "id": samples[i]["id"], "points": int(npoints[i]), "strokes": int(nstrokes[i])}
                                             for i in np.flatnonzero(npoints > threshold)],
                     "largest_point_count_indices": np.argsort(npoints)[-8:][::-1].tolist(),
                     "largest_stroke_count_indices": np.argsort(nstrokes)[-8:][::-1].tolist()},
        "invalid_samples": invalid, "duplicate_id_groups": duplicate_groups(ids),
        "duplicate_geometry_groups": duplicate_groups(hashes), "exact_roundtrips": exact,
        "relative_model_view_roundtrip": delta_error,
    }, hashes


def make_report(audit):
    train, val = audit["train"], audit["val"]
    lines = ["# Auditoría exacta del dataset", "", "Generado por `python scripts/audit_data.py`. Se inspeccionan todas las muestras; los originales no se modifican.", "",
             "## Estructura real", "", "Ambos archivos son pickles sin compresión, aunque el PDF menciona `.pkl.gz`. Contienen una `list[dict]`. Cada diccionario tiene exactamente:", "",
             "- `id`: `int`.", "- `strokes`: lista ordenada de `numpy.ndarray`, cada uno de forma `(N, 2)` y dtype `float32`.",
             "- `parts`: `list[str]`, una etiqueta por stroke.", "- `step_ids`: `list[int]`, una etapa por stroke; las etapas pueden repetirse.",
             "- `description`: `str`.", "",
             "No existe campo pen-up/pen-down, EOS, stage adicional, ni máscara. La separación entre arrays marca los strokes. Las coordenadas son absolutas: el render directo forma dibujos coherentes; acumularlas como deltas desplaza puntos decenas de miles de unidades. Los nombres exactos de metadata son `parts`, `step_ids` y `description`.", "",
             "## Conteos y estadísticas", "", "Los 8187 train + 910 validation = 9097 coinciden con el PDF.", "",
             "| Estadística | Train | Validation |", "|---|---:|---:|"]
    for title, key in (("Sketches", "samples"), ("Strokes", "total_strokes"), ("Puntos", "total_points"), ("Roundtrips exactos", "exact_roundtrips")):
        lines.append(f"| {title} | {train[key]} | {val[key]} |")
    for title, key in (("Strokes/sketch", "strokes_per_sketch"), ("Puntos/sketch", "points_per_sketch"), ("Puntos/stroke", "points_per_stroke"),
                       ("Distancia dentro del stroke", "intra_stroke_step_distance"), ("Reposicionamiento entre strokes", "inter_stroke_move_distance")):
        def cell(split):
            stats = split[key]; p = stats["percentiles"]
            return f"{p['0']:.4g} / {p['50']:.4g} / {p['95']:.4g} / {p['100']:.4g}"
        lines.append(f"| {title}: min / mediana / p95 / max | {cell(train)} | {cell(val)} |")
    lines += ["", "Percentiles adicionales, medias, desviaciones, longitudes de arco, rangos por eje y metadata completa están en `runs/audit/audit.json`.", "", "## Coordenadas y canvas", ""]
    for name, split in (("Train", train), ("Validation", val)):
        c = split["coordinates"]
        lines.append(f"{name}: min x/y `{c['min']}`; max x/y `{c['max']}`; {c['negative_points']} puntos con alguna coordenada negativa; {c['above_504_points']} puntos por encima de 504.")
        lines.append("")
    lines += ["El tamaño original del canvas no figura en el PDF ni en los pickles. El rango observado es compatible con coordenadas de aproximadamente 0–504 y una pequeña cola negativa; no demuestra un canvas original de 500 o 512. Para mostrar todo sin recortar se usa el envelope fijo `[-4,508] × [-4,508]` (512 unidades). La métrica canvas nominal usa `[0,504]` como convención explícita, no como condición de validez del dataset. No se eliminan ni recortan puntos negativos.", "", "## Representación canónica", "",
              "La representación canónica es stroke-5 ABSOLUTA `[x/256, y/256, down, end_stroke, EOS]`, que conserva todos los float32 originales exactamente en float64, incluso puntos cercanos a cero. La vista de entrenamiento secuencial usa `[dx/256, dy/256, down, end_stroke, EOS]` (`encode_deltas` / `decode_deltas`) con deltas globales respecto al punto anterior y origen fijo `(0,0)`. La escala se ajusta únicamente a train: mayor magnitud / 2 redondeada hacia arriba a potencia de dos = 256; ningún prefijo se renormaliza según la continuación. Cada punto conserva su estado de salida y cada sketch añade EOS independiente. El salto al primer punto de otro stroke es movimiento del lápiz levantado y se excluye de penalizaciones de continuidad.", "",
              "La separación entre representación exacta y vista relativa es necesaria: train id=4322 contiene y=3.8224968329503284e-11; al restar el punto anterior y acumular float64 se obtiene 3.822497873784414e-11. Por tanto ni siquiera deltas float64 garantizan roundtrip bit-exact de todo el dataset. Se documenta ese error real, sin usar tolerancia para declarar exactitud canónica.", "",
              "Los batches del modelo usan copias float32, longitudes y máscara booleana, incluido EOS. El padding queda fuera de las pérdidas. No hay truncamiento, resampling, simplificación, rasterización principal ni normalización por sketch. `encode_strokes` / `decode_tokens` son las APIs canónicas absolutas.", ""]
    for name, split in (("Train", train), ("Validation", val)):
        d = split["relative_model_view_roundtrip"]
        lines += [f"{name}: el roundtrip relativo float64 tiene error máximo {d['float64_max_abs_coordinate_error']:.9g} unidades ({d['float64_nonexact_sketches']} sketches no exactos); deltas float32 acumulados en float64 tienen error máximo {d['float32_max_abs_coordinate_error']:.9g} ({d['float32_nonexact_sketches']} sketches no exactos).", ""]
    lines += ["## Integridad, outliers y separación", ""]
    for name, split in (("Train", train), ("Validation", val)):
        lines += [f"- {name}: {len(split['invalid_samples'])} muestras inválidas; {len(split['duplicate_id_groups'])} grupos de IDs duplicados; {len(split['duplicate_geometry_groups'])} grupos de geometría exactamente duplicada; degenerados `{split['degenerate']}`."]
    lines += [f"- IDs compartidos train/val: {len(audit['split_overlap']['ids'])}; geometrías exactas compartidas: {audit['split_overlap']['exact_geometry_count']}.",
              "- Los outliers se marcan (puntos/sketch > p99 y extremos por strokes/puntos); se conservan. Un trazo con un solo punto es un punto válido, no un error estructural. Saltos grandes entre strokes son movimientos válidos.", "",
              "Los hashes SHA256 de ambos originales quedan guardados en `audit.json`. La carga restringida permite exclusivamente los constructores NumPy observados (dtype y frombuffer numérico); bloquea otros globals, buffers object/structured, persistencia y arrays sobredimensionados. No es un deserializador universal de pickles no confiables.", "", "## Muestras y comprobaciones", "",
              "- `train_grid.png`, `val_grid.png`: 16 muestras uniformemente espaciadas por split.", "- `outliers_grid.png`: extremos de train por cantidad de puntos/strokes.", "- `roundtrip_grid.png`: cuatro originales junto a su representación decodificada.",
              "- `metadata_examples.json`: ejemplos completos de partes, etapas y texto.", "", "El script verifica igualdad exacta de todos los puntos y boundaries en las 9097 muestras, además de las pruebas unitarias de carga segura y representación.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("runs/audit"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    data = {name: load_raw(args.raw_dir / f"{name}.pkl") for name in ("train", "val")}
    audit = {"expected_counts_pdf": {"train": 8187, "val": 910, "total": 9097}, "version": 1}
    hashes = {}
    for name, samples in data.items():
        audit[name], hashes[name] = audit_split(samples, args.raw_dir / f"{name}.pkl")
    audit["split_overlap"] = {
        "ids": sorted({s["id"] for s in data["train"]} & {s["id"] for s in data["val"]}),
        "exact_geometry_count": len(set(hashes["train"]) & set(hashes["val"])),
    }
    max_abs_train = max(abs(v) for pair in (audit["train"]["coordinates"]["min"], audit["train"]["coordinates"]["max"]) for v in pair)
    scale = float(2 ** math.ceil(math.log2(max_abs_train / 2)))
    normalization = {"coordinate_mode": "absolute", "model_coordinate_mode": "global_delta", "scale": scale, "origin": [0, 0],
                     "fit_split": "train", "fit_sha256": audit["train"]["sha256"],
                     "fit_rule": "2**ceil(log2(max(abs(train_coordinates))/2))",
                     "canonical_dtype": "float64", "model_dtype": "float32", "resampling": False,
                     "clipping": False, "display_bounds": [-4, 508, -4, 508],
                     "nominal_canvas_bounds": [0, 504, 0, 504], "original_canvas_size": None}
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("data/processed/normalization.json").write_text(json.dumps(normalization, indent=2), encoding="utf-8")
    for name, samples in data.items():
        indices = np.linspace(0, len(samples)-1, 16, dtype=int)
        chosen = [samples[i] for i in indices]
        titles = [f"{name} index={i}, id={s['id']}\n{len(s['strokes'])} strokes / {sum(map(len,s['strokes']))} points" for i, s in zip(indices, chosen)]
        render_grid(chosen, args.output / f"{name}_grid.png", titles=titles)
    indices = list(dict.fromkeys(audit["train"]["outliers"]["largest_point_count_indices"][:4] + audit["train"]["outliers"]["largest_stroke_count_indices"][:4]))
    render_grid([data["train"][i] for i in indices], args.output / "outliers_grid.png",
                titles=[f"id={data['train'][i]['id']}: {len(data['train'][i]['strokes'])} strokes / {sum(map(len,data['train'][i]['strokes']))} points" for i in indices])
    pairs = []; titles = []
    for i in (0, 100, 1000, len(data["train"])-1):
        strokes = data["train"][i]["strokes"]
        pairs.extend([strokes, decode_tokens(encode_strokes(strokes))])
        titles.extend([f"Original id={data['train'][i]['id']}", "Decoded (exact)"])
    render_grid(pairs, args.output / "roundtrip_grid.png", titles=titles)
    audit["elapsed_seconds"] = time.perf_counter() - started
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output / "metadata_examples.json").write_text(json.dumps({k:audit[k]["metadata"]["sample_examples"] for k in data}, indent=2, ensure_ascii=False), encoding="utf-8")
    Path("docs").mkdir(exist_ok=True)
    Path("docs/dataset.md").write_text(make_report(audit), encoding="utf-8")
    print(json.dumps({"status": "ok", "elapsed_seconds": audit["elapsed_seconds"],
                      "counts": {name: audit[name]["samples"] for name in data},
                      "roundtrips": sum(audit[name]["exact_roundtrips"] for name in data),
                      "overlap": audit["split_overlap"], "normalization_scale": scale}, indent=2))


if __name__ == "__main__":
    main()
