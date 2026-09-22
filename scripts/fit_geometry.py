"""Fit geometry on TRAIN once; reuse the persisted statistics on later runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sketchlab.data import load_raw
from sketchlab.geometry import fit_geometry, load_geometry, save_geometry
from sketchlab.rendering import render_grid
from sketchlab.representation import encode_strokes, decode_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/raw/train.pkl"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/geometry.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/geometry.json"))
    parser.add_argument("--force", action="store_true", help="Explicitly refit; otherwise reuse existing calibration")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        stats = load_geometry(args.output)
        print(json.dumps({"status": "reused", "path": str(args.output), "samples": stats["sample_count"]}))
        return
    # This single TRAIN read is required for fitting missing calibration. No VAL
    # access and no second full-schema audit / exhaustive roundtrip are performed.
    samples = load_raw(args.train)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    stats = fit_geometry(samples, config=config, source_path=args.train)
    save_geometry(stats, args.output)  # persist before auxiliary plots/docs
    audit_dir = Path("runs/audit")
    audit_dir.mkdir(parents=True, exist_ok=True)
    indices = np.linspace(0, len(samples)-1, 16, dtype=int)
    render_grid([samples[i] for i in indices], audit_dir/"train_grid.png",
                titles=[f"Train id={samples[i]['id']}: {len(samples[i]['strokes'])} strokes" for i in indices])
    point_counts = [sum(map(len, sample["strokes"])) for sample in samples]
    extremes = list(dict.fromkeys(np.argsort(point_counts)[-4:][::-1].tolist()
                                  + np.argsort([len(s["strokes"]) for s in samples])[-4:][::-1].tolist()))
    render_grid([samples[i] for i in extremes], audit_dir/"outliers_grid.png",
                titles=[f"id={samples[i]['id']}: {len(samples[i]['strokes'])} strokes / {point_counts[i]} points" for i in extremes])
    pairs = []; titles = []
    for index in (0, 100, 1000, len(samples)-1):
        strokes = samples[index]["strokes"]
        pairs.extend([strokes, decode_tokens(encode_strokes(strokes))])
        titles.extend([f"Original id={samples[index]['id']}", "Decoded absolute (exact)"])
    render_grid(pairs, audit_dir/"roundtrip_grid.png", titles=titles)
    examples = [{"index": int(index), **{key: value for key, value in samples[index].items() if key != "strokes"}}
                for index in (0, len(samples)//2, len(samples)-1)]
    (audit_dir/"metadata_examples.json").write_text(json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Geometría y selección de candidatos", "",
             f"Calibración TRAIN-only: {stats['sample_count']} sketches. SHA256: `{stats['source_sha256']}`.", "",
             "Se persiste en `data/processed/geometry.json`; entrenamiento/inferencia la cargan sin releer el dataset. `scripts/fit_geometry.py` reutiliza el archivo si existe; refit requiere `--force`. VALIDATION no participa.", "",
             "## Umbrales", "", "Todos los límites numéricos de geometría salen de TRAIN. Los hiperparámetros del procedimiento están en `configs/geometry.json`: soft=q99.5; severe=max(q99.9 + 6×1.4826×MAD, 1.5×q99.9). El factor 1.4826 estima sigma robusta bajo una distribución normal; no supone que la geometría sea normal. Los factores 6 y 1.5 son tolerancias configurables y documentadas, no distancias en píxeles.", "",
             "Las coordenadas usan extrema observada TRAIN como región blanda y añaden un margen del 10% del rango aprendido para extrema severa. La pequeña cola negativa real permanece válida. El canvas original sigue siendo desconocido.", "",
             "| Medida | Mediana | Límite blando | Límite severo |", "|---|---:|---:|---:|"]
    for name, distribution in stats["distributions"].items():
        lines.append(f"| {name} | {distribution['median']:.6g} | {distribution['soft_upper']:.6g} | {distribution['severe_upper']:.6g} |")
    lines += ["", f"Rango aprendido: `{stats['coordinates']}`.", "", "## Validación y corrección", "",
              "`validate_candidate` devuelve `valid`, `score`, `warnings`, `penalties`, `severe_errors`, `corrections` y métricas. NaN/Inf y estructuras inválidas se rechazan incluso en el prefix. El prefix finito no se evalúa contra los umbrales estadísticos ni se modifica. Las distancias entre strokes no participan en ninguna penalización de continuidad.", "",
              "Los umbrales se aplican al suffix generado: coordenadas, segmento máximo intra-stroke, longitud de arco, puntos/stroke, puntos/sketch, número de strokes, diagonal y degeneración. Curvatura (ángulo/π) y aceleración normalizada detectan dentado; un ángulo por sí solo genera aviso y score, no rechazo, porque los detalles afilados pueden ser válidos. La ausencia explícita de EOS o un límite de generación alcanzado es severa; metadata de terminación ausente queda como desconocida y se avisa.", "",
              "Una escala menor que el p1 TRAIN de la diagonal se avisa en generación completa, sin rechazo automático; no se penaliza por ello un suffix pequeño de completion, que puede añadir solo un detalle. Los límites robustos pueden marcar incluso colas reales legítimas. No se utiliza el validador para borrar datos originales y no garantiza plausibilidad semántica.", "",
              "`postprocess_candidate` conserva raw y postprocessed. Por defecto solo elimina puntos ADYACENTES exactamente duplicados del suffix; no recorta coordenadas ni borra strokes. Suavizado convexo opcional está apagado; conserva extremos y nunca toca prefix. Un candidato grave no se corrige: se conserva para diagnóstico y se rechaza. El wrapper de generación debe muestrear otro z para reemplazarlo si su presupuesto lo permite.", "",
              "## Score, ranking y diversidad", "",
              "Por característica la penalización es max(0, (valor−soft)/(severe−soft)). El score geométrico es 1/(1+máxima penalización), o cero si inválido. Así no se mezclan escalas mediante una suma arbitraria. El score detecta geometría, no calidad semántica o belleza.", "",
              "`rank_candidates` descarta inválidos y ordena lexicográficamente por score geométrico, número de avisos, score del modelo y posición original. `ordering=model_first` permite priorizar score del modelo entre válidos. Después aplica una separación mínima entre suffixes: Chamfer simétrico mayor que 0.05×mediana TRAIN de longitud de segmento (configurable). No usa el prefix compartido para inflar o diluir diversidad. Si faltan válidos/diversos, devuelve menos resultados; nunca rellena con inválidos.", "",
              "El Chamfer auxiliar muestrea posiciones por longitud de arco, hasta 256 puntos para costo acotado, sin conectar strokes distintos. Así un mismo segmento con 2 o 3 vértices no se considera diversidad. Solo cambia la métrica auxiliar; no altera datos ni representación. El reporte incluye candidatos inválidos, orden de calidad, duplicados filtrados, umbral efectivo, Chamfer mínimo/medio, diversidad de endpoints y número de strokes. No afirma que diversidad geométrica sea plausibilidad perceptual.", ""]
    Path("docs").mkdir(exist_ok=True)
    Path("docs/geometry.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "fitted", "path": str(args.output), "samples": stats["sample_count"],
                      "segment_p995": stats["intra_segment_p995"], "source_sha256": stats["source_sha256"]}))


if __name__ == "__main__":
    main()
