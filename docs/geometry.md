# Geometría y selección de candidatos

Calibración TRAIN-only: 8187 sketches. SHA256: `bc2f939ab6ceb816375acdd536d126115c6cf1646ad339d14f6a533321bf39a7`.

Se persiste en `data/processed/geometry.json`; entrenamiento/inferencia la cargan sin releer el dataset. `scripts/fit_geometry.py` reutiliza el archivo si existe; refit requiere `--force`. VALIDATION no participa.

## Umbrales

Todos los límites numéricos de geometría salen de TRAIN. Los hiperparámetros del procedimiento están en `configs/geometry.json`: soft=q99.5; severe=max(q99.9 + 6×1.4826×MAD, 1.5×q99.9). El factor 1.4826 estima sigma robusta bajo una distribución normal; no supone que la geometría sea normal. Los factores 6 y 1.5 son tolerancias configurables y documentadas, no distancias en píxeles.

Las coordenadas usan extrema observada TRAIN como región blanda y añaden un margen del 10% del rango aprendido para extrema severa. La pequeña cola negativa real permanece válida. El canvas original sigue siendo desconocido.

| Medida | Mediana | Límite blando | Límite severo |
|---|---:|---:|---:|
| segment_length | 19.3262 | 154.458 | 314.053 |
| stroke_arc_length | 71.9545 | 1107.22 | 2646.12 |
| turning | 0.152643 | 0.986623 | 1.73003 |
| acceleration | 0.421659 | 0.999818 | 2.69361 |
| stroke_jaggedness | 0.145243 | 0.952721 | 2.28734 |
| stroke_count | 19 | 95.07 | 195.105 |
| point_count | 137 | 519.14 | 1011.66 |
| points_per_stroke | 4 | 37 | 115.5 |
| bbox_diagonal | 560.334 | 705.882 | 1330.15 |
| zero_arc_fraction | 0 | 0 | 0 |

Rango aprendido: `{'min': [-1.9937844276428223, -1.4159601926803589], 'max': [504.0, 504.0], 'central_span': [457.7133184814453, 470.5833299446106], 'severe_min': [-52.593162870407106, -51.9575562119484], 'severe_max': [554.5993784427643, 554.541596019268]}`.

## Validación y corrección

`validate_candidate` devuelve `valid`, `score`, `warnings`, `penalties`, `severe_errors`, `corrections` y métricas. NaN/Inf y estructuras inválidas se rechazan incluso en el prefix. El prefix finito no se evalúa contra los umbrales estadísticos ni se modifica. Las distancias entre strokes no participan en ninguna penalización de continuidad.

Los umbrales se aplican al suffix generado: coordenadas, segmento máximo intra-stroke, longitud de arco, puntos/stroke, puntos/sketch, número de strokes, diagonal y degeneración. Curvatura (ángulo/π) y aceleración normalizada detectan dentado; un ángulo por sí solo genera aviso y score, no rechazo, porque los detalles afilados pueden ser válidos. La ausencia explícita de EOS o un límite de generación alcanzado es severa; metadata de terminación ausente queda como desconocida y se avisa.

Una escala menor que el p1 TRAIN de la diagonal se avisa en generación completa, sin rechazo automático; no se penaliza por ello un suffix pequeño de completion, que puede añadir solo un detalle. Los límites robustos pueden marcar incluso colas reales legítimas: por ejemplo, 211 strokes excede el límite generado conservador de 195.105. No se utiliza el validador para borrar datos originales y no garantiza plausibilidad semántica.

`postprocess_candidate` conserva raw y postprocessed. Por defecto solo elimina puntos ADYACENTES exactamente duplicados del suffix; no recorta coordenadas ni borra strokes. Suavizado convexo opcional está apagado; conserva extremos y nunca toca prefix. Un candidato grave no se corrige: se conserva para diagnóstico y se rechaza. El wrapper de generación debe muestrear otro z para reemplazarlo si su presupuesto lo permite.

## Score, ranking y diversidad

Por característica la penalización es max(0, (valor−soft)/(severe−soft)). El score geométrico es 1/(1+máxima penalización), o cero si inválido. Así no se mezclan escalas mediante una suma arbitraria. El score detecta geometría, no calidad semántica o belleza.

`rank_candidates` descarta inválidos y ordena lexicográficamente por score geométrico, número de avisos, score del modelo y posición original. `ordering=model_first` permite priorizar score del modelo entre válidos. Después aplica una separación mínima entre suffixes: Chamfer simétrico mayor que 0.05×mediana TRAIN de longitud de segmento (configurable). No usa el prefix compartido para inflar o diluir diversidad. Si faltan válidos/diversos, devuelve menos resultados; nunca rellena con inválidos.

El Chamfer auxiliar muestrea posiciones por longitud de arco, hasta 256 puntos para costo acotado, sin conectar strokes distintos. Así un mismo segmento con 2 o 3 vértices no se considera diversidad. Solo cambia la métrica auxiliar; no altera datos ni representación. El reporte incluye candidatos inválidos, orden de calidad, duplicados filtrados, umbral efectivo, Chamfer mínimo/medio, diversidad de endpoints y número de strokes. No afirma que diversidad geométrica sea plausibilidad perceptual.
