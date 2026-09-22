# Dataset y representación verificados

El PDF completo describe 9097 sketches válidos: 8187 TRAIN y 910 VALIDATION. Los archivos suministrados son `train.pkl` y `val.pkl` sin compresión, aunque el PDF menciona `.pkl.gz`. Los conteos reales coinciden.

La inspección inicial de ambos archivos encontró una `list` de `dict`, cada uno con exactamente:

| Campo | Tipo real | Relación |
|---|---|---|
| id | int | Identificador del sketch |
| strokes | list[numpy.ndarray] | Orden de dibujo explícito |
| cada stroke | ndarray float32 (N,2) | Coordenadas absolutas x,y; N variable |
| parts | list[str] | Una etiqueta por stroke |
| step_ids | list[int] | Etapa por stroke; pueden repetirse |
| description | str | Descripción textual |

No hay pen state explícito, campo `stage` separado, EOS ni masks. La separación entre arrays marca pen-up y límites de stroke. No se encontraron errores de esquema/finitud en las 9097 muestras.

Estadísticas iniciales verificadas de TRAIN:

| Medida | Min | p25 | Mediana | p75 | p95 | p99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Strokes/sketch | 8 | 14 | 19 | 27 | 51 | 83.14 | 211 |
| Puntos/sketch | 43 | 104 | 137 | 186 | 314.7 | 458.84 | 1148 |

El rango TRAIN es x=[−1.99378443,504], y=[−1.41596019,504]. El tamaño original del canvas no figura en PDF ni metadata; aproximadamente 0–504 es una observación, no una afirmación sobre la interfaz original. El render usa el envelope fijo [−4,508]² para conservar la cola negativa. Los outliers se conservan.

La auditoría exhaustiva inicial verificó roundtrip canónico en las 9097 muestras. Sus estadísticas intermedias completas no se persistieron por un fallo del renderer antes de escribir el JSON; no se ha repetido esa auditoría tras la instrucción de continuar sin recorrer nuevamente los datos. La calibración geométrica necesaria se realiza una vez sobre TRAIN y persiste distribuciones completas en `data/processed/geometry.json`. No se inventan estadísticas VAL ni resultados de duplicados/overlap que no quedaron guardados.

## Representación

`encode_strokes` → `decode_tokens` usa stroke-5 ABSOLUTO `[x/256,y/256,down,end_stroke,EOS]`, con dtype canónico float64. Cada punto conserva su estado de salida, el final de stroke es explícito y EOS ocupa un token independiente. La escala potencia de dos mantiene exactamente los valores float32 originales. Se conserva el orden, incluso strokes de un punto, sin resampling ni truncamiento.

`encode_sample` → `decode_sample` conserva además metadata y dtype de cada array. Las copias independientes de metadata impiden que una modificación del resultado altere el original.

Los modelos secuenciales usan `encode_deltas` / `decode_deltas`: `[dx/256,dy/256,down,end_stroke,EOS]`, origen fijo (0,0) y movimiento global respecto al punto anterior. Esta vista NO es universalmente exacta: la muestra TRAIN id=4322 contiene y=3.8224968329503284e−11; la resta/acumulación float64 produce 3.822497873784414e−11. Por eso el prefix de completion se copia directamente desde sus coordenadas originales y jamás se reconstruye acumulando deltas. Las copias float32 del entrenamiento introducen adicionalmente redondeo normal.

La escala 256 está ajustada solo a TRAIN mediante `2**ceil(log2(max(abs(train_coordinates))/2))`. No hay recentrado ni normalización según el suffix. El batch usa padding, lengths y masks que incluyen EOS y excluyen padding. `in_stroke_mask` solo marca movimientos que trazan tinta; nunca los reposicionamientos entre strokes.

## Carga y evidencias

`load_raw` usa un Unpickler restringido a `numpy.dtype` y `numpy._core.numeric._frombuffer` (alias de compatibilidad `numpy.core.numeric`), con wrapper que rechaza objetos/structured, formas y tamaños fuera de límites. Rechaza otros globals, referencias persistentes y bytes posteriores al objeto. No usa `pickle.load` genérico. No es un sandbox universal contra agotamiento de recursos en cualquier pickle arbitrario.

El SHA256 TRAIN se conserva en `data/processed/normalization.json` y `geometry.json`; `data/raw` no se modifica. Grids de TRAIN, casos extremos y originales/roundtrip se guardan en `runs/audit`. Las estadísticas y políticas geométricas se documentan en `docs/geometry.md`.
