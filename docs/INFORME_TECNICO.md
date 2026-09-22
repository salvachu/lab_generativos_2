# Informe técnico de implementación

Documento histórico de la fase de infraestructura, actualizado para la publicación del proyecto. Para los resultados posteriores de Round 1 y Round 2 consulta [EXPERIMENTAL_FINDINGS.md](EXPERIMENTAL_FINDINGS.md).

**Estado actual:** A/B/C/D/E están implementados e integrados; 142 tests pasan. Training, resume exacto, evaluación, generación, completion, geometry, corrección, ranking y API local funcionan. Round 1 comparó los cinco modelos y Round 2 continuó C/E hasta 800 updates acumulados. No se publica ningún checkpoint ni se afirma que exista un ganador definitivo.

## Archivos y estructura final

```text
data/raw/                         train.pkl y val.pkl autorizados e incluidos
data/processed/                    normalization.json, geometry.json TRAIN-only
references/                       material original local, excluido de Git
src/sketchlab/
  data.py                         carga restringida y esquema real
  representation.py               archivo exacto y vista relativa
  models/common.py                batching variable y máscaras
  models/sequence.py              A/B/C
  models/hierarchical.py          D/E
  models/__init__.py              create_model(config)
  losses.py                      MDN, KL, free bits, feedback
  training.py                    trainer, sampler reanudable, CSV/JSON
  checkpointing.py               guardado atómico y modos de carga
  config.py                      JSON con herencia explícita
  evaluation.py                  métricas vectoriales y grids
  diagnostics.py                 KL schedules, collapse y sensibilidad
  generation.py                  API común de inferencia
  geometry.py                    calibración y validación independiente
  ranking.py                     ranking y diversidad del sufijo
  orchestration.py                sample_multiple y trazabilidad
  rendering.py                   SVG/PNG, orden y colores
app/                             server.py, index.html, app.js, style.css
configs/                         smoke, short, A/B/C/D/E, Round 2, geometry, prefix_example
scripts/                         train, evaluate, generate, benchmarks y summaries,
                                 fit_geometry, audit_data, setup/run_demo.ps1
tests/                           datos, representación, modelos, generación,
                                 checkpoint, training, geometry, ranking, API
runs/                            evidencias y checkpoints locales, excluidos de Git
docs/                            informe y referencias técnicas
README.md, pyproject.toml, requirements-lock.txt, .gitignore
```

Se creó esta infraestructura a partir de la carpeta con datos y referencias; las iteraciones posteriores continuaron sobre esos mismos archivos. La auditoría completa **no se repitió** después de la nueva indicación. Se realizó una calibración adicional necesaria exclusivamente con TRAIN para geometry. El fallo inicial de persistencia de parte de las estadísticas está documentado en `docs/dataset.md`; no se inventaron valores faltantes.

## Datos y representación

`train.pkl` y `val.pkl` contienen listas de 8187 y910 diccionarios. Keys exactas: `id:int`, `strokes:list[np.ndarray float32(N,2)]`, `parts:list[str]`, `step_ids:list[int]`, `description:str`. No hay columna pen-state: los límites de las listas separan strokes. Medianas TRAIN19 strokes/137 puntos; máximos211 strokes/1148 puntos. Coordenadas absolutas aproximadamente−2..504; canvas original no documentado. No se detectaron muestras inválidas en la revisión inicial.

El archivo canónico es `[x/256,y/256,down,stroke_end,sketch_eos]` en float64, con EOS adicional. `encode_sample/decode_sample` conserva geometría, dtype y metadata. Escala binaria fija y origen0; sin recentrar según el futuro, sin remuestrear ni recortar. Los modelos usan deltas float32 internos: esa vista tiene error numérico pequeño y **no** se considera reversible bit a bit. El caso próximo a cero que lo demuestra está cubierto por un test.

Se conservan todos los puntos y el orden; padding con máscaras, encoders packed y longitudes variables. Un stroke de un punto es representable. Completion copia el prefix original y añade el sufijo: ni el modelo, ni el postproceso, ni SVG redondean sus coordenadas originales.

## Modelos y losses

| Modelo | Arquitectura | Parámetros configuración estándar | Parámetros tiny |
|---|---|---:|---:|
| A | BiGRU→posterior global gaussiano; prior N(0,I); GRU autoregresivo, MDN correlacionada y tres estados del lápiz; beta1 |116929|14021|
| B | Misma red que A; beta, warmup/annealing y free bits configurables |116929|14021|
| C | Encoder de prefix y prior aprendido p(z\|prefix); posterior con sketch completo; decoder condicionado; reconstrucción solo del sufijo |195809|22901|
| D | Encoder variable por stroke→BiGRU global; latent global y local; GRU de strokes, anclaje absoluto MDN, GRU de deltas internos, dos niveles de EOS |306144|36280|
| E | Jerarquía D con contexto del prefix, prior global condicional y losses del sufijo |411232|47752|

Estándar: hidden96, latent32, local16,5 mezclas,1 capa. Tiny: hidden32, latent8, local8,3 mezclas,1 capa. Capas, dropout y teacher forcing configurables. D/E aplican scheduled sampling en puntos; los embeddings anteriores de strokes usan teacher forcing durante entrenamiento. No se añadieron atributos de partes inventados ni una rama raster sin evidencia experimental.

Loss principal: NLL de mezcla gaussiana bivariada + CE de límites/EOS + beta×KL controlado. Se registran reconstrucción, coordenadas, CE, KL global/local bruto, KL con free bits, beta efectivo, loss total y ELBO por sketch. `auxiliary_loss=0` explícito: la suavidad está en el detector/postproceso; no se introdujo una penalización diferencial rígida sin validar su efecto sobre diversidad.

El objetivo normalizado por eventos es un surrogate distinto del ELBO. El ELBO sumado por sketch es estimador Monte Carlo con posterior muestreado y teacher forcing completo; la validación determinista usa q-mean y se etiqueta como diagnóstico. Las medias de validación se agregan por sus denominadores reales, independientemente del tamaño del lote. No comparar directamente NLL entre familias con distintas factorizaciones o máscaras.

Schedules: constante, lineal, coseno y cíclico. Alertas de KL global persistentemente bajo, KL local y dependencia numérica del decoder respecto de z. Sensibilidad latente y diversidad no prueban calidad semántica.

## Completion, random y API común

`load_model`, `encode`, `sample_latent`, `sample`, `complete_sketch`, `generate_random`, `sample_multiple`, `validate_candidate`, `rank_candidates`, `render`; la UI usa el servicio común, sin clases de modelo concretas. `create_model` es la única selección A–E.

Los prefijos de entrenamiento proceden solo de TRAIN: vacío,1stroke,2strokes,25%,50%,75%. C/E usan p(z\|prefix) al generar; no reciben futuro. A/B/D pueden continuar la historia autoregresiva sin prior global condicionado. z explícito y semillas separadas para global, local y decoder permiten estudiar diversidad sin confundirla con cambiar solamente el ruido del decoder. `generate_random` funciona con prefix vacío. Caps afectan únicamente puntos/strokes nuevos y se registran; nunca se presentan como EOS aprendido.

## Geometría, postproceso y ranking

Calibración persistida exclusivamente desde TRAIN, con hash, percentiles y MAD. Ejemplo de umbral de salto interno: q99.5=154.458 unidades; límite severo≈314.053. Se detectan NaN/Inf, estructura malformada, coordenadas extremas, saltos internos, oscilación, degeneración, escala, cantidades excesivas, caps y ausencia de EOS. Los reposicionamientos entre strokes no cuentan como discontinuidades. El prefix se valida estructuralmente, pero no se penaliza por su estilo o posición.

`ValidationResult` contiene `valid`, `score`, `warnings`, `penalties`, `corrections`, errores graves y métricas. Score geométrico: `1/(1+máximo exceso normalizado)`; los inválidos puntúan0. El score no es una probabilidad de plausibilidad.

Postproceso conserva `raw_output` y `postprocessed_output`. Por defecto solo elimina duplicados adyacentes exactos del sufijo; smoothing convexo optativo, desactivado. Los errores graves se rechazan y se exploran otros z dentro del presupuesto; no se ocultan mediante clipping. El prefijo permanece intacto.

Ranking: válidos primero, orden lexicográfico por score geométrico/avisos/score de modelo opcional; después filtrado greedy por Chamfer del sufijo muestreado a lo largo de sus segmentos. El umbral de diversidad deriva de la mediana TRAIN de segmentos. No hay una suma opaca de escalas distintas. La generación actual no proporciona likelihood como score de ranking: la clave opcional permite incorporarlo después. Se conserva cada candidato, incluso rechazados; el sistema puede devolver menos que top-k y lo informa.

## Rendering e interfaz

SVG y grids PNG, orden de strokes, prefijo azul y continuación naranja; exportación y comparación raw/postprocesado. Demo FastAPI+canvas en **http://127.0.0.1:8000**: mouse/touch, Clear, Undo, Random, Complete, Regenerate, presupuesto de candidatos,3/6/9 resultados, selección, SVG/JSON, selector A–E/checkpoint, modo RAW explícito. Presupuesto por defecto20 y top-k6; tiny real disponible, sin muestras ficticias.

QA visual inicial, dibujo y deshacer comprobados. QA HTTP real posterior con E: completion RAW, validada y random devolvieron200, con prefix exacto. Evidencia `runs/demo/http_smoke.json`. En esa última comprobación CUA no ofrecía navegador; por ello no se afirma una nueva captura con resultados reales. `runs/demo/interface.png` es la captura previa a la aparición de checkpoints.

Al cerrar la entrega, la revisión automática rechazó el comando de reinicio del servidor con `blocked by policy`, sin explicar otro motivo. Se conservó la instancia existente y se volvió a comprobar que `/api/models` responde y lista los cinco checkpoints `smoke_verified`. Una nueva ejecución del servidor cargará todas las versiones finales de los módulos; no se intentó eludir el bloqueo.

Renders inspeccionados: `runs/audit/train_grid.png`, `roundtrip_grid.png`, `outliers_grid.png`, y `runs/smoke_verified_inference/tiny_grid.png`. Los tiny producen segmentos/puntos, como corresponde a tres actualizaciones; no equivalen a la demo entrenada del profesor.

## Tests, smoke y hardware

**142 pasados,0 fallidos finales**, en≈8s; `runs/tests.xml`. Dos avisos de deprecación de Starlette/httpx/AnyIO, sin fallo funcional. Cobertura: carga restringida, round-trip y metadata, singletons,211strokes, strokes largos, límites/NaN/Inf, EOS/padding, forward/backward A–E, gradientes finitos, reparametrización, causalidad a z fijo, prior sin futuro, prefix exacto, semillas, geometry/ranking, API, save/load y resume.

Se reprodujo y corrigió una diferencia de resume CUDA con dropout. La prueba continua4steps frente a2+resume2 ahora coincide **bit a bit en CPU y CUDA**. Para GRU multicapa con dropout en CUDA se usa backend recurrente nativo, porque cuDNN mantiene estado de dropout fuera de los RNG guardados; esto puede costar rendimiento. La garantía está acotada al mismo backend/versiones/dispositivo, no a migraciones entre máquinas.

PyTorch2.7.1+cu126, Python3.12.14; CUDA operativo, **GTX1660Ti6GB**. Los cinco tiny usaron32muestras TRAIN/8VAL, batch4,3steps. Tiempo de optimización≈0.14–0.32s/modelo y pico de memoria PyTorch≈37–57MiB; no incluyen reserva del driver/escritorio ni constituyen benchmark. Se preservó un primer smoke en `runs/smoke`; la evidencia final tras las correcciones es **`runs/smoke_verified`**. No se ejecutaron cientos de steps.

Checkpoints actuales: `runs/smoke_verified/{A,B,C,D,E}/best.pt` y `last.pt`. Se guardan pesos, optimizer, scheduler si existe, step/epoch, config, normalización, versión de representación, RNG Python/NumPy/Torch CPU/CUDA y estado del sampler/prefix RNG. Guardado atómico; se rechaza una salida existente. `inference` carga pesos; `warm_start` inicia optimizer/RNG nuevos; `resume` recupera entrenamiento y exige compatibilidad. El schedule KL depende del step restaurado.

## Comandos exactos para la siguiente fase

Ejecutar en PowerShell desde la raíz. **Estos entrenamientos comparativos están preparados, no ejecutados.** A–E heredan `short.json`:1024TRAIN/96VAL,160steps, límite180s de optimización por modelo. Revisar estos presupuestos en la fase siguiente; mantenerlos iguales para comparar.

```powershell
.\.venv\Scripts\python.exe -m scripts.train --config configs/A.json --models A --output runs/comparison_A
.\.venv\Scripts\python.exe -m scripts.train --config configs/B.json --models B --output runs/comparison_B
.\.venv\Scripts\python.exe -m scripts.train --config configs/C.json --models C --output runs/comparison_C
.\.venv\Scripts\python.exe -m scripts.train --config configs/D.json --models D --output runs/comparison_D
.\.venv\Scripts\python.exe -m scripts.train --config configs/E.json --models E --output runs/comparison_E
```

Evaluar, generar y completar usando el E comparativo que produzca el comando anterior:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate runs/comparison_E/E/best.pt --output runs/eval_E --count 16 --samples 3
.\.venv\Scripts\python.exe -m scripts.generate runs/comparison_E/E/best.pt --output runs/random_E --candidates 20 --top-k 6
.\.venv\Scripts\python.exe -m scripts.generate runs/comparison_E/E/best.pt --prefix configs/prefix_example.json --output runs/completion_E --candidates 20 --top-k 6
.\.venv\Scripts\python.exe -m uvicorn app.server:app --host 127.0.0.1 --port 8000
```

Reanudar exactamente E o usar sus pesos como inicio, siempre en salida nueva. `steps` representa actualizaciones adicionales al reanudar; warm-start inicia step0. Las arquitecturas deben coincidir, por lo que el tiny no puede usarse como warm-start del estándar.

```powershell
.\.venv\Scripts\python.exe -m scripts.train --config configs/E.json --models E --resume runs/comparison_E/E/last.pt --output runs/resume_E
.\.venv\Scripts\python.exe -m scripts.train --config configs/E.json --models E --warm-start runs/comparison_E/E/best.pt --output runs/warmstart_E
```

Para repetir solo la verificación, sin torneo y con directorios nuevos:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m scripts.train --config configs/smoke.json --models A,B,C,D,E --output runs/smoke_next
.\.venv\Scripts\python.exe -m scripts.smoke_inference --runs runs/smoke_next --output runs/smoke_next_inference
```

Los comandos de generation/completion/evaluate se verificaron realmente con E tiny en `runs/cli_completion_smoke` y `runs/cli_eval_smoke`; este último solo utilizó1muestra VAL y límites32puntos. El ejemplo de prefijo JSON puede sustituirse por la exportación de la UI.

## Lectura de resultados por el siguiente agente

- `runs/<suite>/manifest.json`: índices/IDs train-val, hashes, seeds, entorno y configuración efectiva.
- `runs/<suite>/<modelo>/history.jsonl`, `metrics.csv`, `metrics.json`: reconstruction/NLL/CE, KL global/local bruto y objetivo, ELBO y flag MC, beta, gradientes, tiempos; `summary.json` y `validation_<step>.json`.
- `runs/<suite>/scoreboard.csv`/`.json`: parámetros, dimensiones, estrategiaKL, validación, diversidad, VRAM, checkpoint y estado. Las tablas de smoke son registros de integridad, no un ranking de calidad.
- `samples/generation_metrics.json`: completion para1/2strokes y25/50/75%, Chamfer/Hausdorff/endpoints, canvas, degeneración, saltos, longitud/cantidad, EOS/caps, diversidad a decoder RNG fijo. `generated_samples.json` conserva vectores.
- `scripts.generate` produce `candidates.json` con raw/postprocesado, validación, correcciones, ranking y descartes; SVG y `raw_vs_processed.png`. `scripts.evaluate` añade `validation_likelihood.json`.
- `diagnostics.collapse_report` y `latent_sensitivity` distinguen KL bajo, dependencia del latente y diversidad. Combinar calidad geométrica, diversidad **entre aceptados**, frecuencias de terminación y revisión visual; no elegir por loss única ni tratar una continuación GT como la única válida.

Los umbrales geométricos pueden rechazar colas legítimas del dataset. Dots y detalles pequeños pueden ser geométricamente aceptados con avisos; **validez geométrica no implica plausibilidad**. No se ha calibrado un juez semántico ni demostrado que E sea superior. Faltan entrenamiento comparativo, selección mediante VALIDATION+renders y después entrenamiento más largo del candidato elegido por la siguiente fase.

## Git y preservación

Se inicializó un repositorio local; no había `.git`. No hay commits, staging ni push. `git status --short`: archivos nuevos sin seguimiento (`.gitignore`, `README.md`, `app/`, `configs/`, `docs/`, `pyproject.toml`, `requirements-lock.txt`, `references/`, `runs/`, `scripts/`, `src/`, `tests/`). **`git diff --stat` vacío**, porque los archivos son untracked, no porque falten cambios.

`.gitignore` excluye raw, video, `.venv`, checkpoints, outputs vectoriales masivos y renders. Los datos conservaron sus hashes, verificados al finalizar los smoke tests:

```text
train.pkl bc2f939ab6ceb816375acdd536d126115c6cf1646ad339d14f6a533321bf39a7
val.pkl   b6f194c60379351933275019fe5204ae1663d63a68490cf5c566b62b9b09fb52
```

No hace falta volver a auditar los datos para continuar. El siguiente agente puede empezar por los comandos comparativos y los CSV/JSON anteriores.
