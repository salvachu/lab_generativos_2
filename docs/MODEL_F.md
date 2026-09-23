# MODEL F — CoSE-inspired Hierarchical Conditional VAE

Implementación original en PyTorch. Es una nueva variante experimental; A–E y sus checkpoints se mantienen. No se consultó ni copió código de repositorios externos. Referencia conceptual: Emre Aksan, Thomas Deselaers, Andrea Tagliasacchi y Otmar Hilliges (2020), **CoSE: Compositional Stroke Embeddings**, NeurIPS 2020. [Paper oficial](https://arxiv.org/html/2006.09930), [ficha arXiv](https://arxiv.org/abs/2006.09930).

## Ideas tomadas de CoSE

Strokes de longitud variable se representan mediante embeddings fijos. Se separan geometría local y posición inicial. Un modelo relacional compone strokes, prediciendo posición y código. El decoder local evalúa una curva condicionada por código y parámetro t sin recurrencia entre puntos. El CoSE principal modela la composición sin orden temporal entre strokes y usa un autoencoder local; su decoder local probabilístico emplea mezclas Gaussianas. Véanse las secciones 3.1–3.3 del paper.

## Adaptaciones nuestras

F conserva el orden real del dataset mediante posiciones sinusoidales y atención causal. Añade un VAE global condicional, EOS entre strokes, una curva local determinista, interpolación por longitud de arco y entrenamiento en dos stages. El autoencoder local de F no es variacional; el modelo completo sí lo es mediante q(z|full,prefix), p(z|prefix) y KL(q||p). Estas decisiones pertenecen a este laboratorio, no se atribuyen al paper.

## Arquitectura exacta V1

| módulo | configuración |
|---|---|
| vista de stroke | anchor inicial absoluto / 256; shape relativa / 256; M=16 posiciones t uniformes en longitud de arco normalizada |
| StrokeEncoder | input (t,x_rel,y_rel); proyección 3→96; Transformer 2 capas, 4 cabezas, FFN 384, GELU, pre-LayerNorm; pooling con máscara; proyección 96→32 |
| StrokeDecoder | MLP (lambda32 + t y 8 features sinusoidales)→96→96→2, GELU; shape(t)=MLP(lambda,t)−MLP(lambda,0) |
| tokens de sketch | concatenación (lambda32,anchor2)→96; BOS cero; posiciones sinusoidales de orden de strokes |
| prefix/global encoder | Transformer causal compartido, 2 capas/4 cabezas/FFN384; resumen del último token observado; flag de prefix vacío |
| prior | Linear(97,64)→mu/logvar de z32; prefix vacío fuerza mu=logvar=0, exactamente N(0,I) |
| posterior | Linear(194,64) sobre resumen completo y resumen del prefix; reparameterization durante TRAIN |
| decoder relacional | Transformer causal, 2 capas/4 cabezas/FFN384; condicionamiento aditivo por contexto y z; predice tras BOS o stroke anterior |
| anchor | MDN bivariado correlacionado de 5 componentes, reutiliza la implementación local existente |
| embedding siguiente | Gaussian diagonal D=32; MLP de hidden96+anchor2→96→64; logvar limitado a [−6,2] |
| EOS | Linear(96,2): continue/EOS exclusivamente a nivel de strokes; ponderación [1,3] |
| dropout | 0.0 inicialmente |

Total: **742,050 parámetros**. Stroke AE: **240,898**. Composición con AE congelado: **501,152 entrenables**.

La factorización es p(stop_k|history,z,prefix), p(anchor_k|history,z,prefix), p(lambda_k|anchor_k,history,z,prefix). El decoder transforma lambda y t directamente en shape; no se suma una cadena de deltas predichos. Durante inference se realimentan el código muestreado y el anchor a nivel de stroke, sin re-encoding de puntos ni feedback punto-a-punto. Todos los puntos de una curva se calculan en una llamada.

E ya poseía anchors y EOS a nivel de strokes. La diferencia decisiva de F es sustituir su decoder local recurrente de deltas por una curva paralela y modelar composición mediante códigos de forma y atención causal. No es un E de mayor tamaño.

## Vista de entrenamiento y máscaras

`src/sketchlab/stroke_view.py` mantiene intactos arrays y dtype canónicos. Interpola la polilínea por longitud de arco, elimina distancias consecutivas nulas solo para interpolación y devuelve anchors, shape, t, máscara de strokes y máscara de puntos. Strokes de un punto o longitud nula producen shape cero; los largos se interpolan completos, sin truncarlos por índice. Cada stroke termina en t=1.

TRAIN: 8,187 sketches, 188,702 strokes; puntos/stroke p50=4, p75=8, p90=16, p95=21, p99=29, máximo=406. M=16 corresponde a p90 y limita memoria. Esta vista puede perder esquinas finas y no es una reconstrucción exacta de raw; M es configurable. Estadísticas y hash TRAIN: `docs/model_f_train_statistics.json`.

Generation usa M puntos por stroke; los conteos de puntos no son directamente comparables con los puntos raw de A–E. El decoder puede evaluarse en cualquier t-grid, pero V1 fija M también en generation. F_short reserva 1,024 puntos para hasta 64 strokes de 16 puntos. El sampler respeta límites y reporta cap si no cabe otro stroke completo. La UI mantiene sus límites configurables existentes.

## VAE y ausencia de future leakage

TRAIN muestrea z del posterior con información completa. Completion muestrea únicamente p(z|prefix); random usa N(0,I). El resumen del prefix excluye tokens futuros mediante máscara y selección del último token observado. El decoder recibe BOS y strokes previos desplazados; la posición k no ve el target k. Las losses de composición se aplican solo al suffix y al slot EOS real, nunca a padding. El posterior sí puede leer el futuro durante entrenamiento, como exige un CVAE; esto no se utiliza durante generation.

El output de completion concatena copias exactas del prefix original con los nuevos strokes. El prefix no pasa por el decoder para su salida. Los tests comprueban bytes, dtype y valores.

## Loss modular

L = lambda_stroke·L_AE + lambda_anchor·NLL_anchor + lambda_embedding·NLL_embedding + lambda_eos·CE_EOS + beta·KL(q||p).

Los cuatro coeficientes iniciales son 1. L_AE es MSE media de coordenadas relativas normalizadas sobre strokes reales, con suma implícita dividida por strokes×M×2. Anchor es NLL media por stroke futuro. Embedding es NLL Gaussian media por stroke y dimensión, con target lambda detached. EOS es CE ponderada por evento stroke/EOS; [1,3] usa sqrt(188702/8187)=4.801 capeado a 3. EOS natural es 4.16% de slots TRAIN; la tasa efectiva aumenta al enmascarar prefixes. No se usan frecuencias VAL para pesos.

KL se suma por dimensión y promedia por sketch, con warmup beta hasta 0.05 y free_bits 0.02/dim. Se registra KL raw y KL controlado. También se registran cada componente, precisión/recall/confusión EOS y P(EOS) final/no-final. Validación agrega cada componente según sus eventos para no depender de la partición en batches.

Esta es una loss VAE modular regularizada sobre códigos aprendidos; por ponderaciones y normalizaciones **no se presenta como una log-likelihood exacta de los puntos raw ni como ELBO raw comparable con A–E**. No se calcula ni publica un ELBO ficticio.

## Stages, optimizer y checkpoints

F_STROKE_AE usa `training_stage: stroke_ae` con `--models F`: entrena solo encoder/decoder local, sin pérdidas de composición ni KL. Stage 2 usa `training_stage: composition` y `--stroke-ae` para transferir esos pesos. Por defecto ambos módulos locales quedan congelados, incluido dropout en eval. `freeze_stroke_ae:false` permite fine-tune; `stroke_ae_lr_factor:0.1` asigna LR diez veces menor a sus parámetros mediante grupo Adam separado. No se activó fine-tune en smoke.

El trainer, BatchStream, splits, prefix sampling, KL schedule y checkpoints son los existentes. Guardan optimizer, config, RNG CPU/CUDA/Python/NumPy, estado del sampler, IDs, paso de KL, normalización y metadata `f-anchor-relative-arclength-v1`. No hay LR scheduler en V1: `scheduler_state=null`, como en A–E; el calendario KL queda en config y step. Resume restaura estados exactamente y rechaza cambios de stage/freeze/LR factor. Warm-start carga pesos con optimizer nuevo. `--stroke-ae` transfiere solo la representación local y comprueba compatibilidad. Cada salida exige un directorio nuevo.

Inference usa el mismo `load_model/sample/generate_random/complete_sketch/sample_multiple`, geometry validator, ranking y rendering. Los checkpoints de solo AE se rechazan para generation y no aparecen en el selector; los de composición F compatibles se descubren automáticamente. No cambió el diseño visual de la UI.

## Smoke realizado, sin entrenamiento de calidad

GTX 1660 Ti, CUDA. Stage 1: 40 updates, 1.487 s de kernels/update loop, 2.323 s del experimento, pico 69.24 MiB. Stage 2: 6 updates, 0.594 s de training, 1.656 s del experimento, pico 45.73 MiB. Tiempos no incluyen arranque de Python y lectura inicial; VRAM es memoria máxima asignada por PyTorch, no ocupación total de la GPU.

MSE AE de validación baja 0.019516→0.005535. En 32 strokes de casos VAL fijados: RMSE medio 19.03 px, Chamfer 11.86 px, endpoint error 42.89 px. Los grids todavía muestran deformación y pérdida de forma: el AE necesita entrenamiento posterior. Se probaron diez generaciones/completions con cap de seis strokes: 4 EOS, 6 caps, todas finitas, prefix bit-exacto. La representación local quedó bit-exactamente congelada en stage 2. El pipeline geometry/ranking funcionó y rechazó 3/3 candidatos de su prueba; no hay certificación de calidad.

Artefactos:

- `runs/model_f_smoke/stroke_ae/F/last.pt`
- `runs/model_f_smoke/composition/F/last.pt`
- `runs/model_f_smoke/stroke_evaluation/stroke_reconstruction.png`
- `runs/model_f_smoke/inference/random.png`
- `runs/model_f_smoke/inference/completion.png`
- `runs/model_f_smoke/inference/summary.json`

## Tests y compatibilidad

Suite final: **166 passed**, incluidos los 144 tests anteriores y 22 casos nuevos F. Se probaron interpolación/máscaras/strokes largos o degenerados, invariancia a traslación, causalidad del prefix y decoder, parámetros prior/posterior, KL y reparameterization, forward/backward CPU y CUDA, composición y AE, prefix exacto, generación sin feedback de puntos, EOS/caps, integración geometry/ranking/render, guardas de metadata, transferencia/congelación/fine-tune y resume bit-exacto de ambos stages en CPU. Los checkpoints persistidos A/B/C/D/E cargaron correctamente. La prueba HTTP excluye el AE del selector e incluye el checkpoint generativo. Dos avisos de deprecación de Starlette/httpx preexistentes, sin fallos.

## Comandos exactos desde la raíz del proyecto

PowerShell; usar directorios nuevos si ya existen resultados. Los siguientes entrenamientos son la recomendación posterior, no se ejecutaron en esta entrega.

```powershell
# 1. Pretrain Stroke AE corto: 300 updates como máximo, 180 s
.\.venv\Scripts\python.exe -m scripts.train --config configs\F_stroke_short.json --models F --output runs\F_stroke_short

# 2. Evaluar Stroke AE sobre los casos VAL persistidos
.\.venv\Scripts\python.exe -m scripts.evaluate_f_stroke runs\F_stroke_short\F\best.pt --output runs\F_stroke_short_eval --count 64

# 3. Entrenar F_short con AE congelado: 400 updates como máximo, 180 s
.\.venv\Scripts\python.exe -m scripts.train --config configs\F_short.json --models F --stroke-ae runs\F_stroke_short\F\best.pt --output runs\F_short

# 4. Evaluar F, exactamente los ocho casos VAL anteriores
.\.venv\Scripts\python.exe -m scripts.evaluate runs\F_short\F\best.pt --output runs\F_short_eval --cases runs\visual_benchmark_round1\selected_validation_cases.json --count 8 --samples 3 --seed 73100 --temperature 0.9 --max-points 1024

# 5. Random generation, raw para inspección de diagnóstico
.\.venv\Scripts\python.exe -m scripts.generate runs\F_short\F\best.pt --output runs\F_short_random --candidates 4 --top-k 4 --seed 73100 --temperature 0.9 --max-points 1024 --max-strokes 64 --raw --no-postprocess

# 6. Completion con prefix de un caso VAL persistido
.\.venv\Scripts\python.exe -m scripts.generate runs\F_short\F\best.pt --prefix runs\model_f_smoke\inference\prefix.json --output runs\F_short_completion --candidates 4 --top-k 4 --seed 73100 --temperature 0.9 --max-points 1024 --max-strokes 64 --raw --no-postprocess
```

Los mismos comandos de generation sin `--raw --no-postprocess` aplican geometry/ranking. Puede haber cero candidatos aceptados; no se sustituyen por inválidos.

Para continuar exactamente un entrenamiento, usar la misma config y `--resume <last.pt> --output <directorio_nuevo>`: `steps` son updates adicionales. Para reiniciar optimizer/RNG desde pesos F, usar `--warm-start` en lugar de `--resume`. Para descongelar AE después, crear una config con `freeze_stroke_ae:false`, LR factor bajo y hacer warm-start; no cambiar el objetivo bajo resume exacto.

## Siguiente entrenamiento recomendado

Ejecutar únicamente el pretrain AE corto de 300 updates y revisar sus reconstrucciones VAL. No iniciar composición seria mientras los strokes sigan perdiendo orientación/curvatura/extremos. Si esa representación es suficiente, ejecutar F_short de 400 updates con AE congelado y revisar geometría, organización y EOS. Ninguno de estos entrenamientos cortos posteriores se ejecutó aquí.
