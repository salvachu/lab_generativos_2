# Arquitecturas y contrato de inferencia

Implementación adaptada al dataset auditado y a las referencias enlazadas en
`docs/research.md`. Son cinco variantes comparables por familias, no reproducciones
de los artículos ni modelos cuya calidad perceptual esté demostrada. En esta fase
solo se ejecutan tests y entrenamiento tiny de verificación.

## Representación y longitud variable

El archivo canónico es `representation.encode_strokes`: coordenadas **absolutas**
en float64, escala binaria 256, límites de stroke y EOS independiente. Conserva
exactamente las coordenadas float32 originales. `encode_sample/decode_sample`
conservan también dtypes y metadata. Las coordenadas casi cero del dataset hacen
que ni siquiera acumular deltas float64 sea universalmente reversible bit a bit.

A/B/C usan explícitamente `encode_deltas`, con tokens
`[dx/256, dy/256, down, stroke_end, sketch_eos]`. El final del stroke pertenece a
su último punto; EOS no aporta coordenada. El desplazamiento al primer punto de
otro stroke **reposiciona** el lápiz y no representa un segmento dibujado. La vista
de entrenamiento se convierte a float32 y admite error numérico, a diferencia del
archivo canónico absoluto. `models/common.tokens` delega a esta conversión común.

Las secuencias de entrenamiento conservan todos los puntos y strokes, sin
remuestreo, recorte ni límites de inferencia. Los encoders GRU usan
`pack_padded_sequence`; los decoders y las losses usan padding y máscaras. En las
redes de varias capas se toma exclusivamente el estado de la última capa de cada
dirección. Los tests incluyen 211 strokes y un stroke de 257 puntos.

## Configuración compartida

`create_model(config)` devuelve un `torch.nn.Module` y conserva la configuración
normalizada en `.config`. Los defaults son `hidden_dim=96`, `latent_dim=32`,
`stroke_latent_dim=16`, `mixtures=5`, `layers=1`, `dropout=0`,
`teacher_forcing=1`, `scale=256`. `model` selecciona A/B/C/D/E.

`layers` afecta a todos los GRU; `dropout` se aplica a sus entradas y entre capas
cuando hay más de una. Todos los parámetros relevantes se guardan en checkpoint.
La cantidad de muestras, límites de puntos/strokes y temperatura pertenecen a la
inferencia; nunca provocan truncamiento silencioso del entrenamiento.

Con esos defaults: A/B tienen 116929 parámetros cada uno, C 195809, D 306144 y
E 411232. Cambiar capas o dimensiones modifica esos conteos; el scoreboard debe
leerlos de la instancia usada.

El método común es:

```python
metrics = model.batch_loss(
    sketches, prefix_counts, beta=0.1, free_bits=0.0,
    deterministic=False, teacher_forcing=None,
)
metrics["loss"].backward()
```

`teacher_forcing=None` usa el valor de configuración. El trainer decide el beta
efectivo según warmup/annealing; A conserva beta=1. `deterministic=True` usa las
medias de los posteriores; para desactivar también dropout se debe llamar
`model.eval()`.

## A: VAE secuencial

Un GRU bidireccional codifica todos los tokens y parametriza
`q(z|sketch)=Normal(mu, diag(exp(logvar)))`. El prior es `Normal(0,I)`.
La muestra reparametrizada inicializa todas las capas del decoder GRU causal y se
concatena a cada entrada. Su salida contiene una mezcla de K gaussianas bivariadas
correlacionadas y tres logits de pen state: continuar stroke, terminar stroke,
terminar sketch. El entrenamiento reconstruye todo el sketch con beta=1.

En completion, A puede consumir el prefijo como historia autoregresiva y seguir
desde él, pero no tiene un prior aprendido condicionado al prefijo. Esto lo
distingue explícitamente de C.

## B: control de KL

Comparte **exactamente la arquitectura** de A. El trainer puede variar beta,
warmup, annealing y free bits; la arquitectura no oculta otro cambio que confunda
esa comparación. El KL bruto se registra separado del KL utilizado en la loss.
El término de free bits aplica un suelo por dimensión latente, no un suelo
artificial sobre el KL reportado.

## C: VAE condicional de completion

Un GRU de contexto recibe solamente los strokes observados. De su representación
`c(prefix)` se obtiene un prior aprendido `p(z|prefix)`. El posterior de
entrenamiento `q(z|sketch,prefix)` combina el encoder bidireccional del sketch
completo con ese contexto. El decoder recibe z y c en su estado inicial y cada
paso. La pérdida de coordenadas y pen state solo cubre el **sufijo y su EOS**.

El prefix se consume como historia verdadera, incluso si se activa scheduled
sampling para la continuación. La selección de prefijos de TRAIN debe incluir
vacío, un stroke y aproximadamente 25/50/75%, para cubrir random generation y
entradas muy cortas. El modelo acepta también un prefijo completo: solo queda
predecir EOS, sin coordenadas objetivo.

En inferencia solo se calcula `p(z|prefix)`; no existe acceso al futuro. Un test
cambia strokes futuros manteniendo idéntico el prefijo y comprueba que el prior
permanece exactamente igual, mientras el posterior sí puede cambiar.

## D: VAE jerárquico por strokes

1. El encoder de un stroke recibe posiciones relativas a su inicio, deltas
   internos y su marcador de final. Es un GRU bidireccional y devuelve un
   embedding fijo; el anclaje absoluto se añade mediante una proyección.
2. Otro GRU bidireccional procesa los embeddings **en el orden de dibujo** y
   parametriza el posterior global. El prior global es `Normal(0,I)`.
3. Un GRU causal de strokes, condicionado al z global, observa únicamente los
   embeddings anteriores. Predice fin de sketch, un anchor **absoluto** mediante
   MDN, y el prior de un latent local del siguiente stroke.
4. El posterior local recibe ese estado causal y el embedding del stroke real.
   Su KL se mide contra el prior local correspondiente.
5. Un GRU de puntos condicionado al estado de stroke, latent local y anchor
   predice los desplazamientos internos mediante MDN y un fin binario de stroke.

El primer punto es el anchor. Un stroke de N puntos genera N-1 desplazamientos
más una decisión final sin coordenada. Por tanto un stroke de longitud 1 está
bien definido y no se duplica su anchor. El fin de sketch se predice en una
posición adicional después de todos los strokes.

La probabilidad del anchor no penaliza su distancia al stroke anterior. No hay
smoothness entre strokes. Los latentes locales y globales son variables de VAE,
no embeddings deterministas que solo se denominen latentes.

## E: VAE jerárquico condicional

Conserva la jerarquía de D y añade contexto de prefijo y prior global aprendido
`p(z|prefix)`. El encoder de contexto lee solo los embeddings observados; cada
embedding se calcula independientemente sobre **un único stroke**, por lo que no
puede transportar información de strokes futuros. Para prefijo vacío se usa un
contexto cero y el prior correspondiente sigue siendo aprendido.

El posterior global sí conoce el sketch completo durante entrenamiento. Los
anchor NLL, trayectoria NLL, final de stroke, KL local y final de sketch solo
se supervisan desde la primera posición no observada. Tanto z global como los
latentes locales tienen efecto sobre la generación. La jerarquía por sí misma
no garantiza superioridad visual: requiere comparación posterior.

## Teacher forcing y causalidad

Con ratio 1 se usa una llamada recurrente batched; con un ratio menor se usa
scheduled sampling por paso, eligiendo historia real o una predicción detached
(media del componente MDN modal y pen state modal). El ratio controla todos los
tokens de A/B/C y los **puntos dentro del stroke** de D/E. En D/E los embeddings
de strokes anteriores continúan con teacher forcing durante entrenamiento;
esta implementación no pretende corregir toda la exposición autoregresiva entre
strokes. En inferencia sí se recodifica cada stroke generado antes del siguiente.

En `eval()` se fuerza teacher forcing para que las métricas de likelihood evalúen
la historia observada. No hay scheduled sampling en validación. Todos los
decoders son unidireccionales; los tests fijan z y contexto, alteran únicamente
entradas futuras y verifican salidas pasadas idénticas. Que el posterior sea
bidireccional no viola la causalidad del decoder generativo.

## Losses, ELBO y unidades

La NLL usa log-sum-exp, log-softmax y gaussianas correlacionadas. Los parámetros
de distribución se estabilizan con `log_sigma` entre -7 y 3, `rho=0.95*tanh(raw)`
y logvar latente entre -10 y 6. Estas cotas son parametrización numérica; **no
recortan coordenadas generadas**. El KL normal-normal se calcula analíticamente.

La loss optimizada, pensada para tamaños variables, es:

```text
reconstruction = media NLL por coordenada + media CE por evento pen/stop
total_loss = reconstruction + beta_effective * KL_objective + auxiliary_loss
```

En D/E las coordenadas incluyen anchors y deltas internos; los eventos CE incluyen
finales de punto y de sketch. KL_objective suma KL global medio por sketch y KL
local medio por stroke objetivo, con free bits si está configurado. Los valores
brutos se registran como `kl`, `global_kl` y `stroke_kl` según corresponda.

`reconstruction_loss`, `KL_loss`, `auxiliary_loss`, `total_loss`,
`beta_effective`, `coordinate_nll`, `pen_ce`, `pen_accuracy` y conteos de eventos
están disponibles por separado. `auxiliary_loss=0` es explícito: no se han
introducido penalizaciones de curvatura o de escala sin evidencia. La validación
geométrica independiente se encarga de detectar artefactos, aprender límites de
TRAIN y permitir corrección/rechazo auditable.

También se registra `negative_elbo` como **suma** de NLL+CE+KL por sketch, con los
KL locales sumados antes de promediar sketches, y `elbo=-negative_elbo`. Su escala
es diferente del objective normalizado. `joint_reconstruction_per_sketch` y
`joint_kl_per_sketch` permiten inspeccionar esa distinción. Es un estimador MC
estándar del ELBO solo con posterior muestreado e historia real. Con medias
posteriores o scheduled sampling es un diagnóstico, indicado mediante
`elbo_is_mc_estimate=0`.

Las densidades continuas pueden tener NLL negativa. La suma de NLL de A/B/C y
D/E no es una puntuación universal: difieren la factorización, los estados y los
espacios coordenados. C/E además puntúan sufijos. Las comparaciones deben informar
familia, longitud observada, unidades y máscaras, junto con geometría y visuales.

## API de generación y checkpoints

`generation.py` ofrece una fachada común para UI/evaluación:

```python
model = load_model("runs/.../best.pt", device="cpu")
posterior = encode(model, strokes)  # dict de np.ndarray(latent_dim,): mu/logvar
z = sample_latent(model, prefix=prefix, seed=42)
result, info = sample(model, prefix, z=z, decoder_seed=9, return_info=True)
results = complete_sketch(model, prefix, n_samples=4)
random_results = generate_random(model, n_samples=4)
svg = render(result, prefix_count=len(prefix))
```

`encode` usa contexto vacío en C/E para inspección posterior de sketches completos;
`sample_latent` usa el prior condicionado al prefijo. `sample(z=posterior['mu'])`
permite reconstrucción autoregresiva posterior sin copiar el objetivo. La función
lazy `sample_multiple` delega al módulo de orquestación geométrica y ranking.

Los checkpoints de inferencia contienen `model_config` y `model_state`. Se cargan
con `torch.load(weights_only=True)` y `load_state_dict(strict=True)`; el módulo
de checkpointing del trainer añade optimizer, scheduler, RNG, representación y
normalización para resume. Cambiar arquitectura requiere warm start explícito,
no se ignoran silenciosamente parámetros incompatibles.

## Prefix exacto, aleatoriedad y terminación

El resultado empieza por copias independientes de las matrices originales del
prefijo, conservando exactamente sus valores, dtype, orden y límites. El prefijo
jamás se reconstruye acumulando deltas, recorta, reescala ni suaviza. Los límites
de inferencia cuentan exclusivamente los puntos/strokes de la continuación.

Existen tres fuentes independientes de aleatoriedad: prior global (`seed`),
latentes locales (`seed+2000003`) y decoder (`decoder_seed`, por defecto
`seed+1000003`). Un `torch.Generator` CPU independiente evita alterar el RNG global
del entrenamiento. Variar seed conservando decoder_seed prueba diversidad
latente; en D/E incluye global y local. Para aislar solo z global se pasan vectores
z distintos con el mismo seed y decoder_seed. `complete_sketch` conserva fijo el
ruido del decoder entre muestras. Con z explícito constante, A/B/C producen las
mismas salidas bajo el mismo ruido; no se simula diversidad añadiendo jitter.

Temperatura 0 selecciona componentes/categorías modales y sus medias; no elimina
el muestreo de los latentes. Temperaturas positivas escalan logits y desviaciones.
Ni temperatura ni sampling arreglan por sí mismos un modelo sin entrenar.

La metadata distingue `termination=eos|max_points|max_strokes|nonfinite_prediction`,
`ended_by_eos`, `capped`, `partial_stroke`, números de puntos y strokes generados,
semillas y controles usados. No se inventa EOS al llegar a un cap. Un fallo no
finito interrumpe la generación y conserva solo puntos finitos ya obtenidos; el
ranking debe rechazar ese estado según sus reglas. La salida geométrica fuera de
canvas permanece sin clamp, para que la validación independiente pueda detectarla.

## Validación y límites conocidos

Tests de modelos/generación: forward/backward finito A–E, una y dos capas,
dropout, scheduled sampling, padding, máscaras de sufijo, EOS sin coordenada,
strokes singleton, 211 strokes, MDN extremo, KL/free bits, reparametrización,
causalidad, prior de prefijo, latente con efecto geométrico bajo decoder fijo,
semillas reproducibles, prefijo bit a bit, límites honestos, ausencia de clipping,
guardado/carga y generación de todas las variantes.

Un smoke exitoso demuestra ejecución, no criaturas plausibles. Persisten exposición
autoregresiva, posible posterior collapse, dependencia del orden, y riesgo de
strokes degenerados o EOS prematuro. Se deben medir KL, sensibilidad latente,
diversidad válida, aceptación geométrica, distribuciones de longitud y grids antes
de seleccionar un modelo o recomendar un entrenamiento largo.
