# Experimental findings

## Teacher forcing y generación free-running

Durante training, **teacher forcing** entrega al decoder el punto y estado reales del paso anterior. Esto permite medir qué tan bien predice el siguiente token cuando la historia de entrada es correcta.

En **free-running generation**, el decoder recibe en cada paso su propia muestra anterior. Una desviación pequeña puede cambiar la siguiente distribución MDN, la decisión de terminar un stroke y la probabilidad de EOS. Esas desviaciones pueden acumularse durante una secuencia larga.

Por eso una reconstruction loss mejor no garantiza samples mejores. La loss teacher-forced mide predicción local condicionada a una historia real; el uso final exige mantener una trayectoria estable bajo la historia producida por el propio modelo. Además intervienen la calibración de EOS, la temperatura, la elección de componentes MDN y el criterio de muestreo.

## Protocolo experimental

Round 1 entrenó A/B/C/D/E durante 160 updates con la misma selección de 1.024 sketches TRAIN, 96 VAL, batch 16 y seed 2026. La evaluación automática se complementó con un benchmark visual fijo:

- ocho sketches VAL elegidos sin consultar resultados de modelos;
- mismos IDs y prefixes para todos los modelos;
- prefixes de 1 stroke, 2 strokes, 25% y 50%;
- tres valores latentes por caso;
- temperatura 0,6, máximo 384 puntos y canvas fijo `[-4, 508]`;
- prefix azul, continuación naranja y GT gris/negro.

La revisión humana eligió C y E para Round 2. La decisión fue deliberadamente distinta del ranking automático: C ofrecía un contraste útil por su objetivo de completion y algunas continuaciones visualmente contenidas; E tenía mejor contención geométrica agregada pero sobre-generaba y producía scribbles.

Round 2 reanudó exactamente `last.pt` de ambos modelos. Conservó optimizer, RNG, sampler, schedule, datos y normalización, y añadió 640 updates para llegar a 800 acumulados. Se repitieron los ocho casos y se añadió el prefix 75%.

## Round 1

Los modelos conservaron el prefix exactamente, pero la calidad free-running fue limitada. A/B/C generaron muchas trayectorias extensas fuera del canvas. D/E mantuvieron más puntos dentro del área útil, aunque con secuencias densas, scribbles y terminación insuficiente.

En el benchmark visual fijo de Round 1:

| Modelo | Mediana puntos | EOS | Cap 384 | Completion válida | Puntos dentro del canvas, media |
|---|---:|---:|---:|---:|---:|
| C | 339 | 57,7% | 42,3% | 2,1% | 19,3% |
| E | 384 | 36,5% | 63,5% | 9,4% | 85,7% |

Estas métricas son diagnósticas. Una continuation puede ser geométricamente válida y no parecer una criatura, o ser semánticamente razonable y diferir del único suffix GT.

## Round 2

La reconstruction de validación mejoró en ambos modelos:

- C: `-0,9263 → -1,7459`.
- E: `-0,9755 → -2,0823`.

El KL global siguió activo y no apareció la alerta implementada de posterior collapse. En E también permaneció activo el KL local por stroke. Sin embargo, las métricas free-running sobre los mismos casos y prefixes comunes evolucionaron así:

| Modelo | Mediana puntos | EOS | Cap 384 | Completion válida | Puntos dentro del canvas, media |
|---|---:|---:|---:|---:|---:|
| C Round 1 | 339 | 57,7% | 42,3% | 2,1% | 19,3% |
| C Round 2 | 384 | 27,9% | 72,1% | 3,1% | 26,0% |
| E Round 1 | 384 | 36,5% | 63,5% | 9,4% | 85,7% |
| E Round 2 | 384 | 16,3% | 82,7% | 5,2% | 56,0% |

C mejoró modestamente la fracción media dentro del canvas y la tasa válida estricta, pero produjo secuencias más largas, alcanzó más el cap y redujo EOS. E también alcanzó el cap con mayor frecuencia, redujo EOS y perdió parte de su contención espacial. Visualmente persisten spaghetti, líneas fuera de escala y continuaciones desconectadas.

![Round 2: C frente a E](assets/round2_C_vs_E.png)

## EOS y sequence cap

`sequence_cap_reached` significa que la generación llegó al máximo configurado de 384 puntos sin emitir EOS. El cap evita ejecuciones indefinidas; no es una terminación aprendida. Una tasa de cap alta puede reflejar EOS mal calibrado, drift hacia estados poco conocidos o un sampling que mantiene demasiado baja la probabilidad de parada.

El caso opuesto también es un fallo: emitir EOS inmediatamente produce un suffix vacío. Por eso no basta con maximizar EOS; interesa una terminación natural después de una continuación organizada.

## Diferencias observadas entre C y E

- C usa un latente global y entrenamiento condicionado por prefixes. Round 2 redujo algunas distancias geométricas agregadas, pero aumentó mucho la longitud y no controló la salida del canvas.
- E añade estructura jerárquica y latentes locales por stroke. Conservó señales latentes activas, pero la mayor cantidad de entrenamiento no corrigió la terminación y redujo su contención en el benchmark fijo.
- En ambos modelos, diferentes `z` cambian la salida. Gran diversidad no implica calidad: parte de la distancia entre samples procede de geometría corrupta.

## Qué sabemos

- Los datos, prefixes y seeds del benchmark visual son reproducibles.
- El prefix se conserva exactamente.
- Las losses y gradientes permanecieron finitos.
- C y E llegaron a 800 updates acumulados mediante resume exacto.
- La reconstruction mejoró mientras EOS, cap y varios indicadores visuales no mejoraron de forma proporcional.
- El postprocesado conservador no repara los fallos graves de los ejemplos representativos.

## Qué todavía no sabemos

- No hemos demostrado que exposure bias sea la causa principal.
- No sabemos si domina la calibración de EOS, el sampling MDN, la temperatura o el drift acumulado.
- No sabemos si un cambio mínimo de curriculum o de objective de termination resolverá el problema.
- No hay evidencia para justificar entrenamiento largo antes de aislar la causa.
- Chamfer, Hausdorff y reconstruction no sustituyen una revisión visual de estructura y semántica.

El siguiente experimento debe aislar teacher-forced frente a free-running, inspeccionar probabilidades de EOS a lo largo de la secuencia y hacer un sweep pequeño de sampling/temperatura antes de modificar la arquitectura.
