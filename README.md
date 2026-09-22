# Lab 2 — VAE for Vector Sketch Generation and Completion

Proyecto reproducible para modelar sketches vectoriales como secuencias ordenadas de strokes. El laboratorio estudia dos tareas: **generación aleatoria** desde una variable latente y **sketch completion**, donde varios valores de `z` producen continuaciones distintas para un mismo prefix.

![Interfaz local de Sketch Lab](docs/assets/interface.png)

## Objetivo

Cada sketch conserva estructura vectorial y orden de dibujo. No se rasteriza para entrenar: un sketch es una secuencia de strokes y cada stroke es una secuencia ordenada de puntos 2D. El objetivo experimental actual es entender por qué una reconstrucción teacher-forced cada vez mejor no se traduce necesariamente en samples free-running estables.

Las dos operaciones principales son:

1. **Random generation:** generar un sketch completo desde `z`.
2. **Sketch completion:** conservar exactamente los strokes observados y generar múltiples suffixes con distintos `z`.

## Dataset

El dataset autorizado ya está incluido en [`data/raw/`](data/raw/):

- `train.pkl`: 8.187 sketches, 17.331.249 bytes.
- `val.pkl`: 910 sketches, 1.891.794 bytes.

Cada archivo contiene una lista de sketches. Un sketch es una secuencia ordenada de strokes; cada stroke es un array `float32 (N, 2)` de puntos absolutos y longitud variable. Los límites de stroke representan pen-up. La codificación stroke-5 reversible añade estados de dibujo, fin de stroke y EOS sin alterar los originales. En completion, el prefix se copia desde sus coordenadas originales y se comprueba punto por punto: el modelo solo genera el suffix.

La estructura y las estadísticas verificadas están documentadas en [docs/dataset.md](docs/dataset.md).

## Modelos implementados

| Modelo | Descripción |
|---|---|
| **A — Sketch-RNN VAE** | Baseline secuencial con un latente global y decoder autoregresivo MDN. |
| **B — KL-controlled / Beta-VAE** | Variante de A con warm-up, beta reducido y free bits para conservar información latente. |
| **C — Conditional Completion VAE** | VAE secuencial entrenado también con prefixes parciales para modelar continuaciones condicionadas. |
| **D — Hierarchical Stroke-VAE** | Representación jerárquica con latente global y latentes locales por stroke. |
| **E — Hierarchical Conditional VAE** | Combina jerarquía por stroke y entrenamiento condicionado por prefixes. |

Los detalles de representación, losses y límites de comparación están en [docs/architectures.md](docs/architectures.md).

## Estado actual

El repositorio incluye representación reversible, pipeline seguro de datos, modelos A/B/C/D/E, training, evaluación, checkpoints con resume exacto, random generation, sketch completion, preservación exacta del prefix, validación geométrica, postprocesado conservador, reranking por diversidad, interfaz local y tests automatizados.

La verificación de publicación ejecuta **142 tests**. Los checkpoints y resultados experimentales se excluyen del repositorio; pueden reproducirse con los comandos siguientes y se guardan bajo `runs/`, que Git ignora.

## Hallazgos actuales

**Round 1** comparó A/B/C/D/E con el mismo presupuesto corto y evaluación homogénea. Después se hizo una revisión visual humana con casos VAL, prefixes y semillas fijos. **Round 2** reanudó exactamente C y E hasta 800 updates acumulados.

La reconstruction de training/validation mejoró, pero la generación autoregresiva free-running no mejoró proporcionalmente. En Round 2 aumentaron la sobre-generación y la frecuencia de llegar al límite de 384 puntos; EOS/termination empeoró y aún aparecen trayectorias fuera del canvas y scribbles.

Esto **no demuestra exposure bias**. Las hipótesis abiertas incluyen calibración de EOS, sampling y temperatura, comportamiento del MDN, drift autoregresivo acumulado y la diferencia entre inputs teacher-forced y estados producidos por el propio decoder. [docs/EXPERIMENTAL_FINDINGS.md](docs/EXPERIMENTAL_FINDINGS.md) separa lo observado de lo que todavía no sabemos.

![Benchmark visual Round 1](docs/assets/round1_visual_benchmark.png)

![Comparación C frente a E en Round 2](docs/assets/round2_C_vs_E.png)

## Interfaz

En la interfaz, **azul** representa el prefix proporcionado y **naranja** la continuación generada. Primero entrena un modelo o coloca un checkpoint compatible dentro de `runs/`; los checkpoints no se publican en Git.

```powershell
.\scripts\run_demo.ps1
```

Después abre [http://127.0.0.1:8000](http://127.0.0.1:8000). El comando equivalente es:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.server:app --host 127.0.0.1 --port 8000
```

## Setup en Windows

Requiere Python 3.11 o 3.12. Para el entorno CUDA 12.6 usado en el laboratorio:

```powershell
git clone https://github.com/salvachu/lab_generativos_2.git
cd lab_generativos_2
.\scripts\setup.ps1 -Python python
```

Para reproducir las versiones bloqueadas manualmente:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

`requirements-lock.txt` fija PyTorch CUDA 12.6. En una máquina sin NVIDIA instala primero la distribución de PyTorch adecuada y después `pip install -e ".[dev]"`. Consulta [docs/QUICK_START.md](docs/QUICK_START.md) para el flujo corto.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Training

Cada comando exige un directorio de salida nuevo:

```powershell
.\.venv\Scripts\python.exe -m scripts.train --config configs\A.json --models A --output runs\comparison_A
.\.venv\Scripts\python.exe -m scripts.train --config configs\B.json --models B --output runs\comparison_B
.\.venv\Scripts\python.exe -m scripts.train --config configs\C.json --models C --output runs\comparison_C
.\.venv\Scripts\python.exe -m scripts.train --config configs\D.json --models D --output runs\comparison_D
.\.venv\Scripts\python.exe -m scripts.train --config configs\E.json --models E --output runs\comparison_E
```

Resume exacto conserva optimizer, RNG, sampler y schedule:

```powershell
.\.venv\Scripts\python.exe -m scripts.train --config configs\round2_C.json --models C --output runs\continued_C --resume runs\comparison_C\C\last.pt
```

## Evaluation

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate runs\comparison_C\C\best.pt --output runs\evaluation_C --count 8 --samples 3 --seed 2026
```

## Generation

Generación aleatoria sin prefix:

```powershell
.\.venv\Scripts\python.exe -m scripts.generate runs\comparison_C\C\best.pt --output runs\random_C --candidates 20 --top-k 6 --seed 42
```

![Problema actual en generación free-running](docs/assets/free_running_generation_problem.png)

## Completion

`configs/prefix_example.json` contiene un prefix mínimo de ejemplo:

```powershell
.\.venv\Scripts\python.exe -m scripts.generate runs\comparison_C\C\best.pt --prefix configs\prefix_example.json --output runs\completion_C --candidates 20 --top-k 6 --seed 42
```

![Ejemplo de completion](docs/assets/completion_example.png)

## Próximos pasos

1. Comparar teacher-forced y free-running con un diagnóstico controlado.
2. Analizar EOS y termination por paso y por condición.
3. Hacer un sweep pequeño de sampling y temperatura.
4. Identificar la causa principal sin asumirla de antemano.
5. Aplicar un cambio mínimo y aislado.
6. Repetir el benchmark visual con los mismos casos.
7. Hacer entrenamiento largo solo después de estabilizar generation.

## Documentación

- [Inicio rápido](docs/QUICK_START.md)
- [Hallazgos experimentales](docs/EXPERIMENTAL_FINDINGS.md)
- [Dataset y representación](docs/dataset.md)
- [Arquitecturas](docs/architectures.md)
- [Geometría, validación y ranking](docs/geometry.md)
- [Informe técnico](docs/INFORME_TECNICO.md)
- [Fuentes primarias](docs/research.md)
