# Referencias y alcance del Laboratorio 2

Revisión de fuentes primarias: 21 de septiembre de 2026. Este documento separa el enunciado, las instrucciones del usuario y las ideas tomadas de artículos. Las propuestas de adaptación son decisiones para este proyecto; no son resultados experimentales publicados sobre este dataset.

## Enunciado local completo

Se leyó íntegramente `references/Laboratorio 2_ Sketch.pdf`: **una página**, incluyendo las dos ilustraciones, y se verificó su render. Copias de revisión: `runs/reference_review/enunciado.txt` y `runs/reference_review/enunciado_page_1.png`. No se modificó el original.

El PDF establece:

- Criaturas imaginarias como secuencias **ordenadas** de trazos, cada uno con un número variable de coordenadas bidimensionales.
- Un generador basado en variables latentes que aprenda la distribución y genere criaturas nuevas.
- Completion a partir de trazos iniciales, con **diferentes continuaciones válidas mediante diferentes muestras latentes**.
- **9097** bocetos válidos: **8187 train y 910 validation**.
- Información adicional de partes, etapa y descripción textual, expresamente **opcional** para la tarea principal.

El PDF denomina los archivos `train.pkl.gz` y `val.pkl.gz`; el material local proporcionado usa `data/raw/train.pkl` y `data/raw/val.pkl`. Esta diferencia de nombre no justifica asumir compresión ni cambiar los originales: debe comprobarse el contenido de los archivos al auditarlos. El PDF no define keys de Python, unidades, canvas, flags del lápiz ni coordenadas relativas; esas propiedades deben obtenerse exclusivamente de la inspección de datos.

La obligación de usar **VAEs o variantes**, comparar A/B/C/D/E, conservar exactamente el prefijo, implementar una interfaz, ejecutar tests y limitar el entrenamiento proviene de la solicitud del usuario y de su indicación del profesor. No está expresada con ese detalle en el PDF. Tampoco prescribe un framework, tamaños latentes, número de mezclas, resolución o presupuesto de GPU.

## Referencias de investigación

### 1. Sketch-RNN: referencia principal para A/B

David Ha y Douglas Eck, *A Neural Representation of Sketch Drawings* (2017). El modelo es un VAE secuencia a secuencia: encoder bidireccional, posterior gaussiano, decoder recurrente autoregresivo con mezcla de gaussianas bivariadas para desplazamientos y distribución categórica para estados del lápiz. Su suplemento incluye escalado de desplazamientos, annealing de KL y un umbral mínimo de KL. [Artículo primario](https://arxiv.org/pdf/1704.03477), [implementación oficial](https://github.com/magenta/magenta/tree/main/magenta/models/sketch_rnn).

**Adaptación:** baseline compacto reproducible con NLL de coordenadas, CE y KL; A/B será una comparación controlada de estrategias de KL dentro de esta familia. B no se debe describir como una técnica ausente del artículo original. El estado del lápiz debe documentarse sin ambigüedad: el punto final de un stroke se conserva y el siguiente comienzo no dibuja una línea entre strokes. El muestreo desde un prior latente es distinto del decoder sin encoder descrito también en el artículo.

### 2. CoSE: embeddings por stroke y relaciones

Emre Aksan, Thomas Deselaers, Andrea Tagliasacchi y Otmar Hilliges, *CoSE: Compositional Stroke Embeddings* (NeurIPS 2020). Codifica strokes de longitud variable en embeddings fijos y usa un modelo relacional para predecir otros strokes. El trabajo trata el dibujo como colección **sin orden**; la separación entre apariencia local y estructura global facilita auto-completion. [Artículo primario](https://arxiv.org/pdf/2006.09930), [página de los autores](https://eth-ait.github.io/cose/).

**Adaptación:** reutilizar la idea de codificación individual y predicción de posición/forma para D/E, pero mantener una secuencia de strokes con orden explícito. El orden es información obligatoria en este laboratorio. No se puede afirmar que una adaptación secuencial pequeña reproduce CoSE, ni que adoptar sus embeddings por sí solo cumple el requisito de VAE: debe existir una distribución latente regularizada y muestreable.

### 3. DeepSVG: jerarquía de estructuras y comandos

Alexandre Carlier, Martin Danelljan, Alexandre Alahi y Radu Timofte, *DeepSVG: A Hierarchical Generative Network for Vector Graphics Animation* (NeurIPS 2020). Es un VAE jerárquico para gráficos SVG que separa paths/shapes de sus comandos y predice estructuras de forma no autoregresiva. [Artículo primario](https://proceedings.neurips.cc/paper/2020/file/bcf9d6bd14a2095866ce8c950b702341-Paper.pdf), [código oficial](https://github.com/alexandre01/deepsvg).

**Adaptación:** jerarquía sketch → stroke → puntos. No introducir curvas Bézier, rellenos, vocabulario SVG ni orden de shapes del artículo si el dataset solo tiene polilíneas. Tampoco hace falta copiar su decoder no autoregresivo; la tarea de completar una secuencia ordenada puede justificar un decoder secuencial. La inspiración es estructural, no una equivalencia de datos ni una reproducción de sus resultados.

### 4. Sketch-HARP: qué trazo, dónde y cómo

Sicong Zang, Shuhui Gao y Zhijun Fang, *Generating Sketches in a Hierarchical Auto-Regressive Process for Flexible Sketch Drawing Manipulation at Stroke-Level* (preprint 2025). Divide la generación en embedding del stroke, posición inicial y trayectoria; incorpora información de strokes y posiciones anteriores. Su objetivo publicado combina reconstrucción secuencial, posición, parada, embeddings y reconstrucción de imagen; no incorpora un término KL de VAE en esa función. [Artículo primario](https://arxiv.org/html/2511.07889v1), [código de autores](https://github.com/SCZang/Sketch-HARP).

**Adaptación:** separar anclaje inicial y desplazamientos dentro del stroke, con parada de stroke y de sketch. D/E necesitan además el posterior, prior y regularización variacional exigidos aquí. No trasladar sus límites de longitud o tamaños de red sin examinar el dataset local. Un salto de anclaje entre strokes no debe tratarse como una discontinuidad dentro de un stroke.

### 5. AI-Sketcher: contexto espacial auxiliar

Nan Cao, Xin Yan, Yang Shi y Chaoran Chen, *AI-Sketcher: A Deep Generative Model for Producing High-Quality Sketches* (AAAI 2019). Extiende la familia Sketch-RNN con información espacial obtenida por CNN, información de clase y una capa de influencia sobre el decoder. [Publicación primaria de AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/4103), [PDF de los autores](https://idvxlab.com/papers/2019AAAI_Sketcher_Cao.pdf).

**Adaptación experimental posible:** encoder de raster auxiliar para la colocación de partes si los resultados muestran ese problema. No sustituir la representación vectorial ni introducir clases que el dataset no tenga. En completion solo se rasterizaría el prefijo disponible para el condicionamiento; rasterizar el futuro durante inferencia sería una fuga de información. La rama espacial requiere una ablation independiente para justificar coste y mejora y no es un requisito del PDF.

## Decisiones que deben comprobarse con experimentos propios

- Una implementación pequeña es una adaptación conceptual, no una reproducción fiel de los cinco artículos.
- No declarar superioridad de D/E por su jerarquía: medir tiempo, KL, validación, diversidad latente y calidad de los renders con el mismo presupuesto.
- Comparar NLL solo cuando representación, unidades, normalización y componentes de loss coincidan. Posición inicial y desplazamientos internos pueden tener escalas diferentes.
- El prefijo devuelto debe copiarse exactamente; el modelo genera el sufijo. La comparación de diversidad debe medir el sufijo y mantener fijo el prefijo.
- Cambiar semillas del decoder no demuestra que se utilice el latente: mantener fijo el ruido del decoder y variar solo z en una evaluación adicional.
- Registrar si se llega a EOS de forma natural o si un límite de seguridad corta el resultado. Un resultado truncado no demuestra que el modelo aprendió cuándo terminar.
- Las métricas contra el único sufijo de validación son diagnósticos, no pruebas de invalidez de otras continuaciones posibles. Combinar geometría, límites del canvas, distribuciones, diversidad y revisión visual.

## Referencia visual de la demo

El video local se utiliza exclusivamente para observar la interacción y el aspecto aproximado de la demo. Se extrajeron y revisaron ocho frames repartidos entre 0 y 22.37 segundos. El archivo dura 22.6 segundos, tiene 226 frames, 10 fps y resolución 728 × 360. Evidencias: `runs/reference_review/video_contact_sheet.png`, `video_frame_00.png` a `video_frame_07.png` y `video_metadata.json`. Extracción reproducible: `.venv/Scripts/python.exe runs/reference_review/extract_video.py`.

Se observa un canvas a la izquierda y nueve alternativas en cuadrícula 3 × 3 a la derecha. Los strokes introducidos son verdes; las continuaciones son violetas/grises. La cuadrícula cambia conforme se añaden un círculo, cuerpo, alas y patas. Incluso con un único trazo inicial se muestran múltiples propuestas. Se ven controles `clear drawing`, un selector con `mosquito`, `random` y `predict`. La experiencia relevante es dibujar pocos strokes, comparar alternativas y añadir información gradualmente. El selector de mosquito no implica que el dataset local tenga clases ni que haya que entrenar un clasificador.

No se deducen del video la arquitectura del modelo, el dataset de entrenamiento, el hardware, el criterio de pérdida ni obligaciones técnicas adicionales. Los frames no prueban preservación numérica exacta del prefijo; eso debe comprobarse con tests en este proyecto.

La ilustración de la propia página del PDF ya muestra un área de dibujo, una cuadrícula de alternativas, prefijo en verde, continuación en otro color y controles de limpiar, random y predict. Esto apoya una interfaz sencilla con varias propuestas comparables; guardar, regenerar y cambiar modelos son requisitos explícitos del usuario, aunque no aparezcan todos en esa captura.
