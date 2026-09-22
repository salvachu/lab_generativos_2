"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const canvas = $("drawing");
  const state = {strokes: [], current: null, pointer: null, busy: false, models: [], samples: [], selectedIndex: null, prefixCount: 0, lastRequest: null, lastResult: null, canvasSize: 512};
  const MAX_POINTS = 4096;
  const MAX_STROKES = 96;
  const COLORS = {prefix: "#2563eb", generated: "#db684b"};

  function message(text, type = "") {
    $("status").textContent = text;
    $("status").className = `status ${type}`;
  }

  function updateControls() {
    const available = Boolean($("model").value);
    $("complete").disabled = state.busy || !available || !state.strokes.length;
    $("random").disabled = state.busy || !available;
    $("regenerate").disabled = state.busy || !state.lastRequest || !available;
    $("clear").disabled = state.busy || !state.strokes.length;
    $("undo").disabled = state.busy || !state.strokes.length;
    for (const id of ["model", "refresh", "temperature", "sample-count", "candidate-budget", "generation-mode"]) $(id).disabled = state.busy;
    const rawMode = $("generation-mode").value === "raw";
    for (const id of ["postprocess", "rerank"]) $(id).disabled = state.busy || rawMode;
    $("raw-notice").hidden = !rawMode;
    for (const id of ["export-svg", "export-json"]) $(id).disabled = state.busy || state.selectedIndex === null;
    $("selection-label").textContent = state.selectedIndex === null ? "Selecciona un resultado" : `Variante ${state.selectedIndex + 1} seleccionada`;
    document.body.classList.toggle("busy", state.busy);
    $("stroke-count").textContent = `${state.strokes.length} ${state.strokes.length === 1 ? "stroke" : "strokes"}`;
    $("point-count").textContent = `${state.strokes.reduce((n, stroke) => n + stroke.length, 0)} puntos · ${state.canvasSize} × ${state.canvasSize}`;
    $("canvas-hint").hidden = state.strokes.length > 0 || Boolean(state.current);
  }

  function drawSketch(target, strokes, prefixCount, order = false) {
    const ctx = target.getContext("2d");
    ctx.clearRect(0, 0, target.width, target.height);
    const scale = target.width / state.canvasSize;
    ctx.save();
    ctx.scale(scale, scale);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineWidth = 2.4;
    strokes.forEach((stroke, index) => {
      if (!stroke.length) return;
      ctx.strokeStyle = ctx.fillStyle = index < prefixCount ? COLORS.prefix : COLORS.generated;
      if (stroke.length === 1) {
        ctx.beginPath();
        ctx.arc(stroke[0][0], stroke[0][1], 1.4, 0, 2 * Math.PI);
        ctx.fill();
      } else {
        ctx.beginPath();
        ctx.moveTo(...stroke[0]);
        for (const point of stroke.slice(1)) ctx.lineTo(...point);
        ctx.stroke();
      }
      if (order) {
        const [x, y] = stroke[0];
        ctx.font = "13px Segoe UI, sans-serif";
        ctx.lineWidth = 3;
        const color = ctx.strokeStyle;
        ctx.strokeStyle = "white";
        ctx.strokeText(String(index + 1), x + 5, y - 5);
        ctx.fillText(String(index + 1), x + 5, y - 5);
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.4;
      }
    });
    ctx.restore();
  }

  function redrawInput() {
    const strokes = state.current ? [...state.strokes, state.current] : state.strokes;
    drawSketch(canvas, strokes, strokes.length, $("stroke-order").checked);
    updateControls();
  }

  function pointFromEvent(event) {
    const rect = canvas.getBoundingClientRect();
    return [
      Math.min(state.canvasSize, Math.max(0, (event.clientX - rect.left) / rect.width * state.canvasSize)),
      Math.min(state.canvasSize, Math.max(0, (event.clientY - rect.top) / rect.height * state.canvasSize)),
    ].map((value) => Math.round(value * 100) / 100);
  }

  canvas.addEventListener("pointerdown", (event) => {
    if (state.busy || state.pointer !== null || (event.pointerType === "mouse" && event.button !== 0)) return;
    if (state.strokes.length >= MAX_STROKES || state.strokes.reduce((n, s) => n + s.length, 0) >= MAX_POINTS) {
      message("Has alcanzado el límite de dibujo. Deshaz o limpia algunos strokes.", "error");
      return;
    }
    event.preventDefault();
    state.pointer = event.pointerId;
    state.current = [pointFromEvent(event)];
    canvas.setPointerCapture(event.pointerId);
    redrawInput();
  });

  canvas.addEventListener("pointermove", (event) => {
    if (state.pointer !== event.pointerId || !state.current) return;
    const point = pointFromEvent(event);
    const last = state.current[state.current.length - 1];
    // Capture vector points directly. This only samples pointer events, and
    // never resamples a completed stroke or an existing completion prefix.
    if (Math.hypot(point[0] - last[0], point[1] - last[1]) < 1.8) return;
    if (state.strokes.reduce((n, s) => n + s.length, state.current.length) >= MAX_POINTS) return;
    state.current.push(point);
    redrawInput();
  });

  function finishStroke(event) {
    if (state.pointer !== event.pointerId || !state.current) return;
    state.strokes.push(state.current);
    state.current = null;
    state.pointer = null;
    if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
    redrawInput();
  }
  canvas.addEventListener("pointerup", finishStroke);
  canvas.addEventListener("pointercancel", finishStroke);
  canvas.addEventListener("lostpointercapture", finishStroke);

  function placeholders() {
    const container = $("results");
    container.replaceChildren();
    for (let index = 0; index < Number($("sample-count").value); index++) {
      const card = document.createElement("article");
      card.className = "sample sample-placeholder";
      const content = document.createElement("div");
      content.className = "placeholder-content";
      const icon = document.createElement("span");
      icon.className = "placeholder-icon";
      icon.textContent = ["α", "β", "γ", "δ", "ε", "ζ", "η", "θ", "ι"][index];
      const caption = document.createElement("span");
      caption.textContent = "Una posibilidad por descubrir";
      content.append(icon, caption);
      const footer = document.createElement("div");
      footer.className = "sample-footer";
      footer.textContent = `VARIANTE ${String(index + 1).padStart(2, "0")}`;
      card.append(content, footer);
      container.append(card);
    }
  }

  function download(content, mimeType, name) {
    const url = URL.createObjectURL(new Blob([content], {type: mimeType}));
    const link = document.createElement("a");
    link.href = url;
    link.download = name;
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function showSamples() {
    const container = $("results");
    container.replaceChildren();
    const useRaw = $("output-view").value === "raw";
    if (!state.samples.length) {
      const empty = document.createElement("div");
      empty.className = "empty-results";
      empty.textContent = "Ningún candidato superó los filtros de selección. Puedes regenerar o usar RAW para inspeccionar el comportamiento de un checkpoint tiny sin presentar sus outputs como válidos.";
      container.append(empty);
    }
    state.samples.forEach((sample, index) => {
      const strokes = useRaw ? sample.raw_strokes : sample.postprocessed_strokes;
      const card = document.createElement("article");
      card.className = `sample${state.selectedIndex === index ? " selected" : ""}`;
      const choose = () => { state.selectedIndex = index; showSamples(); updateControls(); };
      const previewButton = document.createElement("button");
      previewButton.className = "sample-preview";
      previewButton.setAttribute("aria-label", `Seleccionar variante ${index + 1}`);
      previewButton.setAttribute("aria-pressed", String(state.selectedIndex === index));
      previewButton.addEventListener("click", choose);
      const preview = document.createElement("canvas");
      preview.className = "sample-canvas";
      preview.width = preview.height = 700;
      preview.setAttribute("role", "img");
      preview.setAttribute("aria-label", `Variante ${index + 1}: ${strokes.length} strokes. Prefijo azul y continuación coral.`);
      drawSketch(preview, strokes, state.prefixCount, $("stroke-order").checked);
      previewButton.append(preview);
      const footer = document.createElement("div");
      footer.className = "sample-footer";
      const caption = document.createElement("span");
      caption.textContent = `VARIANTE ${String(index + 1).padStart(2, "0")}`;
      const selectButton = document.createElement("button");
      selectButton.className = "text-button";
      selectButton.textContent = state.selectedIndex === index ? "✓ Seleccionada" : "Seleccionar";
      selectButton.addEventListener("click", choose);
      footer.append(caption, selectButton);
      const validation = document.createElement("span");
      const isValid = sample.validation && sample.validation.valid === true;
      validation.className = `sample-validation${!state.lastResult.validated || !isValid ? " invalid" : ""}`;
      validation.textContent = !state.lastResult.validated ? "RAW · sin filtro de validez" : isValid ? "Validación geométrica superada" : "Advertencias geométricas";
      if (useRaw && state.lastResult.postprocess && state.lastResult.validated) validation.textContent = `${isValid ? "Postprocesado válido" : "Postprocesado con advertencias"} · vista RAW original`;
      validation.title = (sample.validation?.warnings || []).join(" · ");
      card.append(previewButton, footer, validation);
      container.append(card);
    });
  }

  function exportSelected(format) {
    if (state.selectedIndex === null) return;
    const sample = state.samples[state.selectedIndex];
    const useRaw = $("output-view").value === "raw";
    const name = `sketch-${state.lastResult.seed}-variante-${state.selectedIndex + 1}-${useRaw ? "raw" : "postprocessed"}.${format}`;
    if (format === "svg") download(useRaw ? sample.raw_svg : sample.postprocessed_svg, "image/svg+xml", name);
    else download(JSON.stringify({model: state.lastResult.model, seed: state.lastResult.seed, candidate_seed: sample.seed, candidate_id: sample.id, temperature: state.lastResult.temperature, prefix_count: state.prefixCount, canvas_size: state.canvasSize, viewed_output: useRaw ? "raw" : "postprocessed", raw_output: sample.raw_strokes, postprocessed_output: sample.postprocessed_strokes, validation: sample.validation, termination: sample.termination, report: state.lastResult.report}, null, 2), "application/json", name);
  }

  async function loadModels() {
    try {
      const response = await fetch("/api/models");
      if (!response.ok) throw new Error("No se pudo consultar la lista de modelos.");
      const payload = await response.json();
      const selected = $("model").value;
      state.models = payload.models;
      state.canvasSize = payload.canvas_size || 512;
      $("model").replaceChildren();
      if (!state.models.length) $("model").add(new Option("Sin checkpoints disponibles", ""));
      for (const model of state.models) $("model").add(new Option(model.label, model.id));
      if (state.models.some((model) => model.id === selected)) $("model").value = selected;
      $("no-models").hidden = state.models.length > 0;
      $("model-help").textContent = state.models.length ? `${state.models.length} checkpoint(s) disponible(s) · inferencia ${payload.device}` : "Variantes A–E disponibles en runs/.";
      redrawInput();
      updateControls();
    } catch (error) {
      message(`${error.message} Inicia el servidor local indicado en README.md.`, "error");
    }
  }

  function nextSeed() {
    return crypto.getRandomValues(new Uint32Array(1))[0];
  }

  async function generate(mode) {
    if (state.busy || !$("model").value) return;
    const nCandidates = Number($("candidate-budget").value);
    const topK = Number($("sample-count").value);
    if (!Number.isInteger(nCandidates) || nCandidates < topK || nCandidates > 64) {
      message(`El presupuesto debe ser un entero entre ${topK} y 64 candidatos.`, "error");
      return;
    }
    const validated = $("generation-mode").value !== "raw";
    const options = {model: $("model").value, n_candidates: nCandidates, top_k: topK, temperature: Number($("temperature").value), validate: validated, postprocess: validated && $("postprocess").checked, rerank: validated && $("rerank").checked, seed: nextSeed()};
    let request;
    if (mode === "regenerate" && state.lastRequest) {
      request = {...state.lastRequest, ...options};
    } else {
      request = {
        ...options,
        prefix: mode === "complete" ? structuredClone(state.strokes) : [],
        max_points: 384,
        max_strokes: 64,
      };
    }
    if (mode === "complete" && !request.prefix.length) return;
    state.busy = true;
    updateControls();
    message(`Generando ${request.n_candidates} candidatos${request.prefix.length ? " del mismo prefijo" : " desde el espacio latente"}${validated ? " y evaluando geometría" : " en modo RAW"}…`, "loading");
    const start = performance.now();
    try {
      const response = await fetch("/api/generate", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(request)});
      const payload = await response.json();
      if (!response.ok) {
        const detail = typeof payload.detail === "string" ? payload.detail : "Entrada no válida. Revisa el prefijo y los controles.";
        throw new Error(detail);
      }
      state.samples = payload.samples;
      state.prefixCount = payload.prefix_count;
      state.lastRequest = request;
      state.lastResult = payload;
      state.selectedIndex = state.samples.length ? 0 : null;
      $("output-view").value = request.postprocess ? "processed" : "raw";
      showSamples();
      $("seed-label").textContent = `Semilla ${payload.seed}`;
      $("result-meta").textContent = `${payload.model} · ${request.n_candidates} candidatos → ${state.samples.length} seleccionados · temperatura ${payload.temperature.toFixed(2)} · ${((performance.now() - start) / 1000).toFixed(1)} s`;
      $("validation-details").hidden = false;
      $("validation-report").textContent = JSON.stringify({report: payload.report, candidates: payload.candidates}, null, 2);
      message(state.samples.length ? `${state.samples.length} variantes listas.${validated ? "" : " RAW: no se ha aplicado filtro de validez."} ${request.prefix.length ? "El prefijo azul es idéntico en todas." : "Muestras del espacio latente."}` : "No hay candidatos seleccionados. Revisa el informe o cambia a RAW para inspeccionar outputs sin filtrar.");
    } catch (error) {
      message(error.message || "No se pudo conectar con el servidor local.", "error");
    } finally {
      state.busy = false;
      updateControls();
    }
  }

  $("clear").addEventListener("click", () => { state.strokes = []; redrawInput(); });
  $("undo").addEventListener("click", () => { state.strokes.pop(); redrawInput(); });
  $("complete").addEventListener("click", () => generate("complete"));
  $("random").addEventListener("click", () => generate("random"));
  $("regenerate").addEventListener("click", () => generate("regenerate"));
  $("refresh").addEventListener("click", loadModels);
  $("model").addEventListener("change", updateControls);
  $("generation-mode").addEventListener("change", updateControls);
  $("output-view").addEventListener("change", () => { if (state.lastResult) showSamples(); });
  $("export-svg").addEventListener("click", () => exportSelected("svg"));
  $("export-json").addEventListener("click", () => exportSelected("json"));
  $("temperature").addEventListener("input", () => { $("temperature-value").value = Number($("temperature").value).toFixed(2); });
  $("sample-count").addEventListener("change", () => { if (!state.samples.length) placeholders(); });
  $("stroke-order").addEventListener("change", () => { redrawInput(); if (state.samples.length) showSamples(); });
  placeholders();
  updateControls();
  loadModels();
})();
