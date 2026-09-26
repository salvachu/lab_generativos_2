"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const MODEL_IDS = {
    H9: "artifacts/models/H9/best.pt",
    H13: "artifacts/models/H13/best.pt",
    H20: "artifacts/models/H20/best.pt",
  };
  const MODELS = ["H9", "H13", "H20"];
  const colors = {prefix: "#2563eb", generated: "#db684b"};
  const state = {strokes: [], current: null, pointer: null, busy: false, serial: 0,
    lastConfig: null, activeBenchmark: false, normal: [], columns: {}, drag: null};

  function message(value, kind = "") {
    $("status").textContent = value;
    $("status").className = `status ${kind}`;
  }

  function paint(canvas, strokes, prefixCount, showOrder) {
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const scale = canvas.width / 512;
    ctx.save();
    ctx.scale(scale, scale);
    ctx.lineWidth = 2.3;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    strokes.forEach((stroke, index) => {
      if (!stroke.length) return;
      ctx.strokeStyle = ctx.fillStyle = index < prefixCount ? colors.prefix : colors.generated;
      if (stroke.length === 1) {
        ctx.beginPath(); ctx.arc(stroke[0][0], stroke[0][1], 1.4, 0, Math.PI * 2); ctx.fill();
      } else {
        ctx.beginPath(); ctx.moveTo(...stroke[0]);
        for (const point of stroke.slice(1)) ctx.lineTo(...point);
        ctx.stroke();
      }
      if (showOrder) {
        ctx.font = "13px Segoe UI, sans-serif";
        ctx.lineWidth = 3;
        ctx.strokeStyle = "white";
        ctx.strokeText(String(index + 1), stroke[0][0] + 4, stroke[0][1] - 5);
        ctx.fillText(String(index + 1), stroke[0][0] + 4, stroke[0][1] - 5);
        ctx.lineWidth = 2.3;
      }
    });
    ctx.restore();
  }

  function updateInput() {
    const strokes = state.current ? [...state.strokes, state.current] : state.strokes;
    paint($("drawing"), strokes, strokes.length, $("show-order").checked);
    $("canvas-hint").hidden = strokes.length > 0;
    $("stroke-count").textContent = `${state.strokes.length} ${state.strokes.length === 1 ? "trazo" : "trazos"}`;
    $("undo").disabled = state.busy || !state.strokes.length;
    $("clear").disabled = state.busy || !state.strokes.length;
    $("complete").disabled = state.busy || !state.strokes.length;
    $("random").disabled = state.busy;
    $("regenerate").disabled = state.busy || !state.lastConfig;
    for (const id of ["model", "samples", "benchmark", "temperature", "candidates", "validate", "postprocess", "rerank"]) $(id).disabled = state.busy;
  }

  function point(event) {
    const rect = $("drawing").getBoundingClientRect();
    return [
      Math.min(512, Math.max(0, (event.clientX - rect.left) / rect.width * 512)),
      Math.min(512, Math.max(0, (event.clientY - rect.top) / rect.height * 512)),
    ].map((value) => Math.round(value * 100) / 100);
  }
  const canvas = $("drawing");
  canvas.addEventListener("pointerdown", (event) => {
    if (state.busy || state.pointer !== null || (event.pointerType === "mouse" && event.button !== 0)) return;
    if (state.strokes.length >= 96 || state.strokes.reduce((n, s) => n + s.length, 0) >= 4096) return;
    event.preventDefault();
    state.pointer = event.pointerId;
    state.current = [point(event)];
    canvas.setPointerCapture(event.pointerId);
    updateInput();
  });
  canvas.addEventListener("pointermove", (event) => {
    if (state.pointer !== event.pointerId || !state.current) return;
    const next = point(event);
    const last = state.current[state.current.length - 1];
    if (Math.hypot(next[0] - last[0], next[1] - last[1]) < 1.8) return;
    if (state.strokes.reduce((n, s) => n + s.length, state.current.length) >= 4096) return;
    state.current.push(next);
    updateInput();
  });
  function finish(event) {
    if (state.pointer !== event.pointerId || !state.current) return;
    state.strokes.push(state.current);
    state.current = null;
    state.pointer = null;
    if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
    updateInput();
  }
  canvas.addEventListener("pointerup", finish);
  canvas.addEventListener("pointercancel", finish);
  canvas.addEventListener("lostpointercapture", finish);

  function reorder(group, from, to) {
    const list = state.activeBenchmark ? state.columns[group] : state.normal;
    if (!list || from < 0 || to < 0 || from >= list.length || to >= list.length || from === to) return;
    const [card] = list.splice(from, 1);
    list.splice(to, 0, card);
    renderResults();
  }

  function resultCard(card, group, index, total) {
    const article = document.createElement("article");
    article.className = "result-card";
    article.tabIndex = 0;
    article.dataset.group = group;
    article.dataset.index = String(index);
    const preview = document.createElement("canvas");
    preview.width = preview.height = 512;
    preview.setAttribute("role", "img");
    preview.setAttribute("aria-label", `${card.model} variante ${index + 1}, prefijo azul y continuación naranja`);
    paint(preview, card.sample.postprocessed_strokes, card.prefixCount, $("show-order").checked);
    const footer = document.createElement("div");
    footer.className = "card-footer";
    const text = document.createElement("div");
    const title = document.createElement("strong");
    title.textContent = `Variante ${index + 1}`;
    const detail = document.createElement("small");
    detail.textContent = ` · ${card.model}${card.sample.validation?.valid === false ? " · inválida" :
      card.sample.validation?.valid == null ? " · sin validar" : ""}`;
    text.append(title, detail);
    const actions = document.createElement("div");
    actions.className = "card-actions";
    for (const [label, offset] of [["↑", -1], ["↓", 1]]) {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = label;
      button.title = `Mover ${offset < 0 ? "arriba" : "abajo"}`;
      button.setAttribute("aria-label", button.title + ` variante ${index + 1} de ${card.model}`);
      button.disabled = index + offset < 0 || index + offset >= total;
      button.addEventListener("click", () => reorder(group, index, index + offset));
      actions.append(button);
    }
    footer.append(text, actions);
    article.append(preview, footer);
    article.addEventListener("pointerdown", (event) => {
      if (event.button !== 0 || event.target.closest("button")) return;
      state.drag = {group, index, x: event.clientX, y: event.clientY, active: false};
      article.setPointerCapture(event.pointerId);
    });
    article.addEventListener("pointermove", (event) => {
      if (!state.drag || state.drag.group !== group || state.drag.index !== index) return;
      if (Math.hypot(event.clientX - state.drag.x, event.clientY - state.drag.y) < 8) return;
      state.drag.active = true;
      document.querySelectorAll(".result-card.drag-over").forEach((element) => element.classList.remove("drag-over"));
      const target = document.elementFromPoint(event.clientX, event.clientY)?.closest(".result-card");
      if (target?.dataset.group === group) target.classList.add("drag-over");
    });
    article.addEventListener("pointerup", (event) => {
      const drag = state.drag;
      state.drag = null;
      document.querySelectorAll(".result-card.drag-over").forEach((element) => element.classList.remove("drag-over"));
      if (!drag?.active || drag.group !== group || drag.index !== index) return;
      const target = document.elementFromPoint(event.clientX, event.clientY)?.closest(".result-card");
      if (target?.dataset.group === group) reorder(group, index, Number(target.dataset.index));
    });
    article.addEventListener("pointercancel", () => { state.drag = null; article.classList.remove("drag-over"); });
    article.addEventListener("keydown", (event) => {
      if (!event.altKey || !["ArrowUp", "ArrowDown"].includes(event.key)) return;
      event.preventDefault();
      reorder(group, index, index + (event.key === "ArrowUp" ? -1 : 1));
    });
    return article;
  }

  function renderResults() {
    const root = $("results");
    root.replaceChildren();
    root.className = state.activeBenchmark ? "benchmark-grid" : "results-grid";
    $("method-note").hidden = !state.activeBenchmark;
    if (state.activeBenchmark) {
      $("result-title").textContent = "Resultados — Benchmark";
      const count = MODELS.reduce((n, model) => n + (state.columns[model]?.length || 0), 0);
      $("result-count").textContent = `${count} variantes`;
      for (const model of MODELS) {
        const column = document.createElement("div");
        column.className = "benchmark-column";
        const heading = document.createElement("h3");
        heading.textContent = model;
        column.append(heading);
        const cards = state.columns[model] || [];
        cards.forEach((card, index) => column.append(resultCard(card, model, index, cards.length)));
        for (let i = cards.length; i < (state.lastConfig?.samples || Number($("samples").value)); i++) {
          const missing = document.createElement("div");
          missing.className = "empty-card";
          missing.textContent = "Sin variante válida";
          column.append(missing);
        }
        root.append(column);
      }
    } else {
      $("result-title").textContent = `Resultados — ${state.lastConfig?.model || $("model").value}`;
      $("result-count").textContent = state.normal.length ? `${state.normal.length} variantes` : "";
      state.normal.forEach((card, index) => root.append(resultCard(card, "normal", index, state.normal.length)));
    }
  }

  function settings(mode) {
    const samples = Number($("samples").value);
    const requestedCandidates = Number($("candidates").value);
    if (![3, 6, 9].includes(samples) ||
        !Number.isInteger(requestedCandidates) || requestedCandidates < 1 || requestedCandidates > 64) {
      throw new Error("Revisa Muestras y Candidatos internos: usa valores válidos.");
    }
    const candidates = Math.max(samples, requestedCandidates);
    if (candidates !== requestedCandidates) $("candidates").value = String(candidates);
    return {mode, model: $("model").value, benchmark: $("benchmark").checked,
      prefix: mode === "complete" ? structuredClone(state.strokes) : [], samples,
      candidates, temperature: Number($("temperature").value),
      validate: $("validate").checked, postprocess: $("postprocess").checked,
      rerank: $("rerank").checked};
  }

  async function generate(mode) {
    if (state.busy) return;
    let config;
    try { config = settings(mode); }
    catch (error) { message(error.message, "error"); return; }
    if (!config || (config.mode === "complete" && !config.prefix.length)) return;
    const serial = ++state.serial;
    state.busy = true;
    updateInput();
    const seed = crypto.getRandomValues(new Uint32Array(1))[0];
    const models = config.benchmark ? MODELS : [config.model];
    const columns = {};
    try {
      for (let i = 0; i < models.length; i++) {
        const model = models[i];
        message(`Generando ${model} (${i + 1}/${models.length})…`, "loading");
        const body = {model: MODEL_IDS[model], prefix: config.prefix,
          n_candidates: config.candidates, top_k: config.samples,
          temperature: config.temperature, seed, max_points: 384, max_strokes: 64,
          validate: config.validate, postprocess: config.validate && config.postprocess,
          rerank: config.validate && config.rerank};
        const response = await fetch("/api/demo/generate", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
        const payload = await response.json();
        if (!response.ok) throw new Error(typeof payload.detail === "string" ? payload.detail : `Falló ${model}.`);
        columns[model] = payload.samples.map((sample) => ({model, sample, prefixCount: config.prefix.length,
          request: body, report: payload.report}));
      }
      if (serial !== state.serial) return;
      state.lastConfig = config;
      state.activeBenchmark = config.benchmark;
      state.columns = columns;
      state.normal = config.benchmark ? [] : columns[config.model];
      renderResults();
      const count = config.benchmark ? MODELS.reduce((n, model) => n + columns[model].length, 0) : state.normal.length;
      message(count ? `${count} variantes listas. Arrastra las tarjetas para ordenarlas.` : "No hubo variantes válidas. Ajusta las opciones o inspecciona otro prefijo.");
    } catch (error) {
      message(error.message || "No se pudo completar la generación.", "error");
    } finally {
      state.busy = false;
      updateInput();
    }
  }

  $("undo").addEventListener("click", () => { state.strokes.pop(); updateInput(); });
  $("clear").addEventListener("click", () => {
    state.strokes = [];
    state.normal = [];
    state.columns = {};
    state.lastConfig = null;
    state.activeBenchmark = false;
    state.serial++;
    renderResults();
    updateInput();
    message("Canvas limpio. Dibuja un nuevo comienzo.");
  });
  $("complete").addEventListener("click", () => generate("complete"));
  $("random").addEventListener("click", () => generate("random"));
  $("regenerate").addEventListener("click", () => generate(state.strokes.length ? "complete" : "random"));
  $("show-order").addEventListener("change", () => { updateInput(); renderResults(); });
  $("benchmark").addEventListener("change", () => {
    if (!state.lastConfig) {
      state.activeBenchmark = $("benchmark").checked;
      renderResults();
    }
  });
  $("model").addEventListener("change", () => { if (!state.lastConfig) renderResults(); });
  $("temperature").addEventListener("input", () => { $("temperature-label").textContent = Number($("temperature").value).toFixed(2); });
  fetch("/api/models").then((r) => r.json()).then((data) => {
    $("device-badge").textContent = `DEMO · VECTORIAL · ${data.demo_device.toUpperCase()}`;
    const ids = new Set(data.models.map((model) => model.id));
    for (const option of $("model").options) option.disabled = !ids.has(MODEL_IDS[option.value]);
    if (!ids.has(MODEL_IDS.H20)) message("No se encontró el checkpoint H20. Revisa artifacts/models/.", "error");
    updateInput();
  }).catch(() => message("No se pudo consultar el servidor local.", "error"));
  updateInput();
  renderResults();
})();
