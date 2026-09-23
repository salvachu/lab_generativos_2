"""Inference-only diagnostics for Round 2 C/E checkpoints.

This script never updates model parameters. It contrasts causal teacher-forced
decoding with modal free-running decoding and evaluates the production sampler.
"""
from __future__ import annotations

import csv
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from sketchlab.data import load_raw
from sketchlab.evaluation import distances, geometry_metrics, prefix_count
from sketchlab.generation import load_model, sample
from sketchlab.geometry import load_geometry, validate_candidate
from sketchlab.losses import mdn_parameters
from sketchlab.models.common import tokens
from sketchlab.orchestration import json_ready
from sketchlab.training import training_threshold


OUT = Path("runs/diagnostics_round2")
MODELS = "CE"
CHECKPOINTS = {m: Path(f"runs/tournament_round2/{m}/best.pt") for m in MODELS}
CONDITIONS = [("1stroke", "one"), ("2strokes", "two"), ("25pct", .25),
              ("50pct", .5), ("75pct", .75)]
TEMPERATURES = [0.3, 0.5, 0.7, 0.9, 1.0]
MAX_POINTS = 384
MAX_STROKES = 64
BOUNDS = (-4.0, 508.0)
BASE_SEED = 73100


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite_mean(values):
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def flatten_points(strokes):
    return np.concatenate([np.asarray(s, dtype=np.float64) for s in strokes]) if strokes else np.empty((0, 2))


def categorical(logits, temperature, rng):
    logits = logits.detach().float().cpu()
    probs = torch.softmax(logits.double(), -1)
    if temperature == 0:
        return int(logits.argmax()), probs.numpy()
    tempered = torch.softmax((logits - logits.max()).double() / temperature, -1)
    return int(torch.multinomial(tempered, 1, generator=rng)), probs.numpy()


def mdn_draw(raw, temperature, rng):
    raw = raw.detach().float().cpu()
    log_weights, means, log_scales, correlations = mdn_parameters(raw)
    component, weights = categorical(log_weights, temperature, rng)
    sigma = log_scales[component].exp()
    if temperature == 0:
        point = means[component]
    else:
        noise = torch.randn(2, generator=rng)
        rho = correlations[component]
        correlated = torch.stack([noise[0], rho * noise[0] + (1-rho.square()).sqrt()*noise[1]])
        point = means[component] + sigma * math.sqrt(temperature) * correlated
    return point.numpy().astype(np.float64), component, sigma.numpy(), float(weights[component])


def state_name(model, event, index):
    if model == "C":
        return ("draw", "stroke_end", "eos")[index]
    if event == "sketch":
        return ("continue", "eos")[index]
    return ("draw", "stroke_end")[index]


def real_suffix_events(sample_data, count, model):
    suffix = sample_data["strokes"][count:]
    if model == "C":
        full = tokens(sample_data["strokes"], 256.0)
        offset = sum(len(s) for s in sample_data["strokes"][:count])
        rows, absolute = [], flatten_points(suffix)
        for i, token in enumerate(full[offset:]):
            label = int(np.argmax(token[2:5]))
            rows.append({"event": "point", "true_dx": float(token[0]*256),
                         "true_dy": float(token[1]*256),
                         "true_x": float(absolute[i, 0]) if i < len(absolute) else None,
                         "true_y": float(absolute[i, 1]) if i < len(absolute) else None,
                         "true_state": state_name(model, "point", label)})
        return rows
    rows = []
    for stroke in suffix:
        stroke = np.asarray(stroke, dtype=np.float64)
        rows.append({"event": "anchor", "true_dx": None, "true_dy": None,
                     "true_x": float(stroke[0, 0]), "true_y": float(stroke[0, 1]),
                     "true_state": "continue"})
        for index in range(len(stroke)):
            is_end = index == len(stroke)-1
            delta = np.zeros(2) if is_end else stroke[index+1]-stroke[index]
            target = stroke[index] if is_end else stroke[index+1]
            rows.append({"event": "point", "true_dx": float(delta[0]), "true_dy": float(delta[1]),
                         "true_x": float(target[0]), "true_y": float(target[1]),
                         "true_state": "stroke_end" if is_end else "draw"})
    rows.append({"event": "sketch_eos", "true_dx": None, "true_dy": None,
                 "true_x": None, "true_y": None, "true_state": "eos"})
    return rows


def add_errors(record, truth):
    record.update({k: truth.get(k) for k in ("true_dx", "true_dy", "true_x", "true_y", "true_state")})
    if truth.get("true_dx") is not None and record.get("dx") is not None:
        record["coordinate_error"] = float(np.linalg.norm(
            np.asarray([record["dx"], record["dy"]]) - np.asarray([truth["true_dx"], truth["true_dy"]])))
    else:
        record["coordinate_error"] = None
    if truth.get("true_x") is not None and record.get("absolute_x") is not None:
        record["absolute_position_drift"] = float(np.linalg.norm(
            np.asarray([record["absolute_x"], record["absolute_y"]]) - np.asarray([truth["true_x"], truth["true_y"]])))
    else:
        record["absolute_position_drift"] = None
    record["state_correct"] = int(record.get("selected_state") == truth.get("true_state"))
    x, y = record.get("absolute_x"), record.get("absolute_y")
    record["inside_canvas"] = int(x is not None and y is not None and BOUNDS[0] <= x <= BOUNDS[1] and BOUNDS[0] <= y <= BOUNDS[1])
    return record


@torch.inference_mode()
def sequence_teacher(model, sample_data, count):
    context = model.context([sample_data["strokes"][:count]])
    z, _ = model.prior(context)
    target_np = tokens(sample_data["strokes"], model.scale)
    target = torch.as_tensor(target_np, dtype=torch.float32, device=model.device).unsqueeze(0)
    inputs = torch.zeros_like(target); inputs[:, 0, 2] = 1; inputs[:, 1:] = target[:, :-1]
    raw, pen, _ = model.decode(inputs, z, context)
    offset = sum(len(s) for s in sample_data["strokes"][:count])
    truth = real_suffix_events(sample_data, count, "C")
    previous = np.asarray(sample_data["strokes"][count-1][-1], dtype=np.float64) if count else np.zeros(2)
    rows = []
    for step, (r, logits, actual) in enumerate(zip(raw[0, offset:], pen[0, offset:], truth)):
        delta, component, sigma, weight = mdn_draw(r, 0, torch.Generator())
        probs = F.softmax(logits, -1).cpu().numpy()
        state = int(np.argmax(probs))
        absolute = previous + model.scale * delta if actual["true_x"] is not None else None
        record = {"step": step, "event": "point", "dx": float(delta[0]*model.scale),
                  "dy": float(delta[1]*model.scale),
                  "absolute_x": float(absolute[0]) if absolute is not None else None,
                  "absolute_y": float(absolute[1]) if absolute is not None else None,
                  "state_probs": json.dumps([float(x) for x in probs]),
                  "eos_probability": float(probs[2]), "stroke_end_probability": float(probs[1]),
                  "selected_state": state_name("C", "point", state), "mdn_component": component,
                  "component_probability": weight, "sigma_x": float(sigma[0]*model.scale),
                  "sigma_y": float(sigma[1]*model.scale), "stroke_count": None,
                  "point_count": step}
        rows.append(add_errors(record, actual))
        if actual["true_x"] is not None:
            previous = np.asarray([actual["true_x"], actual["true_y"]])
    return rows


@torch.inference_mode()
def sequence_free(model, prefix, truth, *, seed, temperature, latent_mean):
    context = model.context([prefix])
    mu, logvar = model.prior(context)
    latent_rng = torch.Generator().manual_seed(seed)
    z = mu if latent_mean else mu + torch.randn(mu.shape, generator=latent_rng) * (0.5*logvar).exp()
    decoder_rng = torch.Generator().manual_seed(seed+900000)
    prefix_tokens = tokens(prefix, model.scale)[:-1]
    bos = np.asarray([[0, 0, 1, 0, 0]], dtype=np.float64)
    inp = torch.as_tensor(np.concatenate([bos, prefix_tokens]), dtype=torch.float32).unsqueeze(0)
    raw, pen, hidden = model.decode(inp, z, context); raw, pen = raw[0, -1], pen[0, -1]
    cursor = np.asarray(prefix[-1][-1], dtype=np.float64).copy() if prefix else np.zeros(2)
    rows, stroke_count = [], 0
    for step in range(MAX_POINTS):
        state, probs = categorical(pen, temperature, decoder_rng)
        delta, component, sigma, weight = mdn_draw(raw, temperature, decoder_rng)
        if state == 2:
            absolute = None
        else:
            absolute = cursor + model.scale*delta
        record = {"step": step, "event": "point", "dx": float(delta[0]*model.scale),
                  "dy": float(delta[1]*model.scale),
                  "absolute_x": float(absolute[0]) if absolute is not None else None,
                  "absolute_y": float(absolute[1]) if absolute is not None else None,
                  "state_probs": json.dumps([float(x) for x in probs]),
                  "eos_probability": float(probs[2]), "stroke_end_probability": float(probs[1]),
                  "selected_state": state_name("C", "point", state), "mdn_component": component,
                  "component_probability": weight, "sigma_x": float(sigma[0]*model.scale),
                  "sigma_y": float(sigma[1]*model.scale), "stroke_count": stroke_count,
                  "point_count": step}
        actual = truth[step] if step < len(truth) else {"true_state": None}
        rows.append(add_errors(record, actual))
        if state == 2:
            break
        cursor = absolute
        if state == 1:
            stroke_count += 1
        token = np.zeros(5, dtype=np.float32); token[:2] = delta; token[2+state] = 1
        raw2, pen2, hidden = model.decode(torch.as_tensor(token)[None, None], z, context, hidden)
        raw, pen = raw2[0, 0], pen2[0, 0]
    return rows


@torch.inference_mode()
def hierarchical_teacher(model, sample_data, count):
    strokes = sample_data["strokes"]
    embeddings = model.embed_strokes(strokes)
    context = model.context([strokes[:count]])
    z, _ = model.prior(context)
    previous = torch.cat([torch.zeros((1, model.hidden_dim)), embeddings], 0).unsqueeze(0)
    states, _ = model.decode_strokes(previous, z, context)
    truth = real_suffix_events(sample_data, count, "E")
    rows, truth_index, step = [], 0, 0
    for stroke_index in range(count, len(strokes)):
        stroke = np.asarray(strokes[stroke_index], dtype=np.float64)
        state = states[:, stroke_index]
        sketch_probs = F.softmax(model.sketch_end_head(state)[0], -1).cpu().numpy()
        anchor_raw = model.anchor_head(state)[0]
        anchor, component, sigma, weight = mdn_draw(anchor_raw, 0, torch.Generator())
        anchor = anchor * model.scale
        record = {"step": step, "event": "anchor", "dx": None, "dy": None,
                  "absolute_x": float(anchor[0]), "absolute_y": float(anchor[1]),
                  "state_probs": json.dumps([float(x) for x in sketch_probs]),
                  "eos_probability": float(sketch_probs[1]), "stroke_end_probability": None,
                  "selected_state": "continue" if sketch_probs.argmax() == 0 else "eos",
                  "mdn_component": component, "component_probability": weight,
                  "sigma_x": float(sigma[0]*model.scale), "sigma_y": float(sigma[1]*model.scale),
                  "stroke_count": stroke_index-count, "point_count": step}
        rows.append(add_errors(record, truth[truth_index])); truth_index += 1; step += 1
        local_mu, _ = model.local_prior(state)
        anchor_target = torch.as_tensor(stroke[0]/model.scale, dtype=torch.float32)[None]
        condition = model.point_condition(state, local_mu, anchor_target)
        target = np.zeros((len(stroke), 3), dtype=np.float32)
        target[:-1, :2] = np.diff(stroke, axis=0)/model.scale; target[-1, 2] = 1
        previous_points = np.zeros_like(target); previous_points[1:] = target[:-1]
        raw, pen, _ = model.decode_points(torch.as_tensor(previous_points)[None], condition)
        previous_abs = stroke[0]
        for r, logits in zip(raw[0], pen[0]):
            delta, component, sigma, weight = mdn_draw(r, 0, torch.Generator())
            probs = F.softmax(logits, -1).cpu().numpy(); selected = int(probs.argmax())
            actual = truth[truth_index]
            absolute = previous_abs + delta*model.scale if actual["true_state"] == "draw" else previous_abs
            record = {"step": step, "event": "point", "dx": float(delta[0]*model.scale),
                      "dy": float(delta[1]*model.scale), "absolute_x": float(absolute[0]),
                      "absolute_y": float(absolute[1]),
                      "state_probs": json.dumps([float(x) for x in probs]),
                      "eos_probability": float(sketch_probs[1]),
                      "stroke_end_probability": float(probs[1]),
                      "selected_state": state_name("E", "point", selected),
                      "mdn_component": component, "component_probability": weight,
                      "sigma_x": float(sigma[0]*model.scale), "sigma_y": float(sigma[1]*model.scale),
                      "stroke_count": stroke_index-count, "point_count": step}
            rows.append(add_errors(record, actual)); truth_index += 1; step += 1
            if actual["true_x"] is not None:
                previous_abs = np.asarray([actual["true_x"], actual["true_y"]])
    sketch_probs = F.softmax(model.sketch_end_head(states[:, len(strokes)])[0], -1).cpu().numpy()
    record = {"step": step, "event": "sketch_eos", "dx": None, "dy": None,
              "absolute_x": None, "absolute_y": None,
              "state_probs": json.dumps([float(x) for x in sketch_probs]),
              "eos_probability": float(sketch_probs[1]), "stroke_end_probability": None,
              "selected_state": state_name("E", "sketch", int(sketch_probs.argmax())),
              "mdn_component": None, "component_probability": None, "sigma_x": None, "sigma_y": None,
              "stroke_count": len(strokes)-count, "point_count": step}
    rows.append(add_errors(record, truth[truth_index]))
    return rows


@torch.inference_mode()
def hierarchical_free(model, prefix, truth, *, seed, temperature, latent_mean):
    context = model.context([prefix]); mu, logvar = model.prior(context)
    latent_rng = torch.Generator().manual_seed(seed)
    z = mu if latent_mean else mu + torch.randn(mu.shape, generator=latent_rng)*(0.5*logvar).exp()
    local_rng = torch.Generator().manual_seed(seed+200000)
    decoder_rng = torch.Generator().manual_seed(seed+900000)
    embeddings = model.embed_strokes(prefix)
    previous = torch.cat([torch.zeros((1, model.hidden_dim)), embeddings], 0).unsqueeze(0)
    states, stroke_hidden = model.decode_strokes(previous, z, context); state = states[:, -1]
    rows, points_used, step = [], 0, 0
    for stroke_index in range(MAX_STROKES):
        stop, sketch_probs = categorical(model.sketch_end_head(state)[0], temperature, decoder_rng)
        if stop == 1:
            record = {"step": step, "event": "sketch_eos", "dx": None, "dy": None,
                      "absolute_x": None, "absolute_y": None,
                      "state_probs": json.dumps([float(x) for x in sketch_probs]),
                      "eos_probability": float(sketch_probs[1]), "stroke_end_probability": None,
                      "selected_state": "eos", "mdn_component": None,
                      "component_probability": None, "sigma_x": None, "sigma_y": None,
                      "stroke_count": stroke_index, "point_count": points_used}
            rows.append(add_errors(record, truth[step] if step < len(truth) else {"true_state": None}))
            break
        anchor, component, sigma, weight = mdn_draw(model.anchor_head(state)[0], temperature, decoder_rng)
        absolute = anchor*model.scale
        record = {"step": step, "event": "anchor", "dx": None, "dy": None,
                  "absolute_x": float(absolute[0]), "absolute_y": float(absolute[1]),
                  "state_probs": json.dumps([float(x) for x in sketch_probs]),
                  "eos_probability": float(sketch_probs[1]), "stroke_end_probability": None,
                  "selected_state": "continue", "mdn_component": component,
                  "component_probability": weight, "sigma_x": float(sigma[0]*model.scale),
                  "sigma_y": float(sigma[1]*model.scale), "stroke_count": stroke_index,
                  "point_count": points_used}
        rows.append(add_errors(record, truth[step] if step < len(truth) else {"true_state": None})); step += 1
        local_mu, local_logvar = model.local_prior(state)
        local_z = local_mu if latent_mean else local_mu + torch.randn(local_mu.shape, generator=local_rng)*(0.5*local_logvar).exp()
        condition = model.point_condition(state, local_z, torch.as_tensor(anchor, dtype=torch.float32)[None])
        prev = torch.zeros((1, 1, 3)); hidden = None; stroke = [absolute.copy()]; points_used += 1
        while points_used < MAX_POINTS:
            raw, pen, hidden = model.decode_points(prev, condition, hidden)
            point_state, point_probs = categorical(pen[0, 0], temperature, decoder_rng)
            delta, component, sigma, weight = mdn_draw(raw[0, 0], temperature, decoder_rng)
            next_abs = stroke[-1] if point_state == 1 else stroke[-1]+delta*model.scale
            record = {"step": step, "event": "point", "dx": float(delta[0]*model.scale),
                      "dy": float(delta[1]*model.scale), "absolute_x": float(next_abs[0]),
                      "absolute_y": float(next_abs[1]),
                      "state_probs": json.dumps([float(x) for x in point_probs]),
                      "eos_probability": float(sketch_probs[1]),
                      "stroke_end_probability": float(point_probs[1]),
                      "selected_state": state_name("E", "point", point_state),
                      "mdn_component": component, "component_probability": weight,
                      "sigma_x": float(sigma[0]*model.scale), "sigma_y": float(sigma[1]*model.scale),
                      "stroke_count": stroke_index, "point_count": points_used}
            rows.append(add_errors(record, truth[step] if step < len(truth) else {"true_state": None})); step += 1
            if point_state == 1:
                break
            stroke.append(next_abs); points_used += 1
            prev = torch.as_tensor([[[*delta, 0.0]]], dtype=torch.float32)
        generated = np.asarray(stroke)
        if points_used >= MAX_POINTS:
            break
        emb = model.embed_strokes([generated]).unsqueeze(0)
        states, stroke_hidden = model.decode_strokes(emb, z, context, stroke_hidden); state = states[:, -1]
    return rows


def attach_metadata(rows, model, mode, sample_data, condition, count):
    for row in rows:
        row.update(model=model, mode=mode, validation_id=sample_data["id"], condition=condition,
                   prefix_strokes=count)
    return rows


def teacher_free_diagnostics(models, cases):
    rows = []
    for model_name, model in models.items():
        for condition_index, (condition, fraction) in enumerate(CONDITIONS):
            for case_index, sample_data in enumerate(cases):
                count = prefix_count(len(sample_data["strokes"]), fraction)
                truth = real_suffix_events(sample_data, count, model_name)
                if model_name == "C":
                    teacher = sequence_teacher(model, sample_data, count)
                    free = sequence_free(model, sample_data["strokes"][:count], truth,
                                         seed=BASE_SEED+condition_index*100+case_index,
                                         temperature=0, latent_mean=True)
                else:
                    teacher = hierarchical_teacher(model, sample_data, count)
                    free = hierarchical_free(model, sample_data["strokes"][:count], truth,
                                             seed=BASE_SEED+condition_index*100+case_index,
                                             temperature=0, latent_mean=True)
                rows.extend(attach_metadata(teacher, model_name, "teacher_forced", sample_data, condition, count))
                rows.extend(attach_metadata(free, model_name, "free_running_modal", sample_data, condition, count))
    columns = ["model", "mode", "validation_id", "condition", "prefix_strokes", "step", "event",
               "true_dx", "true_dy", "dx", "dy", "coordinate_error", "true_x", "true_y",
               "absolute_x", "absolute_y", "absolute_position_drift", "true_state", "selected_state",
               "state_correct", "state_probs", "eos_probability", "stroke_end_probability",
               "inside_canvas", "mdn_component", "component_probability", "sigma_x", "sigma_y",
               "stroke_count", "point_count"]
    normalized = [{k: row.get(k) for k in columns} for row in rows]
    write_csv(OUT/"teacher_forced_vs_free_running.csv", normalized)

    summaries = []
    for model in MODELS:
        for mode in ("teacher_forced", "free_running_modal"):
            selected = [r for r in rows if r["model"] == model and r["mode"] == mode]
            for bucket in range(20):
                bucket_rows = []
                for group_key, group in itertools.groupby(sorted(selected, key=lambda r:(r["validation_id"],r["condition"],r["step"])),
                                                           key=lambda r:(r["validation_id"],r["condition"])):
                    group = list(group)
                    bucket_rows.extend(r for r in group if min(19, int(20*r["step"]/max(1,len(group)))) == bucket)
                summaries.append({"model": model, "mode": mode, "normalized_timestep_bin": bucket,
                                  "coordinate_error": finite_mean(r["coordinate_error"] for r in bucket_rows),
                                  "absolute_position_drift": finite_mean(r["absolute_position_drift"] for r in bucket_rows),
                                  "state_accuracy": finite_mean(r["state_correct"] for r in bucket_rows),
                                  "eos_probability": finite_mean(r["eos_probability"] for r in bucket_rows),
                                  "inside_canvas": finite_mean(r["inside_canvas"] for r in bucket_rows),
                                  "sigma_mean": finite_mean((r["sigma_x"]+r["sigma_y"])/2 for r in bucket_rows if r["sigma_x"] is not None)})
    divergence = {}
    for model in MODELS:
        tf = [r for r in summaries if r["model"] == model and r["mode"] == "teacher_forced"]
        fr = [r for r in summaries if r["model"] == model and r["mode"] == "free_running_modal"]
        first = None
        for a,b in zip(tf,fr):
            if (a["absolute_position_drift"] is not None and b["absolute_position_drift"] is not None and
                b["absolute_position_drift"] > max(25.0, 1.5*a["absolute_position_drift"])):
                first = a["normalized_timestep_bin"]; break
        divergence[model] = {"first_divergence_bin_of_20": first,
                             "approximate_fraction": None if first is None else first/20}
    payload = {"summary_by_normalized_timestep": summaries, "divergence": divergence,
               "method": "same prior latent mean; teacher forced uses real causal history; free running uses modal MDN/argmax feedback"}
    (OUT/"teacher_forced_vs_free_running.json").write_text(json.dumps(json_ready(payload),indent=2),encoding="utf-8")
    plot_teacher_free(summaries)
    return rows, payload


def plot_teacher_free(summary):
    metrics = [("absolute_position_drift","absolute drift"),("state_accuracy","state accuracy"),
               ("eos_probability","EOS probability"),("inside_canvas","inside canvas")]
    fig, axes = plt.subplots(2, 4, figsize=(16,7), constrained_layout=True)
    for row, model in enumerate(MODELS):
        for col,(key,label) in enumerate(metrics):
            ax=axes[row,col]
            for mode,color in (("teacher_forced","#2378d4"),("free_running_modal","#e76427")):
                data=[r for r in summary if r["model"]==model and r["mode"]==mode]
                ax.plot([r["normalized_timestep_bin"]/20 for r in data],[r[key] for r in data],label=mode,color=color)
            ax.set_title(f"{model} · {label}"); ax.set_xlabel("normalized timestep"); ax.grid(alpha=.2)
            if col==0: ax.legend(fontsize=7)
    fig.suptitle("Teacher-forced vs modal free-running · same latent mean")
    fig.savefig(OUT/"visual_grids"/"teacher_forced_vs_free_running.png",dpi=150); plt.close(fig)


def state_counts(dataset, model):
    if model == "C":
        c=Counter()
        for s in dataset:
            for stroke in s["strokes"]:
                c["draw"] += max(0,len(stroke)-1); c["stroke_end"] += 1
            c["eos"] += 1
        return c
    c=Counter()
    for s in dataset:
        c["stroke_continue"] += len(s["strokes"]); c["sketch_eos"] += 1
        c["point_draw"] += sum(max(0,len(stroke)-1) for stroke in s["strokes"])
        c["stroke_end"] += len(s["strokes"])
    return c


def eos_diagnostics(tf_rows, train, val, sampled_records):
    csv_rows=[]; payload={"real_class_balance":{},"models":{}}
    for model in MODELS:
        payload["real_class_balance"][model]={}
        for split,data in (("TRAIN",train),("VAL",val)):
            counts=state_counts(data,model); total=sum(counts.values())
            payload["real_class_balance"][model][split]={k:{"count":v,"fraction":v/total} for k,v in counts.items()}
            for state,value in counts.items():
                csv_rows.append({"section":"real_class_balance","model":model,"split":split,"metric":state,
                                 "bin":None,"value":value,"fraction":value/total})
        length_distributions = {}
        for split, data in (("TRAIN", train), ("VAL", val)):
            stroke_lengths = [len(s["strokes"]) for s in data]
            point_lengths = [sum(len(x) for x in s["strokes"]) for s in data]
            length_distributions[f"real_{split.lower()}"] = {
                "strokes_p25_median_p75": [float(np.quantile(stroke_lengths,q)) for q in (.25,.5,.75)],
                "points_p25_median_p75": [float(np.quantile(point_lengths,q)) for q in (.25,.5,.75)]}
        generated = [r["metrics"] for r in sampled_records
                     if r["model"] == model and r["configuration"] == "T=0.7"]
        length_distributions["generated_T0p7"] = {
            "strokes_p25_median_p75": [float(np.quantile([x["strokes"] for x in generated],q)) for q in (.25,.5,.75)],
            "points_p25_median_p75": [float(np.quantile([x["points"] for x in generated],q)) for q in (.25,.5,.75)]}
        rows=[r for r in tf_rows if r["model"]==model and r["mode"]=="teacher_forced" and r["true_state"] is not None]
        y=np.asarray([r["true_state"]=="eos" for r in rows]); pred=np.asarray([r["selected_state"]=="eos" for r in rows]); p=np.asarray([r["eos_probability"] for r in rows])
        tp=int((y&pred).sum()); fp=int((~y&pred).sum()); fn=int((y&~pred).sum()); tn=int((~y&~pred).sum())
        metrics={"tp":tp,"fp":fp,"fn":fn,"tn":tn,"precision":tp/max(1,tp+fp),"recall":tp/max(1,tp+fn),
                 "mean_eos_probability_positive":float(p[y].mean()) if y.any() else None,
                 "mean_eos_probability_negative":float(p[~y].mean()) if (~y).any() else None,
                 "brier":float(np.mean((p-y.astype(float))**2))}
        bins=[]
        for i in range(10):
            mask=(p>=i/10)&(p<((i+1)/10 if i<9 else 1.000001))
            bins.append({"low":i/10,"high":(i+1)/10,"count":int(mask.sum()),
                         "mean_probability":float(p[mask].mean()) if mask.any() else None,
                         "empirical_eos":float(y[mask].mean()) if mask.any() else None})
        payload["models"][model]={"classification":metrics,"calibration":bins,
                                    "length_distributions":length_distributions}
        for k,v in metrics.items(): csv_rows.append({"section":"teacher_forced_eos","model":model,"split":"fixed_VAL","metric":k,"bin":None,"value":v,"fraction":None})
        for i,b in enumerate(bins):
            csv_rows.append({"section":"calibration","model":model,"split":"fixed_VAL","metric":"eos","bin":i,
                             "value":b["mean_probability"],"fraction":b["empirical_eos"]})
    write_csv(OUT/"eos_diagnostics.csv",csv_rows)
    (OUT/"eos_diagnostics.json").write_text(json.dumps(json_ready(payload),indent=2),encoding="utf-8")
    return payload


def sample_metrics(strokes, count, info, stats):
    suffix=strokes[count:]; geom=geometry_metrics(suffix,training_threshold(None))
    val=validate_candidate(strokes,stats,prefix_count=count,termination=info)
    return {"eos":bool(info["ended_by_eos"]),"cap":info["termination"] in {"max_points","max_strokes"},
            "points":sum(len(s) for s in suffix),"strokes":len(suffix),
            "canvas":geom["canvas_validity"] if geom["canvas_validity"] is not None else 0.0,
            "fully_canvas":bool(geom["canvas_validity"]==1.0),"valid":bool(val.valid),"termination":info["termination"]}


def plot_grid(records, condition):
    configs=["deterministic"]+[f"T={t:.1f}" for t in TEMPERATURES]
    fig,axes=plt.subplots(len(configs),2,figsize=(7,18),constrained_layout=True)
    for row,cfg in enumerate(configs):
        for col,model in enumerate(MODELS):
            rec=next(r for r in records if r["configuration"]==cfg and r["model"]==model and r["condition"]==condition and r["case"]==1 and r["sample"]==0)
            ax=axes[row,col]
            for i,stroke in enumerate(rec["strokes"]):
                a=np.asarray(stroke); color="#2378d4" if i<rec["prefix_count"] else "#e76427"
                ax.plot(a[:,0],a[:,1],color=color,lw=.8)
            ax.set(xlim=BOUNDS,ylim=(BOUNDS[1],BOUNDS[0]),aspect="equal"); ax.set_xticks([]); ax.set_yticks([])
            m=rec["metrics"]; ax.set_title(f"{model} · {cfg}\n{m['points']} pts · {m['termination']} · canvas {100*m['canvas']:.1f}%",fontsize=8)
    fig.suptitle(f"Sampling sweep · {condition} · caso fijo 1")
    fig.savefig(OUT/"visual_grids"/f"sampling_{condition}.png",dpi=145); plt.close(fig)


def sampling_sweep(models,cases,stats):
    records=[]
    configs=[("deterministic",0.0,True)]+[(f"T={t:.1f}",t,False) for t in TEMPERATURES]
    for cfg,temp,deterministic in configs:
        for model_name,model in models.items():
            for condition_index,(condition,fraction) in enumerate(CONDITIONS):
                for case_index,sample_data in enumerate(cases):
                    count=prefix_count(len(sample_data["strokes"]),fraction); prefix=sample_data["strokes"][:count]
                    n=1 if deterministic else 2
                    group=[]
                    for k in range(n):
                        seed=BASE_SEED+condition_index*10000+case_index*100+k
                        z=None
                        if deterministic:
                            context=model.context([prefix]); z=model.prior(context)[0][0].detach().cpu().numpy()
                        strokes,info=sample(model,prefix=prefix,seed=seed,decoder_seed=seed+900000,
                                            temperature=temp,max_points=MAX_POINTS,max_strokes=MAX_STROKES,z=z,return_info=True)
                        metric=sample_metrics(strokes,count,info,stats)
                        rec={"configuration":cfg,"temperature":temp,"model":model_name,"condition":condition,
                             "case":case_index+1,"validation_id":sample_data["id"],"sample":k,"prefix_count":count,
                             "metrics":metric,"strokes":strokes}
                        records.append(rec); group.append(strokes[count:])
                    diversity=distances(group[0],group[1])["chamfer"] if len(group)>1 else 0.0
                    for rec in records[-n:]: rec["metrics"]["diversity_chamfer"]=diversity
            # Eight random generations per configuration/model.
            for k in range(8):
                seed=BASE_SEED+90000+k; z=None
                if deterministic:
                    context=model.context([[]]); z=model.prior(context)[0][0].detach().cpu().numpy()
                strokes,info=sample(model,prefix=[],seed=seed,decoder_seed=seed+900000,temperature=temp,
                                    max_points=MAX_POINTS,max_strokes=MAX_STROKES,z=z,return_info=True)
                records.append({"configuration":cfg,"temperature":temp,"model":model_name,"condition":"random",
                                "case":k+1,"validation_id":None,"sample":0,"prefix_count":0,
                                "metrics":sample_metrics(strokes,0,info,stats),"strokes":strokes})
    rows=[]
    for cfg,temp,_ in configs:
        for model in MODELS:
            selected=[r for r in records if r["configuration"]==cfg and r["model"]==model]
            completion=[r for r in selected if r["condition"]!="random"]
            rows.append({"model":model,"configuration":cfg,"temperature":temp,"samples":len(selected),
                         "eos_rate":float(np.mean([r["metrics"]["eos"] for r in selected])),
                         "cap_rate":float(np.mean([r["metrics"]["cap"] for r in selected])),
                         "median_generated_points":float(np.median([r["metrics"]["points"] for r in selected])),
                         "mean_canvas_validity":float(np.mean([r["metrics"]["canvas"] for r in selected])),
                         "fully_canvas_valid_rate":float(np.mean([r["metrics"]["fully_canvas"] for r in selected])),
                         "completion_valid_rate":float(np.mean([r["metrics"]["valid"] for r in completion])),
                         "diversity_chamfer":finite_mean(r["metrics"].get("diversity_chamfer") for r in completion)})
    write_csv(OUT/"sampling_sweep.csv",rows)
    (OUT/"sampling_sweep.json").write_text(json.dumps(json_ready({"summary":rows,"protocol":{"temperatures":TEMPERATURES,"stochastic_samples_per_case":2,"random_per_model":8,"deterministic":"prior mean + argmax state + modal MDN mean"}}),indent=2),encoding="utf-8")
    for condition,_ in CONDITIONS: plot_grid(records,condition)
    plot_grid(records,"random")
    return records,rows


def prefix_diagnostics(records):
    rows=[]
    for model in MODELS:
        for cfg in ["deterministic"]+[f"T={t:.1f}" for t in TEMPERATURES]:
            for condition,_ in CONDITIONS:
                selected=[r for r in records if r["model"]==model and r["configuration"]==cfg and r["condition"]==condition]
                rows.append({"model":model,"configuration":cfg,"condition":condition,"samples":len(selected),
                             "eos_rate":float(np.mean([r["metrics"]["eos"] for r in selected])),
                             "sequence_cap_rate":float(np.mean([r["metrics"]["cap"] for r in selected])),
                             "median_generated_points":float(np.median([r["metrics"]["points"] for r in selected])),
                             "mean_canvas_validity":float(np.mean([r["metrics"]["canvas"] for r in selected])),
                             "fully_canvas_valid_rate":float(np.mean([r["metrics"]["fully_canvas"] for r in selected]))})
    write_csv(OUT/"prefix_length_diagnostics.csv",rows)
    return rows


def generation_traces(models,cases):
    trace_dir=OUT/"generation_traces"
    for model_name,model in models.items():
        for case_index in (0,4):
            sample_data=cases[case_index]; count=prefix_count(len(sample_data["strokes"]),.25)
            truth=real_suffix_events(sample_data,count,model_name); prefix=sample_data["strokes"][:count]
            seed=BASE_SEED+case_index
            rows=(sequence_free(model,prefix,truth,seed=seed,temperature=.7,latent_mean=False) if model_name=="C"
                  else hierarchical_free(model,prefix,truth,seed=seed,temperature=.7,latent_mean=False))
            columns=["step","event","dx","dy","absolute_x","absolute_y","state_probs","eos_probability",
                     "stroke_end_probability","selected_state","mdn_component","component_probability","sigma_x","sigma_y",
                     "stroke_count","point_count","true_dx","true_dy","coordinate_error","true_x","true_y",
                     "absolute_position_drift","true_state","state_correct","inside_canvas"]
            write_csv(trace_dir/f"{model_name}_case{case_index+1:02d}_25pct_T0p7.csv",[{k:r.get(k) for k in columns} for r in rows])


def render_posthoc():
    """Render aggregate plots and persist exact divergence from existing tables."""
    eos = json.loads((OUT/"eos_diagnostics.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for ax, model in zip(axes, MODELS):
        bins = [b for b in eos["models"][model]["calibration"] if b["count"]]
        ax.plot([b["mean_probability"] for b in bins], [b["empirical_eos"] for b in bins], "o-", label=model)
        ax.plot([0,1],[0,1],"--",color="#888888",lw=.8)
        ax.set(xlim=(0,.2),ylim=(0,1),xlabel="predicted EOS probability",ylabel="empirical EOS",title=f"{model} · EOS calibration")
        ax.grid(alpha=.2)
    fig.savefig(OUT/"visual_grids"/"eos_calibration.png",dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9,4), constrained_layout=True)
    for ax, model in zip(axes, MODELS):
        dist=eos["models"][model]["length_distributions"]
        names=["TRAIN real","VAL real","generated T=.7"]
        values=[dist["real_train"]["points_p25_median_p75"],dist["real_val"]["points_p25_median_p75"],dist["generated_T0p7"]["points_p25_median_p75"]]
        x=np.arange(3); med=np.asarray([v[1] for v in values]); low=med-np.asarray([v[0] for v in values]); high=np.asarray([v[2] for v in values])-med
        ax.errorbar(x,med,yerr=[low,high],fmt="o",capsize=5); ax.set_xticks(x,names,rotation=20); ax.set_ylabel("points"); ax.set_title(f"{model} · length p25/median/p75"); ax.grid(alpha=.2)
    fig.savefig(OUT/"visual_grids"/"length_distributions.png",dpi=150); plt.close(fig)

    rows=list(csv.DictReader((OUT/"teacher_forced_vs_free_running.csv").open(encoding="utf-8")))
    payload=json.loads((OUT/"teacher_forced_vs_free_running.json").read_text(encoding="utf-8"))
    exact={}
    for model in MODELS:
        groups=defaultdict(dict)
        for row in rows:
            if row["model"]==model: groups[(row["validation_id"],row["condition"],int(row["step"]))][row["mode"]]=row
        first={}
        for (vid,condition,step),pair in groups.items():
            if len(pair)!=2: continue
            a,b=pair["teacher_forced"],pair["free_running_modal"]
            if a["absolute_position_drift"] in ("","None") or b["absolute_position_drift"] in ("","None"): continue
            tf,fr=float(a["absolute_position_drift"]),float(b["absolute_position_drift"])
            if fr>max(25.0,1.5*tf): first[(vid,condition)]=min(step,first.get((vid,condition),10**9))
        values=list(first.values())
        exact[model]={"median_timestep":float(np.median(values)),"min_timestep":min(values),"max_timestep":max(values),"groups":len(values)}
    payload["exact_divergence_timestep"]=exact
    (OUT/"teacher_forced_vs_free_running.json").write_text(json.dumps(json_ready(payload),indent=2),encoding="utf-8")


def main():
    if OUT.exists() and any(OUT.iterdir()): raise FileExistsError(f"Refusing to overwrite {OUT}")
    (OUT/"generation_traces").mkdir(parents=True); (OUT/"visual_grids").mkdir()
    train=load_raw("data/raw/train.pkl"); val=load_raw("data/raw/val.pkl")
    manifest=json.loads(Path("runs/visual_benchmark_round1/selected_validation_cases.json").read_text(encoding="utf-8"))
    cases=[val[int(c["validation_index"])] for c in manifest["cases"]]
    if [s["id"] for s in cases] != [c["id"] for c in manifest["cases"]]: raise RuntimeError("Fixed VAL IDs changed")
    models={m:load_model(p) for m,p in CHECKPOINTS.items()}; stats=load_geometry(); torch.set_num_threads(1)
    tf_rows,tf_summary=teacher_free_diagnostics(models,cases)
    records,sweep=sampling_sweep(models,cases,stats)
    eos=eos_diagnostics(tf_rows,train,val,records)
    prefix=prefix_diagnostics(records)
    generation_traces(models,cases)
    render_posthoc()
    raw={"checkpoints":{m:str(p) for m,p in CHECKPOINTS.items()},"case_ids":[s["id"] for s in cases],
         "teacher_free":tf_summary,"eos":eos,"sampling_sweep":sweep,"prefix":prefix}
    (OUT/"diagnostic_summary.json").write_text(json.dumps(json_ready(raw),indent=2),encoding="utf-8")
    print(json.dumps({"status":"completed","output":str(OUT),"trace_files":len(list((OUT/'generation_traces').glob('*.csv'))),"grids":len(list((OUT/'visual_grids').glob('*.png')))},indent=2))


if __name__ == "__main__": main()
