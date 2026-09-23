"""Homogeneous evaluation for the final Model-E scheduled-sampling trial."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from scripts.diagnose_round2 import hierarchical_free, hierarchical_teacher, real_suffix_events
from scripts.evaluate_eos_weighting import generation_metrics, plot_panels, teacher_metrics
from sketchlab.data import load_raw
from sketchlab.evaluation import distances, prefix_count
from sketchlab.generation import load_model, sample
from sketchlab.geometry import load_geometry
from sketchlab.orchestration import json_ready

OUT = Path("runs/scheduled_sampling_experiment")
CHECKPOINTS = {"control": OUT/"control/E/last.pt", "scheduled": OUT/"scheduled/E/last.pt"}
CONDITIONS = [("1stroke", "one"), ("2strokes", "two"), ("25pct", .25),
              ("50pct", .5), ("75pct", .75)]
TARGET_STEPS = (1, 2, 5, 10, 20, 50)
SEED, TEMP, MAX_POINTS, MAX_STROKES = 73100, .9, 384, 64


def mean(values):
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def median(values):
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.median(values)) if values else None


def drift_diagnostics(models, cases):
    rows, first = [], {v: [] for v in models}
    for variant, model in models.items():
        for ci, (condition, fraction) in enumerate(CONDITIONS):
            for vi, case in enumerate(cases):
                count = prefix_count(len(case["strokes"]), fraction)
                truth = real_suffix_events(case, count, "E")
                tf = hierarchical_teacher(model, case, count)
                fr = hierarchical_free(model, case["strokes"][:count], truth,
                    seed=SEED+ci*100+vi, temperature=0, latent_mean=True)
                for mode, source in (("teacher_forced", tf), ("free_running_modal", fr)):
                    for r in source:
                        rows.append({"variant":variant, "mode":mode, "condition":condition,
                            "validation_id":case["id"], "timestep":r["step"]+1,
                            "drift":r["absolute_position_drift"], "coordinate_error":r["coordinate_error"],
                            "inside_canvas":r["inside_canvas"], "state_correct":r["state_correct"]})
                by_tf = {r["step"]+1:r for r in tf if r["absolute_position_drift"] is not None}
                by_fr = {r["step"]+1:r for r in fr if r["absolute_position_drift"] is not None}
                divergence = None
                for step in sorted(set(by_tf)&set(by_fr)):
                    a, b = by_tf[step]["absolute_position_drift"], by_fr[step]["absolute_position_drift"]
                    if b > max(25.0, 1.5*a): divergence=step; break
                if divergence is not None: first[variant].append(divergence)
    summaries=[]
    for variant in models:
        for mode in ("teacher_forced", "free_running_modal"):
            for step in TARGET_STEPS:
                selected=[r for r in rows if r["variant"]==variant and r["mode"]==mode and r["timestep"]==step]
                summaries.append({"variant":variant,"mode":mode,"timestep":step,"samples":len(selected),
                    "median_absolute_drift":median(r["drift"] for r in selected),
                    "mean_absolute_drift":mean(r["drift"] for r in selected),
                    "median_coordinate_error":median(r["coordinate_error"] for r in selected),
                    "inside_canvas":mean(r["inside_canvas"] for r in selected),
                    "state_accuracy":mean(r["state_correct"] for r in selected)})
    divergence={v:{"median_first_strong_divergence_timestep":median(x),
                   "min":min(x) if x else None,"max":max(x) if x else None,"groups":len(x)} for v,x in first.items()}
    return summaries, divergence


def generate(models, cases, stats):
    records=[]
    for variant, model in models.items():
        for ci,(condition,fraction) in enumerate(CONDITIONS):
            for vi,case in enumerate(cases):
                count=prefix_count(len(case["strokes"]),fraction); prefix=case["strokes"][:count]; truth=case["strokes"][count:]
                group=[]
                for k in range(2):
                    seed=SEED+ci*10000+vi*100+k
                    strokes,info=sample(model,prefix=prefix,seed=seed,decoder_seed=seed+900000,
                        temperature=TEMP,max_points=MAX_POINTS,max_strokes=MAX_STROKES,return_info=True)
                    metric=generation_metrics(strokes,count,info,stats,truth)
                    records.append({"variant":variant,"condition":condition,"case":vi+1,"validation_id":case["id"],
                        "sample":k,"prefix_count":count,"metrics":metric,"strokes":strokes})
                    group.append(strokes[count:])
                diversity=distances(group[0],group[1])["chamfer"]
                for r in records[-2:]: r["metrics"]["diversity_chamfer"]=diversity
        for k in range(8):
            seed=SEED+90000+k
            strokes,info=sample(model,prefix=[],seed=seed,decoder_seed=seed+900000,
                temperature=TEMP,max_points=MAX_POINTS,max_strokes=MAX_STROKES,return_info=True)
            records.append({"variant":variant,"condition":"random","case":k+1,"validation_id":None,"sample":0,
                "prefix_count":0,"metrics":{"eos":bool(info["ended_by_eos"]),"cap":bool(info["capped"]),
                "termination":info["termination"],"points":sum(len(s) for s in strokes),"strokes":len(strokes),
                "empty":not strokes,"premature":sum(len(s) for s in strokes)<5},"strokes":strokes})
    return records


def aggregate(records, variant, condition=None):
    selected=[r["metrics"] for r in records if r["variant"]==variant and
              (condition is None and r["condition"]!="random" or r["condition"]==condition)]
    return {"samples":len(selected),"eos_rate":mean(x["eos"] for x in selected),"cap_rate":mean(x["cap"] for x in selected),
        "median_generated_points":median(x["points"] for x in selected),"median_generated_strokes":median(x["strokes"] for x in selected),
        "completion_valid_rate":mean(x.get("completion_valid") for x in selected),
        "premature_rate":mean(x["premature"] for x in selected),"empty_rate":mean(x["empty"] for x in selected),
        "canvas_validity":mean(x.get("canvas") for x in selected),"mean_drift":mean(x.get("drift") for x in selected),
        "mean_chamfer":mean(x.get("chamfer") for x in selected),"mean_hausdorff":mean(x.get("hausdorff") for x in selected),
        "intra_jump_fraction":mean(x.get("intra_jump_fraction") for x in selected),
        "diversity_chamfer":mean(x.get("diversity_chamfer") for x in selected)}


def render(records,cases):
    path=OUT/"visual_grids"; path.mkdir(exist_ok=True)
    for condition,fraction in CONDITIONS:
        case=cases[0]; count=prefix_count(len(case["strokes"]),fraction)
        panels=[("GT",case["strokes"],count)]
        for variant in models_order:
            r=next(x for x in records if x["variant"]==variant and x["condition"]==condition and x["case"]==1 and x["sample"]==0)
            m=r["metrics"]; panels.append((f"{variant}\n{m['points']} pts · {m['termination']} · valid {int(m['completion_valid'])}",r["strokes"],count))
        plot_panels(panels,path/f"{condition}_T0p9.png")
    panels=[]
    for variant in models_order:
        r=next(x for x in records if x["variant"]==variant and x["condition"]=="random" and x["case"]==1)
        m=r["metrics"]; panels.append((f"{variant}\n{m['points']} pts · {m['termination']}",r["strokes"],0))
    plot_panels(panels,path/"random_T0p9.png")


models_order=("control","scheduled")
def main():
    torch.set_num_threads(1); val=load_raw("data/raw/val.pkl")
    manifest=json.loads(Path("runs/visual_benchmark_round1/selected_validation_cases.json").read_text(encoding="utf-8"))
    cases=[val[int(x["validation_index"])] for x in manifest["cases"]]
    models={v:load_model(p) for v,p in CHECKPOINTS.items()}
    teacher={v:teacher_metrics(m,cases) for v,m in models.items()}
    drift,divergence=drift_diagnostics(models,cases)
    records=generate(models,cases,load_geometry())
    termination={v:{"overall":aggregate(records,v),
                    "by_condition":{c:aggregate(records,v,c) for c,_ in CONDITIONS},
                    "random":aggregate(records,v,"random")} for v in models}
    training={v:json.loads((OUT/v/"E/summary.json").read_text(encoding="utf-8")) for v in models}
    scoreboard=[]
    for v in models:
        t=teacher[v]; g=termination[v]["overall"]; tr=training[v]
        scoreboard.append({"variant":v,"updates":tr["steps"],"teacher_forced_reconstruction":tr["validation"]["reconstruction"],
            "teacher_forced_kl":tr["validation"]["KL_loss"],"stroke_end_precision":t["stroke_end"]["precision"],
            "stroke_end_recall":t["stroke_end"]["recall"],"eos_precision":t["eos"]["precision"],"eos_recall":t["eos"]["recall"],
            "eos_probability_final":t["eos_probability_final"],"eos_probability_nonfinal":t["eos_probability_nonfinal"],
            "eos_probability_gap":t["eos_probability_gap"],**g,
            "median_first_strong_divergence_timestep":divergence[v]["median_first_strong_divergence_timestep"],
            "training_time_seconds":tr["training_time_seconds"],"peak_vram_mib":tr["peak_vram_mib"],"checkpoint":str(CHECKPOINTS[v])})
    with (OUT/"scoreboard.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(scoreboard[0]));w.writeheader();w.writerows(scoreboard)
    (OUT/"scoreboard.json").write_text(json.dumps(json_ready(scoreboard),indent=2),encoding="utf-8")
    with (OUT/"drift_comparison.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(drift[0]));w.writeheader();w.writerows(drift)
    (OUT/"termination_metrics.json").write_text(json.dumps(json_ready({"temperature":TEMP,"teacher_forced":teacher,
        "free_running":termination,"divergence":divergence,"protocol":{"case_ids":[c["id"] for c in cases],
        "conditions":[c[0] for c in CONDITIONS],"samples_per_case":2,"random_samples":8,"seed":SEED}}),indent=2),encoding="utf-8")
    render(records,cases)
    print(json.dumps({"status":"completed","rows":len(scoreboard),"drift_rows":len(drift),"grids":6},indent=2))

if __name__=="__main__":main()
