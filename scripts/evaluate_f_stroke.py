"""Evaluate the F representation stage on raw held-out strokes."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sketchlab.checkpointing import load_checkpoint
from sketchlab.data import load_raw
from sketchlab.evaluation import distances
from sketchlab.rendering import render_grid


@torch.inference_mode()
def evaluate(model, strokes, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    if (output/"metrics.json").exists(): raise FileExistsError(output/"metrics.json")
    model.eval(); view = model.view([strokes]); embeddings = model.encode_view(view)[0]
    shape = model.stroke_decoder(embeddings, view["t"]).cpu().numpy()*model.scale
    anchors = view["anchors"][0].cpu().numpy()*model.scale
    target = view["relative"][0].cpu().numpy()*model.scale
    rows=[]; panels=[]; titles=[]
    for i,(raw,relative,anchor,truth) in enumerate(zip(strokes,shape,anchors,target)):
        prediction = relative+anchor
        delta = np.diff(relative, n=2, axis=0)-np.diff(truth, n=2, axis=0)
        rows.append({"stroke":i, "raw_points":len(raw), "point_rmse":float(np.sqrt(np.mean((relative-truth)**2))),
            "endpoint_error":float(np.linalg.norm(prediction[-1]-raw[-1])),
            "second_difference_rmse":float(np.sqrt(np.mean(delta**2))) if len(delta) else 0.,
            **distances([raw],[prediction])})
        if i<8:
            panels.extend([[raw],[prediction]]); titles.extend([f"raw {i+1} ({len(raw)} pts)",f"F_STROKE_AE {i+1}"])
    summary={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in {"stroke","raw_points"}}
    payload={"stage":"F_STROKE_AE", "count":len(rows),"coordinate_units":"canvas pixels", "metrics":summary,"strokes":rows}
    (output/"metrics.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    render_grid(panels,output/"stroke_reconstruction.png",titles=titles,ncols=4)
    return payload


def main():
    p=argparse.ArgumentParser(); p.add_argument("checkpoint"); p.add_argument("--output",required=True)
    p.add_argument("--count",type=int,default=64); args=p.parse_args()
    if args.count<1: p.error("count must be positive")
    torch.set_num_threads(1); model,_=load_checkpoint(args.checkpoint)
    if model.model_name != "F": p.error("F checkpoint required")
    val=load_raw("data/raw/val.pkl",validate=False)
    fixed=json.loads(Path("runs/visual_benchmark_round1/selected_validation_cases.json").read_text(encoding="utf-8"))["cases"]
    strokes=[stroke for c in fixed for stroke in val[c["validation_index"]]["strokes"]][:args.count]
    print(json.dumps(evaluate(model,strokes,args.output)["metrics"],indent=2))

if __name__=="__main__":main()
