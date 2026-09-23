"""Inference-only acceptance smoke for persisted tiny F checkpoints."""
import json
import time
from pathlib import Path

import numpy as np
import torch

from sketchlab.checkpointing import load_checkpoint
from sketchlab.data import load_raw
from sketchlab.evaluation import prefix_count, prefix_preserved
from sketchlab.generation import load_model, sample, sample_multiple
from sketchlab.orchestration import json_ready
from sketchlab.rendering import render_grid


def main():
    root=Path('runs/model_f_smoke'); out=root/'inference'
    if (out/'summary.json').exists(): raise FileExistsError('Persisted smoke already exists')
    out.mkdir(parents=True,exist_ok=True); torch.set_num_threads(1)
    path=root/'composition/F/last.pt'; model=load_model(path)
    ae,_=load_checkpoint(root/'stroke_ae/F/last.pt')
    assert all(torch.equal(v,model.stroke_encoder.state_dict()[k]) for k,v in ae.stroke_encoder.state_dict().items())
    assert all(torch.equal(v,model.stroke_decoder.state_dict()[k]) for k,v in ae.stroke_decoder.state_dict().items())
    val=load_raw('data/raw/val.pkl',validate=False)
    manifest=json.loads(Path('runs/visual_benchmark_round1/selected_validation_cases.json').read_text(encoding='utf-8'))
    fixed=manifest['cases'][:2]; cases=[val[c['validation_index']] for c in fixed]
    assert [s['id'] for s in cases]==[c['id'] for c in fixed]
    records=[]; tick=time.perf_counter()
    random=[]; titles=[]
    for k in range(4):
        strokes,info=sample(model,seed=73100+k,temperature=.9,max_points=96,max_strokes=6,return_info=True)
        assert all(np.isfinite(s).all() for s in strokes)
        random.append(strokes); titles.append(f"F tiny z{k+1}: {info['termination']} / {info['generated_strokes']} strokes")
        records.append({'condition':'random','prefix_exact':True,'info':info,'strokes':strokes})
    render_grid(random,out/'random.png',titles=titles,ncols=4)
    panels=[]; titles=[]; counts=[]
    for ci,case in enumerate(cases):
        count=prefix_count(len(case['strokes']),.25); prefix=case['strokes'][:count]
        panels.append(case['strokes']); titles.append(f"GT fixed VAL {case['id']}"); counts.append(count)
        for k in range(3):
            strokes,info=sample(model,prefix=prefix,seed=73100+ci*100+k,decoder_seed=973100+ci*100,
                temperature=.9,max_points=96,max_strokes=6,return_info=True)
            assert prefix_preserved(prefix,strokes)
            assert all(a.dtype==b.dtype and a.tobytes()==b.tobytes() for a,b in zip(prefix,strokes))
            assert all(np.isfinite(s).all() for s in strokes)
            panels.append(strokes); titles.append(f"F tiny z{k+1}: {info['termination']}"); counts.append(count)
            records.append({'condition':'25pct','validation_id':case['id'],'prefix_exact':True,'info':info,'strokes':strokes})
    render_grid(panels,out/'completion.png',titles=titles,prefix_counts=counts,ncols=4)
    prefix=cases[0]['strokes'][:1]
    ranked=sample_multiple(model,prefix=prefix,n_candidates=3,top_k=2,seed=74001,max_points=96,max_strokes=6)
    (out/'prefix.json').write_text(json.dumps(json_ready(prefix)),encoding='utf-8')
    payload={'checkpoint':str(path),'device':'cpu','seconds':time.perf_counter()-tick,
        'finite_outputs':True,'prefix_bit_exact':True,'stroke_ae_frozen_unchanged':True,'point_autoregression':False,
        'samples':len(records),'learned_eos':sum(r['info']['ended_by_eos'] for r in records),
        'caps':sum(r['info']['capped'] for r in records),
        'ranking_pipeline':ranked['report'],'parameter_count':sum(p.numel() for p in model.parameters()),
        'stroke_ae_parameters':sum(p.numel() for n,p in model.named_parameters() if n.startswith(('stroke_encoder.','stroke_decoder.'))),
        'meaning':'Tiny functional smoke; not a quality benchmark. Early EOS and bad shapes are expected.'}
    (out/'summary.json').write_text(json.dumps(json_ready(payload),indent=2,allow_nan=False),encoding='utf-8')
    (out/'samples.json').write_text(json.dumps(json_ready(records),indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(payload,indent=2))

if __name__=='__main__':main()
