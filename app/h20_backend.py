"""Inference adapter for the final TRAIN-prototype CVAE checkpoint in the local demo."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from sketchlab.data import load_raw, sha256_file
from sketchlab.evaluation import prefix_count, prefix_preserved
from sketchlab.generation import render
from sketchlab.geometry import load_geometry, postprocess_candidate, validate_candidate
from sketchlab.models.common import validate_sketch
from sketchlab.models.exemplar_cvae import ExemplarCVAE
from sketchlab.models.model_g import PARTS
from sketchlab.orchestration import json_ready
from sketchlab.ranking import rank_candidates
from app.demo_geometry import feature, register, register_fit, resample


CONDITIONS = (('1STROKE', 'one'), ('25PCT', .25), ('50PCT', .5), ('75PCT', .75))
SHORTLIST = 48
ROOT = Path(__file__).resolve().parents[1]


class H20Handle:
    def __init__(self, checkpoint: Path, device: str = 'cpu'):
        self.variant = 'H13' if checkpoint.parent.name == 'H13' or 'H13_training' in checkpoint.parts else 'H20'
        self.canvas = 256 if self.variant == 'H13' else 512
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        if not isinstance(saved, dict) or not {'architecture', 'model_state'} <= saved.keys():
            raise ValueError('This is not an H20 checkpoint')
        self.model = ExemplarCVAE(**saved['architecture']).to(device)
        self.model.load_state_dict(saved['model_state'], strict=True)
        self.model.eval()
        self.device = device
        if 'artifacts' in checkpoint.parts:
            bank_root = checkpoint.parent
        else:
            bank_root = ROOT / ('runs/H13_training' if self.variant == 'H13' else 'runs/H15_training')
        manifest = json.loads((bank_root / 'FEATURE_MANIFEST.json').read_text(encoding='utf-8'))
        if manifest['train_sha256'] != sha256_file(ROOT / 'data/raw/train.pkl') or manifest.get('raster_canvas', 256) != self.canvas:
            raise ValueError('Prototype bank does not match the TRAIN source')
        with np.load(bank_root / 'features.npz') as bank:
            indices = bank['library_indices'].copy()
            self.full = bank['library_full_compact'].copy()
        train = load_raw(ROOT / 'data/raw/train.pkl')
        self.library = [train[int(i)] for i in indices]
        self.prefix_banks = {}

    def bank(self, condition: str, fraction):
        if condition not in self.prefix_banks:
            sketches = [s['strokes'][:prefix_count(len(s['strokes']), fraction)] for s in self.library]
            matrix = np.stack([feature(s, self.canvas) for s in sketches]).astype(np.float32)
            norms = matrix.sum(-1)
            parts = np.stack([[float(part in sample['parts'][:prefix_count(len(sample['strokes']), fraction)])
                               for part in PARTS] for sample in self.library]).astype(np.float32)
            self.prefix_banks[condition] = (matrix, norms, parts)
        return self.prefix_banks[condition]


def load_model(checkpoint: Path, device: str = 'cpu') -> H20Handle:
    return H20Handle(Path(checkpoint), device)


@torch.inference_mode()
def _generate(handle: H20Handle, prefix: list[np.ndarray], seed: int, temperature: float,
              max_points: int, max_strokes: int):
    # The training curriculum used four prefix fractions. For an arbitrary
    # drawing, choose the closest phase from the number of observed strokes.
    condition, fraction = CONDITIONS[0 if len(prefix) <= 1 else 1 if len(prefix) <= 4
                                     else 2 if len(prefix) <= 8 else 3]
    matrix, norms, parts = handle.bank(condition, fraction)
    query = feature(prefix, handle.canvas)
    intersection = matrix @ query
    scores = intersection / (norms + query.sum() - intersection + 1e-5)
    if not prefix:
        indices = np.random.default_rng(seed).choice(len(handle.library), SHORTLIST, replace=False)
    else:
        indices = np.argsort(-scores, kind='stable')[:SHORTLIST]
    candidates = torch.from_numpy(handle.full[indices][None]).to(handle.device)
    prefix_tensor = torch.from_numpy(query[None]).to(handle.device)
    score_tensor = torch.from_numpy(scores[indices][None]).to(handle.device)
    part_tensor = torch.from_numpy(parts[indices][None]).to(handle.device) if handle.model.part_dim else None
    rng = np.random.default_rng(seed)
    noise = torch.from_numpy(rng.standard_normal((1, handle.model.latent_dim)).astype(np.float32)).to(handle.device)
    choice_noise = torch.from_numpy(rng.random((1, len(indices))).astype(np.float32)).to(handle.device)
    choice, _ = handle.model.select(prefix_tensor, candidates, score_tensor, noise, choice_noise,
                                    gumbel_scale=.2 * temperature / .6, candidate_parts=part_tensor)
    exemplar = handle.library[int(indices[int(choice.item())])]
    m = prefix_count(len(exemplar['strokes']), fraction) if prefix else 0
    if prefix:
        registration = register if handle.variant == 'H13' else register_fit
        suffix = registration(exemplar['strokes'][:m], prefix, exemplar['strokes'][m:])
    else:
        suffix = exemplar['strokes']
    cap = min(24, max_strokes, max_points // 16)
    capped = len(suffix) > cap
    output = [s.copy() for s in prefix] + [resample(s) for s in suffix[:cap]]
    if not prefix_preserved(prefix, output):
        raise AssertionError('H20 changed the observed prefix')
    info = {'model': handle.variant, 'seed': seed, 'temperature': temperature,
            'termination': 'max_points' if capped else 'eos', 'ended_by_eos': not capped,
            'capped': capped, 'generated_points': 16 * min(len(suffix), cap),
            'generated_strokes': min(len(suffix), cap), 'prototype_id': exemplar['id'],
            'prefix_condition': condition}
    return output, info


def sample_multiple(handle: H20Handle, prefix=None, n_candidates=20, top_k=6, seed=42,
                    temperature=.6, max_points=384, max_strokes=64, validate=True,
                    postprocess=True, rerank=True):
    prefix = [s.copy() for s in validate_sketch([] if prefix is None else prefix)]
    stats = load_geometry(ROOT / 'data/processed/geometry.json') if validate else None
    start = time.perf_counter()
    candidates = []
    for index in range(n_candidates):
        raw, info = _generate(handle, prefix, seed + index, temperature, max_points, max_strokes)
        if validate and postprocess:
            candidate = postprocess_candidate(raw, stats, prefix_count=len(prefix), termination=info)
            candidate['validation'] = json_ready(candidate['validation'])
        else:
            candidate = {'raw_output': [s.copy() for s in raw],
                         'postprocessed_output': [s.copy() for s in raw],
                         'validation': json_ready(validate_candidate(raw, stats, prefix_count=len(prefix), termination=info))
                         if validate else {'valid': None, 'score': None, 'warnings': ['RAW: geometry not validated'],
                                           'penalties': {}, 'corrections': []}}
        if not prefix_preserved(prefix, candidate['postprocessed_output']):
            raise AssertionError('Postprocessing changed the observed prefix')
        candidate.update(id=index, seed=seed + index, termination=info, prefix_count=len(prefix))
        candidates.append(candidate)
    if validate and rerank:
        ranking = rank_candidates(candidates, top_k, stats)
        selected, ranking_report = ranking['selected'], ranking['report']
    elif validate:
        selected = [c for c in candidates if c['validation']['valid']][:top_k]
        ranking_report = {'method': 'sampling order among geometrically valid candidates'}
    else:
        selected = candidates[:top_k]
        ranking_report = {'method': 'RAW diagnostic mode'}
    report = {'n_candidates': len(candidates), 'top_k_requested': top_k,
              'n_selected': len(selected), 'n_valid': sum(c['validation']['valid'] is True for c in candidates) if validate else None,
              'validation_enabled': validate, 'postprocessing_enabled': validate and postprocess,
              'reranking_enabled': validate and rerank, 'prefix_exact': True,
              'seconds': time.perf_counter() - start, 'ranking': ranking_report,
              'limitations': 'H20 uses aligned TRAIN prototypes; prefix phase for a user drawing is estimated from stroke count.'}
    return {'selected': selected, 'candidates': candidates, 'report': report}
