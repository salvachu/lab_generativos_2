"""Independently implemented semantic-plan CVAE; see ARCHITECTURE_DECISION.md.

No future stroke, target part, or target count is ever a decoder input.
Only the variational posterior sees full geometry during training.
"""
from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from sketchlab.losses import gaussian_kl, gaussian_parameters, reparameterize
from sketchlab.stroke_view import VIEW_VERSION, sketch_view
from .common import prefix_counts_checked
from .model_f import StrokeEncoder, StrokeDecoder, transformer, position_encoding


PARTS = ['arms', 'beak', 'body', 'details', 'ears', 'eye', 'feet', 'fin', 'hair',
         'hands', 'head', 'horns', 'initial', 'legs', 'mouth', 'nose', 'paws', 'tail', 'wings']


def decoder(width, heads, layers, dropout):
    layer = nn.TransformerDecoderLayer(width, heads, 4*width, dropout,
                                      activation='gelu', batch_first=True, norm_first=True)
    return nn.TransformerDecoder(layer, layers, norm=nn.LayerNorm(width))


def per_sketch(values, mask):
    """Equal sketch weight, defined as zero for a sketch with no target events."""
    return ((values * mask).sum(-1) / mask.sum(-1).clamp_min(1)).mean()


def stroke_roughness(curves):
    """Dimensionless second difference at the common 16-point stroke view."""
    step=curves.diff(dim=-2)
    mean_step=step.norm(dim=-1).mean(-1).clamp_min(1e-3)
    return step.diff(dim=-2).norm(dim=-1).mean(-1)/mean_step


def stroke_primitive_complexity(curves):
    """Curvature reversals and path excess; a steady arc is allowed."""
    step=curves.diff(dim=-2)
    unit=step/(step.norm(dim=-1,keepdim=True)+1e-4)
    turn=unit[...,:-1,0]*unit[...,1:,1]-unit[...,:-1,1]*unit[...,1:,0]
    reversals=F.relu(-turn[...,:-1]*turn[...,1:]).mean(-1)
    arc=step.norm(dim=-1).sum(-1)
    endpoint=(curves[...,-1,:]-curves[...,0,:]).norm(dim=-1)
    ratio=(arc/(endpoint+.02)).clamp(max=8.)
    return reversals,ratio


def occupancy_grid(resolution, device, dtype):
    """Pixel centers on the same normalized [-4,508] canvas as evaluation."""
    coordinates = (torch.arange(resolution, device=device, dtype=dtype) + .5) / resolution
    coordinates = coordinates * 2. - 4. / 256.
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing='ij')
    return torch.stack([xx, yy], -1)


def future_occupancy(view, resolution):
    """Soft GT raster from future vector points; no raster enters inference."""
    points = view['relative'] + view['anchors'][:, :, None]
    grid = occupancy_grid(resolution, points.device, points.dtype)
    squared = (points[:, :, :, None, None] - grid).square().sum(-1)
    marks = torch.exp(-squared / (2 * .07**2)) * view['stroke_mask'][:, :, None, None, None]
    return 1. - torch.exp(-marks.sum((1, 2)))


def plan_occupancy(plan, resolution):
    """Differentiable coarse occupancy implied by predicted part boxes."""
    boxes = plan['boxes']
    grid = occupancy_grid(resolution, boxes.device, boxes.dtype)
    centers = boxes[:, :, None, None, :2]
    spread = boxes[:, :, None, None, 2:] / 2 + .07
    squared = ((grid - centers) / spread).square().sum(-1)
    coverage = torch.exp(-.5 * squared) * plan['presence'].sigmoid()[:, :, None, None]
    return 1. - torch.prod(1. - coverage.clamp(max=1. - 1e-6), dim=1)


class ModelG(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = {"stroke_samples":16, "stroke_embedding_dim":32, "attention_heads":4,
                       "stroke_encoder_layers":2, "composition_layers":2, "max_stroke_slots":211,
                       "part_vocabulary":PARTS, "anchor_sigma":.15, "code_sigma":.35,
                       "plan_sigma":.2, "curve_weight":5.,
                       "lambda_occ":0., "lambda_parts":1., "lambda_count":1.,
                       "lambda_global":1., "lambda_presence":1.,
                       "raster_resolution":0, "extent_prediction":False,
                       "lambda_extent":1., "code_mixtures":1,
                       "code_mixture_sigma":.18,
                       "lambda_smoothness":0., "lambda_primitive":0.,
                       "group_role_conditioning":False,"lambda_relations":0.,
                       "group_anchor_planner":False,"max_group_slots":20,
                       "group_box_mixtures":1,
                       "lambda_group_box":1.5,"lambda_group_assignment":1.5,
                       "lambda_group_presence":1.,"lambda_group_count":.5, **config}
        self.model_name = 'G'
        self.scale = float(self.config['scale'])
        self.hidden_dim, self.latent_dim = self.config['hidden_dim'], self.config['latent_dim']
        self.stroke_samples, self.embedding_dim = self.config['stroke_samples'], self.config['stroke_embedding_dim']
        self.max_slots = self.config['max_stroke_slots']
        self.parts = tuple(self.config['part_vocabulary'])
        self.part_lookup = {part:i for i,part in enumerate(self.parts)}
        h, d, p = self.hidden_dim, self.embedding_dim, len(self.parts)
        heads, layers, dropout = self.config['attention_heads'], self.config['composition_layers'], self.config['dropout']
        if h % heads or self.stroke_samples < 2 or self.max_slots < 1 or len(set(self.parts)) != p or not p:
            raise ValueError('Invalid G dimensions or part vocabulary')
        if any(not math.isfinite(self.config[k]) or self.config[k] <= 0
               for k in ('anchor_sigma','code_sigma','plan_sigma','curve_weight')):
            raise ValueError('G likelihood scales and curve weight must be positive')
        if any(not math.isfinite(self.config[k]) or self.config[k] < 0
               for k in ('lambda_occ','lambda_parts','lambda_count','lambda_global','lambda_presence',
                         'lambda_extent','lambda_smoothness','lambda_primitive','lambda_relations',
                         'lambda_group_box','lambda_group_assignment','lambda_group_presence',
                         'lambda_group_count')):
            raise ValueError('G loss weights must be finite and nonnegative')
        if not 1 <= self.config['code_mixtures'] <= 8 or not math.isfinite(self.config['code_mixture_sigma']) or self.config['code_mixture_sigma'] <= 0:
            raise ValueError('G code mixture count and sigma must be valid')
        if self.config['lambda_occ'] and self.config['raster_resolution'] < 4:
            raise ValueError('Occupancy loss needs raster_resolution >= 4')
        self.stroke_encoder = StrokeEncoder(h, d, heads, self.config['stroke_encoder_layers'], dropout)
        self.stroke_decoder = StrokeDecoder(d, h)
        self.input_projection = nn.Linear(d+6, h)
        self.context_encoder = transformer(h, heads, layers, dropout)
        self.bos = nn.Parameter(torch.zeros(1, 1, h))
        self.prior_head = nn.Linear(h+1, 2*self.latent_dim)
        self.posterior_head = nn.Linear(2*(h+1), 2*self.latent_dim)
        self.condition = nn.Linear(h+1+self.latent_dim, h)
        self.part_queries = nn.Embedding(p, h)
        self.plan_decoder = decoder(h, heads, layers, dropout)
        self.box_head = nn.Linear(h, 4)
        self.presence_head = nn.Linear(h, 1)
        self.plan_geometry = nn.Linear(5, h)
        if self.config['group_anchor_planner']:
            g=self.config['max_group_slots']
            if not 1<=g<=self.max_slots:raise ValueError('Invalid max_group_slots')
            self.group_queries=nn.Embedding(g,h)
            self.group_decoder=decoder(h,heads,layers,dropout)
            if self.config['group_box_mixtures']>1:
                if self.config['group_box_mixtures']>16:raise ValueError('Too many group box mixtures')
                self.group_box_mixture_head=nn.Linear(h,self.config['group_box_mixtures']*5)
            else:
                self.group_box_head=nn.Linear(h,4)
            self.group_presence_head=nn.Linear(h,1)
            self.group_count_head=nn.Linear(h,1)
            self.group_geometry=nn.Linear(6,h)
            self.group_membership_head=nn.Linear(h,g)
            self.group_state_projection=nn.Linear(h,h)
        self.count_head = nn.Sequential(nn.Linear(h+1+self.latent_dim, h), nn.GELU(), nn.Linear(h, self.max_slots+1))
        self.stroke_queries = nn.Embedding(self.max_slots, h)
        self.slot_decoder = decoder(h, heads, layers, dropout)
        self.part_head = nn.Linear(h, p)
        if self.config['group_role_conditioning']:
            self.group_role = nn.Sequential(nn.Linear(h+3,h),nn.GELU(),nn.Linear(h,h))
        self.anchor_offset = nn.Linear(h, 2)
        self.code_head = nn.Sequential(nn.Linear(h+4, h), nn.GELU(), nn.Linear(h,d))
        if self.config['extent_prediction']:
            self.extent_head = nn.Sequential(nn.Linear(h+4,h),nn.GELU(),nn.Linear(h,1))
            nn.init.zeros_(self.extent_head[-1].weight)
            nn.init.constant_(self.extent_head[-1].bias,-2.5)
        if self.config['code_mixtures'] > 1:
            self.code_mixture_head = nn.Linear(h+4,self.config['code_mixtures']*(d+1))
        self.configure_training()

    @property
    def device(self): return next(self.parameters()).device

    @property
    def representation_metadata(self):
        return {'view_version':VIEW_VERSION, 'stroke_samples':self.stroke_samples,
                'parameterization':'normalized_arc_length', 'scale':self.scale,
                'canonical_unchanged':True, 'generation_points_per_stroke':self.stroke_samples,
                'composition':'semantic-group-anchor-plan-v1' if self.config['group_anchor_planner']
                    else 'semantic-plan-parallel-v1', 'max_stroke_slots':self.max_slots,
                'parts':list(self.parts)}

    def configure_training(self, stage='composition', freeze_stroke_ae=True):
        if stage != 'composition' or not freeze_stroke_ae:
            raise ValueError('G requires composition and the frozen pretrained local AE')
        self.training_stage = stage
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(not name.startswith(('stroke_encoder.','stroke_decoder.')))

    def train(self, mode=True):
        super().train(mode)
        self.stroke_encoder.eval(); self.stroke_decoder.eval()
        return self

    def view(self, sketches):
        return sketch_view(sketches, self.stroke_samples, self.scale, self.device)

    def encode_view(self, view):
        mask = view['stroke_mask']
        embedding = view['relative'].new_zeros((*mask.shape, self.embedding_dim))
        with torch.no_grad():
            if mask.any(): embedding[mask] = self.stroke_encoder(view['relative'][mask], view['t'])
        return embedding

    def context(self, prefixes):
        view = self.view(prefixes); code = self.encode_view(view)
        absolute = view['relative'] + view['anchors'][:,:,None]
        low, high = absolute.amin(-2), absolute.amax(-2)
        bbox = torch.cat([(high+low)/2, high-low], -1)
        features = self.input_projection(torch.cat([code, view['anchors'], bbox], -1))
        inputs = torch.cat([self.bos.expand(len(prefixes),-1,-1), features], 1)
        padding = F.pad(~view['stroke_mask'], (1,0), value=False)
        inputs = inputs + position_encoding(inputs.shape[1], self.hidden_dim, self.device, inputs.dtype)
        memory = self.context_encoder(inputs, src_key_padding_mask=padding)
        counts = view['stroke_mask'].sum(-1)
        summary = torch.cat([memory[:,0], (counts.float()/self.max_slots)[:,None]], -1)
        return {'memory':memory, 'padding':padding, 'summary':summary, 'counts':counts}

    def prior(self, context):
        return gaussian_parameters(self.prior_head(context['summary']))

    def posterior(self, sketches, context):
        full = self.context(sketches)
        return gaussian_parameters(self.posterior_head(torch.cat([full['summary'],context['summary']],-1)))

    def make_plan(self, context, z):
        condition = self.condition(torch.cat([context['summary'],z],-1))
        queries = self.part_queries.weight[None] + condition[:,None]
        state = self.plan_decoder(queries, context['memory'], memory_key_padding_mask=context['padding'])
        raw_box = self.box_head(state)
        boxes = torch.cat([raw_box[...,:2], F.softplus(raw_box[...,2:])+.001],-1)
        presence = self.presence_head(state).squeeze(-1)
        return {'state':state, 'boxes':boxes, 'presence':presence}

    def make_group_plan(self,context,z):
        condition=self.condition(torch.cat([context['summary'],z],-1))
        queries=self.group_queries.weight[None]+condition[:,None]
        state=self.group_decoder(queries,context['memory'],memory_key_padding_mask=context['padding'])
        if self.config['group_box_mixtures']>1:
            k=self.config['group_box_mixtures']
            mixture=self.group_box_mixture_head(state).reshape(*state.shape[:2],k,5)
            logits=mixture[...,0]
            raw_box=mixture[...,1:]
            components=torch.cat([raw_box[...,:2],F.softplus(raw_box[...,2:])+.001],-1)
            if self.training:
                choice=F.gumbel_softmax(logits,tau=1.,hard=True,dim=-1)
            else:
                choice=F.one_hot(logits.argmax(-1),num_classes=k).to(components.dtype)
            boxes=(choice[...,None]*components).sum(-2)
        else:
            raw_box=self.group_box_head(state)
            boxes=torch.cat([raw_box[...,:2],F.softplus(raw_box[...,2:])+.001],-1)
        presence=self.group_presence_head(state).squeeze(-1)
        count=F.softplus(self.group_count_head(state).squeeze(-1))
        result={'state':state,'boxes':boxes,'presence':presence,'count':count}
        if self.config['group_box_mixtures']>1:
            result.update(box_components=components,box_logits=logits)
        return result

    def decode_plan(self, context, z, plan=None, group_plan=None):
        plan = self.make_plan(context,z) if plan is None else plan
        group=(self.make_group_plan(context,z) if group_plan is None else group_plan) if self.config['group_anchor_planner'] else None
        condition = self.condition(torch.cat([context['summary'],z],-1))
        length = self.max_slots + int(context['counts'].max()) + 1
        positions = position_encoding(length, self.hidden_dim,self.device,z.dtype)
        indices = context['counts'][:,None]+torch.arange(self.max_slots,device=self.device)[None]
        queries = self.stroke_queries.weight[None]+condition[:,None]+positions[indices]
        plan_tokens = plan['state']+self.plan_geometry(torch.cat([plan['boxes'],plan['presence'].sigmoid()[...,None]],-1))
        memory = torch.cat([context['memory'],plan_tokens],1)
        padding = F.pad(context['padding'],(0,len(self.parts)),value=False)
        if group is not None:
            group_tokens=group['state']+self.group_geometry(torch.cat([
                group['boxes'],group['presence'].sigmoid()[...,None],
                (group['count']/self.max_slots)[...,None]],-1))
            memory=torch.cat([memory,group_tokens],1)
            padding=F.pad(padding,(0,self.config['max_group_slots']),value=False)
        state = self.slot_decoder(queries,memory,memory_key_padding_mask=padding)
        group_logits=None
        if group is not None:
            group_logits=self.group_membership_head(state)
            group_assignments=(group_logits*3.).softmax(-1)
            group_boxes=group_assignments@group['boxes']
            state=state+self.group_state_projection(group_assignments@group['state'])
        part_logits = self.part_head(state)
        assignments = part_logits.softmax(-1)
        assigned_boxes = assignments @ plan['boxes']
        if self.config['group_role_conditioning']:
            # Soft within-part role and shared part state; inferred from prefix and z only.
            cumulative=assignments.cumsum(1)
            group_total=assignments.sum(1,keepdim=True).clamp_min(1.)
            ordinal=(assignments*cumulative).sum(-1,keepdim=True)/self.max_slots
            fraction=(assignments*(cumulative/group_total)).sum(-1,keepdim=True)
            size=(assignments*group_total).sum(-1,keepdim=True)/self.max_slots
            shared=assignments @ plan['state']
            state=state+self.group_role(torch.cat([shared,ordinal,fraction,size],-1))
        anchor_boxes=group_boxes if group is not None else assigned_boxes
        anchor = anchor_boxes[...,:2] + .5*anchor_boxes[...,2:]*torch.tanh(self.anchor_offset(state))
        code_input = torch.cat([state, anchor_boxes],-1)
        codes = self.code_head(code_input)
        count_logits = self.count_head(torch.cat([context['summary'],z],-1))
        result = {'anchors':anchor,'codes':codes,'part_logits':part_logits,
                  'count_logits':count_logits,'plan':plan}
        if group is not None:
            result.update(group_plan=group,group_logits=group_logits,
                          assigned_group_boxes=anchor_boxes)
        if self.config['extent_prediction']:
            result['extent_log'] = self.extent_head(code_input).squeeze(-1)
        if self.config['code_mixtures'] > 1:
            k=self.config['code_mixtures']
            mixture=self.code_mixture_head(code_input)
            result['code_logits']=mixture[...,:k]
            result['code_means']=codes[...,None,:]+mixture[...,k:].reshape(*codes.shape[:-1],k,self.embedding_dim)
        return result

    @staticmethod
    def scale_curves(curves, extent_log):
        """Set local curve extent without moving its anchored first point."""
        diameter=torch.linalg.norm(curves.amax(-2)-curves.amin(-2),dim=-1)
        desired=extent_log.clamp(-5.,.5).exp()
        factor=(desired/(diameter+1e-4)).clamp(.1,16.)
        return curves*factor[...,None,None]

    def batch_loss(self, sketches, prefix_counts, beta, free_bits=0., deterministic=False,
                   teacher_forcing=None, metadata=None):
        if not math.isfinite(beta) or beta < 0: raise ValueError('beta must be finite and nonnegative')
        if teacher_forcing not in (None,1.): raise ValueError('G has no teacher-forced rollout')
        prefix_counts = prefix_counts_checked(sketches,prefix_counts)
        suffixes = [s[n:] for s,n in zip(sketches,prefix_counts)]
        counts = torch.tensor([len(s) for s in suffixes],device=self.device)
        if (counts > self.max_slots).any(): raise ValueError('Target exceeds max_stroke_slots; never truncate TRAIN')
        c = self.context([s[:n] for s,n in zip(sketches,prefix_counts)])
        pm,pl = self.prior(c); qm,ql = self.posterior(sketches,c)
        z = reparameterize(qm,ql,deterministic)
        output = self.decode_plan(c,z)
        view = self.view(suffixes); target_code = self.encode_view(view)
        k = view['stroke_mask'].shape[1]; mask=view['stroke_mask']
        anchor_sq = ((output['anchors'][:,:k]-view['anchors'])/self.config['anchor_sigma']).square().mean(-1)*.5
        anchors = per_sketch(anchor_sq,mask)
        if self.config['code_mixtures'] > 1:
            means=output['code_means'][:,:k]
            logits=output['code_logits'][:,:k]
            distances=.5*((means-target_code[:,:,None,:])/self.config['code_mixture_sigma']).square().mean(-1)
            log_components=F.log_softmax(logits,-1)-distances
            codes=per_sketch(-torch.logsumexp(log_components,-1),mask)
            selected=log_components.argmax(-1)[...,None,None].expand(-1,-1,1,self.embedding_dim)
            predicted_codes=means.gather(-2,selected).squeeze(-2)
        else:
            code_sq = ((output['codes'][:,:k]-target_code)/self.config['code_sigma']).square().mean(-1)*.5
            codes = per_sketch(code_sq,mask)
            predicted_codes=output['codes'][:,:k]
        pred_curves = self.stroke_decoder(predicted_codes.reshape(-1,self.embedding_dim),view['t'])
        pred_curves = pred_curves.reshape(len(sketches),k,self.stroke_samples,2)
        extent_loss=pred_curves.sum()*0
        if self.config['extent_prediction']:
            target_extent=torch.linalg.norm(view['relative'].amax(-2)-view['relative'].amin(-2),dim=-1).clamp_min(.006)
            extent_loss=per_sketch(F.smooth_l1_loss(output['extent_log'][:,:k],target_extent.log(),reduction='none'),mask)
            pred_curves=self.scale_curves(pred_curves,output['extent_log'][:,:k])
        curve = per_sketch((pred_curves-view['relative']).square().mean((-1,-2)),mask)
        smoothness_loss=primitive_loss=relation_loss=curve*0
        if self.config['lambda_smoothness']:
            predicted=stroke_roughness(pred_curves)
            target=stroke_roughness(view['relative'])
            smoothness_loss=per_sketch(F.relu(predicted-target-.02),mask)
        if self.config['lambda_primitive']:
            pred_flips,pred_ratio=stroke_primitive_complexity(pred_curves)
            target_flips,target_ratio=stroke_primitive_complexity(view['relative'])
            primitive_loss=per_sketch(F.relu(pred_flips-target_flips-.01)+
                                      .2*F.relu(pred_ratio-target_ratio-.15),mask)
        if self.config['lambda_relations'] and k>1:
            pair_mask=(mask[:,1:]*mask[:,:-1]).float()
            pred_delta=output['anchors'][:,1:k]-output['anchors'][:,:k-1]
            true_delta=view['anchors'][:,1:k]-view['anchors'][:,:k-1]
            anchor_relation=F.smooth_l1_loss(pred_delta,true_delta,reduction='none',beta=.1).mean(-1)
            pred_chord=pred_curves[:,:,-1]-pred_curves[:,:,0]
            true_chord=view['relative'][:,:,-1]-view['relative'][:,:,0]
            pred_unit=F.normalize(pred_chord,dim=-1,eps=1e-4)
            true_unit=F.normalize(true_chord,dim=-1,eps=1e-4)
            pred_cos=(pred_unit[:,1:]*pred_unit[:,:-1]).sum(-1)
            true_cos=(true_unit[:,1:]*true_unit[:,:-1]).sum(-1)
            angular_relation=(pred_cos-true_cos).square()
            usable=((true_chord[:,1:].norm(dim=-1)>.02)&
                    (true_chord[:,:-1].norm(dim=-1)>.02)).float()
            relation_loss=per_sketch(anchor_relation+.25*angular_relation*usable,pair_mask)
        count_ce = F.cross_entropy(output['count_logits'],counts)
        zero = curve*0; part_ce=box_loss=presence_loss=part_accuracy=zero
        group_box_loss=group_assignment_loss=group_presence_loss=group_count_loss=zero
        occupancy_loss=plan_occupancy_loss=vector_occupancy_loss=zero
        if self.config['lambda_occ']:
            target_occupancy = future_occupancy(view, self.config['raster_resolution'])
            predicted_plan = plan_occupancy(output['plan'], self.config['raster_resolution'])
            predicted_vector = future_occupancy({
                'relative':pred_curves,'anchors':output['anchors'][:,:k],
                'stroke_mask':mask}, self.config['raster_resolution'])
            plan_occupancy_loss = (predicted_plan-target_occupancy).square().mean()
            vector_occupancy_loss = (predicted_vector-target_occupancy).square().mean()
            occupancy_loss = .5*(plan_occupancy_loss+vector_occupancy_loss)
        if metadata is not None:
            if len(metadata) != len(sketches): raise ValueError('metadata batch must align with sketches')
            target_parts = torch.full(mask.shape,-100,device=self.device,dtype=torch.long)
            boxes = output['plan']['boxes'].new_zeros((len(sketches),len(self.parts),4))
            present = boxes.new_zeros(boxes.shape[:2])
            for b,(sample,suffix,n) in enumerate(zip(metadata,suffixes,prefix_counts)):
                labels = sample['parts'][n:]
                if len(labels) != len(suffix): raise ValueError('parts must align with raw strokes')
                for j,label in enumerate(labels):
                    if label not in self.part_lookup: raise ValueError(f'Unknown part {label!r}')
                    target_parts[b,j]=self.part_lookup[label]
                for label in set(labels):
                    part=self.part_lookup[label]
                    points=np.concatenate([stroke for stroke,t in zip(suffix,labels) if t==label])/self.scale
                    low,high=points.min(0),points.max(0)
                    boxes[b,part]=boxes.new_tensor(np.r_[(low+high)/2,high-low])
                    present[b,part]=1.
            ce=F.cross_entropy(output['part_logits'][:,:k].transpose(1,2),target_parts,reduction='none')
            part_ce=per_sketch(ce,mask)
            part_accuracy=per_sketch((output['part_logits'][:,:k].argmax(-1)==target_parts).float(),mask)
            box_loss=per_sketch(.5*((output['plan']['boxes']-boxes)/self.config['plan_sigma']).square().mean(-1),present)
            presence_loss=F.binary_cross_entropy_with_logits(output['plan']['presence'],present)
            if self.config['group_anchor_planner']:
                group=output['group_plan'];g=self.config['max_group_slots']
                group_targets=torch.full(mask.shape,-100,device=self.device,dtype=torch.long)
                group_boxes=boxes.new_zeros((len(sketches),g,4))
                group_counts=boxes.new_zeros((len(sketches),g))
                group_present=boxes.new_zeros((len(sketches),g))
                for b,(sample,suffix,n) in enumerate(zip(metadata,suffixes,prefix_counts)):
                    steps=sample['step_ids'][n:]
                    if len(steps)!=len(suffix) or any(x>y for x,y in zip(steps,steps[1:])):
                        raise ValueError('step_ids must align and remain ordered')
                    unique=list(dict.fromkeys(steps))
                    if len(unique)>g:raise ValueError('Future exceeds max_group_slots')
                    for index,step in enumerate(unique):
                        positions=[j for j,value in enumerate(steps) if value==step]
                        group_targets[b,positions]=index
                        points=np.concatenate([suffix[j] for j in positions])/self.scale
                        low,high=points.min(0),points.max(0)
                        group_boxes[b,index]=group_boxes.new_tensor(np.r_[(low+high)/2,high-low])
                        group_counts[b,index]=len(positions)
                        group_present[b,index]=1.
                ce=F.cross_entropy(output['group_logits'][:,:k].transpose(1,2),
                                   group_targets,reduction='none',ignore_index=-100)
                group_assignment_loss=per_sketch(ce,mask)
                if self.config['group_box_mixtures']>1:
                    distance=.5*((group['box_components']-group_boxes[:,:,None,:])/
                                self.config['plan_sigma']).square().mean(-1)
                    nll=-torch.logsumexp(F.log_softmax(group['box_logits'],-1)-distance,-1)
                    group_box_loss=per_sketch(nll,group_present)
                else:
                    group_box_loss=per_sketch(.5*((group['boxes']-group_boxes)/
                                       self.config['plan_sigma']).square().mean(-1),group_present)
                group_presence_loss=F.binary_cross_entropy_with_logits(group['presence'],group_present)
                group_count_loss=per_sketch(F.smooth_l1_loss(group['count']/10.,group_counts/10.,reduction='none'),group_present)
        raw_kl=gaussian_kl(qm,ql,pm,pl).mean()
        kl_objective=gaussian_kl(qm,ql,pm,pl,free_bits=free_bits).mean()
        reconstruction=anchors+codes+self.config['lambda_count']*count_ce
        auxiliary=(self.config['curve_weight']*curve+self.config['lambda_parts']*part_ce+
                   self.config['lambda_global']*box_loss+self.config['lambda_presence']*presence_loss+
                   self.config['lambda_occ']*occupancy_loss+self.config['lambda_extent']*extent_loss+
                   self.config['lambda_smoothness']*smoothness_loss+
                   self.config['lambda_primitive']*primitive_loss+
                   self.config['lambda_relations']*relation_loss+
                   self.config['lambda_group_box']*group_box_loss+
                   self.config['lambda_group_assignment']*group_assignment_loss+
                   self.config['lambda_group_presence']*group_presence_loss+
                   self.config['lambda_group_count']*group_count_loss)
        total=reconstruction+auxiliary+beta*kl_objective
        count_accuracy=(output['count_logits'].argmax(-1)==counts).float().mean()
        return {'loss':total,'total_loss':total,'reconstruction':reconstruction+auxiliary,
                'reconstruction_loss':reconstruction+auxiliary,'coordinate_nll':anchors+codes,'pen_ce':count_ce,
                'anchor_nll':anchors,'embedding_nll':codes,'stroke_reconstruction':curve,
                'auxiliary_loss':auxiliary,'plan_box_loss':box_loss,'part_ce':part_ce,
                'occupancy_loss':occupancy_loss,'plan_occupancy_loss':plan_occupancy_loss,
                'vector_occupancy_loss':vector_occupancy_loss,
                'extent_loss':extent_loss,
                'smoothness_loss':smoothness_loss,'primitive_loss':primitive_loss,
                'relation_loss':relation_loss,
                'group_box_loss':group_box_loss,'group_assignment_loss':group_assignment_loss,
                'group_presence_loss':group_presence_loss,'group_count_loss':group_count_loss,
                'part_presence_bce':presence_loss,'part_accuracy':part_accuracy,
                'count_ce':count_ce,'count_accuracy':count_accuracy,'pen_accuracy':count_accuracy,
                'count_mae':(output['count_logits'].argmax(-1)-counts).abs().float().mean(),
                'kl':raw_kl,'KL_loss':raw_kl,'global_kl':raw_kl,'stroke_kl':zero,
                'kl_objective':kl_objective,'global_kl_objective':kl_objective,'stroke_kl_objective':zero,
                'beta_effective':total.new_tensor(beta),'coordinate_events':mask.sum().float(),
                'stroke_events':mask.sum().float(),'pen_events':total.new_tensor(len(sketches))}

    @torch.no_grad()
    def validate_samples(self,samples,batch_size,beta,free_bits):
        from sketchlab.evaluation import prefix_count
        if not samples: raise ValueError('Validation requires samples')
        sums={}
        for start in range(0,len(samples),batch_size):
            batch=samples[start:start+batch_size]; sketches=[s['strokes'] for s in batch]
            counts=[prefix_count(len(s),['one','two',.25,.5,.75][(start+j)%5]) for j,s in enumerate(sketches)]
            metadata=batch if all('parts' in s for s in batch) else None
            result=self.batch_loss(sketches,counts,beta,free_bits,True,metadata=metadata)
            for key,value in result.items():
                sums[key]=sums.get(key,0.)+float(value)*(1 if key.endswith('_events') else len(batch))
        return {key:value if key.endswith('_events') else value/len(samples) for key,value in sums.items()}
