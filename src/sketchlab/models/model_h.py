"""Hierarchical event-autoregressive CVAE with discrete spatial/shape distributions.

The history contains group plans and whole strokes, never individual point deltas.
Future metadata constructs training targets only. Inference accepts prefix + z.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from sketchlab.losses import gaussian_kl,gaussian_parameters,reparameterize
from sketchlab.stroke_view import VIEW_VERSION,sketch_view
from .common import prefix_counts_checked
from .model_f import StrokeEncoder,StrokeDecoder,transformer,position_encoding
from .model_g import PARTS,decoder,per_sketch


class ModelH(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config={'stroke_samples':16,'stroke_embedding_dim':32,'attention_heads':4,
                     'stroke_encoder_layers':2,'composition_layers':2,'decoder_layers':3,
                     'codebook_size':256,'spatial_bins':16,'anchor_bins':16,
                     'factorized_coordinates':False,
                     'joint_mixtures':0,
                     'joint_shape_categories':False,'part_conditioned_groups':False,
                     'group_template_count':0,
                     'group_latent_dim':0,
                     'pointer_groups':False,
                     'pointer_grounding_weight':0.,
                     'semantic_prefix':False,
                     'latent_semantics':False,'planned_parts':False,
                     'max_groups':20,'max_group_strokes':96,'shape_extent':.3,
                     'part_vocabulary':PARTS,**config}
        self.model_name='H';self.scale=float(self.config['scale'])
        self.hidden_dim=self.config['hidden_dim'];self.latent_dim=self.config['latent_dim']
        self.embedding_dim=self.config['stroke_embedding_dim'];self.stroke_samples=self.config['stroke_samples']
        self.parts=tuple(self.config['part_vocabulary']);self.part_lookup={p:i for i,p in enumerate(self.parts)}
        h,d=self.hidden_dim,self.embedding_dim;heads=self.config['attention_heads'];drop=self.config['dropout']
        self.stroke_encoder=StrokeEncoder(h,d,heads,self.config['stroke_encoder_layers'],drop)
        self.stroke_decoder=StrokeDecoder(d,h)
        self.register_buffer('codebook',torch.zeros(self.config['codebook_size'],d))
        self.input_projection=nn.Linear(d+6,h)
        self.context_encoder=transformer(h,heads,self.config['composition_layers'],drop)
        self.bos=nn.Parameter(torch.zeros(1,1,h))
        self.prior_head=nn.Linear(h+1,2*self.latent_dim)
        self.posterior_head=nn.Linear(2*(h+1),2*self.latent_dim)
        self.condition=nn.Linear(h+1+self.latent_dim,h)
        self.event_type=nn.Embedding(3,h)
        self.part_embedding=nn.Embedding(len(self.parts)+1,16)
        self.event_projection=nn.Linear(d+4+2+1+1+2+16,h)
        self.side_projection=nn.Linear(4+2+1+16,h)
        self.event_decoder=decoder(h,heads,self.config['decoder_layers'],drop)
        self.eos_head=nn.Linear(h,2)
        self.part_head=nn.Linear(h,len(self.parts))
        self.count_head=nn.Linear(h,self.config['max_group_strokes'])
        self.center_head=nn.Linear(h,self.config['spatial_bins'] if self.config['factorized_coordinates'] else self.config['spatial_bins']**2)
        if self.config['factorized_coordinates']:
            self.center_y_head=nn.Sequential(nn.Linear(h+1,h),nn.GELU(),nn.Linear(h,self.config['spatial_bins']))
        self.center_offset=nn.Linear(h,2)
        self.size_head=nn.Sequential(nn.Linear(h+2,h),nn.GELU(),nn.Linear(h,2))
        self.anchor_head=nn.Linear(h,self.config['anchor_bins'] if self.config['factorized_coordinates'] else self.config['anchor_bins']**2)
        if self.config['factorized_coordinates']:
            self.anchor_y_head=nn.Sequential(nn.Linear(h+1,h),nn.GELU(),nn.Linear(h,self.config['anchor_bins']))
        self.anchor_offset=nn.Linear(h,2)
        self.shape_head=nn.Sequential(nn.Linear(h+2,h),nn.GELU(),nn.Linear(h,self.config['codebook_size']))
        self.extent_head=nn.Sequential(nn.Linear(h+2,h),nn.GELU(),nn.Linear(h,1))
        if self.config['joint_mixtures']:
            k=self.config['joint_mixtures']
            self.group_mdn=nn.Linear(h,k*(1+2*4))
            self.stroke_mdn=nn.Linear(h,k*(1+2*(3 if self.config['joint_shape_categories'] else d+3)))
            if self.config['joint_shape_categories']:
                self.mixture_shape_head=nn.Linear(h,k*self.config['codebook_size'])
            if self.config['part_conditioned_groups']:
                self.group_part_projection=nn.Linear(16,h)
                self.conditional_count_head=nn.Sequential(nn.Linear(h+16+4,h),nn.GELU(),nn.Linear(h,self.config['max_group_strokes']))
        if self.config['group_template_count']:
            n=self.config['group_template_count']
            self.register_buffer('template_points',torch.zeros(n,self.config['max_group_strokes'],self.stroke_samples,2))
            self.register_buffer('template_counts',torch.ones(n,dtype=torch.long))
            self.register_buffer('template_parts',torch.zeros(n,dtype=torch.long))
            self.template_embedding=nn.Embedding(n,h)
            self.template_head=nn.Sequential(nn.Linear(h+16+4,h),nn.GELU(),nn.Linear(h,n))
            self.template_residual=nn.Linear(h,2*self.stroke_samples)
            nn.init.zeros_(self.template_residual.weight);nn.init.zeros_(self.template_residual.bias)
        if self.config['pointer_groups']:
            self.pointer_key=nn.Sequential(nn.Linear(d+3+(len(self.parts) if self.config['semantic_prefix'] else 0),h),nn.GELU(),nn.Linear(h,h))
            self.pointer_query=nn.Linear(h,h)
            self.pointer_geometry=nn.Linear(h,8)
        if self.config['semantic_prefix']:
            self.prefix_part_head=nn.Linear(h,len(self.parts))
            self.semantic_summary=nn.Linear(2*len(self.parts),h+1)
            self.semantic_memory=nn.Linear(len(self.parts),h)
            for head in (self.semantic_summary,self.semantic_memory):
                nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
            if self.config['latent_semantics']:self.semantic_latent=nn.Linear(self.latent_dim,h)
        if self.config['planned_parts']:
            self.part_plan_gru=nn.GRU(16,h,batch_first=True)
            self.part_plan_head=nn.Linear(h,len(self.parts)+1)
        if self.config['group_latent_dim']:
            gd=self.config['group_latent_dim']
            self.group_shape_encoder=nn.Sequential(nn.Linear(2*self.stroke_samples+2,h),nn.GELU(),nn.Linear(h,h))
            self.group_prior_head=nn.Sequential(nn.Linear(h+16+5,h),nn.GELU(),nn.Linear(h,2*gd))
            self.group_posterior_head=nn.Sequential(nn.Linear(h+16+5+h,h),nn.GELU(),nn.Linear(h,2*gd))
            self.group_curve_head=nn.Sequential(nn.Linear(h+gd,h),nn.GELU(),nn.Linear(h,h),nn.GELU(),
                                                 nn.Linear(h,2*self.stroke_samples))
            if self.config['group_template_count']:
                nn.init.zeros_(self.group_curve_head[-1].weight)
                nn.init.zeros_(self.group_curve_head[-1].bias)
        self.configure_training()

    @property
    def device(self):return next(self.parameters()).device

    @property
    def representation_metadata(self):
        return {'view_version':VIEW_VERSION,'stroke_samples':self.stroke_samples,
                'parameterization':'normalized_arc_length','scale':self.scale,
                'canonical_unchanged':True,'generation_points_per_stroke':self.stroke_samples,
                'composition':'hierarchical-group-event-autoregressive-cvae-v1',
                'codebook_size':self.config['codebook_size'],'shape_extent':self.config['shape_extent']}

    def configure_training(self,stage='composition',freeze_stroke_ae=True):
        if stage!='composition' or not freeze_stroke_ae:raise ValueError('H initially requires frozen Stroke AE')
        self.training_stage=stage
        for name,p in self.named_parameters():p.requires_grad_(not name.startswith(('stroke_encoder.','stroke_decoder.')))

    def train(self,mode=True):
        super().train(mode);self.stroke_encoder.eval();self.stroke_decoder.eval();return self

    def view(self,sketches):return sketch_view(sketches,self.stroke_samples,self.scale,self.device)

    def encode_view(self,view):
        mask=view['stroke_mask'];code=view['relative'].new_zeros((*mask.shape,self.embedding_dim))
        with torch.no_grad():
            if mask.any():code[mask]=self.stroke_encoder(view['relative'][mask],view['t'])
        return code

    def shape_codes(self,view):
        relative=view['relative'];mask=view['stroke_mask']
        extent=(relative.amax(-2)-relative.amin(-2)).norm(dim=-1).clamp_min(.006)
        normalized=relative*(self.config['shape_extent']/extent)[...,None,None]
        codes=relative.new_zeros((*mask.shape,self.embedding_dim))
        with torch.no_grad():
            if mask.any():codes[mask]=self.stroke_encoder(normalized[mask],view['t'])
        ids=(codes[...,None,:]-self.codebook).square().sum(-1).argmin(-1)
        return ids,extent

    def context(self,prefixes):
        view=self.view(prefixes);code=self.encode_view(view)
        absolute=view['relative']+view['anchors'][:,:,None]
        low,high=absolute.amin(-2),absolute.amax(-2)
        bbox=torch.cat([(high+low)/2,high-low],-1)
        features=self.input_projection(torch.cat([code,view['anchors'],bbox],-1))
        inputs=torch.cat([self.bos.expand(len(prefixes),-1,-1),features],1)
        padding=F.pad(~view['stroke_mask'],(1,0),value=False)
        inputs=inputs+position_encoding(inputs.shape[1],self.hidden_dim,self.device,inputs.dtype)
        memory=self.context_encoder(inputs,src_key_padding_mask=padding)
        counts=view['stroke_mask'].sum(-1)
        summary=torch.cat([memory[:,0],(counts.float()/211)[:,None]],-1)
        part_logits=None
        if self.config['semantic_prefix'] and not self.config['latent_semantics']:
            part_logits=self.prefix_part_head(memory[:,1:])
            probability=part_logits.softmax(-1)*view['stroke_mask'][...,None]
            count_parts=probability.sum(1)
            summary=summary+self.semantic_summary(torch.cat([1-(-count_parts).exp(),count_parts/20],-1))
            memory=torch.cat([memory[:,:1],memory[:,1:]+self.semantic_memory(probability)],1)
        result={'memory':memory,'padding':padding,'summary':summary,'counts':counts}
        if part_logits is not None:result.update(part_logits=part_logits,stroke_mask=view['stroke_mask'])
        if self.config['latent_semantics']:
            result['stroke_mask']=view['stroke_mask']
            probability=memory.new_zeros((*view['stroke_mask'].shape,len(self.parts)))
        if self.config['pointer_groups']:
            t=view['t'][None,None,:,None].expand(*absolute.shape[:-1],1)
            point_code=code[:,:,None,:].expand(*absolute.shape[:-1],self.embedding_dim)
            components=[absolute,point_code,t]
            if self.config['semantic_prefix']:components.append(probability[:,:,None,:].expand(*absolute.shape[:-1],len(self.parts)))
            result['point_features']=torch.cat(components,-1).flatten(1,2)
            result['point_mask']=view['point_mask'].flatten(1,2)
        return result

    def semantic_context(self,c,z):
        if not self.config['latent_semantics']:return c
        result=dict(c);memory=c['memory']
        logits=self.prefix_part_head(memory[:,1:]+self.semantic_latent(z)[:,None])
        probability=logits.softmax(-1)*c['stroke_mask'][...,None]
        counts=probability.sum(1)
        result['summary']=c['summary']+self.semantic_summary(torch.cat([1-(-counts).exp(),counts/20],-1))
        result['memory']=torch.cat([memory[:,:1],memory[:,1:]+self.semantic_memory(probability)],1)
        result['part_logits']=logits
        if self.config['pointer_groups']:
            semantic=probability[:,:,None,:].expand(-1,-1,self.stroke_samples,-1).flatten(1,2)
            result['point_features']=torch.cat([c['point_features'][...,:-len(self.parts)],semantic],-1)
        return result

    def part_plan_loss(self,c,z,batch):
        sequences=[torch.cat([row[mask],row.new_tensor([len(self.parts)])]) for row,mask in zip(batch['part'],(batch['type']==0)&batch['mask'])]
        length=max(map(len,sequences));targets=batch['part'].new_full((len(z),length),len(self.parts))
        mask=torch.zeros_like(targets,dtype=torch.bool)
        for i,sequence in enumerate(sequences):targets[i,:len(sequence)]=sequence;mask[i,:len(sequence)]=True
        previous=F.pad(targets[:,:-1],(1,0),value=len(self.parts))
        initial=torch.tanh(self.condition(torch.cat([c['summary'],z],-1)))
        states,_=self.part_plan_gru(self.part_embedding(previous),initial[None])
        logits=self.part_plan_head(states)
        loss=per_sketch(F.cross_entropy(logits.transpose(1,2),targets,reduction='none'),mask)
        accuracy=per_sketch((logits.argmax(-1)==targets).float(),mask)
        return loss,accuracy

    def sample_part_plan(self,c,z,rng,temperature):
        from sketchlab.generation import _category
        hidden=torch.tanh(self.condition(torch.cat([c['summary'],z],-1)))[None]
        previous=torch.tensor([[len(self.parts)]],device=self.device)
        plan=[]
        for i in range(self.config['max_groups']+1):
            state,hidden=self.part_plan_gru(self.part_embedding(previous),hidden)
            part=_category(self.part_plan_head(state[0,0]),temperature,rng)
            if part==len(self.parts):return plan,'eos'
            if i==self.config['max_groups']:break
            plan.append(part);previous.fill_(part)
        return plan,'max_groups'

    def prior(self,c):return gaussian_parameters(self.prior_head(c['summary']))
    def posterior(self,sketches,c):
        full=self.context(sketches)
        return gaussian_parameters(self.posterior_head(torch.cat([full['summary'],c['summary']],-1)))

    def event_embed(self,batch):
        part=self.part_embedding(batch['part'])
        role=batch['role'];code=self.codebook[batch['code']]
        features=torch.cat([code,batch['box'],batch['anchor'],batch['extent'][...,None],
                            batch['count'][...,None]/96.,role,part],-1)
        output=self.event_type(batch['type'])+self.event_projection(features)
        if self.config['group_template_count']:
            output=output+self.template_embedding(batch['template'])*(batch['type']!=2)[...,None]
        return output

    def side_embed(self,batch):
        within=batch['type']==1
        part=torch.where(within,batch['part'],len(self.parts))
        features=torch.cat([batch['box']*within[...,None],batch['role']*within[...,None],
                            within[...,None].float(),self.part_embedding(part)],-1)
        output=self.side_projection(features)
        if self.config['group_template_count']:output=output+self.template_embedding(batch['template'])*within[...,None]
        return output

    def decode_events(self,c,z,batch):
        events=self.event_embed(batch)
        previous=torch.cat([self.bos.expand(len(z),-1,-1),events[:,:-1]],1)
        condition=self.condition(torch.cat([c['summary'],z],-1))
        x=previous+self.side_embed(batch)+condition[:,None]
        x=x+position_encoding(x.shape[1],self.hidden_dim,self.device,x.dtype)
        causal=torch.ones((x.shape[1],x.shape[1]),device=self.device,dtype=torch.bool).triu(1)
        return self.event_decoder(x,c['memory'],tgt_mask=causal,
                  tgt_key_padding_mask=~batch['mask'],memory_key_padding_mask=c['padding'])

    def prepare(self,sketches,prefix_counts,metadata):
        if metadata is None:raise ValueError('H training needs aligned step_ids and parts')
        suffixes=[s[n:] for s,n in zip(sketches,prefix_counts)]
        view=self.view(suffixes);codes,extent=self.shape_codes(view)
        codes=codes.cpu().numpy();extent=extent.cpu().numpy()
        examples=[]
        for b,(sample,suffix,n) in enumerate(zip(metadata,suffixes,prefix_counts)):
            steps=sample['step_ids'][n:];parts=sample['parts'][n:]
            if len(steps)!=len(suffix) or len(parts)!=len(suffix):raise ValueError('Unaligned group metadata')
            if any(a>b for a,b in zip(steps,steps[1:])):raise ValueError('Groups must be ordered')
            events=[]
            for step in dict.fromkeys(steps):
                positions=[i for i,value in enumerate(steps) if value==step]
                if len({parts[j] for j in positions})!=1:raise ValueError('A step must have one part')
                count=len(positions)
                if count>self.config['max_group_strokes']:raise ValueError('Group exceeds count vocabulary')
                pts=np.concatenate([suffix[j] for j in positions])/self.scale
                low,high=pts.min(0),pts.max(0);box=np.r_[(low+high)/2,np.maximum(high-low,.006)]
                part=self.part_lookup[parts[positions[0]]]
                if self.config['group_template_count'] or self.config['group_latent_dim']:
                    absolute=view['relative'][b,positions]+view['anchors'][b,positions,None]
                    box_tensor=absolute.new_tensor(box)
                    normalized=(absolute-(box_tensor[:2]-box_tensor[2:]/2))/box_tensor[2:]
                if self.config['group_template_count']:
                    template=self.nearest_template(normalized,part)
                events.append(dict(type=0,box=box,part=part,count=count,anchor=np.zeros(2),code=0,extent=0.,role=np.zeros(2)))
                if self.config['group_template_count']:events[-1]['template']=template
                for role,j in enumerate(positions):
                    anchor=np.asarray(suffix[j][0])/self.scale
                    events.append(dict(type=1,box=box,part=part,count=count,anchor=anchor,
                        code=int(codes[b,j]),extent=float(extent[b,j]),role=np.array([role/count,(count-role)/count])))
                    if self.config['group_template_count']:
                        index=min(int(role*int(self.template_counts[template])/count),int(self.template_counts[template])-1)
                        events[-1].update(template=template,stroke_index=index,points=normalized[role].cpu().numpy())
                    elif self.config['group_latent_dim']:
                        events[-1]['points']=normalized[role].cpu().numpy()
            events.append(dict(type=2,box=np.zeros(4),part=len(self.parts),count=0,anchor=np.zeros(2),code=0,extent=0.,role=np.zeros(2)))
            examples.append(events)
        return self.tensor_events(examples)

    def tensor_events(self,examples):
        length=max(map(len,examples));batch={}
        keys=['type','part','code','count','box','anchor','extent','role']
        if self.config['group_template_count']:keys+=['template','stroke_index']
        if self.config['group_template_count'] or self.config['group_latent_dim']:keys+=['points']
        for key in keys:
            tail={'box':(4,),'anchor':(2,),'role':(2,),'points':(self.stroke_samples,2)}.get(key,())
            dtype=torch.long if key in ('type','part','code','count','template','stroke_index') else torch.float32
            value=torch.zeros((len(examples),length,*tail),device=self.device,dtype=dtype)
            for b,events in enumerate(examples):
                value[b,:len(events)]=torch.as_tensor(np.asarray([r.get(key,np.zeros(tail) if tail else 0) for r in events]),device=self.device,dtype=dtype)
            batch[key]=value
        batch['mask']=torch.arange(length,device=self.device)[None]<torch.tensor(list(map(len,examples)),device=self.device)[:,None]
        return batch

    @staticmethod
    def grid_target(points,bins,low=0.,span=1.):
        unit=((points-low)/span).clamp(0,1-1e-6)
        cell=(unit*bins).long()
        label=cell[...,1]*bins+cell[...,0]
        offset=unit*bins-(cell.float()+.5)
        return label,offset

    @staticmethod
    def grid_point(label,offset,bins,low=0.,span=1.):
        xy=torch.stack([label%bins,label//bins],-1).to(offset.dtype)
        return low+((xy+.5+offset)/bins)*span

    def spatial_loss(self,state,target,active,kind):
        bins=self.config['spatial_bins' if kind=='center' else 'anchor_bins']
        low,span=(-4/256,2.) if kind=='center' else (0.,1.)
        label,offset=self.grid_target(target,bins,low,span)
        head=getattr(self,kind+'_head');residual=.5*torch.tanh(getattr(self,kind+'_offset')(state))
        if self.config['factorized_coordinates']:
            x_logits=head(state)
            y_logits=getattr(self,kind+'_y_head')(torch.cat([state,target[...,:1]],-1))
            loss=F.cross_entropy(x_logits.transpose(1,2),label%bins,reduction='none')
            loss=loss+F.cross_entropy(y_logits.transpose(1,2),label//bins,reduction='none')
            predicted_x=low+(x_logits.argmax(-1).float()+.5+residual[...,0])/bins*span
            predicted_y_logits=getattr(self,kind+'_y_head')(torch.cat([state,predicted_x[...,None]],-1))
            predicted_label=predicted_y_logits.argmax(-1)*bins+x_logits.argmax(-1)
        else:
            logits=head(state);loss=F.cross_entropy(logits.transpose(1,2),label,reduction='none')
            predicted_label=logits.argmax(-1)
        prediction=self.grid_point(predicted_label,residual,bins,low,span)
        return per_sketch(loss,active),per_sketch((residual-offset).square().mean(-1),active),prediction

    def sample_position(self,state,kind,rng,temperature):
        from sketchlab.generation import _category
        bins=self.config['spatial_bins' if kind=='center' else 'anchor_bins']
        low,span=(-4/256,2.) if kind=='center' else (0.,1.)
        residual=.5*torch.tanh(getattr(self,kind+'_offset')(state))
        if self.config['factorized_coordinates']:
            x=_category(getattr(self,kind+'_head')(state),temperature,rng)
            x_value=state.new_tensor(low+(x+.5)/bins*span)+residual[0]/bins*span
            y=_category(getattr(self,kind+'_y_head')(torch.cat([state,x_value[None]],-1)),temperature,rng)
            label=state.new_tensor(y*bins+x,dtype=torch.long)
        else:
            label=state.new_tensor(_category(getattr(self,kind+'_head')(state),temperature,rng),dtype=torch.long)
        return self.grid_point(label,residual,bins,low,span)

    def losses(self,state,batch,c=None):
        if self.config['group_template_count']:return self.template_losses(state,batch,c)
        if self.config['joint_mixtures']:return self.joint_losses(state,batch)
        mask=batch['mask'];group=(batch['type']==0)&mask;stroke=(batch['type']==1)&mask
        termination=((batch['type']==0)|(batch['type']==2))&mask
        def ce(logits,target,active):
            values=F.cross_entropy(logits.transpose(1,2),target,reduction='none')
            return per_sketch(values,active)
        eos=ce(self.eos_head(state),(batch['type']==2).long(),termination)
        center=batch['box'][...,:2]
        center_loss,center_residual,_=self.spatial_loss(state,center,group,'center')
        size_log=self.size_head(torch.cat([state,center],-1))
        size_loss=per_sketch(F.smooth_l1_loss(size_log,batch['box'][...,2:].clamp_min(.006).log(),reduction='none').mean(-1),group)
        part=ce(self.part_head(state),batch['part'].clamp(max=len(self.parts)-1),group)
        count=ce(self.count_head(state),(batch['count']-1).clamp_min(0),group)
        relative=(batch['anchor']-(center-batch['box'][...,2:]/2))/batch['box'][...,2:].clamp_min(.006)
        anchor,anchor_residual,anchor_pred=self.spatial_loss(state,relative.clamp(0,1),stroke,'anchor')
        shape_input=torch.cat([state,relative.clamp(0,1)],-1)
        shape_logits=self.shape_head(shape_input)
        shape=ce(shape_logits,batch['code'],stroke)
        log_extent=self.extent_head(shape_input).squeeze(-1)
        extent=per_sketch(F.smooth_l1_loss(log_extent,batch['extent'].clamp_min(.006).log(),reduction='none'),stroke)
        total=eos+center_loss+center_residual+size_loss+.5*part+count+anchor+anchor_residual+shape+extent
        anchor_abs=(anchor_pred-.5)*batch['box'][...,2:]+center
        error=per_sketch((anchor_abs-batch['anchor']).square().sum(-1),stroke).sqrt()*self.scale
        return {'reconstruction':total,'eos_loss':eos,'group_center_ce':center_loss,'group_size_loss':size_loss,
                'group_part_ce':part,'group_count_ce':count,'anchor_ce':anchor,'shape_ce':shape,'extent_loss':extent,
                'anchor_rmse_px':error,'shape_accuracy':per_sketch((shape_logits.argmax(-1)==batch['code']).float(),stroke),
                'pen_accuracy':per_sketch((self.eos_head(state).argmax(-1)==(batch['type']==2)).float(),termination),
                'coordinate_events':stroke.sum().float(),'pen_events':termination.sum().float()}

    def joint_parameters(self,state,kind):
        """One component couples location, geometry and scale, avoiding independent draws."""
        d=4 if kind=='group' else (3 if self.config['joint_shape_categories'] else self.embedding_dim+3)
        raw=getattr(self,kind+'_mdn')(state).reshape(*state.shape[:-1],self.config['joint_mixtures'],1+2*d)
        logits,mu,log_sigma=raw[...,0],raw[...,1:1+d],raw[...,1+d:]
        mu=torch.cat([mu[...,:2].sigmoid(),mu[...,2:]],-1)
        return logits,mu,log_sigma.clamp(-3.,1.)

    def joint_nll(self,state,target,kind,shape_target=None):
        logits,mu,log_sigma=self.joint_parameters(state,kind)
        residual=(target[...,None,:]-mu)*(-log_sigma).exp()
        nll=.5*residual.square()+log_sigma+.5*np.log(2*np.pi)
        # Local code has 32 correlated dimensions: average its contribution so
        # anchor likelihood retains equal influence on component assignment.
        if kind=='stroke' and self.config['joint_shape_categories']:
            shape_log=F.log_softmax(self.mixture_shapes(state),-1)
            shape_nll=-shape_log.gather(-1,shape_target[...,None,None].expand(*shape_target.shape,self.config['joint_mixtures'],1)).squeeze(-1)
            nll=nll.sum(-1)+shape_nll
        elif kind=='stroke':nll=nll[...,:2].sum(-1)+nll[...,2:-1].mean(-1)+nll[...,-1]
        else:nll=nll.sum(-1)
        return -torch.logsumexp(F.log_softmax(logits,-1)-nll,-1),mu.gather(-2,logits.argmax(-1)[...,None,None].expand(*logits.shape[:-1],1,mu.shape[-1])).squeeze(-2)

    def code_statistics(self):
        return self.codebook.mean(0),self.codebook.std(0).clamp_min(.05)

    def nearest_template(self,points,part):
        candidates=(self.template_parts==part).nonzero().flatten()
        if not len(candidates):raise ValueError(f'No TRAIN template for part {part}')
        counts=self.template_counts[candidates]
        exact=counts==len(points)
        if exact.any():candidates=candidates[exact];counts=counts[exact]
        role=torch.arange(len(points),device=self.device)[None]*counts[:,None]/len(points)
        index=role.long().minimum(counts[:,None]-1)
        candidates_points=self.template_points[candidates[:,None],index]
        error=(candidates_points-points).square().mean((-1,-2,-3))
        error=error+.05*(counts-len(points)).abs()/max(1,len(points))
        return int(candidates[error.argmin()])

    def template_logits(self,state,part,box):
        logits=self.template_head(torch.cat([state,self.part_embedding(part),box],-1))
        compatible=self.template_parts==part[...,None]
        # EOS/padding have no part; only group targets enter this classifier loss.
        compatible=compatible|(~compatible.any(-1))[...,None]
        return logits.masked_fill(~compatible,-1e4)

    def residual_points(self,state):
        return .1*torch.tanh(self.template_residual(state).reshape(*state.shape[:-1],self.stroke_samples,2))

    def pointer_bank(self,c,batch):
        points=batch['points']*batch['box'][...,None,2:]+batch['box'][...,None,:2]-batch['box'][...,None,2:]/2
        codes=self.codebook[batch['code']][...,None,:].expand(*points.shape[:-1],self.embedding_dim)
        t=torch.linspace(0,1,self.stroke_samples,device=self.device)[None,None,:,None].expand(*points.shape[:-1],1)
        components=[points,codes,t]
        if self.config['semantic_prefix']:
            semantic=F.one_hot(batch['part'].clamp(max=len(self.parts)-1),len(self.parts)).to(points.dtype)
            components.append(semantic[...,None,:].expand(*points.shape[:-1],len(self.parts)))
        features=torch.cat(components,-1).flatten(1,2)
        active=((batch['type']==1)&batch['mask'])[...,None].expand(*points.shape[:-1]).flatten(1,2)
        times=torch.arange(batch['mask'].shape[1],device=self.device).repeat_interleave(self.stroke_samples)
        # A canvas reference supports unconditional generation before any ink.
        fallback=features.new_zeros((len(features),1,features.shape[-1]));fallback[...,:2]=.984375
        bank=torch.cat([fallback,c['point_features'],features],1)
        valid=torch.cat([active.new_ones((len(active),1)),c['point_mask'],active],1)
        times=torch.cat([times.new_full((1+c['point_features'].shape[1],),-1),times])
        return bank,valid,times

    def pointer_components(self,state,bank,query_indices):
        features,valid,times=bank
        score=torch.einsum('bth,bnh->btn',self.pointer_query(state),self.pointer_key(features))/(self.hidden_dim**.5)
        allowed=valid[:,None,:]&(times[None,None,:]<query_indices[None,:,None])
        score=score.masked_fill(~allowed,-1e4)
        raw=self.pointer_geometry(state)
        offset=.75*raw[...,:2].tanh()
        size_log=raw[...,2:4]
        means=features[:,None,:,:2]+offset[:,:,None,:]
        log_sigma=raw[...,4:].clamp(-3.,.5)
        return score,means,size_log,log_sigma,offset

    def pointer_nll(self,state,batch,c,grounding=False):
        bank=self.pointer_bank(c,batch)
        score,means,size_log,log_sigma,offset=self.pointer_components(state,bank,torch.arange(state.shape[1],device=self.device))
        target_center=batch['box'][...,:2]
        coordinate=.5*((target_center[:,:,None,:]-means)*(-log_sigma[...,:2])[:,:,None,:].exp()).square()
        coordinate=(coordinate+log_sigma[...,:2][:,:,None,:]+.5*np.log(2*np.pi)).sum(-1)
        position=-torch.logsumexp(F.log_softmax(score,-1)-coordinate,-1)
        target_size=batch['box'][...,2:].clamp_min(.006).log()
        size=(.5*((target_size-size_log)*(-log_sigma[...,2:]).exp()).square()+log_sigma[...,2:]+.5*np.log(2*np.pi)).sum(-1)
        nll=position+size+.05*offset.square().sum(-1)
        if not grounding:return nll
        features,valid,times=bank
        ink=valid[:,None,:]&(times[None,None,:]<torch.arange(state.shape[1],device=self.device)[None,:,None])
        ink[:,:,0]=False
        distance=(features[:,None,:,:2]-target_center[:,:,None,:]).norm(dim=-1)*self.scale
        near=ink&(distance<64.)
        active=near.any(-1)&(batch['type']==0)&batch['mask']
        log_probability=F.log_softmax(score,-1)
        grounding_ce=-torch.logsumexp(log_probability.masked_fill(~near,-1e4),-1)
        return nll,grounding_ce,active

    def pointer_sample(self,state,c,batch,rng,temperature):
        from sketchlab.generation import _category
        bank=self.pointer_bank(c,batch)
        score,means,size_log,_,_=self.pointer_components(state[None,None],bank,state.new_tensor([batch['mask'].shape[1]-1],dtype=torch.long))
        index=_category(score[0,0],temperature,rng)
        origin=('canvas' if index==0 else 'prefix' if index<=c['point_features'].shape[1] else 'generated')
        reference={'point':bank[0][0,index,:2].cpu().tolist(),'origin':origin,'index':index}
        return means[0,0,index],size_log[0,0].clamp(-5.,.7).exp(),reference

    def group_latent_condition(self,state,part,box,count,template=None):
        """All inputs are known when a group is about to be drawn."""
        if template is not None:state=state+self.template_embedding(template)
        return torch.cat([state,self.part_embedding(part),box,count.to(box.dtype)[...,None]/96.],-1)

    def group_latent_parameters(self,condition,shape=None):
        head=self.group_prior_head(condition) if shape is None else self.group_posterior_head(torch.cat([condition,shape],-1))
        mu,logvar=gaussian_parameters(head)
        return mu,logvar.clamp(-8.,2.)

    def group_latents_for_training(self,state,batch,deterministic):
        group=((batch['type']==0)&batch['mask']);stroke=((batch['type']==1)&batch['mask'])
        index=(batch['type']==0).long().cumsum(-1)-1
        group_count=max(int(group.sum(-1).max()),1)
        conditions=[];shapes=[];active=[]
        features=self.group_shape_encoder(torch.cat([batch['points'].flatten(-2),batch['role']],-1))
        for b in range(len(state)):
            positions=group[b].nonzero().flatten();c=[];f=[]
            for j,position in enumerate(positions.tolist()):
                condition=self.group_latent_condition(state[b,position],batch['part'][b,position],
                                      batch['box'][b,position],batch['count'][b,position],
                                      batch['template'][b,position] if self.config['group_template_count'] else None)
                members=stroke[b]&(index[b]==j)
                c.append(condition)
                f.append(features[b,members].mean(0))
            zero_condition=state.new_zeros((state.shape[-1]+16+5,))
            zero_shape=state.new_zeros((state.shape[-1],))
            conditions.append(torch.stack(c+[zero_condition]*(group_count-len(c))))
            shapes.append(torch.stack(f+[zero_shape]*(group_count-len(f))))
            active.append([True]*len(c)+[False]*(group_count-len(c)))
        condition=torch.stack(conditions);shape=torch.stack(shapes)
        pm,pl=self.group_latent_parameters(condition)
        qm,ql=self.group_latent_parameters(condition,shape)
        latent=reparameterize(qm,ql,deterministic)
        kl=per_sketch(gaussian_kl(qm,ql,pm,pl),torch.tensor(active,device=self.device))
        event_latent=latent.gather(1,index.clamp_min(0).clamp_max(group_count-1)[...,None].expand(-1,-1,latent.shape[-1]))
        return event_latent,kl

    def continuous_group_points(self,state,latent,template=None,stroke_index=None):
        raw=self.group_curve_head(torch.cat([state,latent],-1)).reshape(*state.shape[:-1],self.stroke_samples,2)
        if template is None:return raw.sigmoid(),None
        residual=.5*raw.tanh()
        base=self.template_points[template,stroke_index]
        return (base+residual).clamp(0.,1.),residual

    def continuous_group_losses(self,state,batch,c,deterministic):
        mask=batch['mask'];group=(batch['type']==0)&mask;stroke=(batch['type']==1)&mask
        term=(group|(batch['type']==2))&mask
        def ce(logits,target,active):return per_sketch(F.cross_entropy(logits.transpose(1,2),target,reduction='none'),active)
        eos=ce(self.eos_head(state),(batch['type']==2).long(),term)
        part=ce(self.part_head(state),batch['part'].clamp(max=len(self.parts)-1),group)
        if self.config['pointer_grounding_weight']:
            group_nll,grounding_event,grounded=self.pointer_nll(self.group_state(state,batch['part']),batch,c,True)
            grounding=per_sketch(grounding_event,grounded)
        else:
            group_nll=self.pointer_nll(self.group_state(state,batch['part']),batch,c)
            grounding=state.new_zeros(())
        geometry=per_sketch(group_nll,group)
        count=(state.new_zeros(()) if self.config['group_template_count'] else
               ce(self.group_count_logits(state,batch['part'],batch['box']),
                  (batch['count']-1).clamp_min(0),group))
        latent,kl=self.group_latents_for_training(state,batch,deterministic)
        points,residual=self.continuous_group_points(state,latent,
            batch['template'] if self.config['group_template_count'] else None,
            batch['stroke_index'] if self.config['group_template_count'] else None)
        mse=per_sketch((points-batch['points']).square().mean((-1,-2)),stroke)
        size=batch['box'][...,2:]
        anchor_error=(points[...,0,:]-batch['points'][...,0,:])*size*self.scale
        error=per_sketch(anchor_error.square().sum(-1),stroke).sqrt()
        template=(ce(self.template_logits(state,batch['part'],batch['box']),batch['template'],group)
                  if self.config['group_template_count'] else state.new_zeros(()))
        residual_smooth=(per_sketch(residual.diff(n=2,dim=-2).square().mean((-1,-2)),stroke)
                         if residual is not None else state.new_zeros(()))
        total=eos+.5*part+geometry+count+template+25*mse+.1*residual_smooth+float(self.config['pointer_grounding_weight'])*grounding
        return {'reconstruction':total,'eos_loss':eos,'group_joint_nll':geometry,'group_part_ce':part,
                'group_count_ce':count,'local_point_mse':mse,'anchor_rmse_px':error,
                'template_ce':template,'residual_smoothness':residual_smooth,'pointer_grounding_ce':grounding,
                'group_kl':kl,'pen_accuracy':per_sketch((self.eos_head(state).argmax(-1)==(batch['type']==2)).float(),term),
                'coordinate_events':stroke.sum().float(),'pen_events':term.sum().float()}

    def template_losses(self,state,batch,c=None):
        mask=batch['mask'];group=(batch['type']==0)&mask;stroke=(batch['type']==1)&mask
        term=((batch['type']==0)|(batch['type']==2))&mask
        def ce(logits,target,active):return per_sketch(F.cross_entropy(logits.transpose(1,2),target,reduction='none'),active)
        eos=ce(self.eos_head(state),(batch['type']==2).long(),term)
        part=ce(self.part_head(state),batch['part'].clamp(max=len(self.parts)-1),group)
        center=batch['box'][...,:2];size=batch['box'][...,2:].clamp_min(.006)
        target=torch.cat([(center+4/self.scale)/2,size.log()],-1)
        if self.config['pointer_groups']:
            group_nll=self.pointer_nll(self.group_state(state,batch['part']),batch,c)
        else:group_nll,_=self.joint_nll(self.group_state(state,batch['part']),target,'group')
        gl=per_sketch(group_nll,group)
        logits=self.template_logits(state,batch['part'],batch['box'])
        template=ce(logits,batch['template'],group)
        base=self.template_points[batch['template'],batch['stroke_index']]
        residual=self.residual_points(state);pred=base+residual
        reconstruction=per_sketch((pred-batch['points']).square().mean((-1,-2)),stroke)
        smooth=per_sketch(residual.diff(n=2,dim=-2).square().mean((-1,-2)),stroke)
        anchor_error=(pred[...,0,:]-batch['points'][...,0,:])*size
        error=per_sketch(anchor_error.square().sum(-1),stroke).sqrt()*self.scale
        total=eos+.5*part+gl+template+25*reconstruction+smooth
        return {'reconstruction':total,'eos_loss':eos,'group_joint_nll':gl,'group_part_ce':part,
                'template_ce':template,'local_point_mse':reconstruction,'anchor_rmse_px':error,
                'template_accuracy':per_sketch((logits.argmax(-1)==batch['template']).float(),group),
                'pen_accuracy':per_sketch((self.eos_head(state).argmax(-1)==(batch['type']==2)).float(),term),
                'coordinate_events':stroke.sum().float(),'pen_events':term.sum().float()}

    def mixture_shapes(self,state):
        return self.mixture_shape_head(state).reshape(*state.shape[:-1],self.config['joint_mixtures'],self.config['codebook_size'])

    def group_state(self,state,part):
        if self.config['part_conditioned_groups']:
            return state+self.group_part_projection(self.part_embedding(part))
        return state

    def group_count_logits(self,state,part,box):
        if self.config['part_conditioned_groups']:
            return self.conditional_count_head(torch.cat([state,self.part_embedding(part),box],-1))
        return self.count_head(state)

    def joint_losses(self,state,batch):
        mask=batch['mask'];group=(batch['type']==0)&mask;stroke=(batch['type']==1)&mask
        termination=((batch['type']==0)|(batch['type']==2))&mask
        def ce(head,target,active):return per_sketch(F.cross_entropy(head(state).transpose(1,2),target,reduction='none'),active)
        eos=ce(self.eos_head,(batch['type']==2).long(),termination)
        center=batch['box'][...,:2];size=batch['box'][...,2:].clamp_min(.006)
        group_target=torch.cat([(center+4/self.scale)/2,size.log()],-1)
        group_nll,_=self.joint_nll(self.group_state(state,batch['part']),group_target,'group')
        relative=((batch['anchor']-center)/size+.5).clamp(0,1)
        cm,cs=self.code_statistics()
        code=(self.codebook[batch['code']]-cm)/cs
        extent=batch['extent'].clamp_min(.006).log()[...,None]
        target=torch.cat([relative,extent] if self.config['joint_shape_categories'] else [relative,code,extent],-1)
        stroke_nll,pred=self.joint_nll(state,target,'stroke',batch['code'])
        part=ce(self.part_head,batch['part'].clamp(max=len(self.parts)-1),group)
        count=per_sketch(F.cross_entropy(self.group_count_logits(state,batch['part'],batch['box']).transpose(1,2),(batch['count']-1).clamp_min(0),reduction='none'),group)
        gl=per_sketch(group_nll,group);sl=per_sketch(stroke_nll,stroke)
        total=eos+gl+sl+.5*part+count
        anchor_abs=(pred[...,:2]-.5)*size+center
        error=per_sketch((anchor_abs-batch['anchor']).square().sum(-1),stroke).sqrt()*self.scale
        extra=({'shape_accuracy':per_sketch((self.mixture_shapes(state).mean(-2).argmax(-1)==batch['code']).float(),stroke)}
               if self.config['joint_shape_categories'] else {'code_rmse':per_sketch((pred[...,2:-1]-code).square().mean(-1),stroke).sqrt()})
        return {'reconstruction':total,'eos_loss':eos,'group_joint_nll':gl,'stroke_joint_nll':sl,
                'group_part_ce':part,'group_count_ce':count,'anchor_rmse_px':error,
                **extra,
                'pen_accuracy':per_sketch((self.eos_head(state).argmax(-1)==(batch['type']==2)).float(),termination),
                'coordinate_events':stroke.sum().float(),'pen_events':termination.sum().float()}

    def draw_joint(self,state,kind,rng,temperature):
        from sketchlab.generation import _category
        logits,mu,_=self.joint_parameters(state,kind)
        # Sample mixture identity and global z; use its conditional mode for
        # smooth whole-stroke geometry rather than independent coordinate noise.
        return mu[_category(logits,temperature,rng)]

    def predict_stroke(self,state,box,rng,temperature,template=0,stroke_index=0):
        from sketchlab.generation import _category
        if self.config['group_template_count']:
            points=self.template_points[template,stroke_index]+self.residual_points(state)
            absolute=(points*box[2:]+box[:2]-box[2:]/2)*self.scale
            anchor=absolute[0]/self.scale;relative=(absolute-absolute[0])/self.scale
            extent=(relative.amax(0)-relative.amin(0)).norm().clamp_min(.006)
            local_code=self.stroke_encoder((relative*(self.config['shape_extent']/extent))[None],torch.linspace(0,1,self.stroke_samples,device=self.device))[0]
            code=int((self.codebook-local_code).square().sum(-1).argmin())
            return anchor,code,extent,absolute
        if self.config['joint_mixtures']:
            logits,mu,_=self.joint_parameters(state,'stroke')
            component=_category(logits,temperature,rng);pred=mu[component]
            relative=pred[:2]
            if self.config['joint_shape_categories']:
                code=_category(self.mixture_shapes(state)[component],temperature,rng);local_code=self.codebook[code]
            else:
                cm,cs=self.code_statistics();local_code=pred[2:-1]*cs+cm
                code=int((self.codebook-local_code).square().sum(-1).argmin())
            extent=pred[-1].clamp(-5.,.7).exp()
        else:
            relative=self.sample_position(state,'anchor',rng,temperature).clamp(0,1)
            inputs=torch.cat([state,relative],-1)
            code=_category(self.shape_head(inputs),temperature,rng);local_code=self.codebook[code]
            extent=self.extent_head(inputs).squeeze().clamp(-5.,.7).exp()
        anchor=(relative-.5)*box[2:]+box[:2]
        t=torch.linspace(0,1,self.stroke_samples,device=self.device)
        curve=self.stroke_decoder(local_code[None],t)[0]
        curve=curve*(extent/(curve.amax(0)-curve.amin(0)).norm().clamp_min(.001))
        return anchor,code,extent,(curve+anchor)*self.scale

    def sample_group_latent(self,state,part,box,count,rng,template=None):
        condition=self.group_latent_condition(state,self.part_embedding.weight.new_tensor(part,dtype=torch.long),
                                              box,box.new_tensor(count),
                                              self.part_embedding.weight.new_tensor(template,dtype=torch.long) if template is not None else None)
        mu,logvar=self.group_latent_parameters(condition)
        noise=torch.randn(mu.shape,generator=rng,dtype=torch.float32).to(self.device)
        return mu+noise*(.5*logvar).exp()

    def predict_group_stroke(self,state,box,latent,template=None,stroke_index=None):
        points,_=self.continuous_group_points(state,latent,template,stroke_index)
        absolute=(points*box[2:]+box[:2]-box[2:]/2)*self.scale
        anchor=absolute[0]/self.scale;relative=(absolute-absolute[0])/self.scale
        extent=(relative.amax(0)-relative.amin(0)).norm().clamp_min(.006)
        t=torch.linspace(0,1,self.stroke_samples,device=self.device)
        local_code=self.stroke_encoder((relative*(self.config['shape_extent']/extent))[None],t)[0]
        code=int((self.codebook-local_code).square().sum(-1).argmin())
        return anchor,code,extent,absolute

    def batch_loss(self,sketches,prefix_counts,beta,free_bits=0.,deterministic=False,teacher_forcing=None,metadata=None):
        if teacher_forcing not in (None,1.):raise ValueError('H1 uses measured teacher forcing; no preventive schedule')
        prefix_counts=prefix_counts_checked(sketches,prefix_counts)
        c=self.context([s[:n] for s,n in zip(sketches,prefix_counts)])
        pm,pl=self.prior(c);qm,ql=self.posterior(sketches,c)
        z=reparameterize(qm,ql,deterministic)
        c=self.semantic_context(c,z)
        batch=self.prepare(sketches,prefix_counts,metadata)
        state=self.decode_events(c,z,batch)
        metrics=(self.continuous_group_losses(state,batch,c,deterministic) if self.config['group_latent_dim']
                 else self.losses(state,batch,c))
        if self.config['semantic_prefix']:
            target=torch.zeros_like(c['stroke_mask'],dtype=torch.long)
            for b,(sample,n) in enumerate(zip(metadata,prefix_counts)):
                if n:target[b,:n]=torch.tensor([self.part_lookup[p] for p in sample['parts'][:n]],device=self.device)
            semantic_ce=per_sketch(F.cross_entropy(c['part_logits'].transpose(1,2),target,reduction='none'),c['stroke_mask'])
            metrics.update(decoder_reconstruction=metrics['reconstruction'],prefix_part_ce=semantic_ce,
                prefix_part_accuracy=per_sketch((c['part_logits'].argmax(-1)==target).float(),c['stroke_mask']))
            metrics['reconstruction']=metrics['reconstruction']+semantic_ce
        if self.config['planned_parts']:
            plan_loss,accuracy=self.part_plan_loss(c,z,batch)
            metrics.update(part_plan_ce=plan_loss,part_plan_accuracy=accuracy)
            metrics['reconstruction']=metrics['reconstruction']+plan_loss
        kl=gaussian_kl(qm,ql,pm,pl).mean();objective=gaussian_kl(qm,ql,pm,pl,free_bits=free_bits).mean()
        local_kl=metrics.get('group_kl',objective*0)
        total=metrics['reconstruction']+beta*(objective+local_kl);zero=total*0
        return {**metrics,'loss':total,'total_loss':total,'reconstruction_loss':metrics['reconstruction'],
                'kl':kl+local_kl,'KL_loss':kl+local_kl,'global_kl':kl,'stroke_kl':local_kl,'kl_objective':objective+local_kl,
                'global_kl_objective':objective,'stroke_kl_objective':local_kl,'beta_effective':total.new_tensor(beta)}

    @torch.no_grad()
    def validate_samples(self,samples,batch_size,beta,free_bits):
        from sketchlab.evaluation import prefix_count
        sums={}
        for start in range(0,len(samples),batch_size):
            batch=samples[start:start+batch_size];sketches=[s['strokes'] for s in batch]
            counts=[prefix_count(len(s),['one','two',.25,.5,.75][(start+j)%5]) for j,s in enumerate(sketches)]
            result=self.batch_loss(sketches,counts,beta,free_bits,True,metadata=batch)
            for key,value in result.items():sums[key]=sums.get(key,0.)+float(value)*(1 if key.endswith('_events') else len(batch))
        return {key:value if key.endswith('_events') else value/len(samples) for key,value in sums.items()}

    def next_state(self,c,z,events,query):
        batch=self.tensor_events([events+[query]])
        return self.decode_events(c,z,batch)[0,-1]

    def empty_query(self):
        return dict(type=2,box=np.zeros(4),part=len(self.parts),count=0,anchor=np.zeros(2),code=0,extent=0.,role=np.zeros(2))

    @torch.inference_mode()
    def rollout(self,c,z,rng,temperature,max_points,max_strokes):
        from sketchlab.generation import _category
        c=self.semantic_context(c,z)
        planned,plan_reason=self.sample_part_plan(c,z,rng,temperature) if self.config['planned_parts'] else (None,None)
        events=[];strokes=[];groups=[]
        lo=-4/self.scale;hi=508/self.scale
        for group_index in range(self.config['max_groups']):
            reference=None
            if planned is not None and group_index>=len(planned):return strokes,plan_reason,groups
            query=self.empty_query();state=self.next_state(c,z,events,query)
            if planned is None and _category(self.eos_head(state),temperature,rng)==1:
                return strokes,'eos',groups
            part=(planned[group_index] if planned is not None else _category(self.part_head(state),temperature,rng) if self.config['part_conditioned_groups'] else None)
            if self.config['joint_mixtures']:
                gs=self.group_state(state,state.new_tensor(part,dtype=torch.long)) if part is not None else state
                if self.config['pointer_groups']:
                    center,size,reference=self.pointer_sample(gs,c,self.tensor_events([events+[query]]),rng,temperature)
                else:
                    pred=self.draw_joint(gs,'group',rng,temperature)
                    center=pred[:2]*2-4/self.scale;size=pred[2:].clamp(-5.,.7).exp()
            else:
                center=self.sample_position(state,'center',rng,temperature)
                size=self.size_head(torch.cat([state,center],-1)).clamp(-5.,.7).exp()
            low=(center-size/2).clamp(lo,hi);high=(center+size/2).clamp(lo,hi)
            box=torch.cat([(low+high)/2,(high-low).clamp_min(.006)])
            if part is None:part=_category(self.part_head(state),temperature,rng)
            template=None
            if self.config['group_template_count']:
                template=_category(self.template_logits(state,state.new_tensor(part,dtype=torch.long),box),temperature,rng)
                count=int(self.template_counts[template])
            else:count=1+_category(self.group_count_logits(state,state.new_tensor(part,dtype=torch.long),box),temperature,rng)
            group_latent=(self.sample_group_latent(state,part,box,count,rng,template) if self.config['group_latent_dim'] else None)
            group=dict(type=0,box=box.cpu().numpy(),part=part,count=count,anchor=np.zeros(2),code=0,extent=0.,role=np.zeros(2))
            if template is not None:group['template']=template
            events.append(group);groups.append({'box':group['box'].tolist(),'part':self.parts[part],'planned_strokes':count,'start':len(strokes)})
            if template is not None:groups[-1]['template']=template
            if reference is not None:groups[-1]['reference']=reference
            if planned is not None and group_index==0:
                groups[-1]['semantic_plan']=[self.parts[p] for p in planned]
                groups[-1]['plan_termination']=plan_reason
            for role in range(count):
                if len(strokes)>=max_strokes:return strokes,'max_strokes',groups
                if (len(strokes)+1)*self.stroke_samples>max_points:return strokes,'max_points',groups
                query={**group,'type':1,'role':np.array([role/count,(count-role)/count])}
                if template is not None:query['stroke_index']=role
                state=self.next_state(c,z,events,query)
                anchor,code,extent,absolute=(self.predict_group_stroke(state,box,group_latent,template,role) if group_latent is not None
                                             else self.predict_stroke(state,box,rng,temperature,template or 0,role))
                if not torch.isfinite(absolute).all():return strokes,'nonfinite_prediction',groups
                strokes.append(absolute.cpu().numpy().astype(float))
                event={**query,'anchor':anchor.cpu().numpy(),'code':code,'extent':float(extent)}
                if self.config['group_template_count'] or self.config['group_latent_dim']:
                    event['points']=((absolute/self.scale-box[:2]+box[2:]/2)/box[2:]).cpu().numpy()
                events.append(event)
        return strokes,plan_reason if planned is not None else 'max_groups',groups
