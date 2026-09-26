"""Conditional variational distribution over complete TRAIN sketch prototypes.

The decoder is a categorical likelihood over prefix-compatible prototype plans.
The continuous latent is sampled from a learned conditional prior at inference;
the posterior sees the full TRAIN sketch only during learning.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ExemplarCVAE(nn.Module):
    def __init__(self, raster_side: int = 40, latent_dim: int = 64, part_dim: int = 0):
        super().__init__()
        n=raster_side*raster_side
        self.raster_side=raster_side
        self.latent_dim=latent_dim
        self.part_dim=part_dim
        self.prefix_encoder=nn.Sequential(nn.Linear(n,256),nn.GELU(),nn.Linear(256,128),nn.GELU())
        self.full_encoder=nn.Sequential(nn.Linear(n,256),nn.GELU(),nn.Linear(256,128),nn.GELU())
        self.prior_head=nn.Linear(128,latent_dim*2)
        self.posterior_head=nn.Linear(256,latent_dim*2)
        self.candidate_encoder=nn.Sequential(nn.Linear(n//4,128),nn.GELU(),nn.Linear(128,64))
        self.query_head=nn.Sequential(nn.Linear(128+latent_dim,128),nn.GELU(),nn.Linear(128,64))
        self.prefix_match_weight=nn.Parameter(torch.tensor(1.))
        self.logit_scale=nn.Parameter(torch.tensor(1.))
        if part_dim:
            self.part_head=nn.Linear(128,part_dim)
            self.stage_scale=nn.Parameter(torch.tensor(1.))

    @staticmethod
    def parameters_from(head,inputs):
        mu,logvar=head(inputs).chunk(2,-1)
        return mu,logvar.clamp(-7.,2.)

    @staticmethod
    def sample(mu,logvar,noise=None):
        return mu+(torch.randn_like(mu) if noise is None else noise)*torch.exp(.5*logvar)

    @staticmethod
    def kl(qm,ql,pm,pl):
        return .5*(pl-ql+(ql.exp()+(qm-pm).square())/pl.exp()-1).sum(-1)

    def encode(self,prefix,full=None):
        c=self.prefix_encoder(prefix)
        pm,pl=self.parameters_from(self.prior_head,c)
        if full is None:return c,(pm,pl),None
        f=self.full_encoder(full)
        qm,ql=self.parameters_from(self.posterior_head,torch.cat([c,f],-1))
        return c,(pm,pl),(qm,ql)

    def logits(self,context,z,candidates,prefix_scores,candidate_parts=None):
        query=F.normalize(self.query_head(torch.cat([context,z],-1)),dim=-1)
        key=F.normalize(self.candidate_encoder(candidates),dim=-1)
        structure=torch.einsum('bd,bkd->bk',query,key)
        logits=self.logit_scale.exp().clamp(max=30.)*structure + self.prefix_match_weight*prefix_scores
        if self.part_dim:
            if candidate_parts is None:raise ValueError('Candidate prefix semantics required')
            part_logits=self.part_head(context)[:,None,:].expand_as(candidate_parts)
            stage=-F.binary_cross_entropy_with_logits(part_logits,candidate_parts,reduction='none').mean(-1)
            logits=logits+self.stage_scale.exp().clamp(max=20.)*stage
        return logits

    def loss(self,prefix,full,candidates,prefix_scores,target,kl_weight=.08,prior_weight=.35,
             candidate_parts=None,part_target=None):
        c,(pm,pl),(qm,ql)=self.encode(prefix,full)
        posterior=self.logits(c,self.sample(qm,ql),candidates,prefix_scores,candidate_parts)
        prior=self.logits(c,self.sample(pm,pl),candidates,prefix_scores,candidate_parts)
        q_nll=-(target*F.log_softmax(posterior,-1)).sum(-1).mean()
        p_nll=-(target*F.log_softmax(prior,-1)).sum(-1).mean()
        kl=self.kl(qm,ql,pm,pl).mean()
        part_loss=(F.binary_cross_entropy_with_logits(self.part_head(c),part_target)
                   if self.part_dim else q_nll.new_zeros(()))
        objective=q_nll+prior_weight*p_nll+kl_weight*kl+.5*part_loss
        return {'loss':objective,'posterior_nll':q_nll,'prior_nll':p_nll,'kl':kl,
                'part_bce':part_loss,
                'prior_top1_accuracy':(prior.argmax(-1)==target.argmax(-1)).float().mean()}

    @torch.inference_mode()
    def select(self,prefix,candidates,prefix_scores,noise,choice_noise=None,gumbel_scale=1.,candidate_parts=None):
        c,(pm,pl),_=self.encode(prefix)
        logits=self.logits(c,self.sample(pm,pl,noise),candidates,prefix_scores,candidate_parts)
        if choice_noise is None:return logits.argmax(-1),logits
        # Gumbel categorical sampling keeps distinct, learned prior modes.
        gumbel=-torch.log(-torch.log(choice_noise.clamp(1e-6,1-1e-6)))
        return (logits+gumbel_scale*gumbel).argmax(-1),logits
