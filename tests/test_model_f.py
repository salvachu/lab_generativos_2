import numpy as np
import pytest
import torch

from sketchlab.checkpointing import load_checkpoint, save_checkpoint
from sketchlab.evaluation import prefix_preserved
from sketchlab.generation import complete_sketch, generate_random, load_model, sample, sample_multiple
from sketchlab.losses import gaussian_kl, reparameterize
from sketchlab.models import create_model
from sketchlab.stroke_view import sketch_view, stroke_view
from sketchlab.training import run_experiment, validate


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(1)


def sketches():
    return [[np.array([[12.,30.],[25.,50.],[65.,35.]],dtype=np.float32),
             np.array([[250.,200.],[230.,150.]],dtype=np.float32)],
            [np.array([[24.,30.]],dtype=np.float32),
             np.array([[21.,80.],[36.,95.],[72.,81.],[80.,40.]],dtype=np.float32),
             np.array([[44.,12.],[66.,45.]],dtype=np.float32)]]


def tiny(**override):
    return create_model({"model":"F","hidden_dim":16,"latent_dim":4,"stroke_embedding_dim":8,
        "stroke_samples":8,"attention_heads":2,"stroke_encoder_layers":1,"composition_layers":1,
        "mixtures":2,**override})


def test_view_arc_length_roundtrip_and_translation():
    raw=np.array([[5.,10.],[6.,10.],[9.,10.]])
    original=raw.copy(); v=stroke_view(raw,samples=5,scale=2)
    np.testing.assert_allclose(v['anchor']+v['relative'],np.array([[5+i,10] for i in range(5)])/2)
    np.testing.assert_array_equal(raw,original)
    shifted=stroke_view(raw+[70.,-20.],samples=5,scale=2)
    np.testing.assert_allclose(v['relative'],shifted['relative'])


@pytest.mark.parametrize('raw',[np.array([[8.,9.]]),np.ones((3,2)),np.c_[np.linspace(0,50,4096),np.linspace(0,100,4096)]])
def test_view_single_coincident_and_long_strokes(raw):
    v=stroke_view(raw,16)
    restored=(v['anchor']+v['relative'])*256
    assert restored.shape==(16,2) and np.isfinite(restored).all()
    np.testing.assert_allclose(restored[[0,-1]],raw[[0,-1]],atol=1e-10)


def test_view_masks_and_empty_prefix():
    v=sketch_view([sketches()[0],[],sketches()[1]],samples=8)
    assert v['stroke_mask'].sum().item()==5
    assert v['point_mask'].sum().item()==40
    assert not v['point_mask'][1].any()


def test_encoder_translation_mask_and_nonautoregressive_decoder():
    m=tiny().eval(); raw=sketches()[0][0]; a=m.view([[raw]]); b=m.view([[raw+[100.,-50.]]])
    e=m.encode_view(a)[0]; torch.testing.assert_close(e,m.encode_view(b)[0],rtol=0,atol=0)
    relative=a['relative'][0]
    mask=torch.ones((1,8),dtype=torch.bool); mask[:,4:]=False
    changed=relative.clone(); changed[:,4:]=100
    torch.testing.assert_close(m.stroke_encoder(relative,a['t'],mask),m.stroke_encoder(changed,a['t'],mask))
    full=m.stroke_decoder(e,a['t']); isolated=m.stroke_decoder(e,a['t'][[3]])
    torch.testing.assert_close(full[:,3:4],isolated)
    assert torch.count_nonzero(full[:,0])==0


@pytest.mark.parametrize('stage',['stroke_ae','composition'])
@pytest.mark.parametrize('device',['cpu','cuda'])
def test_forward_backward_stage_and_cuda(stage,device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    m=tiny().to(device); m.configure_training(stage,freeze_stroke_ae=True)
    result=m.batch_loss(sketches(),[1,1],beta=.03,free_bits=.01)
    assert all(v.ndim==0 and torch.isfinite(v) for v in result.values())
    result['loss'].backward()
    gradients={n:p.grad for n,p in m.named_parameters() if p.requires_grad}
    assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients.values())
    assert (m.stroke_encoder.input.weight.grad is not None)==(stage=='stroke_ae')


def test_finetune_allows_ae_gradients_and_local_reconstruction():
    m=tiny(); m.configure_training('composition',freeze_stroke_ae=False)
    result=m.batch_loss(sketches(),[1,1],.05); result['loss'].backward()
    assert m.stroke_encoder.input.weight.grad.abs().sum()>0
    assert m.stroke_decoder.net[0].weight.grad.abs().sum()>0


def test_prior_posterior_causality_and_variational_sampling():
    m=tiny().eval(); data=sketches(); view=m.view(data); code=m.encode_view(view)
    tokens=m.stroke_tokens(code,view['anchors']).detach().requires_grad_()
    context=m.summarize_tokens(tokens,torch.tensor([1,1]))
    context.sum().backward(); assert torch.count_nonzero(tokens.grad[:,1:])==0
    future=tokens.detach().clone(); future[:,1:]+=42
    torch.testing.assert_close(context,m.summarize_tokens(future,torch.tensor([1,1])),rtol=0,atol=0)
    pm,pl=m.prior(context); qm,ql=m.posterior(data,context)
    assert pm.shape==pl.shape==qm.shape==ql.shape==(2,4)
    assert torch.isfinite(gaussian_kl(qm,ql,pm,pl)).all()
    assert not torch.equal(reparameterize(qm,ql),reparameterize(qm,ql))
    torch.testing.assert_close(reparameterize(qm,ql,True),qm)
    empty=m.prior(m.context([[]])); assert all(torch.count_nonzero(x)==0 for x in empty)


def test_relational_mask_next_distributions_and_prefix_exclusion():
    m=tiny().eval(); data=sketches(); v=m.view(data); tokens=m.stroke_tokens(m.encode_view(v),v['anchors'])
    c=m.context([data[0][:1],data[1][:1]]); z=m.prior(c)[0]
    a=m.decode_tokens(tokens,v['stroke_mask'].sum(-1),z,c)
    changed=tokens.clone(); changed[:,1:]+=50
    b=m.decode_tokens(changed,v['stroke_mask'].sum(-1),z,c)
    torch.testing.assert_close(a[:,:2],b[:,:2],rtol=0,atol=0)
    assert m.anchor_head(a).shape==(2,4,2,6)
    assert m.embedding_distribution(a,v['anchors'].new_zeros((2,4,2)))[0].shape==(2,4,8)
    assert m.sketch_end_head(a).shape==(2,4,2)
    r=m.batch_loss(data,[2,3],.05,deterministic=True)
    assert r['coordinate_events']==0 and r['pen_events']==2 and r['anchor_nll']==0


def test_generation_eos_caps_prefix_exact_and_no_point_feedback():
    m=tiny().eval(); prefix=sketches()[0][:1]; original=prefix[0].copy()
    with torch.no_grad():
        m.sketch_end_head.weight.zero_(); m.sketch_end_head.bias.copy_(torch.tensor([-20.,20.]))
    result,info=sample(m,prefix=prefix,temperature=0,max_points=16,max_strokes=2,return_info=True)
    assert info['termination']=='eos' and info['generated_strokes']==0
    assert result[0].tobytes()==original.tobytes() and result[0].dtype==original.dtype
    with torch.no_grad(): m.sketch_end_head.bias.copy_(torch.tensor([20.,-20.]))
    calls=[]; hook=m.stroke_decoder.register_forward_hook(lambda _,args,out:calls.append(out.shape))
    result,info=sample(m,prefix=prefix,temperature=.9,max_points=17,max_strokes=4,return_info=True)
    hook.remove()
    assert calls==[torch.Size([1,8,2])]*2
    assert info['termination']=='max_points' and not info['ended_by_eos']
    assert prefix_preserved(prefix,result) and all(np.isfinite(s).all() for s in result)
    _,info=sample(m,prefix=prefix,max_points=100,max_strokes=1,return_info=True)
    assert info['termination']=='max_strokes'
    assert prefix[0].tobytes()==original.tobytes()


def test_common_generation_ranking_and_render_apis():
    m=tiny().eval(); prefix=sketches()[0][:1]
    assert len(generate_random(m,n_samples=2,max_points=16,max_strokes=2))==2
    assert all(prefix_preserved(prefix,s) for s in complete_sketch(m,prefix,n_samples=2,max_points=16,max_strokes=2))
    result=sample_multiple(m,prefix=prefix,n_candidates=2,top_k=1,max_points=16,max_strokes=2,validate=False)
    assert 'selected' in result and 'report' in result
    from sketchlab.rendering import render
    assert '<svg' in render(prefix)


def test_checkpoint_ae_inference_guard_and_save_load(tmp_path):
    m=tiny(); path=tmp_path/'model.pt'; save_checkpoint(path,m)
    loaded=load_model(path)
    for k,v in m.state_dict().items(): assert torch.equal(v,loaded.state_dict()[k])
    ae=tmp_path/'ae.pt'; save_checkpoint(ae,m,config={'training_stage':'stroke_ae'})
    with pytest.raises(ValueError,match='representation checkpoint'): load_model(ae)
    assert load_checkpoint(ae)[0].training_stage=='stroke_ae'
    assert torch.load(ae,weights_only=True)['model_representation']['stroke_samples']==8
    payload=torch.load(path,weights_only=True); payload['model_representation']['stroke_samples']=99
    corrupt=tmp_path/'incompatible.pt'; torch.save(payload,corrupt)
    with pytest.raises(ValueError,match='stroke-view'): load_model(corrupt)
    with pytest.raises(ValueError,match='stroke-view'): load_checkpoint(corrupt)


@pytest.mark.parametrize('stage',['stroke_ae','composition'])
def test_exact_resume_f(tmp_path,stage):
    data=[{'id':i,'strokes':s} for i,s in enumerate(sketches())]
    config={'seed':19,'device':'cpu','cpu_threads':1,'batch_size':2,'steps':4,'eval_every':2,
        'max_train_seconds':60,'learning_rate':.001,'generation_eval':False,'training_stage':stage,
        'freeze_stroke_ae':True,'architecture':tiny().config}
    run_experiment(config,'F',data,data,tmp_path/'full')
    run_experiment({**config,'steps':2},'F',data,data,tmp_path/'first')
    run_experiment({**config,'steps':2},'F',data,data,tmp_path/'resume',resume=tmp_path/'first/last.pt')
    a=torch.load(tmp_path/'full/last.pt',weights_only=True); b=torch.load(tmp_path/'resume/last.pt',weights_only=True)
    assert a['step']==b['step']==4
    assert a['training_state']['sampler']==b['training_state']['sampler']
    assert torch.equal(a['rng_state']['torch_cpu'],b['rng_state']['torch_cpu'])
    for k in a['model_state']: assert torch.equal(a['model_state'][k],b['model_state'][k]),k


def test_validation_partition_invariance():
    m=tiny().eval(); data=[{'strokes':s} for s in sketches()]
    a=validate(m,data,1,.05,.02); b=validate(m,data,2,.05,.02)
    for k in a: assert a[k]==pytest.approx(b[k],rel=3e-5,abs=1e-5),k


def test_factory_string():
    assert create_model('F').model_name=='F'


def test_stage_transfer_frozen_ae_and_optimizer_groups(tmp_path):
    source=tiny(); source.configure_training('stroke_ae')
    path=tmp_path/'ae.pt'; save_checkpoint(path,source,config={'training_stage':'stroke_ae'})
    data=[{'id':i,'strokes':s} for i,s in enumerate(sketches())]
    config={'seed':19,'device':'cpu','cpu_threads':1,'batch_size':2,'steps':1,'eval_every':1,
        'max_train_seconds':60,'learning_rate':.001,'generation_eval':False,'training_stage':'composition',
        'freeze_stroke_ae':True,'stroke_ae_checkpoint':str(path),'architecture':source.config}
    run_experiment(config,'F',data,data,tmp_path/'composition')
    loaded,_=load_checkpoint(tmp_path/'composition/last.pt')
    for name,p in source.state_dict().items():
        if name.startswith(('stroke_encoder.','stroke_decoder.')):
            assert torch.equal(p,loaded.state_dict()[name])
    assert not torch.equal(source.sketch_end_head.weight,loaded.sketch_end_head.weight)
    run_experiment({**config,'freeze_stroke_ae':False,'stroke_ae_lr_factor':.1},'F',data,data,tmp_path/'finetune')
    state=torch.load(tmp_path/'finetune/last.pt',weights_only=True)
    assert [g['lr'] for g in state['optimizer_state']['param_groups']]==[.001,.0001]


def test_ui_lists_only_generative_f_checkpoints(tmp_path):
    from fastapi.testclient import TestClient
    from app.server import create_app
    m=tiny()
    save_checkpoint(tmp_path/'ae/F/last.pt',m,config={'training_stage':'stroke_ae'})
    save_checkpoint(tmp_path/'composition/F/last.pt',m,config={'training_stage':'composition'})
    response=TestClient(create_app(runs_dir=tmp_path)).get('/api/models')
    assert response.status_code==200
    assert [r['id'] for r in response.json()['models']]==['composition/F/last.pt']
