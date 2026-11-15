"""Meaningful implementation and data checks before formal training."""
import gc
import json
import time
import torch
from torchvision.models import mobilenet_v2
from models import build_model, VARIANTS
from data import ROOT, DataBundle, write_json
from experiment import setup, environment

def main():
    setup(42);results={};device=torch.device('cuda:0');data=DataBundle(device)
    groups={s:{r['group'] for r in data.rows if r['split']==s} for s in ['train','val','test']}
    assert not groups['train']&groups['val'] and not groups['train']&groups['test'] and not groups['val']&groups['test']
    results['split_group_disjoint']=True
    ours=build_model('mobilenet_v2',42).eval()
    reference=mobilenet_v2(weights=None,num_classes=6).eval();reference.load_state_dict(ours.state_dict(),strict=True)
    x=torch.randn(2,3,200,200)
    with torch.no_grad():a=ours(x);b=reference(x)
    assert torch.allclose(a,b,rtol=1e-5,atol=1e-6)
    results['torchvision_forward_max_abs_difference']=float((a-b).abs().max())
    results['baseline_parameter_count']=sum(p.numel() for p in ours.parameters())
    attention=build_model('mobilenet_v2_se',42)
    for name,p in ours.named_parameters():assert torch.equal(p,dict(attention.named_parameters())[name])
    results['shared_initialization_identical']=True
    del ours,reference,attention;gc.collect()
    probes={}
    for variant in VARIANTS:
        torch.cuda.empty_cache();model=build_model(variant,42).to(device,memory_format=torch.channels_last).train()
        optimizer=torch.optim.AdamW(model.parameters(),lr=0.001)
        ids=data.indices['train'][:32];inputs,labels=data.batch(ids)
        old=model.classifier[1].weight.detach().clone();torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize();tick=time.perf_counter()
        with torch.cuda.amp.autocast():logits=model(inputs);loss=torch.nn.functional.cross_entropy(logits,labels)
        assert logits.shape==(32,6) and torch.isfinite(loss)
        loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step();torch.cuda.synchronize()
        assert not torch.equal(old,model.classifier[1].weight)
        probes[variant]={'parameters':sum(p.numel() for p in model.parameters()),'smoke_loss':float(loss),
                         'batch32_forward_backward_step_seconds':time.perf_counter()-tick,
                         'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
        del model,optimizer,logits,loss,old;gc.collect()
    assert probes['mobilenet_v2_dense']['parameters']>probes['mobilenet_v2']['parameters']
    results['cuda_training_probes']=probes
    # A balanced 24-image overfit check proves the training pipeline can learn.
    torch.cuda.empty_cache();model=build_model('mobilenet_v2',42).to(device,memory_format=torch.channels_last).train()
    ids=torch.cat([data.indices['train'][data.labels[data.indices['train']]==c][:4] for c in range(6)])
    inputs,labels=data.batch(ids);optimizer=torch.optim.Adam(model.parameters(),lr=0.003)
    losses=[]
    for step in range(100):
        optimizer.zero_grad(set_to_none=True);logits=model(inputs);loss=torch.nn.functional.cross_entropy(logits,labels)
        loss.backward();optimizer.step();losses.append(float(loss))
    accuracy=float((logits.argmax(1)==labels).float().mean())
    assert losses[-1]<losses[0]*0.35 and accuracy>=0.9, (losses[0],losses[-1],accuracy)
    results['tiny_batch_overfit']={'steps':100,'images':len(ids),'initial_loss':losses[0],
                                 'final_loss':losses[-1],'train_mode_accuracy':accuracy}
    results['environment']=environment();results['status']='passed'
    write_json(ROOT/'outputs'/'implementation_checks.json',results)
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
