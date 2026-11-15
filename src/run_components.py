"""Finite three-run activation/normalization ablation; preserves original runs."""
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from data import ROOT, write_json

def implementation_checks():
    import torch
    from torch import nn
    from torchvision.models import mobilenet_v2
    from models import build_model, InvertedResidual
    from experiment import setup
    setup(42)
    cfg=json.loads((ROOT/'configs'/'config_components.json').read_text())
    original=json.loads((ROOT/'configs'/'config.json').read_text())
    assert {k:v for k,v in cfg.items() if k!='models'}=={k:v for k,v in original.items() if k!='models'}
    base=build_model('mobilenet_v2').eval()
    reference=mobilenet_v2(num_classes=6).eval();reference.load_state_dict(base.state_dict())
    x=torch.randn(2,3,200,200)
    with torch.no_grad():diff=float((base(x)-reference(x)).abs().max())
    assert diff==0
    records=[]
    for variant in cfg['models']:
        model=build_model(variant)
        assert sum(p.numel() for p in model.parameters())==sum(p.numel() for p in base.parameters())
        assert all(torch.equal(p,dict(base.named_parameters())[name]) for name,p in model.named_parameters())
        counts={name:sum(isinstance(m,kind) for m in model.modules()) for name,kind in
                [('relu6',nn.ReLU6),('gelu',nn.GELU),('mish',nn.Mish),('bn',nn.BatchNorm2d),('gn',nn.GroupNorm)]}
        if variant.endswith('_gn'):assert counts['bn']==0 and counts['gn']==52 and counts['relu6']==35
        else:assert counts['bn']==52 and counts['relu6']==0 and counts[variant.rsplit('_',1)[1]]==35
        for block in model.modules():
            if isinstance(block,InvertedResidual):assert isinstance(block.conv[-1],(nn.BatchNorm2d,nn.GroupNorm))
        groups=sorted({(m.num_channels,m.num_groups) for m in model.modules() if isinstance(m,nn.GroupNorm)})
        model=model.cuda().to(memory_format=torch.channels_last).train()
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        before=model.classifier[-1].weight.detach().clone()
        with torch.cuda.amp.autocast():loss=nn.functional.cross_entropy(model(x.cuda()),torch.tensor([0,1],device='cuda'))
        loss.backward();assert torch.isfinite(loss)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step();assert not torch.equal(before,model.classifier[-1].weight)
        records.append({'model':variant,'counts':counts,'gn_channel_group_pairs':groups,'shared_initial_parameters_identical':True,
                        'linear_bottlenecks_preserved':True,'cuda_amp_backward_update':'passed','probe_loss':float(loss)})
        del model,optimizer
    write_json(ROOT/'outputs/components_checks.json',{'status':'passed','torchvision_max_abs_difference':diff,'models':records})

def main():
    os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
    output=ROOT/'outputs';output.mkdir(exist_ok=True);lock=(output/'components.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('Components suite already running')
    cfg=json.loads((ROOT/'configs'/'config_components.json').read_text())
    status={'state':'running','started_unix':time.time(),'current':'checks','completed':[]}
    write_json(output/'components_source_hashes.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src').glob('*.py')})
    stop=threading.Event()
    def snapshot(reason):
        write_json(output/'components_status.json',status)
        with (output/'components_progress.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps({'time':time.time(),'reason':reason,**status})+'\n')
    def heartbeat():
        while not stop.wait(cfg['heartbeat_seconds']):snapshot('scheduled_2_hour_check')
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    try:
        snapshot('start');implementation_checks()
        for action in ['train','evaluate']:
            for model in cfg['models']:
                status['current']={'action':action,'model':model};snapshot('start_stage')
                command=[sys.executable,'-u',str(ROOT/'src'/'experiment.py'),action,'--model',model,'--seed','42','--config','configs/config_components.json']
                if action=='train':command.append('--resume')
                for attempt in range(2):
                    with (output/f'{model}_seed42_{action}.log').open('a') as log:
                        code=subprocess.call(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                    if code==0:break
                    snapshot('retry')
                if code:raise RuntimeError(f'{action} failed: {model}')
                status['completed'].append({'action':action,'model':model});snapshot('stage_complete')
        status.update(state='complete',current=None,finished_unix=time.time());snapshot('complete')
    except BaseException as e:
        status.update(state='failed',error=str(e));snapshot('failed');raise
    finally:stop.set();thread.join(timeout=2)

if __name__=='__main__':main()
