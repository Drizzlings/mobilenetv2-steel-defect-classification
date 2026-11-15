"""Train, evaluate, and profile real NEU-CLS experiments; no synthetic results."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import platform
import random
import time
import numpy as np
import torch
from torch import nn
import torchvision
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from data import ROOT, CLASSES, DataBundle, write_json
from models import build_model

def setup(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)

def environment():
    return {'python':platform.python_version(),'platform':platform.platform(),'torch':torch.__version__,
            'torchvision':torchvision.__version__,'numpy':np.__version__,'cuda':torch.version.cuda,
            'cudnn':torch.backends.cudnn.version(), 'gpu':torch.cuda.get_device_name(0),
            'gpu_memory_bytes':torch.cuda.get_device_properties(0).total_memory,
            'cpu_threads':torch.get_num_threads(),'deterministic_algorithms':True,'tf32':False}

def result_dir(variant,seed): return ROOT/'outputs'/'runs'/f'{variant}_seed{seed}'

def lr_at(epoch,cfg):
    if epoch<cfg['warmup_epochs']: return cfg['learning_rate']*(epoch+1)/cfg['warmup_epochs']
    fraction=(epoch-cfg['warmup_epochs'])/max(1,cfg['epochs']-cfg['warmup_epochs']-1)
    return cfg['min_learning_rate']+(cfg['learning_rate']-cfg['min_learning_rate'])*(1+math.cos(math.pi*fraction))/2

def digest_config(cfg): return hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()

def save_torch(path,value):
    tmp=path.with_suffix('.tmp'); torch.save(value,tmp); tmp.replace(path)

def save_history(path,history):
    tmp=path.with_suffix('.tmp')
    with tmp.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
    tmp.replace(path)

@torch.no_grad()
def predictions(model,data,split,batch_size,corruption=None,strength=None):
    model.eval(); indices=data.indices[split]; logits=[]; targets=[]
    generator=torch.Generator(device=data.device).manual_seed(2026)
    for start in range(0,len(indices),batch_size):
        ids=indices[start:start+batch_size]
        x,y=data.batch(ids,corruption=corruption,strength=strength,generator=generator)
        logits.append(model(x).float().cpu());targets.append(y.cpu())
    logit=torch.cat(logits); y=torch.cat(targets)
    return logit,y,indices.cpu()

def scores(logits,y):
    truth=y.numpy(); pred=logits.argmax(1).numpy()
    precision,recall,f1,support=precision_recall_fscore_support(truth,pred,labels=list(range(6)),zero_division=0)
    return {'accuracy':float(accuracy_score(truth,pred)), 'macro_f1':float(f1.mean()),
            'macro_precision':float(precision.mean()),'macro_recall':float(recall.mean()),
            'cross_entropy':float(nn.functional.cross_entropy(logits,y)),
            'num_samples':len(truth),'correct':int((truth==pred).sum()),
            'confusion_matrix':confusion_matrix(truth,pred,labels=list(range(6))).tolist(),
            'per_class':[{ 'class':CLASSES[i],'precision':float(precision[i]),'recall':float(recall[i]),
                           'f1':float(f1[i]),'support':int(support[i])} for i in range(6)]}

def train(variant,seed,resume=False,config_path='configs/config.json'):
    cfg=json.loads((ROOT/config_path).read_text()); setup(seed)
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda': raise RuntimeError('Formal server experiments require a CUDA GPU')
    output=result_dir(variant,seed);output.mkdir(parents=True,exist_ok=True)
    if (output/'train_summary.json').exists():
        previous=json.loads((output/'train_summary.json').read_text())
        if previous['config_sha256']!=digest_config(cfg): raise RuntimeError('Completed run configuration differs')
        print('Training already complete:',variant,flush=True);return
    model=build_model(variant,seed).to(device,memory_format=torch.channels_last)
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    scaler=torch.cuda.amp.GradScaler(enabled=cfg['amp'])
    criterion=nn.CrossEntropyLoss(label_smoothing=cfg['label_smoothing'])
    data=DataBundle(device);history=[];start_epoch=0;best_key=(-1.0,-float('inf'));best_epoch=-1
    if resume and (output/'last.pt').exists():
        last=torch.load(output/'last.pt',map_location='cpu')
        if last['config_sha256']!=digest_config(cfg): raise RuntimeError('Resume configuration mismatch')
        if last['manifest_sha256']!=data.stats['manifest_sha256']: raise RuntimeError('Resume dataset mismatch')
        model.load_state_dict(last['model']);optimizer.load_state_dict(last['optimizer']);scaler.load_state_dict(last['scaler'])
        start_epoch=last['epoch']+1;history=last['history'];best_key=tuple(last['best_key']);best_epoch=last['best_epoch']
        torch.set_rng_state(last['torch_rng']);torch.cuda.set_rng_state_all(last['cuda_rng'])
        random.setstate(last['python_rng']);np.random.set_state(last['numpy_rng'])
    write_json(output/'config.json',cfg);write_json(output/'environment.json',environment())
    torch.cuda.reset_peak_memory_stats(); training_start=time.perf_counter()
    for epoch in range(start_epoch,cfg['epochs']):
        learning_rate=lr_at(epoch,cfg)
        for group in optimizer.param_groups:group['lr']=learning_rate
        model.train(); generator=torch.Generator(device=device).manual_seed(seed*100000+epoch)
        train_ids=data.indices['train'];order=train_ids[torch.randperm(len(train_ids),generator=generator,device=device)]
        total_loss=0.0;correct=0;count=0
        torch.cuda.synchronize();tick=time.perf_counter()
        for start in range(0,len(order),cfg['batch_size']):
            ids=order[start:start+cfg['batch_size']]
            x,y=data.batch(ids,augment=True,generator=generator)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg['amp']):
                logits=model(x);loss=criterion(logits,y)
            if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss in epoch {epoch+1}')
            scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
            total_loss+=float(loss.detach())*len(y);correct+=int((logits.argmax(1)==y).sum());count+=len(y)
        torch.cuda.synchronize();train_seconds=time.perf_counter()-tick
        val_logits,val_y,_=predictions(model,data,'val',cfg['batch_size'])
        val=scores(val_logits,val_y);key=(val['accuracy'],-val['cross_entropy'])
        if key>best_key:
            best_key=key;best_epoch=epoch+1
            save_torch(output/'best.pt',{'model':model.state_dict(),'variant':variant,'seed':seed,'epoch':best_epoch,
                       'validation':val,'dataset_stats':data.stats,'config':cfg,'config_sha256':digest_config(cfg)})
        row={'epoch':epoch+1,'learning_rate':learning_rate,'train_loss':total_loss/count,
             'train_accuracy':correct/count,'validation_loss':val['cross_entropy'],
             'validation_accuracy':val['accuracy'],'validation_macro_f1':val['macro_f1'],
             'train_seconds':train_seconds,'train_images_per_second':count/train_seconds,
             'elapsed_seconds_this_process':time.perf_counter()-training_start}
        history.append(row);save_history(output/'history.csv',history)
        save_torch(output/'last.pt',{'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                   'scaler':scaler.state_dict(),'epoch':epoch,'history':history,'best_key':best_key,'best_epoch':best_epoch,
                   'config_sha256':digest_config(cfg),'manifest_sha256':data.stats['manifest_sha256'],
                   'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                   'python_rng':random.getstate(),'numpy_rng':np.random.get_state()})
        write_json(output/'status.json',{'stage':'training','variant':variant,'epoch':epoch+1,'epochs':cfg['epochs'],
                   'best_validation_accuracy':best_key[0],'updated_unix':time.time()})
        print(json.dumps({'model':variant,**row,'best_epoch':best_epoch}),flush=True)
    summary={'variant':variant,'seed':seed,'epochs':cfg['epochs'],'best_epoch':best_epoch,
             'best_validation_accuracy':best_key[0], 'config_sha256':digest_config(cfg),
             'train_seconds_total':sum(r['train_seconds'] for r in history),
             'train_images_per_second':len(data.indices['train'])*len(history)/sum(r['train_seconds'] for r in history),
             'peak_training_allocated_bytes':torch.cuda.max_memory_allocated(),
             'num_parameters':sum(p.numel() for p in model.parameters()),'finished_unix':time.time()}
    write_json(output/'train_summary.json',summary)
    write_json(output/'status.json',{'stage':'trained',**summary})

def dump_predictions(path,logits,y,ids,data):
    probs=logits.softmax(1).numpy(); pred=probs.argmax(1)
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['file','true_label','pred_label','true_class','pred_class']+[f'prob_{c}' for c in CLASSES])
        for j,index in enumerate(ids):
            truth=int(y[j]);p=int(pred[j]);writer.writerow([data.rows[int(index)]['file'],truth,p,CLASSES[truth],CLASSES[p],*probs[j].tolist()])

def evaluate(variant,seed,config_path='configs/config.json'):
    cfg=json.loads((ROOT/config_path).read_text());setup(seed)
    # Test access is gated on completion of every planned training run.
    for m in cfg['models']:
        for s in cfg['seeds']:
            if not (result_dir(m,s)/'train_summary.json').exists():raise RuntimeError('Finish all training before opening test results')
    device=torch.device('cuda:0');data=DataBundle(device);output=result_dir(variant,seed)
    checkpoint=torch.load(output/'best.pt',map_location='cpu')
    model=build_model(variant,seed).to(device,memory_format=torch.channels_last)
    model.load_state_dict(checkpoint['model']);model.eval()
    conditions={'clean':None,**cfg['corruptions']};results={}
    for condition,strength in conditions.items():
        logits,y,ids=predictions(model,data,'test',cfg['batch_size'],None if condition=='clean' else condition,strength)
        results[condition]=scores(logits,y)
        dump_predictions(output/f'predictions_{condition}.csv',logits,y,ids,data)
        print(variant,condition,results[condition]['accuracy'],flush=True)
    write_json(output/'evaluation.json',{'variant':variant,'seed':seed,'checkpoint_epoch':checkpoint['epoch'],
               'manifest_sha256':data.stats['manifest_sha256'],'conditions':results})
    profile(model,output,cfg)
    write_json(output/'status.json',{'stage':'complete','variant':variant,'seed':seed,'updated_unix':time.time()})

def profile(model,output,cfg,checkpoint_path=None):
    macs=0;handles=[]
    def hook(module,inputs,output_tensor):
        nonlocal macs
        if isinstance(module,nn.Conv2d):
            macs+=output_tensor.numel()*(module.in_channels//module.groups)*module.kernel_size[0]*module.kernel_size[1]
        elif isinstance(module,nn.Linear):macs+=output_tensor.numel()*module.in_features
    for module in model.modules():
        if isinstance(module,(nn.Conv2d,nn.Linear)):handles.append(module.register_forward_hook(hook))
    x=torch.randn(1,3,cfg['input_size'],cfg['input_size'],device='cuda').contiguous(memory_format=torch.channels_last)
    with torch.inference_mode():model(x)
    for h in handles:h.remove()
    timing={}
    with torch.inference_mode():
        for batch in [1,cfg['profile_batch_size']]:
            inp=x.repeat(batch,1,1,1).contiguous(memory_format=torch.channels_last)
            for _ in range(cfg['latency_warmup']):model(inp)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();times=[]
            for _ in range(cfg['latency_iterations']):
                begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                begin.record();model(inp);end.record();end.synchronize();times.append(begin.elapsed_time(end))
            timing[str(batch)]={'median_ms':float(np.median(times)),'p95_ms':float(np.percentile(times,95)),
                                'mean_ms':float(np.mean(times)),'images_per_second':float(batch*1000/np.mean(times)),
                                'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
    value={'parameters':sum(p.numel() for p in model.parameters()),'conv_linear_macs':macs,
           'conv_linear_flops_2xmacs':2*macs,'input_shape':[1,3,cfg['input_size'],cfg['input_size']],
           'best_checkpoint_bytes':(checkpoint_path or output/'best.pt').stat().st_size,'precision':'float32','memory_format':'channels_last',
           'timing_scope':'CUDA event forward-only; excludes image decoding, transfers and preprocessing',
           'operation_scope':'Conv2d and Linear only; excludes BN, activation, pooling, attention elementwise multiply and residual add',
           'timing':timing,'warmup':cfg['latency_warmup'],'iterations':cfg['latency_iterations']}
    write_json(output/'profile.json',value)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['train','evaluate']);p.add_argument('--model',required=True)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--resume',action='store_true')
    p.add_argument('--config',default='configs/config.json');a=p.parse_args()
    if a.action=='train':train(a.model,a.seed,a.resume,a.config)
    else:evaluate(a.model,a.seed,a.config)

if __name__=='__main__':main()
