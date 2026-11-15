"""CPU/GPU single-image or directory classification with optional Grad-CAM."""
from pathlib import Path
import argparse
import csv
import json
import numpy as np
from PIL import Image
import torch
from data import CLASSES, CHINESE
from models import build_model

def preprocess(path,stats,device):
    with Image.open(path) as im:
        im=im.convert('L').resize(tuple(stats['shape'][::-1]),Image.Resampling.BILINEAR)
        pixels=np.array(im).astype(np.float32)/255
    tensor=torch.from_numpy(pixels)[None,None].to(device)
    tensor=(tensor-stats['train_mean'])/stats['train_std']
    return tensor.expand(-1,3,-1,-1).contiguous(memory_format=torch.channels_last),pixels

def gradcam(model,x,label):
    storage={}
    def capture(module,inputs,output):
        storage['activation']=output
        output.register_hook(lambda g:storage.update(gradient=g))
    handle=model.features[-1].register_forward_hook(capture)
    try:
        model.zero_grad(set_to_none=True);logits=model(x);logits[0,label].backward()
        weights=storage['gradient'].mean((-2,-1),keepdim=True)
        cam=(storage['activation']*weights).sum(1,keepdim=True).relu()
        cam=torch.nn.functional.interpolate(cam,size=x.shape[-2:],mode='bilinear',align_corners=False)
        array=cam.detach().cpu().numpy()[0,0]
        array=(array-array.min())/(array.max()-array.min()+1e-8)
        return array
    finally:handle.remove()

def run(checkpoint_path,input_path,output_dir,device='cpu',cam=False):
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    checkpoint=torch.load(checkpoint_path,map_location='cpu')
    model=build_model(checkpoint['variant'],checkpoint['seed']).to(device,memory_format=torch.channels_last).eval()
    model.load_state_dict(checkpoint['model']);stats=checkpoint['dataset_stats']
    source=Path(input_path);dest=Path(output_dir);dest.mkdir(parents=True,exist_ok=True)
    extensions={'.png','.jpg','.jpeg','.bmp','.tif','.tiff','.webp'}
    if source.is_dir():paths=sorted(p for p in source.iterdir() if p.suffix.lower() in extensions)
    elif source.is_file():paths=[source]
    else:raise FileNotFoundError(f'Input does not exist: {source}')
    if not paths:raise ValueError('No supported images found')
    records=[]
    for index,path in enumerate(paths):
        try:
            x,pixels=preprocess(path,stats,device)
            with torch.no_grad():probs=model(x).softmax(1)[0].cpu().numpy()
            ranked=np.argsort(probs)[::-1][:3];prediction=int(ranked[0])
            record={'file':str(path),'status':'ok','pred_label':prediction,'pred_class':CLASSES[prediction],
                    'pred_class_zh':CHINESE[prediction],'probabilities':probs.tolist(),
                    'top3':[{'class':CLASSES[int(i)],'class_zh':CHINESE[int(i)],'probability':float(probs[i])} for i in ranked]}
            if cam:
                import matplotlib
                heat=gradcam(model,x,prediction)
                color=matplotlib.colormaps['jet'](heat)[...,:3]
                overlay=(np.clip(0.55*pixels[...,None]+0.45*color,0,1)*255).astype(np.uint8)
                target=dest/f'{index:04d}_{path.stem}_gradcam.png';Image.fromarray(overlay).save(target)
                record['gradcam']=target.name
            records.append(record)
        except (OSError,ValueError) as error:
            records.append({'file':str(path),'status':'error','message':str(error)})
    (dest/'predictions.json').write_text(json.dumps({'model':checkpoint['variant'],'checkpoint_epoch':checkpoint['epoch'],
       'note':'Six defect types only; probabilities are uncalibrated softmax scores, not a normal/defective decision.',
       'records':records},ensure_ascii=False,indent=2),encoding='utf-8')
    with (dest/'predictions.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f);writer.writerow(['file','status','predicted_class','predicted_class_zh','top1_probability'])
        for r in records:writer.writerow([r['file'],r['status'],r.get('pred_class',''),r.get('pred_class_zh',''),
                                         max(r['probabilities']) if r['status']=='ok' else ''])
    return records

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--input',required=True)
    p.add_argument('--output',default='outputs/predict');p.add_argument('--device',default='cpu');p.add_argument('--gradcam',action='store_true')
    a=p.parse_args();torch.set_num_threads(4)
    results=run(a.checkpoint,a.input,a.output,a.device,a.gradcam)
    for r in results:
        if r['status']=='ok': print(json.dumps({'file':r['file'],'prediction':r['pred_class_zh'],'top3':r['top3']},ensure_ascii=False))
        else:print(json.dumps(r,ensure_ascii=False))
    if all(r['status']=='error' for r in results):raise SystemExit(2)

if __name__=='__main__':main()
