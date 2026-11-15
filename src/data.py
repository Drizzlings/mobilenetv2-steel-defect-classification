"""Auditable pixel-hash-grouped stratified split and in-memory NEU-CLS loading."""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import random
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ['Cr', 'In', 'Pa', 'PS', 'RS', 'Sc']
ENGLISH = ['Crazing', 'Inclusion', 'Patches', 'Pitted surface', 'Rolled-in scale', 'Scratches']
CHINESE = ['裂纹', '夹杂', '斑块', '麻点', '轧入氧化皮', '划痕']

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)

def prepare():
    cfg = json.loads((ROOT/'configs'/'config.json').read_text())
    folder = ROOT/'data'/'NEU-CLS'
    files = sorted(folder.glob('*.bmp'))
    if len(files) != 1800: raise ValueError(f'Expected official 1800 BMP images, found {len(files)}')
    rows = []; arrays = []; grouped = defaultdict(list)
    for path in files:
        label = CLASSES.index(path.stem.split('_')[0])
        with Image.open(path) as im:
            if im.size != (200,200): raise ValueError(f'Unexpected dimensions: {path.name}: {im.size}')
            gray = np.array(im.convert('L'))
        # Also group exact rotations/reflections, because training uses these transforms.
        hashes = [hashlib.sha256(np.ascontiguousarray(np.rot90(a,k)).tobytes()).hexdigest()
                  for a in (gray, gray[:,::-1]) for k in range(4)]
        group = min(hashes)
        row = {'file': path.name, 'label': label, 'class': CLASSES[label], 'group': group,
               'file_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        rows.append(row); arrays.append(gray); grouped[group].append(len(rows)-1)
    for ids in grouped.values():
        if len({rows[i]['label'] for i in ids}) > 1: raise ValueError('Identical content has conflicting labels')
    rng = random.Random(cfg['split_seed'])
    for label in range(6):
        groups = [ids for ids in grouped.values() if rows[ids[0]]['label'] == label]
        rng.shuffle(groups); counts = Counter()
        for ids in groups:
            split = 'test' if counts['test'] < 60 else 'val' if counts['val'] < 60 else 'train'
            for i in ids: rows[i]['split'] = split
            counts[split] += len(ids)
    data = np.stack(arrays)
    training = data[[i for i,row in enumerate(rows) if row['split']=='train']].astype(np.float64)/255
    mean, std = float(training.mean()), float(training.std())
    stats = {'num_images':len(rows), 'classes':CLASSES, 'class_names_zh':CHINESE,
             'class_names_en':ENGLISH, 'shape':[200,200], 'mode':'grayscale',
             'split_seed':cfg['split_seed'], 'train_mean':mean, 'train_std':std,
             'split_counts':{s:dict(Counter(r['class'] for r in rows if r['split']==s)) for s in ['train','val','test']},
             'duplicate_or_symmetry_groups':[ [rows[i]['file'] for i in ids] for ids in grouped.values() if len(ids)>1],
             'grouping':'exact decoded pixel equality under 90-degree rotations and reflection',
             'limitation':'Original production-board identifiers are not provided; unseen-board generalization is not established.'}
    canonical = json.dumps(rows, sort_keys=True).encode()
    stats['manifest_sha256'] = hashlib.sha256(canonical).hexdigest()
    write_json(ROOT/'data'/'manifest.json',rows); write_json(ROOT/'data'/'dataset_stats.json',stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))

class DataBundle:
    def __init__(self, device):
        self.rows=json.loads((ROOT/'data'/'manifest.json').read_text(encoding='utf-8'))
        self.stats=json.loads((ROOT/'data'/'dataset_stats.json').read_text(encoding='utf-8'))
        arrays=[]
        for r in self.rows:
            with Image.open(ROOT/'data'/'NEU-CLS'/r['file']) as im: arrays.append(np.array(im.convert('L')))
        self.images=torch.from_numpy(np.stack(arrays)[:,None]).to(device)
        self.labels=torch.tensor([r['label'] for r in self.rows], device=device)
        self.indices={s:torch.tensor([i for i,r in enumerate(self.rows) if r['split']==s], device=device)
                      for s in ['train','val','test']}
        self.device=device

    def batch(self, indices, augment=False, generator=None, corruption=None, strength=None):
        x=self.images[indices].float()/255.0
        if augment:
            n=len(x)
            mask=torch.rand(n,1,1,1,device=x.device,generator=generator)<0.5
            x=torch.where(mask,x.flip(-1),x)
            mask=torch.rand(n,1,1,1,device=x.device,generator=generator)<0.5
            x=torch.where(mask,x.flip(-2),x)
            # Each image independently receives one of four rotations.
            rotations=torch.randint(0,4,(n,),device=x.device,generator=generator)
            original=x.clone()
            for k in range(1,4): x[rotations==k]=original[rotations==k].rot90(k,(-2,-1))
            brightness=0.85+0.30*torch.rand(n,1,1,1,device=x.device,generator=generator)
            contrast=0.85+0.30*torch.rand(n,1,1,1,device=x.device,generator=generator)
            center=x.mean((-2,-1),keepdim=True)
            x=((x-center)*contrast+center)*brightness
            x=x.clamp(0,1)
        if corruption=='brightness': x=(x*strength).clamp(0,1)
        elif corruption=='contrast':
            center=x.mean((-2,-1),keepdim=True); x=((x-center)*strength+center).clamp(0,1)
        elif corruption=='gaussian_noise_std':
            noise=torch.randn(x.shape,device=x.device,generator=generator)
            x=(x+noise*strength).clamp(0,1)
        elif corruption=='gaussian_blur_sigma':
            from torchvision.transforms.functional import gaussian_blur
            x=gaussian_blur(x,[9,9],[strength,strength])
        x=(x-self.stats['train_mean'])/self.stats['train_std']
        return x.expand(-1,3,-1,-1).contiguous(memory_format=torch.channels_last), self.labels[indices]

if __name__=='__main__': prepare()
