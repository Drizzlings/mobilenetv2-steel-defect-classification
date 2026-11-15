"""Validate CBAM shape, initialization isolation, gradients and prediction metrics."""
import json, csv
from pathlib import Path
import torch
from models import build_model, CBAM
from experiment import setup
setup(42)
base=build_model('mobilenet_v2');model=build_model('mobilenet_v2_cbam')
shared=dict(base.named_parameters())
assert all(torch.equal(p,shared[n]) for n,p in model.named_parameters() if n in shared)
assert sum(isinstance(m,CBAM) for m in model.modules())==17
x=torch.randn(2,3,200,200);y=model(x)
assert y.shape==(2,6)
loss=torch.nn.functional.cross_entropy(y,torch.tensor([0,1]));loss.backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
r=Path(__file__).resolve().parents[1]/'outputs/runs/mobilenet_v2_cbam_seed42'
evaluation=json.loads((r/'evaluation.json').read_text())
for condition,metrics in evaluation['conditions'].items():
    with (r/f'predictions_{condition}.csv').open() as f: rows=list(csv.DictReader(f))
    correct=sum(v['true_label']==v['pred_label'] for v in rows)
    assert abs(correct/len(rows)-metrics['accuracy'])<1e-12
result={'status':'passed','cbam_blocks':17,'shared_initialization_identical':True,'finite_backward':True,'prediction_conditions_checked':5}
(r/'implementation_checks.json').write_text(json.dumps(result,indent=2));print(result)
