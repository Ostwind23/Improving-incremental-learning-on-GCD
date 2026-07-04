import torch
from mmengine.registry import DefaultScope
DefaultScope.get_instance('mmdet', scope_name='mmdet')
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmdet.registry import MODELS

cfg = Config.fromfile('configs/gdino_inc/70+10/grmi_rms_gate_3e.py')
cfg.default_scope = 'mmdet'
m = MODELS.build(cfg.model)
load_checkpoint(m, 'work_dirs/grmi_rms_gate_3e_v2/epoch_3.pth', map_location='cuda:0')
m.cuda(); m.train()

img = torch.randn(1,3,800,800).cuda()
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData
ds = DetDataSample()
ds.text = 'person . bicycle . car . dog . cat . book'
ds.ori_text = 'person . bicycle . car . dog . cat . book'
ds.gt_instances = InstanceData()
ds.gt_instances.bboxes = torch.tensor([[100,100,200,200]], dtype=torch.float).cuda()
ds.gt_instances.labels = torch.tensor([0]).cuda()

print("gate_net params:")
for n, p in m.residual_inject.gate_net.named_parameters():
    print(f"  {n}: requires_grad={p.requires_grad}")

losses = m.loss([img], [ds])
total = sum(v for v in losses.values() if isinstance(v, torch.Tensor) and v.requires_grad)
print(f"\nTotal loss: {total.item():.4f} requires_grad={total.requires_grad}")
total.backward()

print("\ngate_net grad after backward:")
for n, p in m.residual_inject.gate_net.named_parameters():
    if p.grad is not None:
        print(f"  {n}: grad abs mean = {p.grad.abs().mean():.2e}")
    else:
        print(f"  {n}: grad = None")
