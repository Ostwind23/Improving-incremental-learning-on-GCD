"""Get encoder feature map resolution for gate BCE loss mapping."""
import torch, mmcv
from mmengine.registry import DefaultScope
DefaultScope.get_instance('mmdet', scope_name='mmdet')
from mmengine.config import Config
from mmdet.registry import MODELS

cfg = Config.fromfile('configs/gdino_inc/70+10/grmi_rms_gate_3e.py')
cfg.default_scope = 'mmdet'
m = MODELS.build(cfg.model)
m.cuda(); m.eval()

img = mmcv.imread('data/coco/val2017/000000000139.jpg')
img_t = torch.from_numpy(img).float().permute(2,0,1).unsqueeze(0).cuda()
mean = torch.tensor([123.675,116.28,103.53]).view(1,3,1,1).cuda()
std = torch.tensor([58.395,57.12,57.375]).view(1,3,1,1).cuda()
img_t = (img_t - mean) / std

with torch.no_grad():
    feat = m.backbone(img_t)
    print(f"Backbone: {len(feat)} levels")
    for i, f in enumerate(feat):
        h, w = f.shape[2], f.shape[3]
        print(f"  L{i}: {f.shape[1]}ch, {h}x{w}, tokens={h*w}")
    
    # Now run extract_feat + encoder
    feat2 = m.extract_feat(img_t)
    if isinstance(feat2, dict):
        print(f"\nEncoder output dict keys: {list(feat2.keys())}")
        for k, v in feat2.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape}")
    elif isinstance(feat2, (list, tuple)):
        print(f"\nEncoder output: {len(feat2)} items")
        for i, v in enumerate(feat2):
            if isinstance(v, torch.Tensor):
                print(f"  [{i}]: {v.shape}")
    else:
        print(f"\nEncoder output: {type(feat2).__name__}, shape={feat2.shape if hasattr(feat2,'shape') else 'N/A'}")
