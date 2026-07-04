"""Quantitative diagnosis: actual gradient magnitude at gate_net vs main model.
Loads RMS+Gate 3e checkpoint, runs training forward+backward on val images,
measures grad norms at gate_net.2.bias (the frozen one) vs backbone/decoder."""
import torch, numpy as np, os
from mmengine.registry import DefaultScope
DefaultScope.get_instance('mmdet', scope_name='mmdet')
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmdet.registry import MODELS
import mmcv
from pycocotools.coco import COCO

CKPT = 'work_dirs/grmi_rms_gate_3e_v2/epoch_3.pth'
CONFIG = 'configs/gdino_inc/70+10/grmi_rms_gate_3e.py'

cfg = Config.fromfile(CONFIG)
cfg.default_scope = 'mmdet'
m = MODELS.build(cfg.model)
load_checkpoint(m, CKPT, map_location='cuda:0')
m.cuda()
m.train()  # enable grad tracking

coco = COCO('data/coco/annotations/instances_val2017.json')
img_ids = sorted(coco.getImgIds())[:50]

results = {
    'gate_bias_grad': [],
    'gate_weight_grad': [],
    'W_out_grad': [],
    'W_m_grad': [],
    'backbone_grad': [],
    'decoder_grad': [],
    'loss_vals': [],
}

for n_img, img_id in enumerate(img_ids):
    info = coco.loadImgs([img_id])[0]
    anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
    boxes, labels = [], []
    for a in anns:
        if a.get('iscrowd', 0): continue
        x, y, w, h = a['bbox']
        boxes.append([x, y, x+w, y+h])
        labels.append(a['category_id'] - 1)
    if not boxes: continue
    
    img = mmcv.imread(f'data/coco/val2017/{info["file_name"]}')
    img_t = torch.from_numpy(img).float().permute(2, 0, 1).cuda()
    mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1).cuda()
    std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1).cuda()
    img_t = (img_t - mean) / std
    
    # Build text prompt
    cat_ids = sorted(coco.getCatIds())
    names = [coco.loadCats([c])[0]['name'] for c in cat_ids]
    prompt = ' . '.join(names) + ' .'
    
    from mmdet.structures import DetDataSample
    from mmengine.structures import InstanceData
    
    ds = DetDataSample()
    ds.text = prompt
    ds.ori_text = prompt
    ds.gt_instances = InstanceData()
    ds.gt_instances.bboxes = torch.tensor(boxes).cuda()
    ds.gt_instances.labels = torch.tensor(labels).cuda()
    
    try:
        losses = m.loss([img_t], [ds])
    except Exception as e:
        continue
    
    total_loss = sum(v for v in losses.values() if isinstance(v, torch.Tensor))
    total_loss.backward()
    
    # Record gate grad
    gate_bias = m.residual_inject.gate_net[2].bias
    gate_weight = m.residual_inject.gate_net[2].weight
    if gate_bias.grad is not None:
        results['gate_bias_grad'].append(gate_bias.grad.abs().mean().item())
        results['gate_weight_grad'].append(gate_weight.grad.abs().mean().item())
    else:
        results['gate_bias_grad'].append(0)
        results['gate_weight_grad'].append(0)
    
    # Record W_out grad
    w_out = m.residual_inject.W_out.weight
    if w_out.grad is not None:
        results['W_out_grad'].append(w_out.grad.abs().mean().item())
    else:
        results['W_out_grad'].append(0)
    
    w_m = m.residual_inject.W_m.weight
    if w_m.grad is not None:
        results['W_m_grad'].append(w_m.grad.abs().mean().item())
    else:
        results['W_m_grad'].append(0)
    
    # Average backbone grad (first conv layer)
    bb_param = list(m.backbone.parameters())[0]
    if bb_param.grad is not None:
        results['backbone_grad'].append(bb_param.grad.abs().mean().item())
    else:
        results['backbone_grad'].append(0)
    
    # Average decoder grad (first attn)
    dec_params = list(m.bbox_head.decoder.parameters())
    dec_grads = [p.grad.abs().mean().item() for p in dec_params[:5] if p.grad is not None]
    if dec_grads:
        results['decoder_grad'].append(np.mean(dec_grads))
    
    results['loss_vals'].append(total_loss.item())
    m.zero_grad()
    
    if (n_img + 1) % 5 == 0:
        print(f"  {n_img+1} images done")

print("\n========= Gradient Magnitude Report =========")
for k in ['gate_bias_grad', 'gate_weight_grad', 'W_out_grad', 'W_m_grad', 'backbone_grad', 'decoder_grad']:
    v = results[k]
    if v and any(x > 0 for x in v):
        a = np.array([x for x in v if x > 0])
        print(f"  {k:20s}: mean={a.mean():.2e} median={np.median(a):.2e} std={a.std():.2e} (n={len(a)}/{len(v)})")

# Compute ratios
gb = np.array([x for x in results['gate_bias_grad'] if x > 0])
bb = np.array([x for x in results['backbone_grad'] if x > 0])
dec = np.array([x for x in results['decoder_grad'] if x > 0])
wout = np.array([x for x in results['W_out_grad'] if x > 0])
wm = np.array([x for x in results['W_m_grad'] if x > 0])

print(f"\n  Gate bias / backbone ratio: {gb.mean()/bb.mean() if len(bb)>0 else 0:.2e}")
print(f"  Gate bias / decoder ratio:  {gb.mean()/dec.mean() if len(dec)>0 else 0:.2e}")
print(f"  Gate bias / W_out ratio:    {gb.mean()/wout.mean() if len(wout)>0 else 0:.2e}")
print(f"  Gate bias / W_m ratio:      {gb.mean()/wm.mean() if len(wm)>0 else 0:.2e}")
print(f"\n  Effective LR fraction: {gb.mean()/(5e-5):.1e}")
print(f"  Needed LR multiplier for 1e-4 effective: {1e-4/max(gb.mean(),1e-20):.0f}x")
