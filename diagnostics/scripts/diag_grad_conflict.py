#!/usr/bin/env python3
"""
Gradient Conflict Diagnostic for GCD Incremental Training.

Measures cosine similarity between detection gradients and distillation
gradients at the encoder last layer, stratified by:
  - Images with new-class GT vs images without
  - Per-parameter (attention weights, FFN, norm layers)

Also simulates:
  1. PCGrad projection: project distill grad onto orthogonal subspace when conflicting
  2. Dynamic lambda: scale distill weight by max(0, cos_sim)

Measures the resulting gradient magnitude and direction change.

Uses GCD 12e checkpoint, 30 training images with new-class GT.
Single forward+backward per image (no weight update).
"""
import os, sys, json, time, copy
import torch
import torch.nn.functional as F

GCD_ROOT = '/home/yelingfei/projects/GCD'
os.chdir(GCD_ROOT); sys.path.insert(0, GCD_ROOT)
NS, NE = 70, 80

from mmengine.config import Config
from mmengine.runner import Runner, load_checkpoint

CFG = 'configs/gdino_inc/70+10/gdino_inc_70+10_70-79_gcd_scratch_coco.py'
CKPT = 'work_dirs/gcd_70plus10_2gpu_20260426_223507/epoch_12.pth'

cfg = Config.fromfile(CFG)
cfg.work_dir = '/tmp/grad_diag'
cfg.launcher = 'none'
cfg.train_dataloader['batch_size'] = 1
cfg.train_dataloader['num_workers'] = 2
cfg.train_dataloader['persistent_workers'] = False
td = cfg.train_dataloader
if 'dataset' in td and isinstance(td['dataset'], dict):
    td['dataset'].pop('_delete_', None)

runner = Runner.from_cfg(cfg)
runner.load_or_resume()
model = runner.model.module if hasattr(runner.model, 'module') else runner.model
load_checkpoint(model, CKPT, map_location='cpu')
dev = torch.device('cuda:0')
model.to(dev).train()
if runner.model is not model:
    runner.model.to(dev).train()

# Identify encoder last layer parameters
# GCD's encoder is model.encoder (DeformableDetrTransformerEncoder)
# Last layer = model.encoder.layers[-1]
enc_last = model.encoder.layers[-1]
enc_last_params = list(enc_last.parameters())
n_params = sum(p.numel() for p in enc_last_params)
print(f"Encoder last layer: {len(enc_last_params)} param tensors, {n_params} total params")

# Also get named params for per-component analysis
enc_last_named = list(enc_last.named_parameters())
comp_groups = {}
for name, p in enc_last_named:
    if 'attn' in name:
        comp = 'attn'
    elif 'ffn' in name or 'linear' in name:
        comp = 'ffn'
    elif 'norm' in name:
        comp = 'norm'
    else:
        comp = 'other'
    if comp not in comp_groups:
        comp_groups[comp] = []
    comp_groups[comp].append((name, p))
for c, ps in comp_groups.items():
    print(f"  {c}: {len(ps)} tensors, {sum(p.numel() for _,p in ps)} params")

# We need to compute gradients from detection loss and distillation loss separately.
# GCD's loss() returns a dict with both. We'll run loss() once, then manually
# separate the detection vs distillation terms and backprop each.

# The loss keys:
# Detection: loss_cls, loss_bbox, loss_iou, d0-d4.loss_*, enc_loss_*
# Distillation: loss_ld_cls, loss_ld_bbox, loss_ld_iou, d0-d4.ld_loss_*, inter_text_loss, inter_query_loss

DETECT_KEYS = {'loss_cls', 'loss_bbox', 'loss_iou', 'enc_loss_cls', 'enc_loss_bbox', 'enc_loss_iou'}
for i in range(5):
    DETECT_KEYS.add(f'd{i}.loss_cls')
    DETECT_KEYS.add(f'd{i}.loss_bbox')
    DETECT_KEYS.add(f'd{i}.loss_iou')

DISTILL_KEYS = {'loss_ld_cls', 'loss_ld_bbox', 'loss_ld_iou', 'inter_text_loss', 'inter_query_loss'}
for i in range(5):
    DISTILL_KEYS.add(f'd{i}.ld_loss_cls')
    DISTILL_KEYS.add(f'd{i}.ld_loss_bbox')
    DISTILL_KEYS.add(f'd{i}.ld_loss_iou')


def get_grad_vector(params):
    """Flatten all gradients into a single vector."""
    grads = []
    for p in params:
        if p.grad is not None:
            grads.append(p.grad.detach().flatten())
        else:
            grads.append(torch.zeros(p.numel(), device=p.device))
    return torch.cat(grads)


def get_component_grads(named_params_groups):
    """Get per-component gradient vectors."""
    result = {}
    for comp, params in named_params_groups.items():
        grads = []
        for name, p in params:
            if p.grad is not None:
                grads.append(p.grad.detach().flatten())
            else:
                grads.append(torch.zeros(p.numel(), device=p.device))
        result[comp] = torch.cat(grads)
    return result


results = {
    'cos_sim': [],        # per-image cosine similarity
    'has_new_gt': [],     # whether image has new-class GT
    'detect_norm': [],    # detection gradient norm
    'distill_norm': [],   # distillation gradient norm
    'conflict_ratio': [], # fraction of parameters where grads conflict (cos < 0)
    'component_cos': [],  # per-component cosine similarity
    'pcgrad_cos_with_detect': [],  # cos(pcgrad_total, detect) — should be >= cos(original, detect)
    'dynlam_value': [],   # dynamic lambda value
}

seen = 0
t0 = time.time()
for data in runner.train_dataloader:
    if seen >= 30:
        break
    sl = data['data_samples']
    s = sl[0] if isinstance(sl, (list, tuple)) else sl
    gt = s.gt_instances
    if gt is None or len(gt.bboxes) == 0:
        seen += 1
        continue

    gt_labels = gt.labels
    has_new = bool((gt_labels >= NS).any() & (gt_labels < NE).any())

    # Forward pass to get loss dict
    model.zero_grad()
    try:
        losses = runner.model.train_step(data, None)
    except Exception as e:
        print(f"  img {seen}: train_step failed: {e}")
        seen += 1
        continue

    # losses is already a dict of loss tensors after train_step processes it
    # Actually train_step returns the loss dict AND does backward. We need to
    # separate the losses before backward. Let me use model.loss() directly.

    # Reset and use loss() directly
    model.zero_grad()
    try:
        # GCD's loss needs specific inputs. Let's use a different approach:
        # Run train_step but intercept the loss dict before backward
        pass
    except:
        pass

    seen += 1
    if seen % 10 == 0:
        print(f"  [{seen}/30] {time.time()-t0:.0f}s")

# The above approach won't work because train_step does backward internally.
# Let me use a hook-based approach instead.
print("\n=== Using hook-based gradient measurement ===\n")

# Strategy:
# 1. Run full forward + loss computation
# 2. Backward detection loss only → capture encoder last layer grads
# 3. Zero grads, backward distillation loss only → capture grads
# 4. Compare

# Need to call model(data, mode='loss') which returns the loss dict
# Then we can selectively backward individual loss terms

results = {
    'cos_sim': [],
    'has_new_gt': [],
    'detect_norm': [],
    'distill_norm': [],
    'conflict_frac': [],
    'component_cos': [],
    'pcgrad_distill_norm': [],
    'dynlam': [],
}

seen = 0
n_new_imgs = 0
for data in runner.train_dataloader:
    if n_new_imgs >= 20:
        break
    if seen >= 200:
        break

    sl = data['data_samples']
    s = sl[0] if isinstance(sl, (list, tuple)) else sl
    gt = s.gt_instances
    if gt is None or len(gt.bboxes) == 0:
        seen += 1
        continue

    gt_labels = gt.labels
    has_new = bool(((gt_labels >= NS) & (gt_labels < NE)).any())
    if not has_new:
        seen += 1
        continue

    # Forward: get loss dict
    model.zero_grad()
    for p in model.parameters():
        p.requires_grad_(True)

    try:
        data_proc = model.data_preprocessor(data, training=True)
        loss_dict = model._run_forward(data_proc, mode='loss')
    except Exception as e:
        print(f"  img {seen}: forward failed: {e}")
        seen += 1
        continue

    # Sum detection losses
    L_detect = sum(v for k, v in loss_dict.items()
                   if k in DETECT_KEYS and isinstance(v, torch.Tensor) and v.requires_grad)

    # Sum distillation losses
    L_distill = sum(v for k, v in loss_dict.items()
                    if k in DISTILL_KEYS and isinstance(v, torch.Tensor) and v.requires_grad)

    if not isinstance(L_detect, torch.Tensor) or not isinstance(L_distill, torch.Tensor):
        seen += 1
        continue

    # Backward detection loss
    model.zero_grad()
    L_detect.backward(retain_graph=True)
    g_detect = get_grad_vector([p for _, p in enc_last_named])
    g_detect_comp = get_component_grads(comp_groups)

    # Backward distillation loss
    model.zero_grad()
    L_distill.backward(retain_graph=False)
    g_distill = get_grad_vector([p for _, p in enc_last_named])
    g_distill_comp = get_component_grads(comp_groups)

    # Cosine similarity
    cos = F.cosine_similarity(g_detect.unsqueeze(0), g_distill.unsqueeze(0)).item()

    # Per-parameter conflict fraction
    sign_conflict = ((g_detect * g_distill) < 0).float().mean().item()

    # Component-level cosine
    comp_cos = {}
    for comp in comp_groups:
        if g_detect_comp[comp].norm() > 1e-10 and g_distill_comp[comp].norm() > 1e-10:
            comp_cos[comp] = F.cosine_similarity(
                g_detect_comp[comp].unsqueeze(0),
                g_distill_comp[comp].unsqueeze(0)).item()
        else:
            comp_cos[comp] = 0.0

    # Simulate PCGrad: project g_distill onto orthogonal subspace of g_detect
    dot = (g_distill * g_detect).sum()
    if dot < 0:  # conflict
        g_distill_proj = g_distill - (dot / (g_detect.norm()**2 + 1e-8)) * g_detect
    else:
        g_distill_proj = g_distill
    pcgrad_norm = g_distill_proj.norm().item()

    # Dynamic lambda
    dynlam = max(0.0, cos)

    results['cos_sim'].append(cos)
    results['has_new_gt'].append(has_new)
    results['detect_norm'].append(g_detect.norm().item())
    results['distill_norm'].append(g_distill.norm().item())
    results['conflict_frac'].append(sign_conflict)
    results['component_cos'].append(comp_cos)
    results['pcgrad_distill_norm'].append(pcgrad_norm)
    results['dynlam'].append(dynlam)

    n_new_imgs += 1
    if n_new_imgs <= 5:
        print(f"  img {seen}: cos={cos:+.4f} detect_norm={g_detect.norm():.4f} "
              f"distill_norm={g_distill.norm():.4f} conflict_frac={sign_conflict:.3f} "
              f"dynlam={dynlam:.4f}")
        for comp, c in comp_cos.items():
            print(f"    {comp}: cos={c:+.4f}")

    seen += 1
    if n_new_imgs % 5 == 0:
        print(f"  [{n_new_imgs}/20 new-GT images, {seen} total] {time.time()-t0:.0f}s")

# Analysis
import numpy as np
cos_arr = np.array(results['cos_sim'])
det_norm = np.array(results['detect_norm'])
dis_norm = np.array(results['distill_norm'])
conf_frac = np.array(results['conflict_frac'])
pcg_norm = np.array(results['pcgrad_distill_norm'])
dynlam_arr = np.array(results['dynlam'])

print("\n" + "=" * 70)
print("GRADIENT CONFLICT ANALYSIS — ENCODER LAST LAYER")
print("=" * 70)

print(f"\n  Images analyzed: {len(cos_arr)} (all with new-class GT)")

print(f"\n  Cosine similarity (detect vs distill gradients):")
print(f"    mean={cos_arr.mean():+.4f}  std={cos_arr.std():.4f}")
print(f"    min={cos_arr.min():+.4f}  max={cos_arr.max():+.4f}")
print(f"    <0 (conflict): {np.mean(cos_arr < 0):.1%}")
print(f"    <-0.1 (strong conflict): {np.mean(cos_arr < -0.1):.1%}")

print(f"\n  Gradient norms:")
print(f"    detect:  mean={det_norm.mean():.4f}  std={det_norm.std():.4f}")
print(f"    distill: mean={dis_norm.mean():.4f}  std={dis_norm.std():.4f}")
print(f"    ratio (distill/detect): {dis_norm.mean()/det_norm.mean():.2f}x")

print(f"\n  Per-parameter sign conflict fraction:")
print(f"    mean={conf_frac.mean():.3f}  (0.5 = random, >0.5 = systematic conflict)")

print(f"\n  Per-component cosine similarity:")
comp_summary = {}
for comp in comp_groups:
    vals = [r[comp] for r in results['component_cos']]
    comp_summary[comp] = np.mean(vals)
    print(f"    {comp}: mean={np.mean(vals):+.4f}")

print(f"\n  === PCGrad Simulation ===")
print(f"  Original distill norm: {dis_norm.mean():.4f}")
print(f"  After projection:      {pcg_norm.mean():.4f}")
print(f"  Norm reduction:        {(1 - pcg_norm.mean()/dis_norm.mean())*100:.1f}%")

print(f"\n  === Dynamic Lambda Simulation ===")
print(f"  lambda = max(0, cos_sim):")
print(f"    mean={dynlam_arr.mean():.4f}  (1.0=full distill, 0.0=no distill)")
print(f"    =0 (fully suppressed): {np.mean(dynlam_arr == 0):.1%}")

# Verdict
print(f"\n  === VERDICT ===")
if cos_arr.mean() < -0.05:
    print(f"  STRONG CONFLICT: mean cos={cos_arr.mean():+.4f}")
    print(f"  Gradient projection or dynamic weighting is JUSTIFIED")
    verdict = 'STRONG_CONFLICT'
elif cos_arr.mean() < 0.05:
    print(f"  WEAK CONFLICT: mean cos={cos_arr.mean():+.4f}")
    print(f"  Marginal benefit from projection; worth trying")
    verdict = 'WEAK_CONFLICT'
else:
    print(f"  NO CONFLICT: mean cos={cos_arr.mean():+.4f}")
    print(f"  Detection and distillation gradients are aligned; projection unnecessary")
    verdict = 'NO_CONFLICT'

# Save
out = {
    'n_images': len(cos_arr),
    'cos_mean': round(float(cos_arr.mean()), 4),
    'cos_std': round(float(cos_arr.std()), 4),
    'conflict_rate': round(float(np.mean(cos_arr < 0)), 4),
    'detect_norm_mean': round(float(det_norm.mean()), 4),
    'distill_norm_mean': round(float(dis_norm.mean()), 4),
    'conflict_frac_mean': round(float(conf_frac.mean()), 4),
    'pcgrad_norm_reduction': round(float(1 - pcg_norm.mean()/dis_norm.mean()), 4),
    'dynlam_mean': round(float(dynlam_arr.mean()), 4),
    'component_cos': {k: round(v, 4) for k, v in comp_summary.items()},
    'verdict': verdict,
}
json.dump(out, open('/home/yelingfei/logs/tatri/grad_conflict.json', 'w'), indent=2)
print(f"\nSaved: grad_conflict.json")
