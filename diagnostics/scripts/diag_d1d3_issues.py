#!/usr/bin/env python3
"""
Diagnose D1 truncation + D3 GT-dup failure.

Part 1: Verify D1 truncation IoU computation actually produces non-zero matches.
Part 2: Verify D3 GT duplication InstanceData integrity.

Uses GCD 12e checkpoint, 50 val images.
"""
import os, sys, json, time
import torch
GCD_ROOT = '/home/yelingfei/projects/GCD'
os.chdir(GCD_ROOT); sys.path.insert(0, GCD_ROOT)
NS, NE = 70, 80

from mmengine.config import Config
from mmengine.runner import Runner, load_checkpoint
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_overlaps

CFG = 'configs/gdino_inc/70+10/gdino_inc_70+10_70-79_gcd_scratch_coco.py'
CKPT = 'work_dirs/gcd_70plus10_2gpu_20260426_223507/epoch_12.pth'

cfg = Config.fromfile(CFG)
cfg.work_dir = '/tmp/diag_d1d3'; cfg.launcher = 'none'
cfg.val_dataloader['batch_size'] = 1
vd = cfg.val_dataloader
if 'dataset' in vd and isinstance(vd['dataset'], dict):
    vd['dataset'].pop('_delete_', None)
runner = Runner.from_cfg(cfg); runner.load_or_resume()
model = runner.model.module if hasattr(runner.model, 'module') else runner.model
load_checkpoint(model, CKPT, map_location='cpu')
dev = torch.device('cuda:0')
model.to(dev).eval()
if runner.model is not model: runner.model.to(dev).eval()
for p in model.parameters(): p.requires_grad_(False)

# Hook to capture bbox_preds from last decoder layer (distillation path)
cap = {}
orig_head = model.bbox_head.forward
def head_hook(hs, refs, *a, **k):
    out = orig_head(hs, refs, *a, **k)
    # out = (all_layers_cls_scores, all_layers_bbox_preds)
    cap['all_bbox'] = [b.detach() for b in out[1]]
    return out
model.bbox_head.forward = head_hook

print("=" * 70)
print("PART 1: D1 TRUNCATION IoU VERIFICATION")
print("=" * 70, flush=True)

total_queries_near_new = 0
total_images_with_new = 0
total_new_gt = 0
iou_thr = 0.1

seen = 0
for data in runner.val_dataloader:
    if seen >= 50: break
    sl = data['data_samples']
    s = sl[0] if isinstance(sl, (list, tuple)) else sl
    gt = s.gt_instances
    if gt is None or len(gt.bboxes) == 0: seen += 1; continue
    gt_labels = gt.labels
    gt_bboxes = gt.bboxes.tensor if hasattr(gt.bboxes, 'tensor') else gt.bboxes
    new_mask = (gt_labels >= NS) & (gt_labels < NE)
    if not new_mask.any(): seen += 1; continue

    cap.clear()
    with torch.no_grad():
        _ = runner.model.val_step(data)
    if 'all_bbox' not in cap: seen += 1; continue

    # Last layer bbox_preds: (1, 900, 4) cxcywh normalized [0,1]
    bbox_preds = cap['all_bbox'][-1][0]  # (900, 4)
    pred_xyxy = bbox_cxcywh_to_xyxy(bbox_preds)  # (900, 4) [0,1]

    # GT bboxes: pixel xyxy
    new_gt = gt_bboxes[new_mask].to(dev)
    h, w = s.metainfo['img_shape'][:2]
    factor = new_gt.new_tensor([w, h, w, h]).unsqueeze(0)
    new_gt_norm = new_gt / factor  # [0,1] xyxy

    ious = bbox_overlaps(pred_xyxy, new_gt_norm)  # (900, n_new)
    near_new = (ious >= iou_thr).any(dim=1)
    n_near = int(near_new.sum())

    total_queries_near_new += n_near
    total_images_with_new += 1
    total_new_gt += int(new_mask.sum())

    if seen < 5:
        max_iou_per_gt = ious.max(dim=0)[0]
        print(f"  img {seen}: {int(new_mask.sum())} new GT, "
              f"{n_near}/900 queries near (IoU>={iou_thr}), "
              f"max IoU per GT: {max_iou_per_gt.cpu().tolist()}")

        # Also check raw coordinate ranges
        print(f"    pred_xyxy range: [{pred_xyxy.min():.4f}, {pred_xyxy.max():.4f}]")
        print(f"    new_gt_norm range: [{new_gt_norm.min():.4f}, {new_gt_norm.max():.4f}]")
        print(f"    new_gt_pixel range: [{new_gt.min():.1f}, {new_gt.max():.1f}]")
        print(f"    img_shape: h={h}, w={w}")

    seen += 1

print(f"\n  Summary ({total_images_with_new} images with new-class GT, {total_new_gt} total new GT):")
print(f"  Total queries with IoU>={iou_thr} to any new GT: {total_queries_near_new}")
print(f"  Average per image: {total_queries_near_new/max(total_images_with_new,1):.1f}")
if total_queries_near_new > 0:
    print(f"  VERDICT: D1 truncation WOULD affect {total_queries_near_new} queries across {total_images_with_new} images")
else:
    print(f"  VERDICT: D1 truncation affects ZERO queries — still broken!")

print("\n" + "=" * 70)
print("PART 2: D3 GT DUPLICATION INTEGRITY CHECK")
print("=" * 70, flush=True)

# Check what fields InstanceData has for GT instances in GCD
seen = 0
for data in runner.val_dataloader:
    if seen >= 5: break
    sl = data['data_samples']
    s = sl[0] if isinstance(sl, (list, tuple)) else sl
    gt = s.gt_instances
    if gt is None or len(gt.bboxes) == 0: seen += 1; continue
    gt_labels = gt.labels
    new_mask = (gt_labels >= NS) & (gt_labels < NE)
    if not new_mask.any(): seen += 1; continue

    print(f"\n  Image {seen}: {len(gt_labels)} GT instances, {int(new_mask.sum())} new-class")
    print(f"  InstanceData fields: {list(gt.keys())}")
    for field in gt.keys():
        val = getattr(gt, field)
        if isinstance(val, torch.Tensor):
            print(f"    {field}: shape={val.shape} dtype={val.dtype}")
        else:
            print(f"    {field}: type={type(val).__name__}")

    # Simulate GT duplication and check all fields propagate
    from mmengine.structures import InstanceData
    new_inst = InstanceData()
    n_orig = len(gt_labels)
    for field in gt.keys():
        val = getattr(gt, field)
        if isinstance(val, torch.Tensor) and val.dim() >= 1 and val.size(0) == n_orig:
            sub = val[new_mask]
            extra = sub.repeat(2 - 1, *([1] * (val.dim() - 1)))
            new_inst.set_field(torch.cat([val, extra], dim=0), field)
        else:
            new_inst.set_field(val, field)

    print(f"  After 2x duplication: {len(new_inst.labels)} instances")
    for field in new_inst.keys():
        val = getattr(new_inst, field)
        if isinstance(val, torch.Tensor):
            print(f"    {field}: shape={val.shape}")

    # Verify text_token_mask exists and was duplicated
    if hasattr(new_inst, 'text_token_mask'):
        print(f"  text_token_mask PRESENT after duplication: OK")
    else:
        print(f"  text_token_mask MISSING after duplication: THIS IS THE D3 BUG")

    if hasattr(new_inst, 'positive_maps'):
        print(f"  positive_maps PRESENT after duplication: OK")
    else:
        print(f"  positive_maps MISSING after duplication: POTENTIAL BUG")

    seen += 1

# Part 2b: Check if GT instances during TRAINING have text_token_mask
# This requires looking at what the training pipeline produces
print(f"\n  Checking training-time GT InstanceData fields...")
cfg_train = Config.fromfile(CFG)
cfg_train.work_dir = '/tmp/diag_d1d3_train'; cfg_train.launcher = 'none'
cfg_train.train_dataloader['batch_size'] = 1
td = cfg_train.train_dataloader
if 'dataset' in td and isinstance(td['dataset'], dict):
    td['dataset'].pop('_delete_', None)
# Build just the dataloader
from mmengine.runner import Runner as R2
runner2 = R2.from_cfg(cfg_train)
for data in runner2.train_dataloader:
    sl = data['data_samples']
    s = sl[0] if isinstance(sl, (list, tuple)) else sl
    gt = s.gt_instances
    print(f"\n  Training InstanceData fields: {list(gt.keys())}")
    for field in gt.keys():
        val = getattr(gt, field)
        if isinstance(val, torch.Tensor):
            print(f"    {field}: shape={val.shape} dtype={val.dtype}")
        else:
            print(f"    {field}: type={type(val).__name__}")
    has_ttm = hasattr(gt, 'text_token_mask')
    has_pm = hasattr(gt, 'positive_maps')
    print(f"\n  text_token_mask present in TRAINING GT: {has_ttm}")
    print(f"  positive_maps present in TRAINING GT: {has_pm}")
    if not has_ttm:
        print(f"  >>> text_token_mask is NOT in training GT InstanceData!")
        print(f"  >>> It must be added DURING loss computation, not in the dataset.")
        print(f"  >>> D3 duplication MUST happen AFTER text_token_mask is attached to GT.")
    break

print("\nDone.")
