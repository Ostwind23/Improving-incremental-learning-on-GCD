#!/usr/bin/env python3
"""Precise GT decomposition: no-coverage vs coverage-but-bad-box vs success."""
import torch, json, numpy as np
import os, sys
GCD_ROOT = '/home/yelingfei/projects/GCD'
os.chdir(GCD_ROOT); sys.path.insert(0, GCD_ROOT)
from mmengine.config import Config
from mmengine.runner import Runner, load_checkpoint
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_overlaps
NS, NE = 70, 80

cfg = Config.fromfile("configs/gdino_inc/70+10/gdino_inc_70+10_70-79_gcd_scratch_coco.py")
cfg.work_dir = "/tmp/decomp"; cfg.launcher = "none"
cfg.val_dataloader["batch_size"] = 1
vd = cfg.val_dataloader
if "dataset" in vd and isinstance(vd["dataset"], dict):
    vd["dataset"].pop("_delete_", None)
runner = Runner.from_cfg(cfg); runner.load_or_resume()
model = runner.model.module if hasattr(runner.model, "module") else runner.model
load_checkpoint(model, "work_dirs/gcd_70plus10_2gpu_20260426_223507/epoch_12.pth", map_location="cpu")
dev = torch.device("cuda:0")
model.to(dev).eval()
if runner.model is not model: runner.model.to(dev).eval()
for p in model.parameters(): p.requires_grad_(False)

cap = {}
orig_fe = model.forward_encoder
def fe_hook(*a, **k):
    out = orig_fe(*a, **k)
    for key in ("memory", "spatial_shapes", "memory_mask"):
        v = out.get(key)
        if v is not None: cap[key] = v.detach()
    return out
model.forward_encoder = fe_hook

lgqs_cls = model.bbox_head.cls_branches[model.decoder.num_layers]
def lgqs_hook(module, inputs, output):
    cap["enc_cls"] = output.detach()
lgqs_cls.register_forward_hook(lgqs_hook)

orig_head = model.bbox_head.forward
def head_hook(hs, refs, *a, **k):
    out = orig_head(hs, refs, *a, **k)
    cap["all_bbox"] = [b.detach() for b in out[1]]
    return out
model.bbox_head.forward = head_hook

cats = {"no_coverage": 0, "coverage_but_low_iou": 0, "success": 0}
iou_of_coverage_but_low = []
seen = 0
for data in runner.val_dataloader:
    if seen >= 300: break
    sl = data["data_samples"]
    s = sl[0] if isinstance(sl, (list, tuple)) else sl
    if s.gt_instances is None or len(s.gt_instances.bboxes) == 0: continue
    gt_labels = s.gt_instances.labels
    gt_bboxes = s.gt_instances.bboxes
    if hasattr(gt_bboxes, "tensor"): gt_bboxes = gt_bboxes.tensor
    new_mask = (gt_labels >= NS) & (gt_labels < NE)
    if not new_mask.any(): continue
    cap.clear()
    with torch.no_grad():
        _ = runner.model.val_step(data)
    enc = cap.get("enc_cls"); ss = cap.get("spatial_shapes")
    if enc is None or ss is None or "all_bbox" not in cap: continue
    meta = s.metainfo; ih, iw = meta["img_shape"]
    fac = torch.tensor([iw, ih, iw, ih], device=dev, dtype=torch.float32)
    ssl = ss.cpu().long().tolist()
    li = max(range(len(ssl)), key=lambda k: ssl[k][0]*ssl[k][1])
    H0, W0 = ssl[li]
    off = sum(ssl[k][0]*ssl[k][1] for k in range(li))
    scores_max = enc[0].max(dim=-1)[0]
    _, topk_idx = torch.topk(scores_max, k=min(900, len(scores_max)))
    topk_set = set(topk_idx.cpu().tolist())
    last_bbox = cap["all_bbox"][-1][0]
    pred_boxes = bbox_cxcywh_to_xyxy(last_bbox) * fac
    pred_boxes[:, 0::2].clamp_(0, iw); pred_boxes[:, 1::2].clamp_(0, ih)
    new_gt = gt_bboxes[new_mask].to(dev)
    ious = bbox_overlaps(pred_boxes, new_gt)
    for gi in range(len(new_gt)):
        bx = new_gt[gi]
        gx1 = int(max(0, min(W0-1, bx[0].item()/iw*W0)))
        gx2 = int(max(1, min(W0, bx[2].item()/iw*W0)))
        gy1 = int(max(0, min(H0-1, bx[1].item()/ih*H0)))
        gy2 = int(max(1, min(H0, bx[3].item()/ih*H0)))
        has_coverage = False
        for y in range(gy1, gy2):
            for x in range(gx1, gx2):
                if (off + y*W0 + x) in topk_set:
                    has_coverage = True; break
            if has_coverage: break
        best_iou = ious[:, gi].max().item()
        if not has_coverage:
            cats["no_coverage"] += 1
        elif best_iou >= 0.5:
            cats["success"] += 1
        else:
            cats["coverage_but_low_iou"] += 1
            iou_of_coverage_but_low.append(best_iou)
    seen += 1
    if seen % 100 == 0: print("  [%d/300]" % seen)

total = sum(cats.values())
nc = cats["no_coverage"]; cb = cats["coverage_but_low_iou"]; su = cats["success"]
print("=" * 70)
print("PRECISE GT DECOMPOSITION (%d GT, %d images)" % (total, seen))
print("=" * 70)
print("  No coverage:              %4d (%.1f%%)" % (nc, nc/total*100))
print("  Coverage but IoU<0.5:     %4d (%.1f%%)" % (cb, cb/total*100))
print("  Success (IoU>=0.5):       %4d (%.1f%%)" % (su, su/total*100))
if iou_of_coverage_but_low:
    arr = np.array(iou_of_coverage_but_low)
    print()
    print("  Coverage-but-low-IoU group:")
    print("    IoU mean=%.3f median=%.3f" % (arr.mean(), np.median(arr)))
    print("    IoU>=0.3: %.1f%%" % (np.mean(arr >= 0.3) * 100))
    print("    IoU>=0.1: %.1f%%" % (np.mean(arr >= 0.1) * 100))
    print("    IoU<0.1:  %.1f%%" % (np.mean(arr < 0.1) * 100))
print()
print("  SUMMARY:")
print("    %.0f%% lost to NO LGQS coverage (query never near GT)" % (nc/total*100))
print("    %.0f%% have coverage but prediction box bad (query near but box wrong)" % (cb/total*100))
print("    %.0f%% successfully detected" % (su/total*100))
json.dump(cats, open("/home/yelingfei/logs/tatri/precise_decomp.json", "w"), indent=2)
print("Saved.")
