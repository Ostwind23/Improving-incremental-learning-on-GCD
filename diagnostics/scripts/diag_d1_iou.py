#!/usr/bin/env python3
"""Quick D1 IoU verification: do pred boxes actually overlap normalized GT?"""
import os, sys, torch
os.chdir("/home/yelingfei/projects/GCD"); sys.path.insert(0, ".")
NS, NE = 70, 80
from mmengine.config import Config
from mmengine.runner import Runner, load_checkpoint
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_overlaps

cfg = Config.fromfile("configs/gdino_inc/70+10/gdino_inc_70+10_70-79_gcd_scratch_coco.py")
cfg.work_dir = "/tmp/diag_d1"; cfg.launcher = "none"; cfg.val_dataloader["batch_size"] = 1
vd = cfg.val_dataloader
if "dataset" in vd and isinstance(vd["dataset"], dict): vd["dataset"].pop("_delete_", None)
runner = Runner.from_cfg(cfg); runner.load_or_resume()
model = runner.model.module if hasattr(runner.model, "module") else runner.model
load_checkpoint(model, "work_dirs/gcd_70plus10_2gpu_20260426_223507/epoch_12.pth", map_location="cpu")
dev = torch.device("cuda:0"); model.to(dev).eval()
if runner.model is not model: runner.model.to(dev).eval()
for p in model.parameters(): p.requires_grad_(False)

cap = {}
orig_head = model.bbox_head.forward
def head_hook(hs, refs, *a, **k):
    out = orig_head(hs, refs, *a, **k)
    cap["all_bbox"] = [b.detach() for b in out[1]]
    return out
model.bbox_head.forward = head_hook

total_near = 0; total_imgs = 0; seen = 0
for data in runner.val_dataloader:
    if seen >= 30: break
    sl = data["data_samples"]; s = sl[0] if isinstance(sl,(list,tuple)) else sl
    gt = s.gt_instances
    if gt is None or len(gt.bboxes)==0: seen+=1; continue
    gl = gt.labels; gb = gt.bboxes.tensor if hasattr(gt.bboxes,"tensor") else gt.bboxes
    nm = (gl>=NS)&(gl<NE)
    if not nm.any(): seen+=1; continue
    cap.clear()
    with torch.no_grad(): _ = runner.model.val_step(data)
    if "all_bbox" not in cap: seen+=1; continue
    bp = cap["all_bbox"][-1][0]
    pred_xyxy = bbox_cxcywh_to_xyxy(bp)
    new_gt = gb[nm].to(dev)
    h, w = s.metainfo["img_shape"][:2]
    fac = new_gt.new_tensor([w, h, w, h]).unsqueeze(0)
    new_gt_n = new_gt / fac
    ious = bbox_overlaps(pred_xyxy, new_gt_n)
    near = (ious >= 0.1).any(dim=1)
    nn = int(near.sum())
    total_near += nn; total_imgs += 1
    if total_imgs <= 5:
        mx = ious.max(dim=0)[0]
        print("img%d: %d newGT, %d/900 near, maxIoU=%s, pred[%.3f,%.3f], gt_n[%.3f,%.3f]" % (
            seen, int(nm.sum()), nn, [round(x,3) for x in mx.cpu().tolist()],
            pred_xyxy.min(), pred_xyxy.max(), new_gt_n.min(), new_gt_n.max()))
    seen += 1
print("\nTotal: %d queries near new-GT across %d images, avg=%.1f/img" % (
    total_near, total_imgs, total_near / max(total_imgs, 1)))
if total_near > 0:
    print("VERDICT: truncation WORKS (IoU computation correct)")
else:
    print("VERDICT: truncation STILL BROKEN")
