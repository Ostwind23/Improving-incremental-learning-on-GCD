#!/usr/bin/env python3
"""
Diagnose GRMI ori_ap degradation.
Compare baseline vs GRMI on OLD-class predictions (not new-class).
If R(M) perturbs old-class features, old-class predictions should show
lower scores, lower IoU, or worse classification.
200 val images, both checkpoints.
"""
import os, sys, json, time
import numpy as np
import torch
GCD_ROOT = '/home/yelingfei/projects/GCD'
os.chdir(GCD_ROOT); sys.path.insert(0, GCD_ROOT)
NS, NE = 70, 80

def collect(cfg_path, ckpt_path, n_imgs, dev):
    from mmengine.config import Config
    from mmengine.runner import Runner, load_checkpoint
    from mmdet.structures.bbox import bbox_overlaps
    cfg = Config.fromfile(cfg_path)
    cfg.work_dir = '/tmp/ori_diag'; cfg.launcher = 'none'
    cfg.val_dataloader['batch_size'] = 1
    vd = cfg.val_dataloader
    if 'dataset' in vd and isinstance(vd['dataset'], dict):
        vd['dataset'].pop('_delete_', None)
    runner = Runner.from_cfg(cfg); runner.load_or_resume()
    model = runner.model.module if hasattr(runner.model, 'module') else runner.model
    load_checkpoint(model, ckpt_path, map_location='cpu')
    model.to(dev).eval()
    if runner.model is not model: runner.model.to(dev).eval()
    for p in model.parameters(): p.requires_grad_(False)

    old_records = []  # (score, iou, correct_cls)
    new_records = []
    seen = 0
    for data in runner.val_dataloader:
        if seen >= n_imgs: break
        sl = data['data_samples']
        s = sl[0] if isinstance(sl, (list, tuple)) else sl
        if s.gt_instances is None or len(s.gt_instances.bboxes) == 0:
            seen += 1; continue
        gt_labels = s.gt_instances.labels
        gt_bboxes = s.gt_instances.bboxes
        if hasattr(gt_bboxes, 'tensor'): gt_bboxes = gt_bboxes.tensor

        with torch.no_grad():
            results = runner.model.val_step(data)
        result = results[0] if isinstance(results, (list, tuple)) else results
        pred = result.pred_instances
        pb = pred.bboxes.to(dev)
        ps = pred.scores
        pl = pred.labels

        old_gt_mask = gt_labels < NS
        new_gt_mask = (gt_labels >= NS) & (gt_labels < NE)

        if old_gt_mask.any():
            old_gt = gt_bboxes[old_gt_mask].to(dev)
            old_gt_lab = gt_labels[old_gt_mask]
            ious = bbox_overlaps(pb, old_gt)
            for qi in range(len(pb)):
                best_iou, best_gi = ious[qi].max(dim=0)
                bi = best_iou.item()
                if bi < 0.1: continue
                tl = int(old_gt_lab[best_gi.item()])
                old_records.append((float(ps[qi]), bi, int(pl[qi]) == tl))

        if new_gt_mask.any():
            new_gt = gt_bboxes[new_gt_mask].to(dev)
            new_gt_lab = gt_labels[new_gt_mask]
            ious_n = bbox_overlaps(pb, new_gt)
            for qi in range(len(pb)):
                best_iou, best_gi = ious_n[qi].max(dim=0)
                bi = best_iou.item()
                if bi < 0.1: continue
                tl = int(new_gt_lab[best_gi.item()])
                new_records.append((float(ps[qi]), bi, int(pl[qi]) == tl))

        seen += 1
        if seen % 100 == 0:
            print("  [%d/%d]" % (seen, n_imgs), flush=True)

    del runner, model
    import gc; gc.collect(); torch.cuda.empty_cache()
    return old_records, new_records

dev = torch.device('cuda:0')
N = 500

print("=== Collecting BASELINE ===", flush=True)
base_old, base_new = collect(
    'configs/gdino_inc/70+10/gdino_inc_70+10_70-79_gcd_scratch_coco.py',
    'work_dirs/gcd_70plus10_2gpu_20260426_223507/epoch_12.pth', N, dev)

print("=== Collecting GRMI ===", flush=True)
grmi_old, grmi_new = collect(
    'configs/gdino_inc/70+10/gdino_inc_70+10_70-79_gcd_grmi_12e_coco.py',
    'work_dirs/_preserved/grmi_first_best_ep12.pth', N, dev)

def analyze(records, label):
    scores = np.array([r[0] for r in records])
    ious = np.array([r[1] for r in records])
    correct = np.array([r[2] for r in records])
    hi = ious >= 0.5
    print("  %s: %d total, %d IoU>=0.5" % (label, len(records), hi.sum()))
    if hi.sum() > 0:
        hs = scores[hi]; hc = correct[hi]
        print("    IoU>=0.5: score mean=%.4f median=%.4f cls_correct=%.1f%%" % (
            hs.mean(), np.median(hs), hc.mean()*100))
        for thr in [0.1, 0.2, 0.3, 0.5]:
            print("      score>=%.1f: %d (%.1f%%)" % (thr, (hs>=thr).sum(), (hs>=thr).mean()*100))
    return scores, ious, correct

print("\n" + "=" * 70)
print("OLD-CLASS PREDICTIONS (IoU with old GT)")
print("=" * 70)
bs, bi, bc = analyze(base_old, "Baseline old-class")
gs, gi, gc_ = analyze(grmi_old, "GRMI old-class")

bh = bs[bi >= 0.5]; gh = gs[gi >= 0.5]
if len(bh) > 0 and len(gh) > 0:
    print("\n  OLD-CLASS IoU>=0.5 comparison:")
    print("    n_preds: baseline=%d GRMI=%d delta=%+d (%.1f%%)" % (
        len(bh), len(gh), len(gh)-len(bh), (len(gh)-len(bh))/len(bh)*100))
    print("    score mean: baseline=%.4f GRMI=%.4f delta=%+.4f" % (
        bh.mean(), gh.mean(), gh.mean()-bh.mean()))
    print("    cls_correct: baseline=%.1f%% GRMI=%.1f%% delta=%+.1fpp" % (
        bc[bi>=0.5].mean()*100, gc_[gi>=0.5].mean()*100,
        (gc_[gi>=0.5].mean()-bc[bi>=0.5].mean())*100))

print("\n" + "=" * 70)
print("NEW-CLASS PREDICTIONS (IoU with new GT)")
print("=" * 70)
analyze(base_new, "Baseline new-class")
analyze(grmi_new, "GRMI new-class")

print("\n" + "=" * 70)
print("DIAGNOSIS: WHY DOES ori_ap DROP?")
print("=" * 70)
if len(bh) > 0 and len(gh) > 0:
    score_drop = gh.mean() - bh.mean()
    n_drop = len(gh) - len(bh)
    cls_drop = gc_[gi>=0.5].mean() - bc[bi>=0.5].mean()
    print("  Old-class IoU>=0.5 predictions:")
    print("    Score change: %+.4f" % score_drop)
    print("    Count change: %+d (%+.1f%%)" % (n_drop, n_drop/len(bh)*100))
    print("    Cls accuracy change: %+.1fpp" % (cls_drop*100))
    if score_drop < -0.005:
        print("  -> Score DROP on old-class = R(M) perturbs old-class features")
    if n_drop < -len(bh)*0.05:
        print("  -> Fewer old-class predictions = old-class recall drops")
    if cls_drop < -0.02:
        print("  -> Worse classification = feature confusion between old/new")

result = {
    'baseline_old_n_iou05': int((bi>=0.5).sum()),
    'grmi_old_n_iou05': int((gi>=0.5).sum()),
    'baseline_old_score_mean': round(float(bh.mean()),4) if len(bh)>0 else None,
    'grmi_old_score_mean': round(float(gh.mean()),4) if len(gh)>0 else None,
    'baseline_old_cls_correct': round(float(bc[bi>=0.5].mean()),4) if (bi>=0.5).sum()>0 else None,
    'grmi_old_cls_correct': round(float(gc_[gi>=0.5].mean()),4) if (gi>=0.5).sum()>0 else None,
}
json.dump(result, open('/home/yelingfei/logs/tatri/ori_ap_diagnosis.json', 'w'), indent=2)
print("\nSaved: ori_ap_diagnosis.json")
