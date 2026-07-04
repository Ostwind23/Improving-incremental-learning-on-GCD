#!/usr/bin/env python3
"""
GRMI Score Distribution Analysis.

Compare GCD baseline vs GRMI 12e: how does the score distribution change
for new-class predictions? This explains the +1.8pt AP gain mechanism.

Measures:
  1. For predictions matched to new-GT (IoU>0.5): score distribution
  2. For predictions matched to new-GT (IoU>0.3): score distribution
  3. Top-K new-class score comparison (what are the highest-scoring new-class preds?)
  4. Score at precision-recall operating points
"""
import argparse, os, sys, json
import numpy as np
import torch

GCD_ROOT = '/home/yelingfei/projects/GCD'
os.chdir(GCD_ROOT); sys.path.insert(0, GCD_ROOT)
NS, NE = 70, 80

def collect_scores(cfg_path, ckpt_path, n_imgs, dev):
    from mmengine.config import Config
    from mmengine.runner import Runner, load_checkpoint
    from mmdet.structures.bbox import bbox_overlaps

    cfg = Config.fromfile(cfg_path)
    cfg.work_dir = '/tmp/score_dist'
    cfg.launcher = 'none'
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

    # Collect per-prediction records for new-class GT matches
    records = []  # (score, iou, correct_cls, pred_label, true_label)
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
        new_mask = (gt_labels >= NS) & (gt_labels < NE)
        if not new_mask.any():
            seen += 1; continue

        with torch.no_grad():
            results = runner.model.val_step(data)
        result = results[0] if isinstance(results, (list, tuple)) else results
        pred_inst = result.pred_instances
        pred_bboxes = pred_inst.bboxes.to(dev)
        pred_scores = pred_inst.scores
        pred_labels = pred_inst.labels
        new_gt = gt_bboxes[new_mask].to(dev)
        new_gt_labels = gt_labels[new_mask]
        if len(pred_bboxes) == 0 or len(new_gt) == 0:
            seen += 1; continue
        ious = bbox_overlaps(pred_bboxes, new_gt)
        for qi in range(len(pred_bboxes)):
            best_iou, best_gi = ious[qi].max(dim=0)
            best_iou = best_iou.item()
            if best_iou < 0.1: continue
            true_label = int(new_gt_labels[best_gi.item()])
            pred_label = int(pred_labels[qi])
            score = float(pred_scores[qi])
            records.append((score, best_iou, pred_label == true_label, pred_label, true_label))
        seen += 1
        if seen % 500 == 0:
            print("  [%d/%d] records=%d" % (seen, n_imgs, len(records)), flush=True)
    return records

def main():
    dev = torch.device('cuda:0')
    print("=== Collecting GCD baseline scores ===", flush=True)
    base_records = collect_scores(
        'configs/gdino_inc/70+10/gdino_inc_70+10_70-79_gcd_scratch_coco.py',
        'work_dirs/gcd_70plus10_2gpu_20260426_223507/epoch_12.pth',
        5000, dev)
    print("=== Collecting GRMI scores ===", flush=True)
    grmi_records = collect_scores(
        'configs/gdino_inc/70+10/gdino_inc_70+10_70-79_gcd_grmi_12e_coco.py',
        'work_dirs/_preserved/grmi_first_best_ep12.pth',
        5000, dev)

    print("\n" + "=" * 70)
    print("SCORE DISTRIBUTION COMPARISON")
    print("=" * 70)

    def analyze(records, label):
        scores = np.array([r[0] for r in records])
        ious = np.array([r[1] for r in records])
        correct = np.array([r[2] for r in records])
        # IoU>0.5 subset
        hi = ious >= 0.5
        lo = (ious >= 0.1) & (ious < 0.5)
        print("\n  %s (total %d records):" % (label, len(records)))
        print("    All new-match (IoU>0.1): score mean=%.4f median=%.4f p90=%.4f max=%.4f" %
              (scores.mean(), np.median(scores), np.percentile(scores, 90), scores.max()))
        if hi.sum() > 0:
            hs = scores[hi]
            hc = correct[hi]
            print("    IoU>=0.5 (n=%d): score mean=%.4f median=%.4f p90=%.4f" %
                  (hi.sum(), hs.mean(), np.median(hs), np.percentile(hs, 90)))
            print("      cls_correct=%.1f%%, score_if_correct=%.4f, score_if_wrong=%.4f" %
                  (hc.mean()*100,
                   hs[hc.astype(bool)].mean() if hc.any() else 0,
                   hs[~hc.astype(bool)].mean() if (~hc).any() else 0))
            # Score > threshold analysis
            for thr in [0.1, 0.2, 0.3, 0.5]:
                above = (hs >= thr).sum()
                print("      score>=%.1f: %d (%.1f%%)" % (thr, above, above/len(hs)*100))
        return scores, ious, correct

    base_s, base_i, base_c = analyze(base_records, "GCD BASELINE")
    grmi_s, grmi_i, grmi_c = analyze(grmi_records, "GRMI 12e")

    # Direct comparison at IoU>=0.5
    print("\n" + "=" * 70)
    print("DIRECT COMPARISON (IoU>=0.5 predictions)")
    print("=" * 70)
    bh = base_s[base_i >= 0.5]
    gh = grmi_s[grmi_i >= 0.5]
    if len(bh) > 0 and len(gh) > 0:
        print("  Score mean: baseline=%.4f  GRMI=%.4f  delta=%+.4f" %
              (bh.mean(), gh.mean(), gh.mean()-bh.mean()))
        print("  Score median: baseline=%.4f  GRMI=%.4f" % (np.median(bh), np.median(gh)))
        print("  Score p90: baseline=%.4f  GRMI=%.4f" % (np.percentile(bh, 90), np.percentile(gh, 90)))
        for thr in [0.1, 0.2, 0.3]:
            b_above = (bh >= thr).mean()
            g_above = (gh >= thr).mean()
            print("  score>=%.1f: baseline=%.1f%%  GRMI=%.1f%%  delta=%+.1fpp" %
                  (thr, b_above*100, g_above*100, (g_above-b_above)*100))

    result = {
        'baseline_n_iou05': int((base_i >= 0.5).sum()),
        'grmi_n_iou05': int((grmi_i >= 0.5).sum()),
        'baseline_score_mean_iou05': float(bh.mean()) if len(bh) > 0 else None,
        'grmi_score_mean_iou05': float(gh.mean()) if len(gh) > 0 else None,
        'baseline_score_p90_iou05': float(np.percentile(bh, 90)) if len(bh) > 0 else None,
        'grmi_score_p90_iou05': float(np.percentile(gh, 90)) if len(gh) > 0 else None,
    }
    json.dump(result, open('/home/yelingfei/logs/tatri/score_distribution.json', 'w'), indent=2)
    print("\nSaved: score_distribution.json")

if __name__ == '__main__':
    main()
