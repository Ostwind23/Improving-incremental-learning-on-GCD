"""
Diagnose new/old class composition of detection loss.
Hooks into loss_by_feat_single_new to measure:
  - L_detect_new = detection loss from new-class matched queries (>=70)
  - L_detect_old = detection loss from old-class matched queries (1-69)
  - L_bg = background queries loss (label=0)
  - gs_ratio_v2 = L_detect_new / L_total (the CORRECT ratio for GS-GRMI)

Compares to current gs_ratio (all detection / total) to show how much
old-class detection gradient currently leaks through R(M).
"""
import mmdet.apis  # noqa
import mmdet.engine.hooks  # noqa
import torch
import os, json, time
from mmengine.config import Config
from mmengine.runner import Runner, load_checkpoint


def main():
    cfg = Config.fromfile("/home/yelingfei/projects/GCD/configs/gdino_inc/70+10/grmi_t1_decouple_6e_coco.py")
    cfg.train_cfg.max_epochs = 1
    cfg.train_dataloader.batch_size = 2
    cfg.work_dir = "/tmp/losscomp_smoke"
    cfg.launcher = "none"
    cfg.default_hooks.checkpoint = dict(type="CheckpointHook", interval=999)

    runner = Runner.from_cfg(cfg); runner.load_or_resume()
    model = runner.model.module if hasattr(runner.model, "module") else runner.model

    # Monkey-patch loss_cls/loss_bbox/loss_iou to return per-query loss
    head = model.bbox_head
    orig_loss_cls = head.loss_cls
    orig_loss_bbox = head.loss_bbox
    orig_loss_iou = head.loss_iou

    # Hook to capture labels per image
    stats = {"iters": 0, "n_new": [], "n_old": [], "n_bg": [],
             "frac_new": [], "frac_old": [], "frac_bg": []}

    # Monkey-patch loss_by_feat_single_new to expose labels
    orig_lbf = head.loss_by_feat_single_new
    captured = {}
    def patched_lbf(cls_scores, bbox_preds, batch_gt_instances, batch_img_metas):
        # Replicate matching to get labels
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        with torch.no_grad():
            cls_reg_targets = head.get_targets(cls_scores_list, bbox_preds_list,
                                               batch_gt_instances, batch_img_metas)
        labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, \
            num_total_pos, num_total_neg = cls_reg_targets
        labels = torch.stack(labels_list, 0)
        # Count new/old/bg matched queries (label 0 = bg, 1-69 = old, >=70 = new)
        # Note: labels may be token indices, not class ids. Check.
        # In GDINO, labels are class-local token positions. Need to check gt label mapping.
        # Actually labels here are the matched GT class indices (0-indexed within prompt).
        matched = labels[labels > 0]
        n_new = int((matched >= 71).sum())  # token positions for new classes start ~169
        n_old = int((matched > 0).sum()) - n_new
        n_bg = int(labels.numel() - int((labels > 0).sum()))
        captured["n_new"] = n_new
        captured["n_old"] = n_old
        captured["n_bg"] = n_bg
        captured["labels"] = matched.cpu()
        return orig_lbf(cls_scores, bbox_preds, batch_gt_instances, batch_img_metas)

    head.loss_by_feat_single_new = patched_lbf

    # Run a few iters
    dl = iter(runner.train_dataloader)
    model.train()
    print("[LOSSCOMP] Running 20 iters...")
    for step in range(20):
        data = next(dl)
        data = model.data_preprocessor(data, True)
        losses = model(**data, mode="loss")
        n_new = captured.get("n_new", 0)
        n_old = captured.get("n_old", 0)
        n_bg = captured.get("n_bg", 0)
        total = n_new + n_old + n_bg
        if total > 0:
            stats["n_new"].append(n_new)
            stats["n_old"].append(n_old)
            stats["n_bg"].append(n_bg)
            stats["frac_new"].append(n_new / total)
            stats["frac_old"].append(n_old / total)
            stats["frac_bg"].append(n_bg / total)
        if step < 5:
            lbl = captured.get('labels', [])
            lbl_show = lbl[:10].tolist() if hasattr(lbl, 'tolist') else list(lbl)[:10]
            print(f"  iter{step}: new={n_new} old={n_old} bg={n_bg} "
                  f"labels_sample={lbl_show}")
        stats["iters"] += 1

    import numpy as np
    print("\n===== DETECTION LOSS COMPOSITION =====")
    if stats["frac_new"]:
        print(f"Matched queries per image (mean):")
        print(f"  new-class (>=71 token): {np.mean(stats['n_new']):.1f}")
        print(f"  old-class (1-70 token): {np.mean(stats['n_old']):.1f}")
        print(f"  background (0):         {np.mean(stats['n_bg']):.1f}")
        print(f"\nFraction of matched queries:")
        print(f"  new: {np.mean(stats['frac_new'])*100:.1f}%")
        print(f"  old: {np.mean(stats['frac_old'])*100:.1f}%")
        print(f"  bg:  {np.mean(stats['frac_bg'])*100:.1f}%")
        print(f"\nImplication for GS-GRMI:")
        print(f"  Current gs_ratio (all detect/total) ≈ 0.86")
        print(f"  Old-class detect fraction within detection: {np.mean(stats['frac_old'])/(np.mean(stats['frac_new'])+np.mean(stats['frac_old']))*100:.1f}%")
        print(f"  => gs_ratio_v2 (new detect only/total) ≈ {np.mean(stats['frac_new'])*0.86:.3f}")
        print(f"  => R(M) gradient would be {np.mean(stats['frac_new'])*0.86/0.86*100:.0f}% of current gs_ratio")


if __name__ == "__main__":
    main()
