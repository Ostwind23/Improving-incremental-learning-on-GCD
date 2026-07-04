#!/usr/bin/env python3
"""
Bottleneck A-B Coupling Diagnosis.

Question: Is classification confusion (B=30.1%) caused by poor LGQS coverage (A)?
Or are they independent?

Test by measuring classification accuracy STRATIFIED by localization quality:

Group 1: Queries with IoU > 0.5 to new-class GT (well-localized)
  → If classification confusion is STILL 30% here, B is independent of A
  → If confusion drops to <10% here, B is caused by A (bad position → bad cls)

Group 2: Queries with 0.1 < IoU < 0.3 (poorly localized, edge of GT)
  → If confusion is much higher here, confirms A causes B

Group 3: Queries with IoU < 0.1 (essentially background)
  → Baseline: what does the model predict for background queries?

Also measures the REVERSE direction:
  Among correctly classified new-class queries, what is their IoU distribution?
  Among misclassified queries, what is their IoU distribution?
  → If misclassified queries have LOWER IoU, A→B coupling confirmed
  → If misclassified queries have SIMILAR IoU, B is independent

Finally: for the 51% of GT that have "some position" in top-900 but
fail to produce IoU>0.5 predictions — WHY?
  Is it because the query at that position classifies wrong (B)?
  Or because the query at that position can't refine its box (localization)?
"""
import argparse, os, sys, time, json
import numpy as np
import torch

GCD_ROOT = os.environ.get('GCD_ROOT', '/home/yelingfei/projects/GCD')
if os.path.isdir(os.path.join(GCD_ROOT, 'mmdet')):
    os.chdir(GCD_ROOT)
sys.path.insert(0, GCD_ROOT)
NS, NE = 70, 80


def run(args):
    from mmengine.config import Config
    from mmengine.runner import Runner, load_checkpoint
    from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_overlaps
    from mmdet.models.dense_heads.atss_vlfusion_head import convert_grounding_to_cls_scores

    cfg = Config.fromfile(args.cfg)
    cfg.work_dir = '/tmp/coupling'
    cfg.launcher = 'none'
    cfg.val_dataloader['batch_size'] = 1
    vd = cfg.val_dataloader
    if 'dataset' in vd and isinstance(vd['dataset'], dict):
        vd['dataset'].pop('_delete_', None)
    runner = Runner.from_cfg(cfg); runner.load_or_resume()
    model = runner.model.module if hasattr(runner.model, 'module') else runner.model
    load_checkpoint(model, args.ckpt, map_location='cpu')
    dev = torch.device('cuda:0')
    model.to(dev).eval()
    if runner.model is not model: runner.model.to(dev).eval()
    for p in model.parameters(): p.requires_grad_(False)

    # Build tpm
    first_data = next(iter(runner.val_dataloader))
    sl = first_data['data_samples']
    s0 = sl[0] if isinstance(sl, (list, tuple)) else sl
    tt = s0.text
    if isinstance(tt, str): tt = (tt,)
    _, caption, tok_pos, _ = model.get_tokens_and_prompts(tt, True)
    tpm_tok = model.language_model.tokenizer(
        [caption], padding='max_length' if model.language_model.pad_to_max else 'longest',
        return_tensors='pt')
    tpm, _ = model.get_positive_map(tpm_tok, tok_pos)
    model.token_positive_maps = tpm

    # Hook to capture final predictions
    cap = {}
    orig_head = model.bbox_head.forward
    def head_hook(hs, refs, *a, **k):
        out = orig_head(hs, refs, *a, **k)
        cap['all_cls'] = [c.detach() for c in out[0]]
        cap['all_bbox'] = [b.detach() for b in out[1]]
        return out
    model.bbox_head.forward = head_hook

    # Per-query records: (iou_to_nearest_new_gt, true_class, pred_class, pred_score_correct, pred_score_best_old)
    records = []

    # Per-GT records: for each new-class GT, what is the best prediction's IoU and cls
    gt_records = []

    seen = 0
    for data in runner.val_dataloader:
        if seen >= args.n_imgs: break
        sl = data['data_samples']
        s = sl[0] if isinstance(sl, (list, tuple)) else sl
        if s.gt_instances is None or len(s.gt_instances.bboxes) == 0: continue
        gt_labels = s.gt_instances.labels
        gt_bboxes = s.gt_instances.bboxes
        if hasattr(gt_bboxes, 'tensor'): gt_bboxes = gt_bboxes.tensor
        new_mask = (gt_labels >= NS) & (gt_labels < NE)
        if not new_mask.any(): continue

        cap.clear()
        with torch.no_grad():
            _ = runner.model.val_step(data)
        if 'all_bbox' not in cap: continue

        meta = s.metainfo; ih, iw = meta['img_shape']
        fac = torch.tensor([iw, ih, iw, ih], device=dev, dtype=torch.float32)
        last_bbox = cap['all_bbox'][-1][0]
        last_cls = cap['all_cls'][-1]
        pred_boxes = bbox_cxcywh_to_xyxy(last_bbox) * fac
        pred_boxes[:, 0::2].clamp_(0, iw)
        pred_boxes[:, 1::2].clamp_(0, ih)

        new_gt = gt_bboxes[new_mask].to(dev)
        new_gt_labels = gt_labels[new_mask]

        # Per-class scores
        cls_logits = last_cls[0].sigmoid().unsqueeze(0)
        cls_by_class = convert_grounding_to_cls_scores(
            logits=cls_logits, positive_maps=[tpm])[0]  # (nq, 80)

        # IoU matrix: all queries vs all new GT
        ious = bbox_overlaps(pred_boxes, new_gt)  # (nq, n_new)

        # For each query that has ANY overlap with new GT
        for qi in range(pred_boxes.shape[0]):
            max_iou, best_gi = ious[qi].max(dim=0) if ious.shape[1] > 0 else (torch.tensor(0.), torch.tensor(0))
            max_iou = max_iou.item()
            best_gi = best_gi.item()
            if max_iou < 0.01: continue  # skip pure background

            true_cls = int(new_gt_labels[best_gi])
            score_correct = cls_by_class[qi, true_cls].item()
            old_scores = cls_by_class[qi, :NS]
            best_old_score = old_scores.max().item()
            best_old_cls = old_scores.argmax().item()
            # What does the model predict overall?
            pred_cls = cls_by_class[qi].argmax().item()

            records.append({
                'iou': max_iou,
                'true_cls': true_cls,
                'pred_cls': pred_cls,
                'score_correct': score_correct,
                'score_best_old': best_old_score,
                'best_old_cls': best_old_cls,
                'correct': pred_cls == true_cls,
            })

        # Per-GT: best prediction for each GT
        for gi in range(len(new_gt)):
            best_qi = ious[:, gi].argmax().item()
            best_iou = ious[best_qi, gi].item()
            true_cls = int(new_gt_labels[gi])
            pred_cls = cls_by_class[best_qi].argmax().item()
            score_correct = cls_by_class[best_qi, true_cls].item()
            gt_records.append({
                'iou': best_iou,
                'true_cls': true_cls,
                'pred_cls': pred_cls,
                'correct': pred_cls == true_cls,
                'score_correct': score_correct,
            })

        seen += 1
        if seen % 100 == 0:
            print(f"  [{seen}/{args.n_imgs}]")

    # ═══════ Analysis ═══════
    print("\n" + "=" * 80)
    print("BOTTLENECK A-B COUPLING ANALYSIS (%d images, %d query records, %d GT records)" %
          (seen, len(records), len(gt_records)))
    print("=" * 80)

    recs = records
    iou_arr = np.array([r['iou'] for r in recs])
    correct_arr = np.array([r['correct'] for r in recs])

    print("\n╔══ TEST 1: Classification accuracy STRATIFIED by IoU ══╗")
    print("  (If B is caused by A, low-IoU queries should have much worse classification)")
    bins = [(0.5, 1.0, 'IoU≥0.5 (well-localized)'),
            (0.3, 0.5, '0.3≤IoU<0.5 (moderate)'),
            (0.1, 0.3, '0.1≤IoU<0.3 (edge)'),
            (0.01, 0.1, '0.01≤IoU<0.1 (near-background)')]
    for lo, hi, label in bins:
        mask = (iou_arr >= lo) & (iou_arr < hi)
        n = mask.sum()
        if n == 0:
            print(f"  {label}: n=0")
            continue
        acc = correct_arr[mask].mean()
        # Also compute mean score_correct and score_best_old
        sc = np.mean([r['score_correct'] for r, m in zip(recs, mask) if m])
        so = np.mean([r['score_best_old'] for r, m in zip(recs, mask) if m])
        print(f"  {label}: n={n:>5}  cls_correct={acc:.1%}  score_correct={sc:.4f}  score_old={so:.4f}  margin={sc-so:+.4f}")

    print("\n╔══ TEST 2: IoU distribution of correct vs misclassified queries ══╗")
    print("  (If A causes B, misclassified should have lower IoU)")
    correct_ious = iou_arr[correct_arr.astype(bool)]
    wrong_ious = iou_arr[~correct_arr.astype(bool)]
    if len(correct_ious) > 0 and len(wrong_ious) > 0:
        print(f"  Correctly classified: IoU mean={correct_ious.mean():.3f} median={np.median(correct_ious):.3f} ≥0.5={np.mean(correct_ious>=0.5):.1%}")
        print(f"  Misclassified:        IoU mean={wrong_ious.mean():.3f} median={np.median(wrong_ious):.3f} ≥0.5={np.mean(wrong_ious>=0.5):.1%}")
        print(f"  IoU gap (correct - wrong): {correct_ious.mean() - wrong_ious.mean():+.3f}")
        if abs(correct_ious.mean() - wrong_ious.mean()) < 0.02:
            print(f"  >>> INDEPENDENT: IoU similar → classification error NOT caused by position")
        else:
            print(f"  >>> COUPLED: IoU differs → poor localization contributes to misclassification")

    print("\n╔══ TEST 3: Per-GT analysis — WHY does 51% coverage → 17.3% match? ══╗")
    gt_ious = np.array([r['iou'] for r in gt_records])
    gt_correct = np.array([r['correct'] for r in gt_records])
    print(f"  Total new-class GT: {len(gt_records)}")
    print(f"  GT with best-pred IoU≥0.5: {np.mean(gt_ious>=0.5):.1%} (= C4 match coverage)")
    print(f"  GT with best-pred IoU≥0.3: {np.mean(gt_ious>=0.3):.1%}")
    print(f"  GT with best-pred IoU≥0.1: {np.mean(gt_ious>=0.1):.1%}")
    print(f"  GT with best-pred IoU<0.1: {np.mean(gt_ious<0.1):.1%} (completely missed)")
    print()
    # Among GT with IoU≥0.5, what fraction classified correctly?
    good_loc = gt_ious >= 0.5
    if good_loc.sum() > 0:
        print(f"  Among well-localized GT (IoU≥0.5, n={good_loc.sum()}):")
        print(f"    Classified correctly: {gt_correct[good_loc].mean():.1%}")
        print(f"    → These would contribute to AP if classification were fixed")
    bad_loc = gt_ious < 0.1
    if bad_loc.sum() > 0:
        print(f"  Among missed GT (IoU<0.1, n={bad_loc.sum()}):")
        print(f"    Classified correctly: {gt_correct[bad_loc].mean():.1%}")
        print(f"    → These need LGQS fix, classification irrelevant")

    print("\n╔══ TEST 4: Decomposition — how much AP is lost to each bottleneck? ══╗")
    n_gt = len(gt_records)
    n_loc_ok = (gt_ious >= 0.5).sum()
    n_loc_ok_cls_ok = ((gt_ious >= 0.5) & gt_correct).sum()
    n_loc_ok_cls_wrong = ((gt_ious >= 0.5) & ~gt_correct).sum()
    n_loc_bad = (gt_ious < 0.5).sum()
    print(f"  Total GT: {n_gt}")
    print(f"  ├── Localized (IoU≥0.5): {n_loc_ok} ({n_loc_ok/n_gt:.1%})")
    print(f"  │   ├── Correctly classified: {n_loc_ok_cls_ok} ({n_loc_ok_cls_ok/n_gt:.1%}) → SUCCESS")
    print(f"  │   └── Misclassified: {n_loc_ok_cls_wrong} ({n_loc_ok_cls_wrong/n_gt:.1%}) → LOST TO B")
    print(f"  └── Not localized (IoU<0.5): {n_loc_bad} ({n_loc_bad/n_gt:.1%}) → LOST TO A")
    print()
    print(f"  DECOMPOSITION:")
    print(f"    Fixing A (all GT localized): could recover up to {n_loc_bad/n_gt:.1%} of GT")
    print(f"    Fixing B (all cls correct): could recover {n_loc_ok_cls_wrong/n_gt:.1%} of GT")
    print(f"    Both independent contributions sum to: {(n_loc_bad+n_loc_ok_cls_wrong)/n_gt:.1%}")

    # Coupling coefficient
    print("\n╔══ COUPLING VERDICT ══╗")
    if len(correct_ious) > 0 and len(wrong_ious) > 0:
        iou_gap = abs(correct_ious.mean() - wrong_ious.mean())
        if iou_gap < 0.02:
            coupling = "INDEPENDENT"
            print(f"  A and B are {coupling} (IoU gap = {iou_gap:.3f} < 0.02)")
            print(f"  → Classification errors occur equally at good and bad positions")
            print(f"  → Can attack B independently (VLM re-cls) without fixing A")
        elif iou_gap < 0.05:
            coupling = "WEAKLY COUPLED"
            print(f"  A and B are {coupling} (IoU gap = {iou_gap:.3f})")
            print(f"  → Some correlation but can still attack B independently")
        else:
            coupling = "STRONGLY COUPLED"
            print(f"  A and B are {coupling} (IoU gap = {iou_gap:.3f} ≥ 0.05)")
            print(f"  → Poor localization causes misclassification")
            print(f"  → Must fix A first, then B will partially resolve")

    result = {
        'n_query_records': len(records),
        'n_gt_records': len(gt_records),
        'iou_gap_correct_vs_wrong': float(correct_ious.mean() - wrong_ious.mean()) if len(correct_ious) > 0 and len(wrong_ious) > 0 else None,
        'cls_acc_iou_ge05': float(correct_arr[(iou_arr >= 0.5)].mean()) if (iou_arr >= 0.5).sum() > 0 else None,
        'cls_acc_iou_01_03': float(correct_arr[(iou_arr >= 0.1) & (iou_arr < 0.3)].mean()) if ((iou_arr >= 0.1) & (iou_arr < 0.3)).sum() > 0 else None,
        'gt_loc_ok': int(n_loc_ok),
        'gt_loc_ok_cls_ok': int(n_loc_ok_cls_ok),
        'gt_loc_ok_cls_wrong': int(n_loc_ok_cls_wrong),
        'gt_loc_bad': int(n_loc_bad),
        'n_images': seen,
    }
    outpath = '/home/yelingfei/logs/tatri/coupling_diagnosis.json'
    with open(outpath, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {outpath}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', default='configs/gdino_inc/70+10/gdino_inc_70+10_70-79_gcd_scratch_coco.py')
    p.add_argument('--ckpt', default='work_dirs/gcd_70plus10_2gpu_20260426_223507/epoch_12.pth')
    p.add_argument('--n-imgs', type=int, default=300)
    run(p.parse_args())
