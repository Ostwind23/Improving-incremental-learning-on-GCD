#!/usr/bin/env python3
"""
Six-Node Pipeline Diagnostic for GCD 70+10.

Phase 1 (GCD 12e checkpoint): Nodes A, B, C, F
Phase 2 (Base 0-69 checkpoint): Nodes D, E

Node A: NMS cross-class suppression rate for new classes
Node B: Hungarian matching quality — new vs old class assignment IoU
Node C: Decoder per-layer cls score gap evolution
Node D: Teacher/distillation target quality at new-class GT positions
Node E: Teacher pseudo-label recall on new-class GT
Node F: Box regression quality — new vs old class IoU comparison

300 images each phase, consistent with prior diagnostics.
Cross-checks with known values at the end.
"""
import os, sys, json, gc, time
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

GCD_ROOT = '/home/yelingfei/projects/GCD'
os.chdir(GCD_ROOT); sys.path.insert(0, GCD_ROOT)
NS, NE = 70, 80


def load_model(cfg_path, ckpt_path):
    from mmengine.config import Config
    from mmengine.runner import Runner, load_checkpoint
    cfg = Config.fromfile(cfg_path)
    cfg.work_dir = '/tmp/sixnode'
    cfg.launcher = 'none'
    cfg.val_dataloader['batch_size'] = 1
    vd = cfg.val_dataloader
    if 'dataset' in vd and isinstance(vd['dataset'], dict):
        vd['dataset'].pop('_delete_', None)
    runner = Runner.from_cfg(cfg)
    runner.load_or_resume()
    model = runner.model.module if hasattr(runner.model, 'module') else runner.model
    load_checkpoint(model, ckpt_path, map_location='cpu')
    dev = torch.device('cuda:0')
    model.to(dev).eval()
    if runner.model is not model:
        runner.model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return runner, model, dev


def build_tpm(runner, model):
    first_data = next(iter(runner.val_dataloader))
    sl = first_data['data_samples']
    s0 = sl[0] if isinstance(sl, (list, tuple)) else sl
    tt = s0.text
    if isinstance(tt, str): tt = (tt,)
    _, caption, tok_pos, _ = model.get_tokens_and_prompts(tt, True)
    tpm_tok = model.language_model.tokenizer(
        [caption],
        padding='max_length' if model.language_model.pad_to_max else 'longest',
        return_tensors='pt')
    tpm, _ = model.get_positive_map(tpm_tok, tok_pos)
    model.token_positive_maps = tpm
    return tpm


# ═══════════════════════════════════════════════════════════════════════
# PHASE 1: GCD 12e — Nodes A, B, C, F
# ═══════════════════════════════════════════════════════════════════════

def phase1(n_imgs=300):
    from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_overlaps
    from mmdet.models.dense_heads.atss_vlfusion_head import convert_grounding_to_cls_scores

    print("\n" + "=" * 80)
    print("PHASE 1: GCD 12e — Nodes A, B, C, F (%d images)" % n_imgs)
    print("=" * 80 + "\n", flush=True)

    CFG = 'configs/gdino_inc/70+10/gdino_inc_70+10_70-79_gcd_scratch_coco.py'
    CKPT = 'work_dirs/gcd_70plus10_2gpu_20260426_223507/epoch_12.pth'
    runner, model, dev = load_model(CFG, CKPT)
    tpm = build_tpm(runner, model)

    cap = {}
    orig_head = model.bbox_head.forward
    def head_hook(hs, refs, *a, **k):
        out = orig_head(hs, refs, *a, **k)
        cap['all_cls'] = [c.detach() for c in out[0]]
        cap['all_bbox'] = [b.detach() for b in out[1]]
        return out
    model.bbox_head.forward = head_hook

    # ─── Collectors ───
    # A: NMS
    a_pre_total = 0          # pre-NMS queries with IoU≥0.5 to any new-GT (unique queries)
    a_post_total = 0         # post-NMS preds with IoU≥0.5 to any new-GT
    a_pre_cls_correct = 0    # of pre-NMS matched, how many have correct cls
    a_pre_cls_new = 0        # of pre-NMS matched, how many have ANY new-class argmax
    a_pre_scores = []        # max class score of pre-NMS matched queries
    a_pre_per_gt = []        # per new-GT: how many pre-NMS queries match it
    a_post_per_gt = []       # per new-GT: how many post-NMS preds match it

    # B: Matching
    b_new_ious = []
    b_old_ious = []

    # C: Per-layer gap
    c_gaps = [[] for _ in range(6)]

    # F: Box quality
    f_new_ious = []
    f_old_ious = []

    seen = 0
    t0 = time.time()
    for data in runner.val_dataloader:
        if seen >= n_imgs: break
        sl = data['data_samples']
        s = sl[0] if isinstance(sl, (list, tuple)) else sl
        gt = s.gt_instances
        if gt is None or len(gt.bboxes) == 0:
            seen += 1; continue
        gt_labels = gt.labels
        gt_bboxes = gt.bboxes.tensor if hasattr(gt.bboxes, 'tensor') else gt.bboxes
        new_mask = (gt_labels >= NS) & (gt_labels < NE)
        old_mask = gt_labels < NS
        if not new_mask.any():
            seen += 1; continue

        cap.clear()
        with torch.no_grad():
            results = runner.model.val_step(data)
        if 'all_bbox' not in cap:
            seen += 1; continue

        result = results[0] if isinstance(results, (list, tuple)) else results
        ih, iw = s.metainfo['img_shape']
        fac = gt_bboxes.new_tensor([iw, ih, iw, ih]).to(dev)

        # Pre-NMS: 900 queries (last decoder layer)
        pre_box = bbox_cxcywh_to_xyxy(cap['all_bbox'][-1][0]) * fac
        pre_box[:, 0::2].clamp_(0, iw)
        pre_box[:, 1::2].clamp_(0, ih)
        pre_cls = convert_grounding_to_cls_scores(
            cap['all_cls'][-1][0].sigmoid().unsqueeze(0), [tpm])[0]  # (900, 80)

        # Post-NMS
        post_box = result.pred_instances.bboxes.to(dev)
        post_labels = result.pred_instances.labels
        post_scores = result.pred_instances.scores

        new_gt = gt_bboxes[new_mask].to(dev)
        new_gt_lab = gt_labels[new_mask]
        old_gt = gt_bboxes[old_mask].to(dev) if old_mask.any() else torch.zeros(0, 4, device=dev)
        old_gt_lab = gt_labels[old_mask] if old_mask.any() else torch.tensor([], dtype=torch.long)
        all_gt = gt_bboxes.to(dev)
        all_gt_lab = gt_labels

        # ═══ Node A: NMS cross-class suppression ═══
        ious_pre_new = bbox_overlaps(pre_box, new_gt)  # (900, n_new)
        # Unique pre-NMS queries matched to ANY new-GT
        pre_any_match = (ious_pre_new >= 0.5).any(dim=1)  # (900,)
        n_pre = int(pre_any_match.sum())
        a_pre_total += n_pre

        for qi in torch.where(pre_any_match)[0]:
            qi = int(qi)
            best_gi = int(ious_pre_new[qi].argmax())
            true_cls = int(new_gt_lab[best_gi])
            pred_cls = int(pre_cls[qi].argmax())
            a_pre_scores.append(float(pre_cls[qi].max()))
            if pred_cls == true_cls:
                a_pre_cls_correct += 1
            if NS <= pred_cls < NE:
                a_pre_cls_new += 1

        # Per-GT counts
        for gi in range(len(new_gt)):
            a_pre_per_gt.append(int((ious_pre_new[:, gi] >= 0.5).sum()))

        # Post-NMS
        if len(post_box) > 0:
            ious_post_new = bbox_overlaps(post_box, new_gt)
            post_any_match = (ious_post_new >= 0.5).any(dim=1)
            a_post_total += int(post_any_match.sum())
            for gi in range(len(new_gt)):
                a_post_per_gt.append(int((ious_post_new[:, gi] >= 0.5).sum()))
        else:
            a_post_total += 0
            for gi in range(len(new_gt)):
                a_post_per_gt.append(0)

        # ═══ Node B: Hungarian matching ═══
        if len(all_gt) > 0:
            ious_all = bbox_overlaps(pre_box, all_gt).cpu().numpy()
            cost = 1.0 - ious_all
            row_idx, col_idx = linear_sum_assignment(cost)
            for r, c in zip(row_idx, col_idx):
                iou_val = float(ious_all[r, c])
                lab = int(all_gt_lab[c])
                if NS <= lab < NE:
                    b_new_ious.append(iou_val)
                elif lab < NS:
                    b_old_ious.append(iou_val)

        # ═══ Node C: Per-layer score gap ═══
        # Select queries near new-GT (IoU≥0.3 at last layer)
        mq_mask = (ious_pre_new >= 0.3).any(dim=1)
        mq_idx = torch.where(mq_mask)[0]
        if len(mq_idx) > 0:
            mq_true = []
            for qi in mq_idx:
                best_gi = int(ious_pre_new[int(qi)].argmax())
                mq_true.append(int(new_gt_lab[best_gi]))
            for l in range(min(6, len(cap['all_cls']))):
                l_cls = convert_grounding_to_cls_scores(
                    cap['all_cls'][l][0].sigmoid().unsqueeze(0), [tpm])[0]
                for qi in mq_idx:
                    qi = int(qi)
                    ns = float(l_cls[qi, NS:NE].max())
                    os_ = float(l_cls[qi, :NS].max())
                    c_gaps[l].append(ns - os_)

        # ═══ Node F: Box quality ═══
        for gi in range(len(new_gt)):
            best_iou = float(ious_pre_new[:, gi].max())
            if best_iou >= 0.5:
                f_new_ious.append(best_iou)
        if len(old_gt) > 0:
            ious_pre_old = bbox_overlaps(pre_box, old_gt)
            for gi in range(len(old_gt)):
                best_iou = float(ious_pre_old[:, gi].max())
                if best_iou >= 0.5:
                    f_old_ious.append(best_iou)

        seen += 1
        if seen % 50 == 0:
            print(f"  Phase 1 [{seen}/{n_imgs}] {time.time()-t0:.0f}s", flush=True)

    # ═══ PHASE 1 RESULTS ═══
    print("\n" + "=" * 80)
    print("NODE A: NMS CROSS-CLASS SUPPRESSION (%d images)" % seen)
    print("=" * 80)
    kill = a_pre_total - a_post_total
    kill_rate = kill / max(a_pre_total, 1) * 100
    print(f"  Pre-NMS queries matched to new-GT (IoU≥0.5):  {a_pre_total}")
    print(f"  Post-NMS preds matched to new-GT (IoU≥0.5):   {a_post_total}")
    print(f"  Killed by threshold+NMS: {kill} ({kill_rate:.1f}%)")
    if a_pre_total > 0:
        print(f"  Among pre-NMS matched queries:")
        print(f"    Correct class:   {a_pre_cls_correct}/{a_pre_total} ({a_pre_cls_correct/a_pre_total:.1%})")
        print(f"    Any new-cls argmax: {a_pre_cls_new}/{a_pre_total} ({a_pre_cls_new/a_pre_total:.1%})")
        sa = np.array(a_pre_scores)
        print(f"    Score mean={sa.mean():.4f} median={np.median(sa):.4f}")
        print(f"    Score ≥ 0.3: {np.mean(sa>=0.3):.1%},  ≥ 0.1: {np.mean(sa>=0.1):.1%}")
    pa = np.array(a_pre_per_gt)
    qa = np.array(a_post_per_gt)
    print(f"  Per new-GT: pre-NMS mean={pa.mean():.2f}, post-NMS mean={qa.mean():.2f}")
    print(f"  New-GT with zero pre-NMS match: {np.mean(pa==0):.1%}")
    print(f"  New-GT with zero post-NMS match: {np.mean(qa==0):.1%}")

    print("\n" + "=" * 80)
    print("NODE B: HUNGARIAN MATCHING QUALITY (%d images)" % seen)
    print("=" * 80)
    bn = np.array(b_new_ious) if b_new_ious else np.array([0.0])
    bo = np.array(b_old_ious) if b_old_ious else np.array([0.0])
    print(f"  New-class GT matched ({len(b_new_ious)} instances):")
    print(f"    IoU mean={bn.mean():.3f}  median={np.median(bn):.3f}")
    print(f"    IoU≥0.5: {np.mean(bn>=0.5):.1%}  ≥0.3: {np.mean(bn>=0.3):.1%}  <0.1: {np.mean(bn<0.1):.1%}")
    print(f"  Old-class GT matched ({len(b_old_ious)} instances):")
    print(f"    IoU mean={bo.mean():.3f}  median={np.median(bo):.3f}")
    print(f"    IoU≥0.5: {np.mean(bo>=0.5):.1%}  ≥0.3: {np.mean(bo>=0.3):.1%}  <0.1: {np.mean(bo<0.1):.1%}")
    print(f"  Gap (old−new) IoU mean: {bo.mean()-bn.mean():+.3f}")

    print("\n" + "=" * 80)
    print("NODE C: PER-LAYER CLS SCORE GAP EVOLUTION (%d images)" % seen)
    print("=" * 80)
    print(f"  (Queries with IoU≥0.3 to new-GT at last layer; gap = max_new_score − max_old_score)")
    for l in range(6):
        if c_gaps[l]:
            g = np.array(c_gaps[l])
            print(f"  d{l}: gap={g.mean():+.4f}  median={np.median(g):+.4f}  "
                  f"std={g.std():.4f}  frac<0={np.mean(g<0):.1%}  n={len(g)}")
    if c_gaps[0] and c_gaps[5]:
        g0 = np.mean(c_gaps[0])
        g5 = np.mean(c_gaps[5])
        print(f"  d0→d5 gap change: {g5-g0:+.4f} ({'widening' if g5<g0 else 'narrowing'})")

    print("\n" + "=" * 80)
    print("NODE F: BOX REGRESSION QUALITY new vs old (%d images)" % seen)
    print("=" * 80)
    fn = np.array(f_new_ious) if f_new_ious else np.array([0.0])
    fo = np.array(f_old_ious) if f_old_ious else np.array([0.0])
    print(f"  New-class GT with best-pred IoU≥0.5: {len(f_new_ious)} instances")
    print(f"    IoU mean={fn.mean():.3f}  median={np.median(fn):.3f}  p10={np.percentile(fn,10):.3f}  p90={np.percentile(fn,90):.3f}")
    print(f"  Old-class GT with best-pred IoU≥0.5: {len(f_old_ious)} instances")
    print(f"    IoU mean={fo.mean():.3f}  median={np.median(fo):.3f}  p10={np.percentile(fo,10):.3f}  p90={np.percentile(fo,90):.3f}")
    print(f"  Gap (old−new) IoU mean: {fo.mean()-fn.mean():+.3f}")

    r1 = {
        'A_pre_nms': a_pre_total, 'A_post_nms': a_post_total,
        'A_kill_rate_pct': round(kill_rate, 2),
        'A_pre_correct_cls_rate': round(a_pre_cls_correct / max(a_pre_total, 1), 4),
        'A_pre_new_argmax_rate': round(a_pre_cls_new / max(a_pre_total, 1), 4),
        'A_pre_score_mean': round(float(np.mean(a_pre_scores)), 4) if a_pre_scores else None,
        'A_gt_zero_pre_rate': round(float(np.mean(pa == 0)), 4),
        'A_gt_zero_post_rate': round(float(np.mean(qa == 0)), 4),
        'B_new_iou_mean': round(float(bn.mean()), 4),
        'B_old_iou_mean': round(float(bo.mean()), 4),
        'B_new_ge05': round(float(np.mean(bn >= 0.5)), 4),
        'B_old_ge05': round(float(np.mean(bo >= 0.5)), 4),
        'B_new_lt01': round(float(np.mean(bn < 0.1)), 4),
        'B_old_lt01': round(float(np.mean(bo < 0.1)), 4),
        'B_new_n': len(b_new_ious), 'B_old_n': len(b_old_ious),
        'C_gaps': {f'd{l}': round(float(np.mean(c_gaps[l])), 4) for l in range(6) if c_gaps[l]},
        'F_new_iou_mean': round(float(fn.mean()), 4),
        'F_old_iou_mean': round(float(fo.mean()), 4),
        'F_new_n': len(f_new_ious), 'F_old_n': len(f_old_ious),
        'n_images': seen,
    }
    del runner, model
    gc.collect(); torch.cuda.empty_cache()
    return r1


# ═══════════════════════════════════════════════════════════════════════
# PHASE 2: Base 0-69 checkpoint — Nodes D, E
# ═══════════════════════════════════════════════════════════════════════

def phase2(n_imgs=300):
    from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_overlaps
    from mmdet.models.dense_heads.atss_vlfusion_head import convert_grounding_to_cls_scores

    print("\n" + "=" * 80)
    print("PHASE 2: Base 0-69 checkpoint — Nodes D, E (%d images)" % n_imgs)
    print("=" * 80 + "\n", flush=True)

    CFG = 'configs/gdino_inc/70+10/gdino_inc_70+10_70-79_gcd_scratch_coco.py'
    CKPT_BASE = 'work_dirs/gdino_inc_70+10_0-69_scratch_coco/epoch_12.pth'
    runner, model, dev = load_model(CFG, CKPT_BASE)
    tpm = build_tpm(runner, model)

    cap = {}
    orig_head = model.bbox_head.forward
    def head_hook(hs, refs, *a, **k):
        out = orig_head(hs, refs, *a, **k)
        cap['all_cls'] = [c.detach() for c in out[0]]
        cap['all_bbox'] = [b.detach() for b in out[1]]
        return out
    model.bbox_head.forward = head_hook

    # ─── Collectors ───
    # D: Teacher prediction at new-GT positions
    d_teacher_ious = []      # best IoU between teacher pred and new-GT
    d_teacher_pred_cls = []   # teacher's predicted class at best-IoU query
    d_teacher_true_cls = []   # true new-class label
    d_teacher_score_correct = []  # teacher's score for the correct new class
    d_teacher_score_best_old = [] # teacher's best old-class score at that query

    # E: Teacher recall
    e_new_gt_total = 0
    e_new_gt_recalled_05 = 0
    e_new_gt_recalled_03 = 0
    e_new_gt_recalled_01 = 0

    # Also measure teacher on OLD classes for comparison
    e_old_gt_total = 0
    e_old_gt_recalled_05 = 0

    seen = 0
    t0 = time.time()
    for data in runner.val_dataloader:
        if seen >= n_imgs: break
        sl = data['data_samples']
        s = sl[0] if isinstance(sl, (list, tuple)) else sl
        gt = s.gt_instances
        if gt is None or len(gt.bboxes) == 0:
            seen += 1; continue
        gt_labels = gt.labels
        gt_bboxes = gt.bboxes.tensor if hasattr(gt.bboxes, 'tensor') else gt.bboxes
        new_mask = (gt_labels >= NS) & (gt_labels < NE)
        old_mask = gt_labels < NS
        if not new_mask.any():
            seen += 1; continue

        cap.clear()
        with torch.no_grad():
            results = runner.model.val_step(data)
        if 'all_bbox' not in cap:
            seen += 1; continue

        ih, iw = s.metainfo['img_shape']
        fac = gt_bboxes.new_tensor([iw, ih, iw, ih]).to(dev)

        pre_box = bbox_cxcywh_to_xyxy(cap['all_bbox'][-1][0]) * fac
        pre_box[:, 0::2].clamp_(0, iw)
        pre_box[:, 1::2].clamp_(0, ih)
        pre_cls = convert_grounding_to_cls_scores(
            cap['all_cls'][-1][0].sigmoid().unsqueeze(0), [tpm])[0]

        new_gt = gt_bboxes[new_mask].to(dev)
        new_gt_lab = gt_labels[new_mask]
        old_gt = gt_bboxes[old_mask].to(dev) if old_mask.any() else torch.zeros(0, 4, device=dev)

        # ═══ Node D: Teacher quality at new-class GT ═══
        if len(new_gt) > 0:
            ious = bbox_overlaps(pre_box, new_gt)  # (900, n_new)
            for gi in range(len(new_gt)):
                best_qi = int(ious[:, gi].argmax())
                best_iou = float(ious[best_qi, gi])
                true_cls = int(new_gt_lab[gi])
                pred_cls = int(pre_cls[best_qi].argmax())
                score_correct = float(pre_cls[best_qi, true_cls])
                score_best_old = float(pre_cls[best_qi, :NS].max())

                d_teacher_ious.append(best_iou)
                d_teacher_pred_cls.append(pred_cls)
                d_teacher_true_cls.append(true_cls)
                d_teacher_score_correct.append(score_correct)
                d_teacher_score_best_old.append(score_best_old)

        # ═══ Node E: Teacher recall ═══
        if len(new_gt) > 0:
            ious_new = bbox_overlaps(pre_box, new_gt)
            for gi in range(len(new_gt)):
                best_iou = float(ious_new[:, gi].max())
                e_new_gt_total += 1
                if best_iou >= 0.5: e_new_gt_recalled_05 += 1
                if best_iou >= 0.3: e_new_gt_recalled_03 += 1
                if best_iou >= 0.1: e_new_gt_recalled_01 += 1

        if len(old_gt) > 0:
            ious_old = bbox_overlaps(pre_box, old_gt)
            for gi in range(len(old_gt)):
                best_iou = float(ious_old[:, gi].max())
                e_old_gt_total += 1
                if best_iou >= 0.5: e_old_gt_recalled_05 += 1

        seen += 1
        if seen % 50 == 0:
            print(f"  Phase 2 [{seen}/{n_imgs}] {time.time()-t0:.0f}s", flush=True)

    # ═══ PHASE 2 RESULTS ═══
    print("\n" + "=" * 80)
    print("NODE D: TEACHER PREDICTION AT NEW-CLASS GT (%d images)" % seen)
    print("=" * 80)
    ti = np.array(d_teacher_ious)
    tsc = np.array(d_teacher_score_correct)
    tso = np.array(d_teacher_score_best_old)
    tp = np.array(d_teacher_pred_cls)
    tt = np.array(d_teacher_true_cls)

    print(f"  Total new-class GT examined: {len(ti)}")
    print(f"  Teacher best-query IoU to new-GT:")
    print(f"    mean={ti.mean():.3f}  median={np.median(ti):.3f}")
    print(f"    ≥0.5: {np.mean(ti>=0.5):.1%}  ≥0.3: {np.mean(ti>=0.3):.1%}  <0.1: {np.mean(ti<0.1):.1%}")

    # What does teacher predict at these positions?
    teacher_pred_correct = np.mean(tp == tt)
    teacher_pred_old = np.mean(tp < NS)
    teacher_pred_new_wrong = np.mean((tp >= NS) & (tp < NE) & (tp != tt))
    print(f"  Teacher predicted class at best-IoU query:")
    print(f"    Correct new class: {teacher_pred_correct:.1%}")
    print(f"    Wrong (mapped to old class): {teacher_pred_old:.1%}")
    print(f"    Wrong (different new class): {teacher_pred_new_wrong:.1%}")
    print(f"  Teacher score for correct new class: mean={tsc.mean():.4f}")
    print(f"  Teacher best old-class score:        mean={tso.mean():.4f}")
    print(f"  Margin (correct_new − best_old):     {(tsc-tso).mean():+.4f}")

    # Top confusion pairs
    confused = [(int(tp[i]), int(tt[i])) for i in range(len(tp)) if tp[i] != tt[i] and tp[i] < NS]
    if confused:
        from collections import Counter
        ALL_CLS = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
            "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog",
            "horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella",
            "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
            "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
            "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
            "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant",
            "bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
            "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors",
            "teddy bear","hair drier","toothbrush"]
        cnt = Counter(confused)
        print(f"  Top teacher confusion (new→old):")
        for (pred, true), n in cnt.most_common(10):
            pn = ALL_CLS[pred] if pred < len(ALL_CLS) else str(pred)
            tn = ALL_CLS[true] if true < len(ALL_CLS) else str(true)
            print(f"    {tn}→{pn}: {n}")

    # Distillation signal analysis
    print(f"\n  DISTILLATION SIGNAL INTERPRETATION:")
    print(f"    Teacher sees new-class GT as:")
    hi = ti >= 0.5
    if hi.sum() > 0:
        print(f"      Well-located (IoU≥0.5, n={hi.sum()}): teacher cls correct={np.mean((tp==tt)[hi]):.1%}")
    lo = ti < 0.1
    if lo.sum() > 0:
        print(f"      Missed (IoU<0.1, n={lo.sum()}): distillation = 'predict background here'")
    mid = (ti >= 0.1) & (ti < 0.5)
    if mid.sum() > 0:
        print(f"      Partial (0.1≤IoU<0.5, n={mid.sum()}): distillation = 'weak/wrong signal'")

    print("\n" + "=" * 80)
    print("NODE E: TEACHER RECALL ON NEW vs OLD CLASSES (%d images)" % seen)
    print("=" * 80)
    print(f"  New-class GT: {e_new_gt_total}")
    print(f"    Recalled (IoU≥0.5): {e_new_gt_recalled_05} ({e_new_gt_recalled_05/max(e_new_gt_total,1)*100:.1f}%)")
    print(f"    Recalled (IoU≥0.3): {e_new_gt_recalled_03} ({e_new_gt_recalled_03/max(e_new_gt_total,1)*100:.1f}%)")
    print(f"    Recalled (IoU≥0.1): {e_new_gt_recalled_01} ({e_new_gt_recalled_01/max(e_new_gt_total,1)*100:.1f}%)")
    print(f"  Old-class GT: {e_old_gt_total}")
    print(f"    Recalled (IoU≥0.5): {e_old_gt_recalled_05} ({e_old_gt_recalled_05/max(e_old_gt_total,1)*100:.1f}%)")
    print(f"  RECALL GAP (old−new at IoU≥0.5): {e_old_gt_recalled_05/max(e_old_gt_total,1)*100 - e_new_gt_recalled_05/max(e_new_gt_total,1)*100:+.1f}pp")

    r2 = {
        'D_teacher_iou_mean': round(float(ti.mean()), 4),
        'D_teacher_iou_ge05': round(float(np.mean(ti >= 0.5)), 4),
        'D_teacher_cls_correct': round(float(teacher_pred_correct), 4),
        'D_teacher_cls_old': round(float(teacher_pred_old), 4),
        'D_score_correct_mean': round(float(tsc.mean()), 4),
        'D_score_old_mean': round(float(tso.mean()), 4),
        'D_n': len(ti),
        'E_new_recall_05': round(e_new_gt_recalled_05 / max(e_new_gt_total, 1), 4),
        'E_new_recall_03': round(e_new_gt_recalled_03 / max(e_new_gt_total, 1), 4),
        'E_new_recall_01': round(e_new_gt_recalled_01 / max(e_new_gt_total, 1), 4),
        'E_old_recall_05': round(e_old_gt_recalled_05 / max(e_old_gt_total, 1), 4),
        'E_new_total': e_new_gt_total, 'E_old_total': e_old_gt_total,
        'n_images': seen,
    }
    del runner, model
    gc.collect(); torch.cuda.empty_cache()
    return r2


# ═══════════════════════════════════════════════════════════════════════
# CROSS-CHECKS
# ═══════════════════════════════════════════════════════════════════════

def cross_check(r1, r2):
    print("\n" + "=" * 80)
    print("CROSS-CHECKS WITH PRIOR DIAGNOSTICS")
    print("=" * 80)

    checks = []

    # Check 1: Node F new-class count should match precise_decomp success count
    # precise_decomp: 79 new-class GT with IoU≥0.5 (300 images)
    f_new_n = r1.get('F_new_n', 0)
    checks.append(('F_new_n vs precise_decomp success',
                    f_new_n, 79, abs(f_new_n - 79)))

    # Check 2: Node B new-class GT total should match precise_decomp total
    # precise_decomp: 722 total GT (300 images)
    b_new_n = r1.get('B_new_n', 0)
    b_old_n = r1.get('B_old_n', 0)
    b_total = b_new_n + b_old_n
    checks.append(('B total GT vs expected ~4000+',
                    b_total, 'variable', 'n/a'))

    # Check 3: Node B new-class IoU≥0.5 rate vs precise_decomp 10.9%
    b_new_ge05 = r1.get('B_new_ge05', 0)
    checks.append(('B_new IoU≥0.5 vs precise_decomp 10.9%',
                    f'{b_new_ge05:.1%}', '10.9%', abs(b_new_ge05 - 0.109)))

    # Check 4: Node C d5 gap vs coupling_diagnosis score gap
    # coupling_diagnosis: margin at IoU≥0.5 = +0.011
    c_d5 = r1.get('C_gaps', {}).get('d5', None)
    if c_d5 is not None:
        checks.append(('C d5 gap sign', 'negative' if c_d5 < 0 else 'positive',
                        'expected negative (old > new)', ''))

    # Check 5: Node E teacher recall vs S2 pseudo-label contamination 0.9%
    e_new_05 = r2.get('E_new_recall_05', 0)
    checks.append(('E teacher new-recall@0.5 (base ckpt)',
                    f'{e_new_05:.1%}', 'expected LOW (base never saw 70-79)', ''))

    print()
    for name, observed, expected, diff in checks:
        print(f"  [{name}]")
        print(f"    Observed: {observed}")
        print(f"    Expected: {expected}")
        if diff != 'n/a' and diff != '':
            status = '✓ CONSISTENT' if (isinstance(diff, (int, float)) and diff < 0.05) else '⚠ CHECK'
            print(f"    Diff: {diff}  {status}")
        print()


def main():
    print("Six-Node Pipeline Diagnostic for GCD 70+10")
    print("Starting at", time.strftime('%Y-%m-%d %H:%M:%S'), flush=True)

    r1 = phase1(300)
    r2 = phase2(300)
    cross_check(r1, r2)

    combined = {'phase1': r1, 'phase2': r2}
    outpath = '/home/yelingfei/logs/tatri/six_node_diagnostic.json'
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, 'w') as f:
        json.dump(combined, f, indent=2)
    print(f"\nSaved: {outpath}")
    print("Done at", time.strftime('%Y-%m-%d %H:%M:%S'))


if __name__ == '__main__':
    main()
