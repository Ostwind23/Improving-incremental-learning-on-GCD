#!/usr/bin/env python3
"""
Three-Direction Feasibility Diagnostic for GCD 70+10.

D1: Distillation truncation — student vs teacher cls conflict at new-class GT
D2: Score distribution sensitivity — fine score histogram for IoU≥0.5 preds
D3: GT duplication — Hungarian matching simulation with 1x/2x/3x new-class GT

Phase A: GCD 12e (student) → D1 student side, D2, D3
Phase B: Base 0-69 (teacher) → D1 teacher side
Phase C: D3 matching simulation (no model)
Phase D: Analysis + cross-checks

300 val images.
"""
import os, sys, json, gc, time, copy
import numpy as np
import torch
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

GCD_ROOT = '/home/yelingfei/projects/GCD'
os.chdir(GCD_ROOT); sys.path.insert(0, GCD_ROOT)
NS, NE = 70, 80


def load_model(cfg_path, ckpt_path):
    from mmengine.config import Config
    from mmengine.runner import Runner, load_checkpoint
    cfg = Config.fromfile(cfg_path)
    cfg.work_dir = '/tmp/d123'; cfg.launcher = 'none'
    cfg.val_dataloader['batch_size'] = 1
    vd = cfg.val_dataloader
    if 'dataset' in vd and isinstance(vd['dataset'], dict):
        vd['dataset'].pop('_delete_', None)
    runner = Runner.from_cfg(cfg); runner.load_or_resume()
    model = runner.model.module if hasattr(runner.model, 'module') else runner.model
    load_checkpoint(model, ckpt_path, map_location='cpu')
    dev = torch.device('cuda:0')
    model.to(dev).eval()
    if runner.model is not model: runner.model.to(dev).eval()
    for p in model.parameters(): p.requires_grad_(False)
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


def run_phase(ckpt_path, n_imgs, collect_d2d3=False):
    """Run model on n_imgs, collect per-new-GT and per-query diagnostics."""
    from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_overlaps
    from mmdet.models.dense_heads.atss_vlfusion_head import convert_grounding_to_cls_scores

    CFG = 'configs/gdino_inc/70+10/gdino_inc_70+10_70-79_gcd_scratch_coco.py'
    runner, model, dev = load_model(CFG, ckpt_path)
    tpm = build_tpm(runner, model)

    cap = {}
    orig_head = model.bbox_head.forward
    def head_hook(hs, refs, *a, **k):
        out = orig_head(hs, refs, *a, **k)
        cap['all_cls'] = [c.detach() for c in out[0]]
        cap['all_bbox'] = [b.detach() for b in out[1]]
        return out
    model.bbox_head.forward = head_hook

    # D1: per new-GT records
    d1_records = []

    # D2: per IoU≥0.5 prediction (score, iou)
    d2_score_iou = []

    # D3: per-image data for matching simulation
    d3_images = []

    seen = 0; t0 = time.time()
    for data in runner.val_dataloader:
        if seen >= n_imgs: break
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

        ih, iw = s.metainfo['img_shape']
        fac = gt_bboxes.new_tensor([iw, ih, iw, ih]).to(dev)
        pre_box = bbox_cxcywh_to_xyxy(cap['all_bbox'][-1][0]) * fac
        pre_box[:, 0::2].clamp_(0, iw); pre_box[:, 1::2].clamp_(0, ih)
        pre_cls = convert_grounding_to_cls_scores(
            cap['all_cls'][-1][0].sigmoid().unsqueeze(0), [tpm])[0]

        new_gt = gt_bboxes[new_mask].to(dev)
        new_gt_lab = gt_labels[new_mask]
        all_gt = gt_bboxes.to(dev)
        all_gt_lab = gt_labels

        ious_new = bbox_overlaps(pre_box, new_gt)

        # D1: per new-class GT
        for gi in range(len(new_gt)):
            best_qi = int(ious_new[:, gi].argmax())
            best_iou = float(ious_new[best_qi, gi])
            true_cls = int(new_gt_lab[gi])
            pred_cls = int(pre_cls[best_qi].argmax())
            score_correct = float(pre_cls[best_qi, true_cls])
            score_best_old = float(pre_cls[best_qi, :NS].max())
            score_best_new = float(pre_cls[best_qi, NS:NE].max())
            full_scores = pre_cls[best_qi].cpu().numpy().tolist()
            d1_records.append({
                'true_cls': true_cls, 'pred_cls': pred_cls,
                'iou': best_iou,
                'score_correct': score_correct,
                'score_best_old': score_best_old,
                'score_best_new': score_best_new,
                'cls80_scores': full_scores,
            })

        if collect_d2d3:
            # D2: ALL queries vs new-GT, record score + best IoU
            for qi in range(pre_box.shape[0]):
                best_iou_qi = float(ious_new[qi].max()) if ious_new.shape[1] > 0 else 0.0
                max_score = float(pre_cls[qi].max())
                if best_iou_qi >= 0.3:
                    d2_score_iou.append((max_score, best_iou_qi))

            # D3: store query boxes + GT for matching sim
            d3_images.append({
                'query_boxes': pre_box.cpu(),
                'gt_boxes': all_gt.cpu(),
                'gt_labels': all_gt_lab.cpu(),
                'n_queries': pre_box.shape[0],
            })

        seen += 1
        if seen % 50 == 0:
            print(f"  [{seen}/{n_imgs}] {time.time()-t0:.0f}s", flush=True)

    del runner, model; gc.collect(); torch.cuda.empty_cache()
    return d1_records, d2_score_iou, d3_images, seen


def d3_matching_sim(d3_images):
    """Simulate Hungarian matching with 1x/2x/3x GT duplication."""
    from mmdet.structures.bbox import bbox_overlaps

    results = {}
    for dup in [1, 2, 3]:
        new_assigned_ious = []
        old_assigned_ious = []
        new_gt_total = 0
        new_gt_matched_05 = 0
        new_gt_matched_03 = 0

        for img in d3_images:
            qb = img['query_boxes']
            gb = img['gt_boxes']
            gl = img['gt_labels']

            new_mask = (gl >= NS) & (gl < NE)
            if dup > 1 and new_mask.any():
                new_gb = gb[new_mask]
                new_gl = gl[new_mask]
                extra_gb = new_gb.repeat(dup - 1, 1)
                extra_gl = new_gl.repeat(dup - 1)
                gb = torch.cat([gb, extra_gb], dim=0)
                gl = torch.cat([gl, extra_gl], dim=0)
                new_mask = (gl >= NS) & (gl < NE)

            ious = bbox_overlaps(qb, gb).numpy()
            cost = 1.0 - ious
            nq, ng = cost.shape
            if ng == 0: continue
            row_idx, col_idx = linear_sum_assignment(cost)

            for r, c in zip(row_idx, col_idx):
                iou_val = float(ious[r, c])
                lab = int(gl[c])
                if NS <= lab < NE:
                    new_assigned_ious.append(iou_val)
                    new_gt_total += 1
                    if iou_val >= 0.5: new_gt_matched_05 += 1
                    if iou_val >= 0.3: new_gt_matched_03 += 1
                else:
                    old_assigned_ious.append(iou_val)

        ni = np.array(new_assigned_ious) if new_assigned_ious else np.array([0.0])
        oi = np.array(old_assigned_ious) if old_assigned_ious else np.array([0.0])
        results[dup] = {
            'new_n': len(new_assigned_ious),
            'new_iou_mean': float(ni.mean()),
            'new_ge05': int(new_gt_matched_05),
            'new_ge03': int(new_gt_matched_03),
            'new_lt01': int(np.sum(ni < 0.1)),
            'old_n': len(old_assigned_ious),
            'old_iou_mean': float(oi.mean()),
        }
    return results


def main():
    N = 300
    print("=" * 80)
    print("THREE-DIRECTION FEASIBILITY DIAGNOSTIC")
    print("=" * 80, flush=True)

    # ═══ Phase A: Student (GCD 12e) ═══
    print("\n>>> Phase A: Student (GCD 12e) ...", flush=True)
    CKPT_STUDENT = 'work_dirs/gcd_70plus10_2gpu_20260426_223507/epoch_12.pth'
    d1_stu, d2_data, d3_data, n_stu = run_phase(CKPT_STUDENT, N, collect_d2d3=True)
    print(f"  Student done: {n_stu} images, {len(d1_stu)} new-GT records", flush=True)

    # ═══ Phase B: Teacher (base 0-69) ═══
    print("\n>>> Phase B: Teacher (base 0-69) ...", flush=True)
    CKPT_TEACHER = 'work_dirs/gdino_inc_70+10_0-69_scratch_coco/epoch_12.pth'
    d1_tea, _, _, n_tea = run_phase(CKPT_TEACHER, N, collect_d2d3=False)
    print(f"  Teacher done: {n_tea} images, {len(d1_tea)} new-GT records", flush=True)

    # ═══ Phase C: D3 Matching Simulation ═══
    print("\n>>> Phase C: GT Duplication Matching Simulation ...", flush=True)
    d3_res = d3_matching_sim(d3_data)

    # ═══════════════════════════════════════════════════════════════
    # ANALYSIS
    # ═══════════════════════════════════════════════════════════════

    # ─── D1: Distillation Truncation Feasibility ───
    print("\n" + "=" * 80)
    print("D1: DISTILLATION TRUNCATION FEASIBILITY")
    print("=" * 80)

    n_gt = min(len(d1_stu), len(d1_tea))
    if n_gt == 0:
        print("  No data!"); return

    # Align by index (same dataloader order, same image filter)
    print(f"\n  New-class GT instances: student={len(d1_stu)}, teacher={len(d1_tea)}")
    if len(d1_stu) != len(d1_tea):
        print(f"  ⚠ Count mismatch — using min({n_gt})")

    # Per-GT comparison
    conflicts = 0
    stu_correct = 0
    tea_correct = 0
    stu_scores_new = []
    tea_scores_old = []
    pressure_magnitude = []

    for i in range(n_gt):
        s = d1_stu[i]; t = d1_tea[i]
        true_cls = s['true_cls']

        s_pred = s['pred_cls']
        t_pred = t['pred_cls']
        s_correct = (s_pred == true_cls)
        t_correct = (t_pred == true_cls)

        if s_correct: stu_correct += 1
        if t_correct: tea_correct += 1

        # Conflict: student says new, teacher says old
        if s_correct and not t_correct:
            conflicts += 1

        # Distillation pressure = how much teacher pushes toward old class
        # At this GT position: teacher's old-class score vs student's new-class score
        pressure = t['score_best_old'] - s['score_correct']
        pressure_magnitude.append(pressure)
        stu_scores_new.append(s['score_correct'])
        tea_scores_old.append(t['score_best_old'])

    pm = np.array(pressure_magnitude)
    sn = np.array(stu_scores_new)
    to = np.array(tea_scores_old)

    print(f"\n  Student cls correct at new-GT: {stu_correct}/{n_gt} ({stu_correct/n_gt:.1%})")
    print(f"  Teacher cls correct at new-GT: {tea_correct}/{n_gt} ({tea_correct/n_gt:.1%})")
    print(f"  Direct conflicts (student=correct, teacher=wrong): {conflicts}/{n_gt} ({conflicts/n_gt:.1%})")

    print(f"\n  Distillation pressure (teacher_old_score − student_new_score):")
    print(f"    mean={pm.mean():+.4f}  median={np.median(pm):+.4f}")
    print(f"    max={pm.max():+.4f}  min={pm.min():+.4f}")
    print(f"    >0 (teacher wins): {np.mean(pm > 0):.1%}")
    print(f"    <0 (student wins): {np.mean(pm < 0):.1%}")

    print(f"\n  Student new-class score at GT: mean={sn.mean():.4f}")
    print(f"  Teacher old-class score at GT: mean={to.mean():.4f}")
    print(f"  Score ratio (teacher_old / student_new): {to.mean()/max(sn.mean(),1e-6):.1f}x")

    # Stratify by student IoU
    stu_ious = np.array([d1_stu[i]['iou'] for i in range(n_gt)])
    for lo, hi, label in [(0.5, 1.0, 'IoU≥0.5'), (0.3, 0.5, '0.3-0.5'), (0.1, 0.3, '0.1-0.3'), (0.0, 0.1, '<0.1')]:
        mask = (stu_ious >= lo) & (stu_ious < hi)
        nm = mask.sum()
        if nm == 0: continue
        print(f"\n  [{label}] n={nm}")
        print(f"    Student correct: {np.mean([d1_stu[i]['pred_cls']==d1_stu[i]['true_cls'] for i in range(n_gt) if mask[i]]):.1%}")
        print(f"    Teacher correct: {np.mean([d1_tea[i]['pred_cls']==d1_tea[i]['true_cls'] for i in range(n_gt) if mask[i]]):.1%}")
        print(f"    Pressure: {pm[mask].mean():+.4f}")

    # D1 verdict
    print(f"\n  ═══ D1 VERDICT ═══")
    if conflicts / n_gt > 0.3 and pm.mean() > 0.01:
        print(f"  HIGH feasibility: {conflicts/n_gt:.0%} direct conflicts, pressure={pm.mean():+.4f}")
        print(f"  Truncating distillation at new-GT regions would relieve {conflicts} queries")
        print(f"  Expected gain: moderate (student already winning at 84.5%, but low scores)")
        d1_verdict = 'HIGH'
    elif conflicts / n_gt > 0.1:
        print(f"  MODERATE feasibility: {conflicts/n_gt:.0%} conflicts, pressure={pm.mean():+.4f}")
        d1_verdict = 'MODERATE'
    else:
        print(f"  LOW feasibility: only {conflicts/n_gt:.0%} conflicts")
        d1_verdict = 'LOW'

    # ─── D2: Score Distribution Sensitivity ───
    print("\n" + "=" * 80)
    print("D2: SCORE DISTRIBUTION SENSITIVITY (GRMI Enhancement Headroom)")
    print("=" * 80)

    if d2_data:
        scores_arr = np.array([x[0] for x in d2_data])
        ious_arr = np.array([x[1] for x in d2_data])

        # IoU≥0.5 subset
        hi_mask = ious_arr >= 0.5
        hi_scores = scores_arr[hi_mask]
        all_scores = scores_arr

        print(f"\n  Queries matched to new-GT (IoU≥0.3): {len(d2_data)}")
        print(f"  Of which IoU≥0.5: {hi_mask.sum()}")

        if hi_mask.sum() > 0:
            print(f"\n  Score distribution of IoU≥0.5 predictions:")
            bins = [0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
            hist, _ = np.histogram(hi_scores, bins=bins)
            cumsum = 0
            for i in range(len(bins)-1):
                cumsum += hist[i]
                print(f"    [{bins[i]:.3f}, {bins[i+1]:.3f}): {hist[i]:>4d} ({hist[i]/len(hi_scores)*100:>5.1f}%)  cumulative: {cumsum/len(hi_scores)*100:.1f}%")

            print(f"\n  Summary:")
            print(f"    mean={hi_scores.mean():.4f}  median={np.median(hi_scores):.4f}")
            print(f"    p10={np.percentile(hi_scores,10):.4f}  p90={np.percentile(hi_scores,90):.4f}")

            # Sensitivity: if all scores boosted by X%, how many move above threshold?
            print(f"\n  Sensitivity to score boost (IoU≥0.5 preds):")
            for boost in [1.5, 2.0, 3.0, 5.0]:
                boosted = hi_scores * boost
                above_01 = np.sum(boosted >= 0.1)
                above_03 = np.sum(boosted >= 0.3)
                print(f"    {boost:.1f}x boost: {above_01} above 0.1 ({above_01/len(hi_scores)*100:.1f}%), "
                      f"{above_03} above 0.3 ({above_03/len(hi_scores)*100:.1f}%)")

        # Also show IoU 0.3-0.5 group (the "almost there" predictions)
        mid_mask = (ious_arr >= 0.3) & (ious_arr < 0.5)
        if mid_mask.sum() > 0:
            mid_scores = scores_arr[mid_mask]
            print(f"\n  IoU 0.3-0.5 ('almost there' predictions): n={mid_mask.sum()}")
            print(f"    score mean={mid_scores.mean():.4f}  median={np.median(mid_scores):.4f}")

        # D2 verdict
        print(f"\n  ═══ D2 VERDICT ═══")
        if hi_mask.sum() > 0:
            frac_below_01 = np.mean(hi_scores < 0.1)
            if frac_below_01 > 0.8:
                print(f"  HIGH headroom: {frac_below_01:.0%} of IoU≥0.5 preds have score<0.1")
                print(f"  Score improvement would directly increase effective recall")
                d2_verdict = 'HIGH'
            elif frac_below_01 > 0.5:
                print(f"  MODERATE headroom: {frac_below_01:.0%} below 0.1")
                d2_verdict = 'MODERATE'
            else:
                print(f"  LOW headroom: only {frac_below_01:.0%} below 0.1")
                d2_verdict = 'LOW'
        else:
            d2_verdict = 'UNKNOWN'
    else:
        d2_verdict = 'NO_DATA'

    # ─── D3: GT Duplication Matching ───
    print("\n" + "=" * 80)
    print("D3: GT DUPLICATION MATCHING SIMULATION")
    print("=" * 80)

    for dup in [1, 2, 3]:
        r = d3_res[dup]
        print(f"\n  {dup}x GT duplication:")
        print(f"    New-class matched queries: {r['new_n']}")
        print(f"    New-class IoU mean: {r['new_iou_mean']:.3f}")
        print(f"    New-class IoU≥0.5: {r['new_ge05']}")
        print(f"    New-class IoU≥0.3: {r['new_ge03']}")
        print(f"    New-class IoU<0.1: {r['new_lt01']}")
        print(f"    Old-class matched queries: {r['old_n']} (IoU mean={r['old_iou_mean']:.3f})")

    # D3 analysis
    r1 = d3_res[1]; r2 = d3_res[2]; r3 = d3_res[3]
    if r1['new_n'] > 0:
        gain_2x = r2['new_n'] - r1['new_n']
        gain_3x = r3['new_n'] - r1['new_n']
        gain_2x_05 = r2['new_ge05'] - r1['new_ge05']
        gain_3x_05 = r3['new_ge05'] - r1['new_ge05']
        old_loss_2x = r1['old_n'] - r2['old_n']
        old_loss_3x = r1['old_n'] - r3['old_n']

        print(f"\n  Gains from duplication:")
        print(f"    2x: +{gain_2x} new-class queries (+{gain_2x/r1['new_n']*100:.0f}%), "
              f"+{gain_2x_05} at IoU≥0.5, old-class displaced: {old_loss_2x}")
        print(f"    3x: +{gain_3x} new-class queries (+{gain_3x/r1['new_n']*100:.0f}%), "
              f"+{gain_3x_05} at IoU≥0.5, old-class displaced: {old_loss_3x}")

        # Quality of extra assignments
        if gain_2x > 0:
            extra_iou_2x = (r2['new_iou_mean'] * r2['new_n'] - r1['new_iou_mean'] * r1['new_n']) / gain_2x
            print(f"    Extra 2x queries avg IoU: {extra_iou_2x:.3f}")
        if gain_3x > gain_2x and (gain_3x - gain_2x) > 0:
            extra_iou_3x = (r3['new_iou_mean'] * r3['new_n'] - r2['new_iou_mean'] * r2['new_n']) / (gain_3x - gain_2x)
            print(f"    Extra 3x (beyond 2x) queries avg IoU: {extra_iou_3x:.3f}")

    # D3 verdict
    print(f"\n  ═══ D3 VERDICT ═══")
    if r1['new_n'] > 0:
        iou_quality_2x = r2['new_iou_mean']
        if gain_2x > r1['new_n'] * 0.5 and iou_quality_2x > 0.1:
            print(f"  HIGH feasibility: +{gain_2x/r1['new_n']*100:.0f}% more queries, quality IoU={iou_quality_2x:.3f}")
            d3_verdict = 'HIGH'
        elif gain_2x > r1['new_n'] * 0.2:
            print(f"  MODERATE feasibility: +{gain_2x/r1['new_n']*100:.0f}% more queries")
            d3_verdict = 'MODERATE'
        else:
            print(f"  LOW feasibility: only +{gain_2x/r1['new_n']*100:.0f}% more queries")
            d3_verdict = 'LOW'
    else:
        d3_verdict = 'UNKNOWN'

    # ─── FINAL RANKING ───
    print("\n" + "=" * 80)
    print("DIRECTION RANKING")
    print("=" * 80)
    print(f"  D1 Distillation Truncation: {d1_verdict}")
    print(f"  D2 GRMI Enhancement:        {d2_verdict}")
    print(f"  D3 GT Duplication:           {d3_verdict}")

    # Cross-checks
    print("\n  Cross-checks with prior data:")
    # Student correct at IoU≥0.5 should match Node A (84.5%)
    hi_ious = stu_ious >= 0.5
    if hi_ious.sum() > 0:
        stu_hi_correct = np.mean([d1_stu[i]['pred_cls']==d1_stu[i]['true_cls'] for i in range(n_gt) if hi_ious[i]])
        print(f"    Student cls@IoU≥0.5: {stu_hi_correct:.1%} (Node A was 84.5%) {'✓' if abs(stu_hi_correct-0.845)<0.1 else '⚠'}")
    # Teacher correct should be ~0% (Node D)
    print(f"    Teacher cls correct: {tea_correct/n_gt:.1%} (Node D was 9.4%) {'✓' if abs(tea_correct/n_gt-0.094)<0.1 else '⚠'}")
    # D3 1x new_ge05 should match precise_decomp 10.9% → ~13 of 117
    print(f"    D3 1x matched IoU≥0.5: {r1['new_ge05']} of {r1['new_n']} (precise_decomp was 10.9%) {'✓' if abs(r1['new_ge05']/max(r1['new_n'],1)-0.109)<0.05 else '⚠'}")

    # Save
    result = {
        'd1': {
            'n_gt': n_gt,
            'student_correct': stu_correct,
            'teacher_correct': tea_correct,
            'conflicts': conflicts,
            'pressure_mean': round(float(pm.mean()), 4),
            'student_new_score_mean': round(float(sn.mean()), 4),
            'teacher_old_score_mean': round(float(to.mean()), 4),
            'verdict': d1_verdict,
        },
        'd2': {
            'n_iou05': int(hi_mask.sum()) if d2_data else 0,
            'score_mean_iou05': round(float(hi_scores.mean()), 4) if d2_data and hi_mask.sum() > 0 else None,
            'frac_below_01': round(float(frac_below_01), 4) if d2_data and hi_mask.sum() > 0 else None,
            'verdict': d2_verdict,
        },
        'd3': d3_res,
        'd3_verdict': d3_verdict,
    }
    outpath = '/home/yelingfei/logs/tatri/d123_feasibility.json'
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {outpath}")


if __name__ == '__main__':
    main()
