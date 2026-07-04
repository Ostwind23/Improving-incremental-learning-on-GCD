#!/usr/bin/env python3
"""
Risk Verification for Three Directions.

R1: D1 risk — new-class GT vs old-class GT spatial overlap
    If queries near new-GT are also near old-GT, truncation damages old-class.
    Measure: for each new-class GT, how many old-class GT overlap it?
    And for each query matched to new-GT, what's its IoU with nearest old-GT?

R2: D2 risk — GRMI residual R(M) effect on old-class features
    Measure actual R(M) magnitude at old-class vs new-class GT positions
    in the GRMI 12e checkpoint (where R(M) module exists).
    If R(M) perturbs old-class features significantly, larger γ is risky.

R3: D3 risk — gradient quality at different IoU thresholds
    For GT-duplicated matching at 2x/3x, stratify extra matches by IoU
    and measure: what fraction of extra gradient is useful (IoU≥0.1/0.2)?
    Simulate: if we filter extra matches below IoU threshold, how many
    good matches remain?

All use GCD 12e checkpoint (300 images) unless noted.
"""
import os, sys, json, gc, time
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
    cfg.work_dir = '/tmp/risk'; cfg.launcher = 'none'
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


def main():
    from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_overlaps
    from mmdet.models.dense_heads.atss_vlfusion_head import convert_grounding_to_cls_scores

    N = 300
    CFG = 'configs/gdino_inc/70+10/gdino_inc_70+10_70-79_gcd_scratch_coco.py'
    CKPT = 'work_dirs/gcd_70plus10_2gpu_20260426_223507/epoch_12.pth'

    print("=" * 80)
    print("RISK VERIFICATION FOR THREE DIRECTIONS")
    print("=" * 80, flush=True)

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

    # ═══ R1 collectors ═══
    # Per new-GT: IoU with nearest old-GT
    r1_new_old_max_iou = []
    # Per new-GT: number of old-GT with IoU>0 / >0.3 / >0.5
    r1_n_old_overlap = []
    # Per query matched to new-GT (IoU≥0.3): IoU with nearest old-GT
    r1_query_old_iou = []
    # Per query matched to new-GT (IoU≥0.3): which has HIGHER IoU — new or old GT?
    r1_query_primary = []  # 'new' or 'old'

    # ═══ R3 collectors ═══
    # Store per-image data for matching simulation with IoU filtering
    r3_images = []

    seen = 0; t0 = time.time()
    for data in runner.val_dataloader:
        if seen >= N: break
        sl = data['data_samples']
        s = sl[0] if isinstance(sl, (list, tuple)) else sl
        gt = s.gt_instances
        if gt is None or len(gt.bboxes) == 0: seen += 1; continue
        gt_labels = gt.labels
        gt_bboxes = gt.bboxes.tensor if hasattr(gt.bboxes, 'tensor') else gt.bboxes
        new_mask = (gt_labels >= NS) & (gt_labels < NE)
        old_mask = gt_labels < NS
        if not new_mask.any(): seen += 1; continue

        cap.clear()
        with torch.no_grad():
            _ = runner.model.val_step(data)
        if 'all_bbox' not in cap: seen += 1; continue

        ih, iw = s.metainfo['img_shape']
        fac = gt_bboxes.new_tensor([iw, ih, iw, ih]).to(dev)
        pre_box = bbox_cxcywh_to_xyxy(cap['all_bbox'][-1][0]) * fac
        pre_box[:, 0::2].clamp_(0, iw); pre_box[:, 1::2].clamp_(0, ih)

        new_gt = gt_bboxes[new_mask].to(dev)
        old_gt = gt_bboxes[old_mask].to(dev) if old_mask.any() else torch.zeros(0, 4, device=dev)
        all_gt = gt_bboxes.to(dev)
        all_gt_lab = gt_labels

        # ═══ R1: New-GT vs Old-GT spatial overlap ═══
        if len(old_gt) > 0 and len(new_gt) > 0:
            gt_cross_iou = bbox_overlaps(new_gt, old_gt)  # (n_new, n_old)
            for gi in range(len(new_gt)):
                max_old_iou = float(gt_cross_iou[gi].max())
                r1_new_old_max_iou.append(max_old_iou)
                n_overlap_any = int((gt_cross_iou[gi] > 0).sum())
                n_overlap_03 = int((gt_cross_iou[gi] > 0.3).sum())
                n_overlap_05 = int((gt_cross_iou[gi] > 0.5).sum())
                r1_n_old_overlap.append((n_overlap_any, n_overlap_03, n_overlap_05))

            # Per query matched to new-GT: also measure IoU with old-GT
            ious_new = bbox_overlaps(pre_box, new_gt)
            ious_old = bbox_overlaps(pre_box, old_gt)
            for qi in range(pre_box.shape[0]):
                best_new_iou = float(ious_new[qi].max())
                if best_new_iou < 0.3: continue
                best_old_iou = float(ious_old[qi].max())
                r1_query_old_iou.append(best_old_iou)
                r1_query_primary.append('new' if best_new_iou > best_old_iou else 'old')
        elif len(new_gt) > 0:
            for gi in range(len(new_gt)):
                r1_new_old_max_iou.append(0.0)
                r1_n_old_overlap.append((0, 0, 0))

        # ═══ R3: Store for matching sim ═══
        r3_images.append({
            'query_boxes': pre_box.cpu(),
            'gt_boxes': all_gt.cpu(),
            'gt_labels': all_gt_lab.cpu(),
        })

        seen += 1
        if seen % 50 == 0:
            print(f"  [{seen}/{N}] {time.time()-t0:.0f}s", flush=True)

    # ═══════════════════════════════════════════════════════════════
    # R1 ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("R1: NEW-GT vs OLD-GT SPATIAL OVERLAP (Distillation Truncation Risk)")
    print("=" * 80)

    ov = np.array(r1_new_old_max_iou)
    print(f"\n  Total new-class GT: {len(ov)}")
    print(f"\n  Max IoU between each new-GT and its nearest old-GT:")
    print(f"    mean={ov.mean():.3f}  median={np.median(ov):.3f}")
    print(f"    IoU=0 (no overlap at all): {np.mean(ov==0):.1%}")
    print(f"    IoU>0 (any overlap):       {np.mean(ov>0):.1%}")
    print(f"    IoU>0.3:                   {np.mean(ov>0.3):.1%}")
    print(f"    IoU>0.5:                   {np.mean(ov>0.5):.1%}")

    # Distribution of n_old_overlap
    if r1_n_old_overlap:
        any_ov = np.array([x[0] for x in r1_n_old_overlap])
        ov03 = np.array([x[1] for x in r1_n_old_overlap])
        ov05 = np.array([x[2] for x in r1_n_old_overlap])
        print(f"\n  Number of old-GT overlapping each new-GT:")
        print(f"    Any overlap:  mean={any_ov.mean():.1f}  max={any_ov.max()}")
        print(f"    IoU>0.3:      mean={ov03.mean():.2f}  max={ov03.max()}")
        print(f"    IoU>0.5:      mean={ov05.mean():.2f}  max={ov05.max()}")

    # Per-query analysis
    if r1_query_old_iou:
        qo = np.array(r1_query_old_iou)
        qp = np.array(r1_query_primary)
        print(f"\n  Queries matched to new-GT (IoU≥0.3): {len(qo)}")
        print(f"    Also matched to old-GT (IoU>0):   {np.mean(qo>0):.1%}")
        print(f"    Also matched to old-GT (IoU>0.3): {np.mean(qo>0.3):.1%}")
        print(f"    Also matched to old-GT (IoU>0.5): {np.mean(qo>0.5):.1%}")
        print(f"    Primary target is new-GT:         {np.mean(qp=='new'):.1%}")
        print(f"    Primary target is old-GT:         {np.mean(qp=='old'):.1%}")

    # R1 verdict
    print(f"\n  ═══ R1 RISK ASSESSMENT ═══")
    overlap_rate = float(np.mean(ov > 0.3))
    if overlap_rate > 0.3:
        print(f"  HIGH RISK: {overlap_rate:.0%} of new-GT have IoU>0.3 with old-GT")
        print(f"  Truncating distillation at these positions WILL damage old-class protection")
        print(f"  RECOMMENDATION: Use soft truncation (reduce weight, don't zero)")
        print(f"  Or: only truncate for queries where new-GT IoU >> old-GT IoU")
        r1_risk = 'HIGH'
    elif overlap_rate > 0.1:
        print(f"  MODERATE RISK: {overlap_rate:.0%} overlap")
        print(f"  RECOMMENDATION: Selective truncation (only where no old-GT nearby)")
        r1_risk = 'MODERATE'
    else:
        print(f"  LOW RISK: only {overlap_rate:.0%} overlap")
        print(f"  Safe to truncate at most new-GT positions")
        r1_risk = 'LOW'

    # Additional: what fraction of queries are "pure new" (new IoU > old IoU)?
    if r1_query_primary:
        pure_new_rate = np.mean(qp == 'new')
        print(f"\n  Queries primarily serving new-GT: {pure_new_rate:.1%}")
        print(f"  → These are SAFE to truncate distillation on")
        print(f"  → The other {1-pure_new_rate:.1%} also serve old-GT, risky to truncate")

    # ═══════════════════════════════════════════════════════════════
    # R2 ANALYSIS — check if GRMI checkpoint exists
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("R2: GRMI RESIDUAL MAGNITUDE AT OLD vs NEW POSITIONS")
    print("=" * 80)

    GRMI_CKPT = 'work_dirs/_preserved/grmi_first_best_ep12.pth'
    grmi_exists = os.path.isfile(GRMI_CKPT)

    if not grmi_exists:
        print(f"  GRMI checkpoint not found at {GRMI_CKPT}")
        print(f"  Skipping R2 — need GRMI model with R(M) module")
        r2_data = None
    else:
        # Load GRMI model to measure R(M) magnitudes
        GRMI_CFG = 'configs/gdino_inc/70+10/gdino_inc_70+10_70-79_gcd_grmi_12e_coco.py'
        if not os.path.isfile(GRMI_CFG):
            print(f"  GRMI config not found at {GRMI_CFG}")
            print(f"  Attempting with base config + manual R(M) measurement")
            r2_data = None
        else:
            del runner, model; gc.collect(); torch.cuda.empty_cache()
            runner2, model2, dev2 = load_model(GRMI_CFG, GRMI_CKPT)

            # Find R(M) module
            r_module = None
            for name, mod in model2.named_modules():
                if 'grmi' in name.lower() or 'residual' in name.lower():
                    print(f"  Found R(M) module: {name} -> {type(mod)}")
                    r_module = mod

            # Hook encoder to capture memory before and after R(M)
            r2_cap = {}
            orig_fe2 = model2.forward_encoder
            def fe_hook2(*a, **k):
                out = orig_fe2(*a, **k)
                r2_cap['memory'] = out.get('memory', None)
                if r2_cap['memory'] is not None:
                    r2_cap['memory'] = r2_cap['memory'].detach()
                for key in ('spatial_shapes',):
                    v = out.get(key)
                    if v is not None: r2_cap[key] = v.detach()
                return out
            model2.forward_encoder = fe_hook2

            # Measure R(M) at GT positions
            r2_new_norms = []
            r2_old_norms = []
            r2_bg_norms = []  # background positions

            r2_seen = 0
            for data in runner2.val_dataloader:
                if r2_seen >= 200: break
                sl = data['data_samples']
                s = sl[0] if isinstance(sl, (list, tuple)) else sl
                gt = s.gt_instances
                if gt is None or len(gt.bboxes) == 0: r2_seen += 1; continue
                gt_labels = gt.labels
                gt_bboxes = gt.bboxes.tensor if hasattr(gt.bboxes, 'tensor') else gt.bboxes
                new_mask = (gt_labels >= NS) & (gt_labels < NE)
                if not new_mask.any(): r2_seen += 1; continue

                r2_cap.clear()
                with torch.no_grad():
                    _ = runner2.model.val_step(data)
                mem = r2_cap.get('memory')
                ss = r2_cap.get('spatial_shapes')
                if mem is None or ss is None: r2_seen += 1; continue

                ih, iw = s.metainfo['img_shape']
                ssl = ss.cpu().long().tolist()
                li = max(range(len(ssl)), key=lambda k: ssl[k][0]*ssl[k][1])
                H0, W0 = ssl[li]
                off = sum(ssl[k][0]*ssl[k][1] for k in range(li))

                mem_finest = mem[0, off:off+H0*W0, :]  # (H*W, 256)
                mem_norm = mem_finest.norm(dim=-1)  # (H*W,)

                # Build masks
                new_grid = torch.zeros(H0, W0, device=dev2)
                old_grid = torch.zeros(H0, W0, device=dev2)
                old_mask_gt = gt_labels < NS
                for i in range(len(gt_labels)):
                    lab = int(gt_labels[i])
                    bx = gt_bboxes[i]
                    gx1 = int(max(0, min(W0-1, bx[0].item()/iw*W0)))
                    gx2 = int(max(1, min(W0, bx[2].item()/iw*W0)))
                    gy1 = int(max(0, min(H0-1, bx[1].item()/ih*H0)))
                    gy2 = int(max(1, min(H0, bx[3].item()/ih*H0)))
                    if NS <= lab < NE:
                        new_grid[gy1:gy2, gx1:gx2] = 1
                    elif lab < NS:
                        old_grid[gy1:gy2, gx1:gx2] = 1

                new_flat = new_grid.reshape(-1) > 0
                old_flat = old_grid.reshape(-1) > 0
                bg_flat = (~new_flat) & (~old_flat)

                if new_flat.any():
                    r2_new_norms.extend(mem_norm[new_flat].cpu().tolist())
                if old_flat.any():
                    r2_old_norms.extend(mem_norm[old_flat].cpu().tolist())
                if bg_flat.any():
                    # Sample background to avoid huge lists
                    bg_idx = torch.where(bg_flat)[0]
                    if len(bg_idx) > 50:
                        bg_idx = bg_idx[torch.randperm(len(bg_idx))[:50]]
                    r2_bg_norms.extend(mem_norm[bg_idx].cpu().tolist())

                r2_seen += 1

            nn = np.array(r2_new_norms) if r2_new_norms else np.array([0.0])
            on = np.array(r2_old_norms) if r2_old_norms else np.array([0.0])
            bn = np.array(r2_bg_norms) if r2_bg_norms else np.array([0.0])

            print(f"\n  GRMI encoder memory norm at different regions (200 images):")
            print(f"    New-class GT positions (n={len(r2_new_norms)}): mean={nn.mean():.3f} std={nn.std():.3f}")
            print(f"    Old-class GT positions (n={len(r2_old_norms)}): mean={on.mean():.3f} std={on.std():.3f}")
            print(f"    Background positions   (n={len(r2_bg_norms)}):  mean={bn.mean():.3f} std={bn.std():.3f}")
            print(f"    New/Old ratio: {nn.mean()/on.mean():.4f}")
            print(f"    New/BG ratio:  {nn.mean()/bn.mean():.4f}")

            # The question: does R(M) distort old-class features?
            # If norms are similar → R(M) is uniform → larger γ perturbs everything equally
            # If new > old → R(M) is selective → larger γ might be safer
            r2_data = {
                'new_norm_mean': round(float(nn.mean()), 4),
                'old_norm_mean': round(float(on.mean()), 4),
                'bg_norm_mean': round(float(bn.mean()), 4),
                'new_old_ratio': round(float(nn.mean()/on.mean()), 4),
            }

            print(f"\n  ═══ R2 RISK ASSESSMENT ═══")
            ratio = nn.mean() / on.mean()
            if abs(ratio - 1.0) < 0.02:
                print(f"  R(M) is UNIFORM across regions (ratio={ratio:.4f})")
                print(f"  Increasing γ perturbs old-class features equally")
                print(f"  RISK: old-class features will drift with larger γ")
                print(f"  RECOMMENDATION: Need region-selective R(M) or very small γ increase")
                r2_risk = 'HIGH'
            else:
                print(f"  R(M) shows {abs(ratio-1)*100:.1f}% selectivity")
                r2_risk = 'MODERATE' if abs(ratio - 1.0) < 0.05 else 'LOW'

            del runner2, model2; gc.collect(); torch.cuda.empty_cache()

            # Reload student model for R3
            runner, model, dev = load_model(CFG, CKPT)
            tpm = build_tpm(runner, model)
            model.bbox_head.forward = head_hook

    # ═══════════════════════════════════════════════════════════════
    # R3 ANALYSIS — GT duplication quality filtering
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("R3: GT DUPLICATION — MATCHING QUALITY vs IoU THRESHOLD")
    print("=" * 80)

    # Simulate matching with 2x GT duplication at different IoU thresholds
    # For each extra match, record its IoU
    # Then simulate: what if we only keep extra matches above threshold?

    for dup in [2, 3]:
        print(f"\n  --- {dup}x GT Duplication ---")
        extra_ious_all = []
        orig_new_ious = []
        orig_old_ious = []
        n_images_used = 0

        for img in r3_images:
            qb = img['query_boxes']
            gb = img['gt_boxes']
            gl = img['gt_labels']
            new_m = (gl >= NS) & (gl < NE)
            if not new_m.any(): continue

            # 1x matching (baseline)
            ious_1x = bbox_overlaps(qb, gb).numpy()
            cost_1x = 1.0 - ious_1x
            r1x, c1x = linear_sum_assignment(cost_1x)
            matched_1x = set()
            for r, c in zip(r1x, c1x):
                lab = int(gl[c])
                if NS <= lab < NE:
                    orig_new_ious.append(float(ious_1x[r, c]))
                    matched_1x.add(r)
                else:
                    orig_old_ious.append(float(ious_1x[r, c]))

            # Nx matching
            new_gb = gb[new_m]
            new_gl = gl[new_m]
            extra_gb = new_gb.repeat(dup - 1, 1)
            extra_gl = new_gl.repeat(dup - 1)
            gb_dup = torch.cat([gb, extra_gb], dim=0)
            gl_dup = torch.cat([gl, extra_gl], dim=0)

            ious_nx = bbox_overlaps(qb, gb_dup).numpy()
            cost_nx = 1.0 - ious_nx
            rnx, cnx = linear_sum_assignment(cost_nx)

            for r, c in zip(rnx, cnx):
                lab = int(gl_dup[c])
                if NS <= lab < NE and c >= len(gb):
                    # This is an EXTRA match (from duplicated GT)
                    extra_ious_all.append(float(ious_nx[r, c]))

            n_images_used += 1

        ea = np.array(extra_ious_all) if extra_ious_all else np.array([0.0])
        oa = np.array(orig_new_ious) if orig_new_ious else np.array([0.0])

        print(f"  Images: {n_images_used}")
        print(f"  Original new-class matches: {len(orig_new_ious)} (IoU mean={oa.mean():.3f})")
        print(f"  Extra matches from duplication: {len(extra_ious_all)} (IoU mean={ea.mean():.3f})")

        # IoU distribution of extra matches
        print(f"\n  Extra match IoU distribution:")
        for thr in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
            n_above = int(np.sum(ea >= thr))
            print(f"    IoU≥{thr:.2f}: {n_above:>4d} ({n_above/max(len(ea),1)*100:>5.1f}%)")

        # Effective gradient: at different IoU filters, how many useful extra matches?
        print(f"\n  If we filter extra matches by IoU threshold:")
        for thr in [0.0, 0.05, 0.1, 0.2, 0.3]:
            kept = int(np.sum(ea >= thr))
            avg_iou = float(ea[ea >= thr].mean()) if kept > 0 else 0.0
            # Gradient quality proxy: IoU-weighted count
            weighted = float(np.sum(ea[ea >= thr])) if kept > 0 else 0.0
            print(f"    thr={thr:.2f}: kept={kept:>4d} ({kept/max(len(ea),1)*100:>5.1f}%), "
                  f"avg_IoU={avg_iou:.3f}, IoU-weighted_sum={weighted:.1f}")

        # Compare: what fraction of TOTAL new-class gradient comes from extra?
        if len(orig_new_ious) > 0:
            orig_gradient_proxy = sum(orig_new_ious)
            for thr in [0.0, 0.1, 0.2]:
                extra_gradient = float(np.sum(ea[ea >= thr]))
                total = orig_gradient_proxy + extra_gradient
                print(f"\n    thr={thr:.1f}: extra gradient share = {extra_gradient/total*100:.1f}% of total new-class signal")

    # R3 verdict
    print(f"\n  ═══ R3 RISK ASSESSMENT ═══")
    if len(extra_ious_all) > 0:
        frac_below_01 = float(np.mean(ea < 0.1))
        frac_below_005 = float(np.mean(ea < 0.05))
        if frac_below_01 > 0.7:
            print(f"  HIGH noise risk: {frac_below_01:.0%} of extra matches have IoU<0.1")
            print(f"  Most extra gradient is NOISE (query far from GT)")
            print(f"  RECOMMENDATION: Must filter extra matches (IoU≥0.1 or ≥0.15)")
            r3_risk = 'HIGH_NOISE'
        else:
            print(f"  MODERATE noise: {frac_below_01:.0%} below 0.1")
            r3_risk = 'MODERATE'
    else:
        r3_risk = 'NO_DATA'

    # ═══ FINAL SUMMARY ═══
    print("\n" + "=" * 80)
    print("FINAL RISK SUMMARY")
    print("=" * 80)
    print(f"  R1 (D1 truncation → old-class damage): {r1_risk}")
    print(f"  R2 (D2 larger γ → old-class drift):    {r2_data['new_old_ratio'] if r2_data else 'SKIPPED'}")
    print(f"  R3 (D3 GT dup → gradient noise):       {r3_risk}")

    # Save
    result = {
        'r1': {
            'n_new_gt': len(r1_new_old_max_iou),
            'overlap_any': round(float(np.mean(ov > 0)), 4),
            'overlap_03': round(float(np.mean(ov > 0.3)), 4),
            'overlap_05': round(float(np.mean(ov > 0.5)), 4),
            'query_also_old_03': round(float(np.mean(np.array(r1_query_old_iou) > 0.3)), 4) if r1_query_old_iou else None,
            'query_primary_new': round(float(np.mean(np.array(r1_query_primary) == 'new')), 4) if r1_query_primary else None,
            'risk': r1_risk,
        },
        'r2': r2_data,
        'r3': {
            'extra_ious_mean_2x': round(float(ea.mean()), 4) if extra_ious_all else None,
            'frac_below_01': round(float(frac_below_01), 4) if extra_ious_all else None,
            'risk': r3_risk,
        },
    }
    outpath = '/home/yelingfei/logs/tatri/risk_verification.json'
    with open(outpath, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {outpath}")


if __name__ == '__main__':
    main()
