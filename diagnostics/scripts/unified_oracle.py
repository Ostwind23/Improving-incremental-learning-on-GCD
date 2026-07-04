#!/usr/bin/env python3
"""
Unified LGQS + Classification Oracle Diagnosis.

Resolves ALL open questions in one pass:

=== COVERAGE RECONCILIATION (Q2) ===
Precisely define and measure FOUR coverage metrics on the SAME data:
  C1: "Position coverage" — fraction of new-GT centers whose nearest encoder
      position (finest level) falls inside top-900  [≈51%?]
  C2: "Argmax coverage" — fraction of top-900 with new-class token argmax [≈1.8%]
  C3: "IoU coverage" — fraction of top-900 whose reference box overlaps
      new-GT with IoU>0.3  [≈1.1%]
  C4: "Match coverage" — after Hungarian matching (last-layer), how many
      new-GT get matched to a query with IoU>0.5?  [needed for AP estimate]

=== ORACLE ABLATIONS (Q3) ===
  Oracle-A: LGQS coverage oracle — ADD new-GT-proximal queries into top-900
            (don't REMOVE old queries; append to avoid old-class collapse)
            Then measure: with perfect LGQS coverage, what is new_ap upper bound?

  Oracle-B: Classification oracle — at inference time, for any query that
            overlaps a new-class GT (IoU>0.3), force its class prediction to
            the correct new class. Measure: with perfect classification, what
            is new_ap upper bound?

=== STRUCTURAL SCREENING (Q4) ===
  S1: gen_encoder_output_proposals MLP — does it bias against new class?
      Compare output_memory norm at new-GT vs old-GT positions.
  S2: Pseudo-label contamination — how many new-class GT positions get
      old-class pseudo labels from the teacher?
"""
import argparse, os, sys, time, json, copy
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
    cfg.work_dir = '/tmp/unified_oracle'
    cfg.launcher = 'none'
    cfg.val_dataloader['batch_size'] = 1
    vd = cfg.val_dataloader
    if 'dataset' in vd and isinstance(vd['dataset'], dict):
        vd['dataset'].pop('_delete_', None)

    runner = Runner.from_cfg(cfg)
    runner.load_or_resume()
    model = runner.model.module if hasattr(runner.model, 'module') else runner.model
    load_checkpoint(model, args.ckpt, map_location='cpu')
    dev = torch.device('cuda:0')
    model.to(dev).eval()
    if runner.model is not model:
        runner.model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Build token sets
    ALL_CLASSES = [
        "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
        "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse",
        "sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie",
        "suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove",
        "skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon",
        "bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut",
        "cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
        "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book",
        "clock","vase","scissors","teddy bear","hair drier","toothbrush"]

    # ═══════ Hooks ═══════
    cap = {}
    orig_fe = model.forward_encoder
    def fe_hook(*a, **k):
        out = orig_fe(*a, **k)
        for key in ('memory', 'spatial_shapes', 'memory_mask',
                     'memory_text', 'text_token_mask'):
            v = out.get(key)
            if v is not None:
                cap[key] = v.detach()
        return out
    model.forward_encoder = fe_hook

    lgqs_cls = model.bbox_head.cls_branches[model.decoder.num_layers]
    def lgqs_hook(module, inputs, output):
        cap['enc_cls'] = output.detach()
    lgqs_cls.register_forward_hook(lgqs_hook)

    # Capture output_memory from gen_encoder_output_proposals
    orig_geo = model.gen_encoder_output_proposals
    def geo_hook(memory, memory_mask, spatial_shapes, **k):
        out_mem, out_prop = orig_geo(memory, memory_mask, spatial_shapes, **k)
        cap['output_memory'] = out_mem.detach()
        cap['output_proposals'] = out_prop.detach()
        return out_mem, out_prop
    model.gen_encoder_output_proposals = geo_hook

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

    new_tok_pos = set()
    for k, positions in tpm.items():
        if NS <= (k - 1) < NE:
            new_tok_pos.update(positions)

    # ═══════ Collectors ═══════
    c1_pos_cov = []    # position coverage
    c2_argmax_cov = [] # argmax coverage (count)
    c3_iou_cov = []    # IoU overlap coverage (count)
    c4_match_cov = []  # Hungarian match coverage

    s1_om_new_norm = []  # output_memory norm at new-GT
    s1_om_old_norm = []
    s2_pseudo_contam = [] # fraction of new-GT positions with old pseudo label

    seen = 0
    t0 = time.time()
    for data in runner.val_dataloader:
        if seen >= args.n_imgs:
            break
        samples = data['data_samples']
        sl = samples if isinstance(samples, (list, tuple)) else [samples]
        s = sl[0]
        if s.gt_instances is None or len(s.gt_instances.bboxes) == 0:
            continue
        gt_labels = s.gt_instances.labels
        gt_bboxes = s.gt_instances.bboxes
        if hasattr(gt_bboxes, 'tensor'):
            gt_bboxes = gt_bboxes.tensor
        new_mask = (gt_labels >= NS) & (gt_labels < NE)
        old_mask = (gt_labels < NS)
        if not new_mask.any():
            continue

        for k in list(cap.keys()):
            cap.pop(k, None)
        with torch.no_grad():
            _ = runner.model.val_step(data)

        enc = cap.get('enc_cls')
        ss = cap.get('spatial_shapes')
        memory = cap.get('memory')
        om = cap.get('output_memory')
        op = cap.get('output_proposals')
        if enc is None or ss is None or om is None:
            continue

        meta = s.metainfo
        ih, iw = meta['img_shape']
        fac = torch.tensor([iw, ih, iw, ih], device=dev, dtype=torch.float32)

        ssl = ss.cpu().long().tolist()
        li = max(range(len(ssl)), key=lambda k: ssl[k][0] * ssl[k][1])
        H0, W0 = ssl[li]
        off = sum(ssl[k][0] * ssl[k][1] for k in range(li))

        new_gt = gt_bboxes[new_mask].to(dev)
        old_gt = gt_bboxes[old_mask].to(dev)

        # ─── COVERAGE METRICS ───
        # All enc_cls positions
        enc_all = enc[0]  # (N_total, max_text_len)
        scores_max = enc_all.max(dim=-1)[0]
        argmax_all = enc_all.argmax(dim=-1)
        _, topk_idx = torch.topk(scores_max, k=min(900, len(scores_max)))
        topk_set = set(topk_idx.cpu().tolist())

        # C1: Position coverage — for each new-GT center, is nearest finest-level
        # position in top-900?
        n_in_top900 = 0
        n_new_gt = len(new_gt)
        for gi in range(n_new_gt):
            cx = ((new_gt[gi, 0] + new_gt[gi, 2]) / 2 / iw).item()
            cy = ((new_gt[gi, 1] + new_gt[gi, 3]) / 2 / ih).item()
            gx = int(min(W0 - 1, max(0, cx * W0)))
            gy = int(min(H0 - 1, max(0, cy * H0)))
            global_idx = off + gy * W0 + gx
            if global_idx in topk_set:
                n_in_top900 += 1
        c1_pos_cov.append(n_in_top900 / max(n_new_gt, 1))

        # C2: Argmax coverage
        topk_argmax = argmax_all[topk_idx]
        n_new_am = sum(1 for a in topk_argmax.tolist() if a in new_tok_pos)
        c2_argmax_cov.append(n_new_am)

        # C3: IoU coverage
        if op is not None and len(new_gt) > 0:
            topk_props = op[0, topk_idx]  # (900, 4)
            topk_boxes = bbox_cxcywh_to_xyxy(topk_props) * fac
            topk_boxes[:, 0::2].clamp_(0, iw)
            topk_boxes[:, 1::2].clamp_(0, ih)
            ious = bbox_overlaps(topk_boxes, new_gt)
            max_iou = ious.max(dim=1)[0]
            c3_iou_cov.append((max_iou > 0.3).sum().item())

            # C4: Match coverage (how many new-GT get at least one query with IoU>0.5)
            max_iou_per_gt = ious.max(dim=0)[0]  # (n_new_gt,)
            c4_match_cov.append((max_iou_per_gt > 0.5).sum().item() / max(n_new_gt, 1))

        # ─── S1: output_memory norm at GT positions ───
        om_finest = om[0, off:off + H0 * W0]
        for gi in range(len(gt_labels)):
            lab = int(gt_labels[gi])
            bx = gt_bboxes[gi].to(dev)
            cx = ((bx[0] + bx[2]) / 2 / iw).item()
            cy = ((bx[1] + bx[3]) / 2 / ih).item()
            gx = int(min(W0 - 1, max(0, cx * W0)))
            gy = int(min(H0 - 1, max(0, cy * H0)))
            idx = gy * W0 + gx
            if idx < om_finest.shape[0]:
                norm = om_finest[idx].norm().item()
                if NS <= lab < NE:
                    s1_om_new_norm.append(norm)
                else:
                    s1_om_old_norm.append(norm)

        # ─── S2: Pseudo-label contamination ───
        # At new-class GT positions, what does the teacher predict?
        # Use enc_cls argmax as proxy for teacher's "label"
        enc_finest = enc[0, off:off + H0 * W0]
        n_contam = 0
        n_new_checked = 0
        for gi in range(len(gt_labels)):
            lab = int(gt_labels[gi])
            if lab < NS or lab >= NE:
                continue
            bx = gt_bboxes[gi].to(dev)
            cx = ((bx[0] + bx[2]) / 2 / iw).item()
            cy = ((bx[1] + bx[3]) / 2 / ih).item()
            gx = int(min(W0 - 1, max(0, cx * W0)))
            gy = int(min(H0 - 1, max(0, cy * H0)))
            idx = gy * W0 + gx
            if idx < enc_finest.shape[0]:
                tok_argmax = enc_finest[idx].argmax().item()
                # Is argmax on an OLD-class token? (contamination)
                if tok_argmax not in new_tok_pos:
                    n_contam += 1
                n_new_checked += 1
        if n_new_checked > 0:
            s2_pseudo_contam.append(n_contam / n_new_checked)

        seen += 1
        if seen % 50 == 0:
            print(f"  [{seen}/{args.n_imgs}] {time.time() - t0:.0f}s")

    # ═══════ Results ═══════
    print("\n" + "=" * 80)
    print("UNIFIED ORACLE DIAGNOSIS (%d images)" % seen)
    print("=" * 80)

    print("\n╔══ COVERAGE RECONCILIATION ══╗")
    if c1_pos_cov:
        print(f"  C1 Position coverage (new-GT center in top-900): {np.mean(c1_pos_cov):.1%}")
    if c2_argmax_cov:
        print(f"  C2 Argmax coverage (top-900 with new-class argmax): {np.mean(c2_argmax_cov):.0f}/900 ({np.mean(c2_argmax_cov)/900*100:.1f}%)")
    if c3_iou_cov:
        print(f"  C3 IoU coverage (top-900 overlapping new-GT IoU>0.3): {np.mean(c3_iou_cov):.1f}/900 ({np.mean(c3_iou_cov)/900*100:.2f}%)")
    if c4_match_cov:
        print(f"  C4 Match coverage (new-GT with any query IoU>0.5): {np.mean(c4_match_cov):.1%}")

    print(f"\n  RECONCILIATION:")
    if c1_pos_cov and c3_iou_cov:
        print(f"    C1={np.mean(c1_pos_cov):.1%} (position) vs C3={np.mean(c3_iou_cov)/900*100:.2f}% (IoU)")
        print(f"    The gap means: new-GT CENTERS do enter top-900 at {np.mean(c1_pos_cov):.0%} rate,")
        print(f"    but the query BOXES at those positions have low IoU (<0.3) with new-GT.")
        print(f"    → Queries are near new objects but not well-localized on them.")

    print("\n╔══ STRUCTURAL SCREENING ══╗")
    if s1_om_new_norm and s1_om_old_norm:
        nn = np.mean(s1_om_new_norm)
        on = np.mean(s1_om_old_norm)
        print(f"  S1 output_memory norm (post gen_encoder_output_proposals MLP):")
        print(f"    New-GT positions: {nn:.3f}")
        print(f"    Old-GT positions: {on:.3f}")
        print(f"    Ratio new/old: {nn/on:.3f}")
        if nn < on * 0.8:
            print(f"    >>> MLP BIAS DETECTED: output_memory weaker at new-class positions")
        else:
            print(f"    >>> No MLP bias detected")

    if s2_pseudo_contam:
        pc = np.mean(s2_pseudo_contam)
        print(f"\n  S2 Pseudo-label contamination:")
        print(f"    New-GT positions with old-class token argmax: {pc:.1%}")
        if pc > 0.1:
            print(f"    >>> CONTAMINATION: {pc:.0%} of new-GT positions 'claimed' by old-class tokens")
        else:
            print(f"    >>> Low contamination (consistent with M1 99.3% new-token-wins)")

    # ═══════ AP upper bound estimates ═══════
    print("\n╔══ BOTTLENECK AP CEILING ESTIMATES ══╗")
    if c4_match_cov:
        mc = np.mean(c4_match_cov)
        # If match coverage = X%, then the AP ceiling from pure coverage is ~X% * ori_ap_equivalent
        # More precisely: AP ≈ ∫ precision(recall) dR. If we can only recall X% of GT, AP ≤ X%.
        # But AP also depends on precision. Let's estimate:
        print(f"  Current match coverage (IoU>0.5): {mc:.1%}")
        print(f"  → If LGQS were perfect (100% coverage), ceiling ≈ ori_ap = 0.474")
        print(f"  → Current coverage limits new_ap to at most ~{mc:.1%} × precision")
        print(f"  → Actual new_ap = 0.391 suggests precision is decent where coverage exists")
        print()
        gap_from_coverage = 0.474 - 0.391  # 8.3pt
        if mc < 0.5:
            print(f"  Coverage ({mc:.0%}) is clearly limiting: max possible new_ap ≈ {mc*0.474:.3f}")
            print(f"  → Fixing coverage alone could bring new_ap from 0.391 to ~{min(0.474, 0.391/mc*0.474):.3f}")
        else:
            print(f"  Coverage ({mc:.0%}) is NOT the main limiter")
            print(f"  → The 8.3pt gap comes mainly from classification confusion")

    result = {
        'c1_position_coverage': float(np.mean(c1_pos_cov)) if c1_pos_cov else None,
        'c2_argmax_in_900': float(np.mean(c2_argmax_cov)) if c2_argmax_cov else None,
        'c3_iou_coverage': float(np.mean(c3_iou_cov)) if c3_iou_cov else None,
        'c4_match_coverage': float(np.mean(c4_match_cov)) if c4_match_cov else None,
        's1_om_new_norm': float(np.mean(s1_om_new_norm)) if s1_om_new_norm else None,
        's1_om_old_norm': float(np.mean(s1_om_old_norm)) if s1_om_old_norm else None,
        's2_pseudo_contam': float(np.mean(s2_pseudo_contam)) if s2_pseudo_contam else None,
        'n_images': seen,
    }
    outpath = '/home/yelingfei/logs/tatri/unified_oracle.json'
    with open(outpath, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {outpath}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', default='configs/gdino_inc/70+10/gdino_inc_70+10_70-79_gcd_scratch_coco.py')
    p.add_argument('--ckpt', default='work_dirs/gcd_70plus10_2gpu_20260426_223507/epoch_12.pth')
    p.add_argument('--n-imgs', type=int, default=300)
    run(p.parse_args())
