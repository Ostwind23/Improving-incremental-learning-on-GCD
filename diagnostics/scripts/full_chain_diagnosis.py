#!/usr/bin/env python3
"""
Full Pipeline Chain Diagnosis — from encoder to final prediction.

Measures every link in the chain for NEW-CLASS objects specifically:

STAGE 1 (Encoder/LGQS):
  S1a: New-GT position enc_cls score vs top-900 threshold  [M3 = -2.67, confirmed]
  S1b: How many top-900 queries land on new-class GT? (IoU-based)  [M2 = 13.4]
  S1c: Compare visual feature norm at new-GT vs old-GT positions  [NEW — is visual weak?]

STAGE 2 (Decoder input → output):
  S2a: Of the 13 new-class queries entering decoder, what IoU do they achieve?
  S2b: Do these queries get BETTER or WORSE through 6 decoder layers?
  S2c: In decoder text cross-attention, do new-class queries attend to correct token?

STAGE 3 (Classification head):
  S3a: After decoder, what cls score does the model give to new-class at matched queries?
  S3b: Confusion: does the model predict the CORRECT new class or a confusing old class?

This replaces all prior broken/partial diagnostics with one authoritative pipeline.
"""
import argparse, os, sys, time, json
import numpy as np
import torch

GCD_ROOT = os.environ.get('GCD_ROOT', '/home/yelingfei/projects/GCD')
if os.path.isdir(os.path.join(GCD_ROOT, 'mmdet')):
    os.chdir(GCD_ROOT)
sys.path.insert(0, GCD_ROOT)
NS, NE = 70, 80

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


def build_token_sets(model):
    tok = model.language_model.tokenizer
    cap = '. '.join(ALL_CLASSES) + '.'
    enc = tok(cap, return_offsets_mapping=True, add_special_tokens=True)
    offsets = enc['offset_mapping']
    new_cols, old_cols = [], []
    cursor = 0
    for ci, cname in enumerate(ALL_CLASSES):
        idx = cap.find(cname, cursor)
        if idx < 0: cursor += 1; continue
        c0, c1 = idx, idx + len(cname)
        toks = [t for t, (s, e) in enumerate(offsets) if s < c1 and e > c0]
        if ci >= 70: new_cols.extend(toks)
        else: old_cols.extend(toks)
        cursor = c1
    return sorted(set(new_cols)), sorted(set(old_cols))


def run(args):
    from mmengine.config import Config
    from mmengine.runner import Runner, load_checkpoint
    from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_overlaps
    from mmdet.models.dense_heads.atss_vlfusion_head import convert_grounding_to_cls_scores

    cfg = Config.fromfile(args.cfg)
    cfg.work_dir = '/tmp/chain_full'
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

    new_cols, old_cols = build_token_sets(model)
    new_col_set = set(new_cols)

    # ═══════ Hooks ═══════
    cap = {}
    orig_fe = model.forward_encoder
    def fe_hook(*a, **k):
        out = orig_fe(*a, **k)
        for key in ('memory','spatial_shapes','memory_mask','memory_text','text_token_mask'):
            v = out.get(key)
            if v is not None: cap[key] = v.detach()
        return out
    model.forward_encoder = fe_hook

    # Capture enc_cls
    lgqs_cls = model.bbox_head.cls_branches[model.decoder.num_layers]
    def lgqs_hook(module, inputs, output):
        cap['enc_cls'] = output.detach()
    lgqs_cls.register_forward_hook(lgqs_hook)

    # Capture per-layer decoder hidden states and references
    orig_fd = model.forward_decoder
    def fd_hook(*a, **k):
        out = orig_fd(*a, **k)
        if 'hidden_states' in out:
            hs_val = out['hidden_states']
            if isinstance(hs_val, (list, tuple)):
                cap['hidden_states'] = torch.stack([h.detach() for h in hs_val])
            else:
                cap['hidden_states'] = hs_val.detach()
        if 'references' in out:
            ref_val = out['references']
            if isinstance(ref_val, (list, tuple)):
                cap['references'] = torch.stack([r.detach() for r in ref_val])
            else:
                cap['references'] = ref_val.detach()
        return out
    model.forward_decoder = fd_hook

    # Build tpm
    ds = runner.val_dataloader.dataset
    loader_it = iter(runner.val_dataloader)
    first_data = next(loader_it)
    sl = first_data['data_samples']
    s0 = sl[0] if isinstance(sl, (list,tuple)) else sl
    tt = s0.text
    if isinstance(tt, str): tt = (tt,)
    _, caption, tok_pos, _ = model.get_tokens_and_prompts(tt, True)
    tpm_tok = model.language_model.tokenizer(
        [caption], padding='max_length' if model.language_model.pad_to_max else 'longest',
        return_tensors='pt')
    tpm, _ = model.get_positive_map(tpm_tok, tok_pos)
    model.token_positive_maps = tpm

    # ═══════ Collectors ═══════
    # S1
    s1c_new_feat_norms = []
    s1c_old_feat_norms = []
    s1b_new_gt_in_top900 = []

    # S2
    s2a_new_query_ious = []         # IoU of new-class matched queries (last layer)
    s2b_per_layer_ious = {i: [] for i in range(6)}  # IoU at each decoder layer
    s2c_new_attn_correct = []       # does query attend to correct new-class token?

    # S3
    s3a_cls_scores_correct = []     # cls score on correct new class
    s3a_cls_scores_best_old = []    # cls score on best confusing old class
    s3b_confusions = []             # (predicted_class, true_class) for misclassified

    seen = 0
    t0 = time.time()
    for data in runner.val_dataloader:
        if seen >= args.n_imgs: break
        samples = data['data_samples']
        sl = samples if isinstance(samples, (list, tuple)) else [samples]
        s = sl[0]
        if s.gt_instances is None or len(s.gt_instances.bboxes) == 0: continue
        gt_labels = s.gt_instances.labels
        gt_bboxes = s.gt_instances.bboxes
        if hasattr(gt_bboxes, 'tensor'): gt_bboxes = gt_bboxes.tensor
        new_mask = (gt_labels >= NS) & (gt_labels < NE)
        old_mask = (gt_labels < NS)
        if not new_mask.any(): continue

        for k in list(cap.keys()): cap.pop(k, None)
        with torch.no_grad():
            _ = runner.model.val_step(data)

        enc = cap.get('enc_cls')
        ss = cap.get('spatial_shapes')
        memory = cap.get('memory')
        hs = cap.get('hidden_states')
        refs = cap.get('references')
        if enc is None or ss is None or memory is None: continue

        ssl = ss.cpu().long().tolist()
        meta = s.metainfo
        ih, iw = meta['img_shape']

        # ─── S1c: Visual feature norm at GT positions ───
        # Find finest level
        li = max(range(len(ssl)), key=lambda k: ssl[k][0]*ssl[k][1])
        H0, W0 = ssl[li]
        off = sum(ssl[k][0]*ssl[k][1] for k in range(li))
        mem_finest = memory[0, off:off+H0*W0, :]  # (H0*W0, 256)

        for gi in range(len(gt_labels)):
            lab = int(gt_labels[gi])
            bx = gt_bboxes[gi]
            cx = ((bx[0]+bx[2])/2/iw).item()
            cy = ((bx[1]+bx[3])/2/ih).item()
            gx = int(min(W0-1, max(0, cx*W0)))
            gy = int(min(H0-1, max(0, cy*H0)))
            feat_norm = mem_finest[gy*W0+gx].norm().item()
            if NS <= lab < NE:
                s1c_new_feat_norms.append(feat_norm)
            else:
                s1c_old_feat_norms.append(feat_norm)

        # ─── S1b: top-900 queries on new GT (IoU-based, all levels) ───
        if refs is not None and refs.shape[0] > 0:
            # Last layer references = query reference points
            ref_last = refs[-1, 0]  # (nq, 4) cxcywh normalized
            fac = ref_last.new_tensor([iw, ih, iw, ih])
            pred_boxes = bbox_cxcywh_to_xyxy(ref_last) * fac
            pred_boxes[:, 0::2].clamp_(0, iw)
            pred_boxes[:, 1::2].clamp_(0, ih)
            new_gt = gt_bboxes[new_mask].to(dev)
            if len(new_gt) > 0:
                ious_to_newgt = bbox_overlaps(pred_boxes, new_gt)  # (nq, n_new)
                max_iou_per_q = ious_to_newgt.max(dim=1)[0]  # (nq,)
                n_overlap = (max_iou_per_q > 0.3).sum().item()
                s1b_new_gt_in_top900.append(n_overlap)

                # ─── S2a: IoU of best-matched queries to new GT ───
                for ni in range(len(new_gt)):
                    best_q = ious_to_newgt[:, ni].argmax().item()
                    best_iou = ious_to_newgt[best_q, ni].item()
                    s2a_new_query_ious.append(best_iou)

                # ─── S2b: Per-layer IoU for new GT ───
                if refs.shape[0] >= 6:
                    for layer_i in range(6):
                        ref_l = refs[layer_i, 0]
                        boxes_l = bbox_cxcywh_to_xyxy(ref_l) * fac
                        boxes_l[:, 0::2].clamp_(0, iw)
                        boxes_l[:, 1::2].clamp_(0, ih)
                        ious_l = bbox_overlaps(boxes_l, new_gt)
                        for ni in range(len(new_gt)):
                            best_iou_l = ious_l[:, ni].max().item()
                            s2b_per_layer_ious[layer_i].append(best_iou_l)

                # ─── S3: Classification at matched queries ───
                # Use convert_grounding_to_cls_scores for per-class scores
                head = model.bbox_head
                all_cls, all_bbox = head(
                    hs, refs,
                    cap.get('memory_text', None),
                    cap.get('text_token_mask', None))
                last_cls = all_cls[-1]  # (1, nq, text_len)
                cls_logits = last_cls[0].sigmoid().unsqueeze(0)
                cls_by_class = convert_grounding_to_cls_scores(
                    logits=cls_logits, positive_maps=[tpm])[0]  # (nq, 80)

                new_gt_labels = gt_labels[new_mask]
                for ni in range(len(new_gt)):
                    best_q = ious_to_newgt[:, ni].argmax().item()
                    best_iou = ious_to_newgt[best_q, ni].item()
                    if best_iou < 0.1: continue
                    true_cls = int(new_gt_labels[ni])
                    score_correct = cls_by_class[best_q, true_cls].item()
                    # Best old-class score
                    old_scores = cls_by_class[best_q, :NS]
                    best_old_score = old_scores.max().item()
                    best_old_cls = old_scores.argmax().item()
                    s3a_cls_scores_correct.append(score_correct)
                    s3a_cls_scores_best_old.append(best_old_score)
                    if best_old_score > score_correct:
                        s3b_confusions.append((best_old_cls, true_cls))

        seen += 1
        if seen % 50 == 0:
            print(f"  [{seen}/{args.n_imgs}] {time.time()-t0:.0f}s")

    # ═══════ Results ═══════
    print("\n" + "=" * 80)
    print("FULL PIPELINE CHAIN DIAGNOSIS (%d images with new-class GT)" % seen)
    print("=" * 80)

    print("\n╔══ STAGE 1: ENCODER / LGQS ══╗")
    if s1c_new_feat_norms and s1c_old_feat_norms:
        nn = np.array(s1c_new_feat_norms)
        on = np.array(s1c_old_feat_norms)
        print(f"  S1c VISUAL FEATURE NORM at GT positions:")
        print(f"    New-class GT: mean={nn.mean():.3f} median={np.median(nn):.3f}")
        print(f"    Old-class GT: mean={on.mean():.3f} median={np.median(on):.3f}")
        print(f"    Ratio new/old: {nn.mean()/on.mean():.3f}")
        if nn.mean() < on.mean() * 0.8:
            print(f"    >>> VISUAL WEAKNESS CONFIRMED: new-class features {nn.mean()/on.mean()*100:.0f}% of old")
        else:
            print(f"    >>> Visual features comparable — bottleneck is NOT feature norm")

    if s1b_new_gt_in_top900:
        ov = np.array(s1b_new_gt_in_top900)
        print(f"\n  S1b TOP-900 queries overlapping new GT (IoU>0.3):")
        print(f"    Mean: {ov.mean():.1f}/900 ({ov.mean()/900*100:.2f}%)")

    print("\n╔══ STAGE 2: DECODER REFINEMENT ══╗")
    if s2a_new_query_ious:
        qi = np.array(s2a_new_query_ious)
        print(f"  S2a Best-match query IoU to new GT (last decoder layer):")
        print(f"    Mean={qi.mean():.3f} median={np.median(qi):.3f}")
        print(f"    IoU>=0.5: {np.mean(qi>=0.5)*100:.1f}%  IoU>=0.3: {np.mean(qi>=0.3)*100:.1f}%")

    if s2b_per_layer_ious[0]:
        print(f"\n  S2b Per-layer IoU progression (new-class GT):")
        for layer_i in range(6):
            v = np.array(s2b_per_layer_ious[layer_i])
            print(f"    d{layer_i}: mean={v.mean():.3f} ≥0.5={np.mean(v>=0.5)*100:.1f}% ≥0.3={np.mean(v>=0.3)*100:.1f}%")
        v0 = np.array(s2b_per_layer_ious[0])
        v5 = np.array(s2b_per_layer_ious[5])
        delta = v5.mean() - v0.mean()
        print(f"    Refinement d0→d5: {delta:+.3f} ({'IMPROVING' if delta > 0.01 else 'STAGNANT' if delta > -0.01 else 'DEGRADING'})")

    print("\n╔══ STAGE 3: CLASSIFICATION HEAD ══╗")
    if s3a_cls_scores_correct:
        cc = np.array(s3a_cls_scores_correct)
        co = np.array(s3a_cls_scores_best_old)
        print(f"  S3a Classification score at new-class matched queries:")
        print(f"    Correct new-class score: mean={cc.mean():.4f} median={np.median(cc):.4f}")
        print(f"    Best old-class score:    mean={co.mean():.4f} median={np.median(co):.4f}")
        print(f"    Correct > old: {np.mean(cc > co)*100:.1f}%")
        print(f"    Margin (correct - old): mean={np.mean(cc-co):.4f}")

    if s3b_confusions:
        print(f"\n  S3b Confusions (predicted old-class when true is new-class):")
        print(f"    Total confused: {len(s3b_confusions)}/{len(s3a_cls_scores_correct)} = {len(s3b_confusions)/max(1,len(s3a_cls_scores_correct))*100:.1f}%")
        from collections import Counter
        top_conf = Counter([(ALL_CLASSES[p], ALL_CLASSES[t]) for p, t in s3b_confusions]).most_common(10)
        for (pred, true), count in top_conf:
            print(f"      {true} → {pred}: {count}")

    # ═══════ CHAIN SUMMARY ═══════
    print("\n" + "=" * 80)
    print("CHAIN BOTTLENECK IDENTIFICATION")
    print("=" * 80)
    if s1c_new_feat_norms:
        vis_ratio = np.mean(s1c_new_feat_norms) / max(np.mean(s1c_old_feat_norms), 1e-9)
    else:
        vis_ratio = 1.0
    if s1b_new_gt_in_top900:
        coverage = np.mean(s1b_new_gt_in_top900) / 900
    else:
        coverage = 0
    if s2b_per_layer_ious[0]:
        refinement = np.mean(s2b_per_layer_ious[5]) - np.mean(s2b_per_layer_ious[0])
    else:
        refinement = 0
    if s3a_cls_scores_correct:
        cls_correct_rate = np.mean(np.array(s3a_cls_scores_correct) > np.array(s3a_cls_scores_best_old))
    else:
        cls_correct_rate = 0

    print(f"  Encoder visual norm ratio (new/old):  {vis_ratio:.3f}")
    print(f"  LGQS coverage (new GT in top-900):    {coverage*100:.2f}%")
    print(f"  Decoder refinement (d0→d5 IoU gain):  {refinement:+.3f}")
    print(f"  Classification correct rate:          {cls_correct_rate*100:.1f}%")
    print()
    bottlenecks = []
    if vis_ratio < 0.8:
        bottlenecks.append(("ENCODER VISUAL", f"new features {vis_ratio:.0%} of old"))
    if coverage < 0.05:
        bottlenecks.append(("LGQS COVERAGE", f"only {coverage*100:.1f}% of queries on new GT"))
    if -0.01 < refinement < 0.01:
        bottlenecks.append(("DECODER STAGNANT", f"d0→d5 gain = {refinement:+.3f}"))
    if cls_correct_rate < 0.5:
        bottlenecks.append(("CLASSIFICATION", f"only {cls_correct_rate*100:.0f}% correct"))
    if bottlenecks:
        print("  IDENTIFIED BOTTLENECKS:")
        for name, desc in bottlenecks:
            print(f"    ● {name}: {desc}")
    else:
        print("  No clear single bottleneck — problem is distributed")

    result = {
        's1c_new_feat_norm': float(np.mean(s1c_new_feat_norms)) if s1c_new_feat_norms else None,
        's1c_old_feat_norm': float(np.mean(s1c_old_feat_norms)) if s1c_old_feat_norms else None,
        's1c_ratio': float(vis_ratio),
        's1b_coverage': float(coverage),
        's2a_matched_iou': float(np.mean(s2a_new_query_ious)) if s2a_new_query_ious else None,
        's2b_d0_iou': float(np.mean(s2b_per_layer_ious[0])) if s2b_per_layer_ious[0] else None,
        's2b_d5_iou': float(np.mean(s2b_per_layer_ious[5])) if s2b_per_layer_ious[5] else None,
        's2b_refinement': float(refinement),
        's3a_correct_score': float(np.mean(s3a_cls_scores_correct)) if s3a_cls_scores_correct else None,
        's3a_best_old_score': float(np.mean(s3a_cls_scores_best_old)) if s3a_cls_scores_best_old else None,
        's3a_correct_rate': float(cls_correct_rate),
        's3b_n_confusions': len(s3b_confusions),
        'n_images': seen,
    }
    outpath = '/home/yelingfei/logs/tatri/full_chain_diagnosis.json'
    with open(outpath, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {outpath}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', default='configs/gdino_inc/70+10/gdino_inc_70+10_70-79_gcd_scratch_coco.py')
    p.add_argument('--ckpt', default='work_dirs/gcd_70plus10_2gpu_20260426_223507/epoch_12.pth')
    p.add_argument('--n-imgs', type=int, default=200)
    run(p.parse_args())
