#!/usr/bin/env python3
"""
LGQS Coverage Reconciliation — resolve the 51% vs 2.5% vs 0% conflict.

Three different "coverage" metrics measured by prior probes:

  A. check1_hook (51%): "at new-class GT positions, is new-class token the argmax?"
     → Position-level: does enc_cls[gt_pos] peak at new token?
     → This measures: if we COULD look at gt_pos, does text help? YES

  B. grmi_mechanism_diag (2.5%): "of top-900 queries, what fraction argmax to new token?"
     → Query-level: after topk selection, how many queries are "new-class queries"?
     → This measures: does the LGQS pipeline actually select new-class regions? RARELY

  C. benefit_chain (0%): same as B but with a possible bug
     → Need to verify vs B

The key insight: A and B measure different things and are BOTH correct.
- A=51% means new-class text embeddings DO respond at gt positions (text is working)
- B=2.5% means those positions don't make it into top-900 (absolute score too low)
- A is necessary but not sufficient for B.

This script measures ALL THREE on the same images to confirm consistency,
then adds:
  D. "How many top-900 queries overlap with new-class GT boxes?" (IoU-based)
  E. "What is the absolute score gap between new-class gt positions and the 900th position?"
     (measures how far away new-class regions are from entering top-900)
"""
import argparse, json, os, sys
import numpy as np
import torch

GCD_ROOT = os.environ.get('GCD_ROOT', os.path.dirname(os.path.abspath(__file__)))
if os.path.isdir(os.path.join(GCD_ROOT, 'mmdet')):
    os.chdir(GCD_ROOT)
sys.path.insert(0, GCD_ROOT)
NS, NE = 70, 80


def run(args):
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmdet.registry import DATASETS
    from torch.utils.data import DataLoader
    from mmengine.dataset import pseudo_collate
    from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_overlaps

    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cfg = Config.fromfile(args.config)
    cfg.work_dir = '/tmp/lgqs_reconcile'
    runner = Runner.from_cfg(cfg)
    model = runner.model
    ckpt = torch.load(args.ckpt, map_location='cpu')
    sd = ckpt.get('state_dict', ckpt)
    sd = {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model = model.to(dev); model.eval()

    ds = DATASETS.build(cfg.val_dataloader.dataset)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=pseudo_collate)
    first = next(iter(loader))
    tt = first['data_samples'][0].text
    if isinstance(tt, str): tt = (tt,)
    _, caption, tok_pos, _ = model.get_tokens_and_prompts(tt, True)
    tpm_tok = model.language_model.tokenizer(
        [caption], padding='max_length' if model.language_model.pad_to_max else 'longest',
        return_tensors='pt')
    tpm, _ = model.get_positive_map(tpm_tok, tok_pos)
    model.token_positive_maps = tpm

    with torch.no_grad():
        text_dict = model.language_model([caption])
        if model.text_feat_map is not None:
            text_dict['embedded'] = model.text_feat_map(text_dict['embedded'])

    # Build new-class token mask
    new_tok_pos = []
    for k, positions in tpm.items():
        if NS <= (k - 1) < NE:
            new_tok_pos.extend(positions)
    new_tok_pos = sorted(set(new_tok_pos))
    new_tok_set = set(new_tok_pos)

    N_imgs = min(args.max_images, len(ds))
    loader2 = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                         collate_fn=pseudo_collate)

    # Hook into the REAL pre_decoder to capture LGQS internals
    import types
    captured = {}
    orig_pre_decoder = type(model).pre_decoder.__wrapped__ if hasattr(type(model).pre_decoder, '__wrapped__') else None

    # Instead of hooking, call the model's actual LGQS path
    # The key function is gen_encoder_output_proposals + cls_branches

    metric_A = []  # check1_hook style: argmax at gt positions
    metric_B = []  # grmi_diag style: fraction of top-900 with new argmax
    metric_D = []  # IoU-based: top-900 positions overlapping new GT
    metric_E = []  # score gap: new-gt-position score vs 900th threshold

    processed = 0
    for item in loader2:
        if processed >= N_imgs: break
        data = model.data_preprocessor(
            {'inputs': item['inputs'], 'data_samples': item['data_samples']},
            training=False)
        gt_inst = data['data_samples'][0].gt_instances
        gt_labels = gt_inst.labels
        gt_bboxes = gt_inst.bboxes
        if hasattr(gt_bboxes, 'tensor'): gt_bboxes = gt_bboxes.tensor
        new_mask = (gt_labels >= NS) & (gt_labels < NE)
        if not new_mask.any():
            processed += 1; continue

        with torch.no_grad():
            img_feats = model.extract_feat(data['inputs'])
            enc_in, dec_in = model.pre_transformer(img_feats, data['data_samples'])
            enc_out = model.forward_encoder(**enc_in, text_dict=text_dict)

            memory = enc_out['memory']  # (bs, N_pos, 256)
            spatial_shapes = enc_out['spatial_shapes']
            memory_mask = enc_out.get('memory_mask', None)
            if memory_mask is None:
                memory_mask = torch.zeros(memory.shape[:2], dtype=torch.bool, device=dev)

            # Call gen_encoder_output_proposals (the real LGQS path)
            output_memory, output_proposals = model.gen_encoder_output_proposals(
                memory, memory_mask, spatial_shapes)

            # Compute enc_cls using the REAL cls_branches
            N_layers = len(model.bbox_head.cls_branches)
            memory_text = text_dict['embedded'].to(dev)
            text_mask = text_dict['text_token_mask'].to(dev)
            enc_cls = model.bbox_head.cls_branches[N_layers - 1](
                output_memory, memory_text, text_mask)  # (bs, N_pos, n_text_tokens)

            # === Metric A: at new-class GT positions, is new-class token the argmax? ===
            # Map GT boxes to encoder positions
            meta = data['data_samples'][0].metainfo
            ih, iw = meta['img_shape']
            # Convert GT boxes to normalized coords for position lookup
            new_gt_bboxes = gt_bboxes[new_mask]
            new_gt_cx = (new_gt_bboxes[:, 0] + new_gt_bboxes[:, 2]) / 2 / iw
            new_gt_cy = (new_gt_bboxes[:, 1] + new_gt_bboxes[:, 3]) / 2 / ih

            # Find nearest encoder position to each GT center
            # Reconstruct position coords from spatial_shapes
            all_positions = []
            for lvl, (h, w) in enumerate(spatial_shapes):
                for yi in range(h):
                    for xi in range(w):
                        all_positions.append(((xi + 0.5) / w, (yi + 0.5) / h))
            all_positions = torch.tensor(all_positions, device=dev)

            n_new_gt_argmax_new = 0
            n_new_gt_total = 0
            new_gt_scores = []
            for gi in range(len(new_gt_cx)):
                cx, cy = new_gt_cx[gi].item(), new_gt_cy[gi].item()
                dists = ((all_positions[:, 0] - cx)**2 + (all_positions[:, 1] - cy)**2)
                nearest_pos = dists.argmin().item()
                if nearest_pos < enc_cls.shape[1]:
                    logits_at_pos = enc_cls[0, nearest_pos]  # (n_text_tokens,)
                    argmax_tok = logits_at_pos.argmax().item()
                    score_at_pos = logits_at_pos.max().item()
                    if argmax_tok in new_tok_set:
                        n_new_gt_argmax_new += 1
                    n_new_gt_total += 1
                    new_gt_scores.append(score_at_pos)

            if n_new_gt_total > 0:
                metric_A.append(n_new_gt_argmax_new / n_new_gt_total)

            # === Metric B: fraction of top-900 with new-class argmax ===
            scores_max = enc_cls[0].max(dim=-1)[0]  # (N_pos,)
            argmax_all = enc_cls[0].argmax(dim=-1)   # (N_pos,)
            _, topk_idx = torch.topk(scores_max, k=min(900, len(scores_max)))
            topk_argmax = argmax_all[topk_idx]
            n_new_argmax_in_top900 = sum(1 for a in topk_argmax.tolist() if a in new_tok_set)
            metric_B.append(n_new_argmax_in_top900 / 900)

            # === Metric D: top-900 positions overlapping new GT (IoU > 0.3) ===
            # Get reference boxes for top-900 positions
            topk_proposals = output_proposals[0, topk_idx]  # (900, 4) normalized cxcywh
            fac = topk_proposals.new_tensor([iw, ih, iw, ih])
            topk_boxes = bbox_cxcywh_to_xyxy(topk_proposals) * fac
            topk_boxes[:, 0::2].clamp_(0, iw)
            topk_boxes[:, 1::2].clamp_(0, ih)

            if len(new_gt_bboxes) > 0:
                ious = bbox_overlaps(topk_boxes, new_gt_bboxes)  # (900, n_new_gt)
                max_iou_per_query = ious.max(dim=1)[0]  # (900,)
                n_overlap = (max_iou_per_query > 0.3).sum().item()
                metric_D.append(n_overlap / 900)

            # === Metric E: score gap ===
            threshold_900 = scores_max[topk_idx[-1]].item()  # 900th score
            if new_gt_scores:
                avg_new_gt_score = float(np.mean(new_gt_scores))
                metric_E.append(avg_new_gt_score - threshold_900)

        processed += 1
        if processed % 50 == 0:
            print(f"  {processed}/{N_imgs}")

    print("\n" + "=" * 80)
    print("LGQS COVERAGE RECONCILIATION")
    print("=" * 80)

    if metric_A:
        mA = float(np.mean(metric_A))
        print(f"\n  Metric A (check1_hook style):")
        print(f"    At new-class GT positions, new-class token is argmax: {mA:.1%}")
        print(f"    (Prior result: 98.9% via hook, 51% coverage via different method)")

    if metric_B:
        mB = float(np.mean(metric_B))
        print(f"\n  Metric B (grmi_diag style):")
        print(f"    Fraction of top-900 with new-class argmax: {mB:.2%}")
        print(f"    = {mB*900:.1f} out of 900 queries")
        print(f"    (Prior result: 2.5% via grmi_mechanism_diag)")

    if metric_D:
        mD = float(np.mean(metric_D))
        print(f"\n  Metric D (IoU-based, NEW):")
        print(f"    Top-900 positions overlapping new GT (IoU>0.3): {mD:.2%}")
        print(f"    = {mD*900:.1f} out of 900 queries")

    if metric_E:
        mE = float(np.mean(metric_E))
        print(f"\n  Metric E (score gap, NEW):")
        print(f"    New-GT-position score vs 900th threshold: {mE:+.4f}")
        if mE > 0:
            print(f"    → New-class GT positions ALREADY ABOVE threshold (in top-900)")
        else:
            print(f"    → New-class GT positions BELOW threshold by {abs(mE):.4f}")
            print(f"    → Text perturbation needs to raise score by {abs(mE):.4f} to enter top-900")

    # Reconciliation
    print("\n" + "=" * 80)
    print("RECONCILIATION")
    print("=" * 80)
    if metric_A and metric_B and metric_D:
        print(f"  A={np.mean(metric_A):.1%} (argmax correct at GT pos)")
        print(f"  B={np.mean(metric_B):.2%} (new argmax in top-900)")
        print(f"  D={np.mean(metric_D):.2%} (top-900 overlapping new GT)")
        print()
        if np.mean(metric_D) > 0.01:
            print("  INTERPRETATION: top-900 queries DO land on new-class objects")
            print("  (even if their argmax isn't on new-class token)")
            print("  → LGQS is NOT the bottleneck for coverage")
            print("  → The problem is downstream (decoder can't refine these queries)")
        else:
            print("  INTERPRETATION: top-900 queries DON'T land on new-class objects")
            print("  → LGQS IS the bottleneck")

    with open("lgqs_reconciliation.json", "w") as f:
        json.dump({
            "metric_A_argmax_at_gt": float(np.mean(metric_A)) if metric_A else None,
            "metric_B_new_argmax_in_top900": float(np.mean(metric_B)) if metric_B else None,
            "metric_D_iou_overlap": float(np.mean(metric_D)) if metric_D else None,
            "metric_E_score_gap": float(np.mean(metric_E)) if metric_E else None,
            "n_images": processed,
        }, f, indent=2)
    print(f"\nSaved: lgqs_reconciliation.json")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--max-images', type=int, default=200)
    run(p.parse_args())
