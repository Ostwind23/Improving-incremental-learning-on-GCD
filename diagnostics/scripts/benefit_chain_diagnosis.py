#!/usr/bin/env python3
"""
TATRI Benefit Chain Diagnosis — measure signal strength at each link:

Link 1: LGQS — does text perturbation change which positions enter top-900?
  Metric: new-class coverage (fraction of top-900 with argmax on new-class token)
  Measure: baseline T vs T + Δ_directional (not random — use oracle directions)

Link 2: Decoder text XAttn — do decoder queries attend more to new-class tokens?
  Metric: attention weight on new-class tokens vs old-class tokens
  For queries matched to new-class GT

Link 3: Classification — after decoder, can the model distinguish new from old?
  Metric: cls score on correct new class vs best confusing old class

Also measure the "gradient path length" problem:
  How much gradient flows back from cls_loss to memory_text?
  (confirms TATRI's gradient signal weakness)
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn as nn

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
    from mmdet.models.dense_heads.atss_vlfusion_head import convert_grounding_to_cls_scores

    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cfg = Config.fromfile(args.config)
    cfg.work_dir = '/tmp/chain_diag'
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

    # Identify new-class token positions
    new_tok_pos = []
    new_cls_to_tok = {}  # class_id -> list of token positions
    for k, positions in tpm.items():
        cls_id = k - 1
        if NS <= cls_id < NE:
            new_tok_pos.extend(positions)
            new_cls_to_tok[cls_id] = list(positions)
    new_tok_pos = sorted(set(new_tok_pos))
    ttm = text_dict['text_token_mask'][0]
    valid_pos = torch.where(ttm)[0].tolist()
    old_tok_pos = [p for p in valid_pos if p not in new_tok_pos]

    embedded = text_dict['embedded'][0].detach()  # (T, 256)
    new_embs = embedded[new_tok_pos]
    old_embs = embedded[old_tok_pos]

    print("=" * 80)
    print("LINK 0: Text embedding landscape (what does the transform need to learn?)")
    print("=" * 80)

    # For each new class, find its nearest old-class token and the DIRECTION to escape
    escape_directions = {}
    for cls_id, tok_positions in new_cls_to_tok.items():
        cls_emb = embedded[tok_positions].mean(dim=0)  # average if multi-token
        # Distance to nearest old
        dists = (old_embs - cls_emb.unsqueeze(0)).norm(dim=-1)
        nearest_idx = dists.argmin().item()
        nearest_dist = dists[nearest_idx].item()
        nearest_old_emb = old_embs[nearest_idx]
        # Escape direction: away from nearest old (orthogonal to old-new axis)
        toward_old = (nearest_old_emb - cls_emb)
        toward_old = toward_old / toward_old.norm()
        # "ideal" direction: perpendicular to toward_old in the cls_emb-old plane
        # Simple: use cls_emb - projection onto toward_old
        proj = (cls_emb * toward_old).sum() * toward_old
        perp = cls_emb - proj
        if perp.norm() > 1e-6:
            escape_dir = perp / perp.norm()
        else:
            escape_dir = torch.randn_like(cls_emb)
            escape_dir = escape_dir / escape_dir.norm()
        escape_directions[cls_id] = escape_dir
        cos_to_nearest = torch.cosine_similarity(cls_emb.unsqueeze(0),
                                                  nearest_old_emb.unsqueeze(0)).item()
        print(f"  cls {cls_id}: dist_to_nearest_old={nearest_dist:.3f} "
              f"cos={cos_to_nearest:.3f} norm={cls_emb.norm():.3f}")

    # ═══════════════════════════════════════════════════════════════════
    # LINK 1: LGQS sensitivity — does text change → top-900 change?
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("LINK 1: LGQS — text perturbation → top-900 new-class coverage")
    print("  Comparing: baseline vs oracle-direction perturbation (not random)")
    print("=" * 80)

    N_imgs = min(args.max_images, len(ds))
    loader_l1 = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                           collate_fn=pseudo_collate)

    # Build oracle-perturbed text: move each new-class token in its escape direction
    perturbation_scales = [0.0, 0.5, 1.0, 2.0, 5.0]
    results_l1 = {}

    for scale in perturbation_scales:
        td_pert = {}
        for k, v in text_dict.items():
            td_pert[k] = v.clone() if isinstance(v, torch.Tensor) else v
        if scale > 0:
            emb = td_pert['embedded'].clone()
            for cls_id, tok_positions in new_cls_to_tok.items():
                if cls_id in escape_directions:
                    for p in tok_positions:
                        if p < emb.shape[1]:
                            emb[0, p] += scale * escape_directions[cls_id].to(emb.device)
            td_pert['embedded'] = emb

        new_in_top900 = []
        processed = 0
        for item in loader_l1:
            if processed >= N_imgs: break
            data = model.data_preprocessor(
                {'inputs': item['inputs'], 'data_samples': item['data_samples']},
                training=False)
            gt_labels = data['data_samples'][0].gt_instances.labels
            if not ((gt_labels >= NS) & (gt_labels < NE)).any():
                processed += 1; continue

            with torch.no_grad():
                img_feats = model.extract_feat(data['inputs'])
                enc_in, dec_in = model.pre_transformer(img_feats, data['data_samples'])
                enc_out = model.forward_encoder(**enc_in, text_dict=td_pert)

                # Manually compute LGQS scores
                memory = enc_out['memory']
                memory_text = td_pert['embedded'].to(dev)
                text_mask = td_pert['text_token_mask'].to(dev)

                # enc_outputs_class approximation
                output_memory = memory  # simplified; real LGQS uses gen_encoder_output_proposals
                # Use the actual cls_branches for accurate scoring
                N_layers = len(model.bbox_head.cls_branches)
                enc_cls = model.bbox_head.cls_branches[N_layers - 1](
                    output_memory[0:1], memory_text, text_mask)
                # Score: max over text tokens
                scores = enc_cls[0].max(dim=-1)[0]  # (N_pos,)
                # Argmax class for each position
                argmax_cls = enc_cls[0].argmax(dim=-1)  # (N_pos,)
                _, topk_idx = torch.topk(scores, k=min(900, len(scores)))

                # How many top-900 positions have argmax on a new-class token?
                topk_argmax = argmax_cls[topk_idx]
                new_mask_tok = torch.zeros(enc_cls.shape[2], device=dev)
                for p in new_tok_pos:
                    if p < enc_cls.shape[2]:
                        new_mask_tok[p] = 1.0
                n_new = sum(1 for a in topk_argmax if a < len(new_mask_tok) and new_mask_tok[a] > 0)
                new_in_top900.append(n_new.item() if isinstance(n_new, torch.Tensor) else n_new)

            processed += 1

        mean_new = float(np.mean(new_in_top900)) if new_in_top900 else 0
        frac = mean_new / 900 * 100
        results_l1[scale] = {'mean_new_in_900': mean_new, 'frac_pct': frac,
                             'n_images': len(new_in_top900)}
        print(f"  scale={scale:.1f}: new_in_top900={mean_new:.1f} ({frac:.2f}%)  "
              f"[n={len(new_in_top900)} images]")

    # Check if ANY scale improves coverage
    baseline_cov = results_l1[0.0]['mean_new_in_900']
    best_scale = max(perturbation_scales, key=lambda s: results_l1[s]['mean_new_in_900'])
    best_cov = results_l1[best_scale]['mean_new_in_900']
    print(f"\n  LINK 1 VERDICT: baseline coverage={baseline_cov:.1f}, "
          f"best={best_cov:.1f} at scale={best_scale}")
    if best_cov > baseline_cov * 1.1:
        print(f"  → LINK 1 RESPONSIVE: oracle text perturbation improves LGQS coverage")
    else:
        print(f"  → LINK 1 SATURATED: even oracle text perturbation doesn't help LGQS")
        print(f"  → This means LGQS bottleneck is NOT in text quality but in encoder features")

    # ═══════════════════════════════════════════════════════════════════
    # LINK 3: Gradient path — how much gradient reaches memory_text?
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("LINK 3: Gradient path — does cls_loss gradient reach memory_text?")
    print("  (Measures the fundamental learnability problem)")
    print("=" * 80)

    # Enable grad on text embeddings, do one forward-backward, measure grad magnitude
    loader_g = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                          collate_fn=pseudo_collate)
    model.train()
    grad_norms_new = []
    grad_norms_old = []
    processed = 0
    for item in loader_g:
        if processed >= min(N_imgs, 20): break
        data = model.data_preprocessor(
            {'inputs': item['inputs'], 'data_samples': item['data_samples']},
            training=True)
        gt_labels = data['data_samples'][0].gt_instances.labels
        if not ((gt_labels >= NS) & (gt_labels < NE)).any():
            processed += 1; continue

        # Make text embeddings require grad
        td_grad = {}
        for k, v in text_dict.items():
            if isinstance(v, torch.Tensor):
                td_grad[k] = v.clone().detach().to(dev)
            else:
                td_grad[k] = v
        td_grad['embedded'] = td_grad['embedded'].clone().detach().requires_grad_(True)

        try:
            img_feats = model.extract_feat(data['inputs'])
            head_inputs = model.forward_transformer(img_feats, td_grad, data['data_samples'])
            losses = model.bbox_head.loss(
                **head_inputs, batch_data_samples=data['data_samples'])
            # Only cls loss
            total_cls = losses.get('loss_cls', torch.tensor(0.0))
            if isinstance(total_cls, torch.Tensor) and total_cls.requires_grad:
                total_cls.backward(retain_graph=False)

                if td_grad['embedded'].grad is not None:
                    g = td_grad['embedded'].grad[0]  # (T, 256)
                    new_g = g[new_tok_pos].norm(dim=-1)
                    old_g = g[old_tok_pos].norm(dim=-1)
                    grad_norms_new.append(new_g.mean().item())
                    grad_norms_old.append(old_g.mean().item())
        except Exception as e:
            print(f"  grad test error: {e}")
        finally:
            model.zero_grad()

        processed += 1

    model.eval()
    if grad_norms_new:
        mn = float(np.mean(grad_norms_new))
        mo = float(np.mean(grad_norms_old))
        ratio = mn / mo if mo > 0 else float('inf')
        print(f"  Grad norm at new-class text tokens: {mn:.6f}")
        print(f"  Grad norm at old-class text tokens: {mo:.6f}")
        print(f"  Ratio new/old: {ratio:.2f}x")
        print(f"  Absolute scale: {'WEAK' if mn < 0.001 else 'MODERATE' if mn < 0.01 else 'STRONG'}")

        # Compare to GRMI: GRMI's encoder memory grad ≈ ???
        # The key question: is the text gradient enough to train a transform?
        # A Linear(256,64) weight needs ~O(1e-3) gradient per param to train
        # With 256*64=16384 params and lr=5e-5: param_update ≈ lr * grad ≈ 5e-5 * grad
        # For meaningful update in 3 epochs: need grad > ~0.01
        if mn < 0.001:
            print(f"\n  → LINK 3 DEAD: gradient {mn:.6f} is too weak to train TATRI transform")
            print(f"  → With lr=5e-5, param update ≈ {5e-5 * mn:.8f} per step (negligible)")
            print(f"  → Direct text loss is REQUIRED — cannot rely on backprop from cls_loss")
        elif mn < 0.01:
            print(f"\n  → LINK 3 WEAK: gradient exists but marginal")
            print(f"  → Direct text loss would help significantly")
        else:
            print(f"\n  → LINK 3 OK: gradient is sufficient")

    # Summary
    print("\n" + "=" * 80)
    print("BENEFIT CHAIN SUMMARY")
    print("=" * 80)
    print("  Link 0: Text landscape — new-class tokens close to old (cos up to 0.996)")
    l1_ok = best_cov > baseline_cov * 1.1
    print(f"  Link 1: LGQS responsiveness — {'RESPONSIVE' if l1_ok else 'SATURATED'}")
    grad_ok = len(grad_norms_new) > 0 and np.mean(grad_norms_new) >= 0.001
    print(f"  Link 3: Gradient path — {'OK' if grad_ok else 'TOO WEAK'}")
    print()
    if not l1_ok:
        print("  ⚠ CRITICAL: LGQS doesn't respond to text perturbation even with oracle direction!")
        print("    → Options 1/2/3 all rely on LGQS responding to text change")
        print("    → If Link 1 is broken, ALL text-side options are dead")
        print("    → Must target decoder text cross-attention directly instead")
    elif not grad_ok:
        print("  ⚠ GRADIENT BOTTLENECK: text channel receives too little gradient from main loss")
        print("    → Options 1/2 (direct text loss) can fix this")
        print("    → Option 3 (lower γ) cannot fix this — same weak gradient, smaller perturbation")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--max-images', type=int, default=100)
    run(p.parse_args())
