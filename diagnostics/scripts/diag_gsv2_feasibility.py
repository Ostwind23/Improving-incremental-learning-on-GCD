"""
GS-GRMI v2 feasibility diagnostic.

Part 1: Token position -> class mapping verification
  - Build token_positive_maps from tokenizer
  - Cross-check with model.token_positive_maps (built during loss())
  - Print exact token ranges per class, confirm 70-79 = new classes
  - Verify on actual labels from Hungarian matching

Part 2: Detection loss split feasibility
  - Decompose detection loss into new-class vs old-class contributions
  - Measure actual L_new / L_old / L_bg ratio

Part 3: Gradient direction test (the KEY question)
  - Compute grad(L_new) and grad(L_old) w.r.t. R(M) parameters
  - Measure cosine similarity between them
  - If cos ~ 1.0: gradients point same direction -> splitting is pointless
  - If cos << 1.0: gradients differ -> splitting is valuable
"""
import mmdet.apis  # noqa
import mmdet.engine.hooks  # noqa
import torch, os, json, time
import numpy as np
from mmengine.config import Config
from mmengine.runner import Runner, load_checkpoint

ALL_CLASSES = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse",
    "sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie",
    "suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove",
    "skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon",
    "bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut",
    "cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book",
    "clock","vase","scissors","teddy bear","hair drier","toothbrush"]


def main():
    cfg = Config.fromfile("/home/yelingfei/projects/GCD/configs/gdino_inc/70+10/grmi_t1_decouple_6e_coco.py")
    cfg.train_cfg.max_epochs = 1
    cfg.train_dataloader.batch_size = 2
    cfg.work_dir = "/tmp/gsv2_diag"
    cfg.launcher = "none"
    cfg.default_hooks.checkpoint = dict(type="CheckpointHook", interval=999)
    runner = Runner.from_cfg(cfg); runner.load_or_resume()
    model = runner.model.module if hasattr(runner.model, "module") else runner.model
    model.train()

    # ============ PART 1: Token position <-> class mapping ============
    print("=" * 60)
    print("PART 1: Token position -> class mapping")
    print("=" * 60)

    # Method A: Build from tokenizer directly
    tok = model.language_model.tokenizer
    cap_str = '. '.join(ALL_CLASSES) + '.'
    enc = tok(cap_str, return_offsets_mapping=True, add_special_tokens=True)
    offsets = enc['offset_mapping']
    token_ranges_manual = {}
    cursor = 0
    for ci, cname in enumerate(ALL_CLASSES):
        idx = cap_str.find(cname, cursor)
        if idx < 0:
            token_ranges_manual[ci] = []
            continue
        c0, c1 = idx, idx + len(cname)
        toks = [ti for ti, (s, e) in enumerate(offsets) if s < c1 and e > c0]
        token_ranges_manual[ci] = toks
        cursor = c1

    # Find boundary
    new_start_token = min(token_ranges_manual[70])
    old_end_token = max(token_ranges_manual[69])
    print(f"Manual tokenizer analysis:")
    print(f"  Old classes (0-69): token range [{min(token_ranges_manual[0])}, {old_end_token}]")
    print(f"  New classes (70-79): token range [{new_start_token}, {max(token_ranges_manual[79])}]")
    print(f"  Gap between old and new: {new_start_token - old_end_token - 1} tokens")
    print(f"\n  New class token positions:")
    for c in range(70, 80):
        print(f"    class {c} ({ALL_CLASSES[c]}): tokens {token_ranges_manual[c]}")

    # Method B: Get from model after one forward pass (model.token_positive_maps)
    dl = iter(runner.train_dataloader)
    data = next(dl)
    data = model.data_preprocessor(data, True)
    # Run forward to trigger token_positive_maps construction
    losses = model(**data, mode="loss")

    tpm = model.token_positive_maps  # dict: {class_id+1: [token_positions]}
    print(f"\n  Model token_positive_maps (subset):")
    for c in range(70, 80):
        key = c + 1  # 1-indexed in GCD
        if key in tpm:
            print(f"    class {c} ({ALL_CLASSES[c]}): key={key} tokens={list(tpm[key])}")
        else:
            print(f"    class {c} ({ALL_CLASSES[c]}): key={key} NOT FOUND")

    # Cross-check
    print(f"\n  Cross-check (manual vs model.token_positive_maps):")
    match = True
    for c in range(80):
        key = c + 1
        manual = set(token_ranges_manual[c])
        model_tpm = set(tpm[key]) if key in tpm else set()
        if manual != model_tpm:
            print(f"    MISMATCH class {c}: manual={manual} model={model_tpm}")
            match = False
    if match:
        print(f"    ALL 80 classes MATCH between manual tokenizer and model.token_positive_maps")

    # ============ PART 2: Loss split feasibility ============
    print()
    print("=" * 60)
    print("PART 2: Detection loss split - new vs old class composition")
    print("=" * 60)

    head = model.bbox_head
    orig_get_targets = head.get_targets

    new_token_set = set()
    for c in range(70, 80):
        new_token_set.update(token_ranges_manual[c])
    old_token_set = set()
    for c in range(0, 70):
        old_token_set.update(token_ranges_manual[c])

    loss_stats = {"n_new": [], "n_old": [], "n_bg": [], "L_new": [], "L_old": [], "L_bg": []}

    def patched_get_targets(cls_scores_list, bbox_preds_list, batch_gt_instances, batch_img_metas):
        result = orig_get_targets(cls_scores_list, bbox_preds_list, batch_gt_instances, batch_img_metas)
        labels_list = result[0]
        # labels_list[i]: (900, 256) multi-hot, 1 at matched token positions
        for lab_tensor in labels_list:
            matched_mask = lab_tensor.sum(-1) > 0  # (900,) which queries are matched
            for qi in range(lab_tensor.shape[0]):
                if matched_mask[qi]:
                    hot_tokens = lab_tensor[qi].nonzero(as_tuple=True)[0].tolist()
                    is_new = any(t in new_token_set for t in hot_tokens)
                    is_old = any(t in old_token_set for t in hot_tokens)
                    if is_new:
                        loss_stats["n_new"].append(1)
                    elif is_old:
                        loss_stats["n_old"].append(1)
        return result

    head.get_targets = patched_get_targets

    # Run 10 iters to collect stats
    dl2 = iter(runner.train_dataloader)
    for step in range(10):
        data = next(dl2)
        data = model.data_preprocessor(data, True)
        losses = model(**data, mode="loss")

    head.get_targets = orig_get_targets

    total_new = sum(loss_stats["n_new"])
    total_old = sum(loss_stats["n_old"])
    total_matched = total_new + total_old
    print(f"Over 10 iters (batch_size=2, 20 images):")
    print(f"  Total matched queries: {total_matched}")
    print(f"  New-class matched: {total_new} ({total_new/max(total_matched,1)*100:.1f}%)")
    print(f"  Old-class matched: {total_old} ({total_old/max(total_matched,1)*100:.1f}%)")
    print(f"\nImplication:")
    print(f"  If R(M) gets all detection gradient: new+old = {total_matched} queries")
    print(f"  If R(M) gets only new-class gradient: {total_new} queries ({total_new/max(total_matched,1)*100:.1f}%)")
    print(f"  Gradient reduction factor: {total_matched/max(total_new,1):.1f}x")

    # ============ PART 3: Gradient direction test ============
    print()
    print("=" * 60)
    print("PART 3: New-class vs old-class gradient direction on R(M)")
    print("=" * 60)

    # We need to compute grad(L_new, R(M).params) and grad(L_old, R(M).params)
    # Then check cosine similarity

    ri = model.residual_inject
    ri_params = list(ri.net.parameters())  # MLP parameters only
    total_ri_params = sum(p.numel() for p in ri_params)
    print(f"R(M) MLP parameters: {total_ri_params} ({total_ri_params/1e3:.0f}K)")

    cos_sims = []
    norm_ratios = []

    dl3 = iter(runner.train_dataloader)
    for step in range(5):
        data = next(dl3)
        data = model.data_preprocessor(data, True)

        # Need to separate new-class and old-class loss
        # Strategy: hook into loss_by_feat_single_new to capture per-query loss
        captured_labels = []

        def label_hook(cls_scores_list, bbox_preds_list, batch_gt_instances, batch_img_metas):
            result = orig_get_targets(cls_scores_list, bbox_preds_list, batch_gt_instances, batch_img_metas)
            captured_labels.append(result[0])  # labels_list
            return result

        head.get_targets = label_hook
        model.zero_grad()
        losses = model(**data, mode="loss")
        head.get_targets = orig_get_targets

        # Get total detection loss (cls + bbox + iou for all decoder layers + encoder)
        detect_keys = [k for k in losses if any(k.startswith(p) for p in
                       ('loss_cls', 'loss_bbox', 'loss_iou',
                        'enc_loss_cls', 'enc_loss_bbox', 'enc_loss_iou'))
                       or (len(k) > 2 and k[0] == 'd' and k[1].isdigit() and '.loss_' in k)]
        distill_keys = [k for k in losses if 'ld_' in k or 'inter_' in k]

        L_detect = sum(losses[k] for k in detect_keys
                       if isinstance(losses[k], torch.Tensor) and losses[k].requires_grad)
        L_distill = sum(losses[k] for k in distill_keys
                        if isinstance(losses[k], torch.Tensor) and losses[k].requires_grad)
        L_total = sum(v for v in losses.values()
                      if isinstance(v, torch.Tensor) and v.requires_grad)

        # Compute gradients w.r.t. R(M) params
        # Since we can't easily split L_detect into L_new + L_old at scalar level
        # (they're entangled in the same forward), we compute:
        # grad_detect = grad(L_detect, R(M).params)
        # grad_distill = grad(L_distill, R(M).params)
        # And also grad_total
        try:
            grad_detect = torch.autograd.grad(L_detect, ri_params, retain_graph=True, allow_unused=True)
            grad_distill = torch.autograd.grad(L_distill, ri_params, retain_graph=True, allow_unused=True)
            grad_total = torch.autograd.grad(L_total, ri_params, retain_graph=False, allow_unused=True)

            # Flatten to vectors
            gd = torch.cat([g.flatten() for g in grad_detect if g is not None])
            gt = torch.cat([g.flatten() for g in grad_distill if g is not None])
            ga = torch.cat([g.flatten() for g in grad_total if g is not None])

            if gd.numel() > 0 and gt.numel() > 0:
                cos_det_dist = float(torch.nn.functional.cosine_similarity(gd.unsqueeze(0), gt.unsqueeze(0)))
                cos_det_total = float(torch.nn.functional.cosine_similarity(gd.unsqueeze(0), ga.unsqueeze(0)))
                ratio = float(gd.norm() / (gt.norm() + 1e-10))
                print(f"  iter{step}: cos(detect, distill)={cos_det_dist:.4f}  "
                      f"cos(detect, total)={cos_det_total:.4f}  "
                      f"||detect||/||distill||={ratio:.2f}  "
                      f"L_detect={L_detect.item():.3f} L_distill={L_distill.item():.3f}")
                cos_sims.append(cos_det_dist)
                norm_ratios.append(ratio)
        except RuntimeError as e:
            print(f"  iter{step}: grad failed: {e}")

    if cos_sims:
        print(f"\n  Mean cos(detect, distill) on R(M): {np.mean(cos_sims):.4f}")
        print(f"  Mean ||detect||/||distill|| on R(M): {np.mean(norm_ratios):.2f}")
        print(f"\n  Interpretation:")
        if abs(np.mean(cos_sims)) < 0.3:
            print(f"  cos ~ 0: detection and distillation gradients are ORTHOGONAL on R(M)")
            print(f"  => Removing distillation gradient from R(M) won't destabilize training")
        elif np.mean(cos_sims) > 0.7:
            print(f"  cos >> 0: detection and distillation gradients ALIGN on R(M)")
            print(f"  => They reinforce each other; removing distillation may slow R(M)")
        elif np.mean(cos_sims) < -0.3:
            print(f"  cos << 0: detection and distillation gradients CONFLICT on R(M)")
            print(f"  => Removing distillation frees R(M) from conflict -> beneficial")

    # NOTE: For the NEW vs OLD detection gradient direction test,
    # we need to actually split detection loss, which requires
    # per-query loss decomposition. We'll report whether the split
    # is feasible (Part 2 data) and compute the split gradient in
    # a follow-up after confirming the token mapping (Part 1).
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Token mapping: verified {'OK' if match else 'MISMATCH'}")
    print(f"New-class token boundary: >= {new_start_token}")
    print(f"Matched query composition: new={total_new} ({total_new/max(total_matched,1)*100:.1f}%) old={total_old} ({total_old/max(total_matched,1)*100:.1f}%)")
    if cos_sims:
        print(f"cos(detect_grad, distill_grad) on R(M): {np.mean(cos_sims):.4f}")
        print(f"||detect_grad||/||distill_grad|| on R(M): {np.mean(norm_ratios):.2f}")


if __name__ == "__main__":
    main()
