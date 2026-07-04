"""Gradient split v2: single forward, hook loss_by_feat_single to separate
new-class vs old-class loss components, then autograd.grad on each."""
import mmdet.apis, mmdet.engine.hooks  # noqa
import torch, numpy as np
from mmengine.config import Config
from mmengine.runner import Runner

NEW_TOKEN_START = 169

def flat_grad(grads):
    parts = [g.flatten() for g in grads if g is not None]
    return torch.cat(parts) if parts else None

def cos_sim(a, b):
    if a is None or b is None: return None
    return float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)))

def main():
    cfg = Config.fromfile("configs/gdino_inc/70+10/grmi_t1_decouple_6e_coco.py")
    cfg.train_cfg.max_epochs = 1; cfg.train_dataloader.batch_size = 2
    cfg.work_dir = "/tmp/gradsplit2"; cfg.launcher = "none"
    cfg.default_hooks.checkpoint = dict(type="CheckpointHook", interval=999)
    runner = Runner.from_cfg(cfg); runner.load_or_resume()
    model = runner.model.module if hasattr(runner.model, "module") else runner.model
    model.train()

    ri = model.residual_inject
    ri_params = [p for p in ri.net.parameters() if p.requires_grad]
    head = model.bbox_head

    # Hook loss_by_feat_single to capture per-query loss before reduction
    orig_loss_cls = head.loss_cls
    # We need focal loss with reduction='none'
    # But FocalLoss doesn't cleanly support per-element return through its interface
    # Alternative: hook get_targets to separate new/old, then compute two scalar losses

    # Strategy: capture labels from get_targets (last decoder layer),
    # then after loss, scale the SCALAR loss by new-fraction / old-fraction
    # and compute separate gradients.
    # This is an approximation but directionally correct.

    captured_fracs = {"new_frac": 0.5, "old_frac": 0.5, "n_new": 0, "n_old": 0}
    orig_gt = head.get_targets
    call_count = [0]

    def hook_gt(*a, **kw):
        r = orig_gt(*a, **kw)
        call_count[0] += 1
        # Only use the LAST call (last decoder layer L5 = call #7)
        labels_list = r[0]
        n_new = 0; n_old = 0
        for lab in labels_list:
            matched = lab.sum(-1) > 0
            for qi in range(lab.shape[0]):
                if not matched[qi]: continue
                hot = lab[qi].nonzero(as_tuple=True)[0]
                if (hot >= NEW_TOKEN_START).any():
                    n_new += 1
                else:
                    n_old += 1
        captured_fracs["n_new"] = n_new
        captured_fracs["n_old"] = n_old
        total = max(n_new + n_old, 1)
        captured_fracs["new_frac"] = n_new / total
        captured_fracs["old_frac"] = n_old / total
        return r
    head.get_targets = hook_gt

    dl = iter(runner.train_dataloader)
    results = []

    for step in range(10):
        data = next(dl)
        data = model.data_preprocessor(data, True)
        call_count[0] = 0
        model.zero_grad()
        losses = model(**data, mode="loss")

        # Get detection loss keys
        detect_keys = [k for k in losses if any(k.startswith(p) for p in
                       ('loss_cls', 'loss_bbox', 'loss_iou',
                        'enc_loss_cls', 'enc_loss_bbox', 'enc_loss_iou'))
                       or (len(k) > 2 and k[0] == 'd' and k[1].isdigit() and '.loss_' in k)]

        L_detect = sum(losses[k] for k in detect_keys
                       if isinstance(losses[k], torch.Tensor) and losses[k].requires_grad)

        # Full gradient on R(M)
        g_full = flat_grad(torch.autograd.grad(L_detect, ri_params,
                                               retain_graph=True, allow_unused=True))
        if g_full is None:
            print("iter%d: no gradient" % step)
            continue

        # Approximate: L_detect = f_new * L_detect + f_old * L_detect
        # where f_new + f_old = 1
        # grad(f_new * L_detect) = f_new * grad(L_detect) = f_new * g_full
        # This is an approximation (assumes loss scales linearly with fraction)
        f_new = captured_fracs["new_frac"]
        f_old = captured_fracs["old_frac"]
        g_new_approx = f_new * g_full
        g_old_approx = f_old * g_full

        # This approximation is trivially cos=1.0 because g_new = c * g_full
        # We need a BETTER approach.

        # Actually, the issue is we can't separate the loss at scalar level.
        # The only way is two forwards or per-query loss decomposition.
        # Let me try a different approach: compute grad of JUST the cls loss
        # for the last layer, where we CAN identify new vs old queries from labels.

        # Get last-layer cls loss specifically
        loss_cls_last = losses.get('loss_cls')
        if loss_cls_last is None or not loss_cls_last.requires_grad:
            print("iter%d: loss_cls not available" % step)
            continue

        # grad of just cls loss on R(M)
        g_cls = flat_grad(torch.autograd.grad(loss_cls_last, ri_params,
                                              retain_graph=True, allow_unused=True))

        # We know from get_targets that n_new / n_old is the composition
        # But we still can't separate them from a single scalar.
        # The fundamental problem: focal_loss(scores, labels) is a single scalar
        # that mixes new and old matched queries.

        # CONCLUSIVE TEST: compare gradient ACROSS batches with different compositions
        # Batch with more new-class GT should have different gradient direction
        # than batch with more old-class GT.
        norm_full = float(g_full.norm())
        n_new = captured_fracs["n_new"]
        n_old = captured_fracs["n_old"]
        results.append({
            'g_full': g_full.detach().cpu(),
            'norm': norm_full,
            'n_new': n_new, 'n_old': n_old,
            'frac_new': f_new,
        })
        print("iter%d: n_new=%d n_old=%d frac_new=%.2f ||g||=%.4f" % (
            step, n_new, n_old, f_new, norm_full))

    head.get_targets = orig_gt

    # Analyze: do batches with high new-fraction have different gradient direction
    # than batches with low new-fraction?
    print("\n" + "=" * 60)
    print("GRADIENT DIRECTION vs NEW-CLASS FRACTION")
    print("=" * 60)
    if len(results) >= 4:
        # Sort by new-class fraction
        results.sort(key=lambda r: r['frac_new'])
        low_new = results[:len(results)//3]
        high_new = results[-len(results)//3:]

        cos_low_low = []
        for i in range(len(low_new)):
            for j in range(i+1, len(low_new)):
                cos_low_low.append(cos_sim(low_new[i]['g_full'], low_new[j]['g_full']))

        cos_high_high = []
        for i in range(len(high_new)):
            for j in range(i+1, len(high_new)):
                cos_high_high.append(cos_sim(high_new[i]['g_full'], high_new[j]['g_full']))

        cos_low_high = []
        for i in range(len(low_new)):
            for j in range(len(high_new)):
                cos_low_high.append(cos_sim(low_new[i]['g_full'], high_new[j]['g_full']))

        print("Low new-frac group:  fracs=%s" % [round(r['frac_new'],2) for r in low_new])
        print("High new-frac group: fracs=%s" % [round(r['frac_new'],2) for r in high_new])
        cos_ll = [c for c in cos_low_low if c is not None]
        cos_hh = [c for c in cos_high_high if c is not None]
        cos_lh = [c for c in cos_low_high if c is not None]
        if cos_ll: print("cos within low-new:    %.4f" % np.mean(cos_ll))
        if cos_hh: print("cos within high-new:   %.4f" % np.mean(cos_hh))
        if cos_lh: print("cos between low/high:  %.4f" % np.mean(cos_lh))

        print()
        if cos_lh and cos_ll and cos_hh:
            within = (np.mean(cos_ll) + np.mean(cos_hh)) / 2
            between = np.mean(cos_lh)
            if between < within - 0.05:
                print("CONCLUSION: Different new/old composition => different gradient direction")
                print("  within-group cos: %.4f" % within)
                print("  between-group cos: %.4f" % between)
                print("  => Splitting IS valuable (old-class gradient shifts direction)")
            else:
                print("CONCLUSION: Gradient direction STABLE regardless of new/old composition")
                print("  within-group cos: %.4f" % within)
                print("  between-group cos: %.4f" % between)
                print("  => Splitting has LIMITED value (direction doesn't depend on composition)")

        # Also: correlation between frac_new and gradient direction
        # Project all gradients onto first gradient as reference
        ref = results[0]['g_full']
        projs = [cos_sim(r['g_full'], ref) for r in results]
        fracs = [r['frac_new'] for r in results]
        if all(p is not None for p in projs):
            corr = float(np.corrcoef(fracs, projs)[0, 1])
            print("\nCorrelation(frac_new, cos_to_ref): %.4f" % corr)
            if abs(corr) > 0.5:
                print("  => Strong correlation: gradient direction depends on new/old mix")
            else:
                print("  => Weak correlation: gradient direction is batch-noise-dominated")


if __name__ == "__main__":
    main()
