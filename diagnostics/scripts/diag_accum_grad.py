"""Accumulated gradient direction test.
Instead of comparing single-step gradients (noise-dominated),
accumulate gradients over 50 steps to approximate E[grad],
then compare accumulated new-class vs old-class gradient directions.

Method: hook generate_pseudo_label to optionally suppress pseudo-labels.
Run 50 steps normally (full gradient), accumulate on R(M).
Run 50 steps with empty pseudo-labels (new-only gradient), accumulate.
Compare accumulated directions.
"""
import mmdet.apis, mmdet.engine.hooks  # noqa
import torch, numpy as np, copy
from mmengine.config import Config
from mmengine.runner import Runner

def flat_params(params):
    return torch.cat([p.flatten() for p in params])

def main():
    cfg = Config.fromfile("configs/gdino_inc/70+10/grmi_t1_decouple_6e_coco.py")
    cfg.train_cfg.max_epochs = 1; cfg.train_dataloader.batch_size = 2
    cfg.work_dir = "/tmp/accum_grad"; cfg.launcher = "none"
    cfg.default_hooks.checkpoint = dict(type="CheckpointHook", interval=999)
    runner = Runner.from_cfg(cfg); runner.load_or_resume()
    model = runner.model.module if hasattr(runner.model, "module") else runner.model
    model.train()

    ri = model.residual_inject
    ri_params = [p for p in ri.net.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in ri_params)

    head = model.bbox_head
    orig_gen_pseudo = head.generate_pseudo_label
    suppress_pseudo = [False]

    def hook_gen_pseudo(*args, **kwargs):
        result = orig_gen_pseudo(*args, **kwargs)
        if suppress_pseudo[0]:
            topk_query, batch_pseudo, batch_all = result
            from mmengine.structures import InstanceData
            # Replace batch_all with GT only
            batch_data_samples = args[5] if len(args) > 5 else kwargs.get('batch_data_samples')
            new_batch_all = []
            for gt_inst, pseudo_inst in zip(
                [ds.gt_instances for ds in batch_data_samples], batch_pseudo):
                all_inst = InstanceData()
                all_inst.bboxes = gt_inst.bboxes
                all_inst.labels = gt_inst.labels
                if hasattr(gt_inst, 'positive_maps'):
                    all_inst.positive_maps = gt_inst.positive_maps
                if hasattr(gt_inst, 'text_token_mask'):
                    all_inst.text_token_mask = gt_inst.text_token_mask
                new_batch_all.append(all_inst)
            empty_pseudo = []
            for p in batch_pseudo:
                ep = InstanceData()
                for k in ['bboxes', 'labels', 'positive_maps', 'text_token_mask']:
                    if hasattr(p, k):
                        setattr(ep, k, getattr(p, k)[:0])
                empty_pseudo.append(ep)
            return topk_query, empty_pseudo, new_batch_all
        return result
    head.generate_pseudo_label = hook_gen_pseudo

    N_STEPS = 50

    # Pass A: full gradient (new + old pseudo) accumulated
    print("Pass A: Full gradient (new + old pseudo), %d steps..." % N_STEPS)
    suppress_pseudo[0] = False
    accum_full = torch.zeros(n_params, device='cuda')
    dl_a = iter(runner.train_dataloader)
    for step in range(N_STEPS):
        data = next(dl_a)
        data = model.data_preprocessor(data, True)
        model.zero_grad()
        losses = model(**data, mode="loss")
        L = sum(v for v in losses.values()
                if isinstance(v, torch.Tensor) and v.requires_grad)
        grads = torch.autograd.grad(L, ri_params, retain_graph=False, allow_unused=True)
        g = torch.cat([x.flatten() for x in grads if x is not None])
        accum_full += g
        if step % 20 == 0:
            print("  step %d: ||accum||=%.4f" % (step, accum_full.norm().item()))
    accum_full_norm = accum_full / N_STEPS

    # Pass B: new-only gradient (suppress pseudo-labels) accumulated
    print("Pass B: New-only gradient (suppress pseudo), %d steps..." % N_STEPS)
    suppress_pseudo[0] = True
    accum_new = torch.zeros(n_params, device='cuda')
    dl_b = iter(runner.train_dataloader)
    for step in range(N_STEPS):
        data = next(dl_b)
        data = model.data_preprocessor(data, True)
        model.zero_grad()
        try:
            losses = model(**data, mode="loss")
            L = sum(v for v in losses.values()
                    if isinstance(v, torch.Tensor) and v.requires_grad)
            grads = torch.autograd.grad(L, ri_params, retain_graph=False, allow_unused=True)
            g = torch.cat([x.flatten() for x in grads if x is not None])
            accum_new += g
        except Exception as e:
            print("  step %d: failed (%s)" % (step, str(e)[:80]))
            continue
        if step % 20 == 0:
            print("  step %d: ||accum||=%.4f" % (step, accum_new.norm().item()))
    accum_new_norm = accum_new / N_STEPS

    head.generate_pseudo_label = orig_gen_pseudo

    # Compute old-pseudo accumulated gradient by subtraction
    accum_old = accum_full - accum_new
    accum_old_norm = accum_old / N_STEPS

    # Compare directions
    cos_fn = torch.nn.functional.cosine_similarity

    cos_new_full = float(cos_fn(accum_new.unsqueeze(0), accum_full.unsqueeze(0)))
    cos_new_old = float(cos_fn(accum_new.unsqueeze(0), accum_old.unsqueeze(0)))
    cos_old_full = float(cos_fn(accum_old.unsqueeze(0), accum_full.unsqueeze(0)))

    print("\n" + "=" * 60)
    print("ACCUMULATED GRADIENT DIRECTION (50 steps each)")
    print("=" * 60)
    print("||E[g_full]||  = %.6f" % accum_full_norm.norm().item())
    print("||E[g_new]||   = %.6f" % accum_new_norm.norm().item())
    print("||E[g_old]||   = %.6f (by subtraction)" % accum_old_norm.norm().item())
    print()
    print("cos(E[g_new], E[g_full]) = %.4f" % cos_new_full)
    print("cos(E[g_new], E[g_old])  = %.4f" % cos_new_old)
    print("cos(E[g_old], E[g_full]) = %.4f" % cos_old_full)
    print()
    print("Norm ratio: ||E[g_new]|| / ||E[g_old]|| = %.2f" % (
        accum_new_norm.norm().item() / max(accum_old_norm.norm().item(), 1e-10)))
    print()

    m = cos_new_old
    if m > 0.7:
        print("CONCLUSION: E[g_new] and E[g_old] ALIGN (cos=%.2f)" % m)
        print("  => Expected gradient directions are similar")
        print("  => Removing old gradient won't change R(M)'s learning direction much")
        print("  => Splitting has LIMITED value for direction, only reduces magnitude")
    elif m > 0.3:
        print("CONCLUSION: E[g_new] and E[g_old] PARTIALLY aligned (cos=%.2f)" % m)
        print("  => Removing old gradient SHIFTS R(M) direction moderately")
        print("  => Splitting has MODERATE value")
    elif m > -0.1:
        print("CONCLUSION: E[g_new] and E[g_old] ORTHOGONAL (cos=%.2f)" % m)
        print("  => Old gradient is noise relative to new-class signal")
        print("  => Removing it DOUBLES signal-to-noise ratio on R(M)")
        print("  => Splitting IS valuable")
    else:
        print("CONCLUSION: E[g_new] and E[g_old] CONFLICT (cos=%.2f)" % m)
        print("  => Old gradient ACTIVELY FIGHTS new-class optimization")
        print("  => Splitting is CRITICAL for improvement")
        print("  => Potential gain: removing a NEGATIVE contribution frees R(M)")


if __name__ == "__main__":
    main()
