"""Accumulated gradient direction test v2: hook head.loss instead of generate_pseudo_label.
Cleaner approach: intercept batch_all_instances right before matching.
"""
import mmdet.apis, mmdet.engine.hooks  # noqa
import torch, numpy as np
from mmengine.config import Config
from mmengine.runner import Runner

def flat_params(params):
    return torch.cat([p.flatten() for p in params])

def main():
    cfg = Config.fromfile("configs/gdino_inc/70+10/grmi_t1_decouple_6e_coco.py")
    cfg.train_cfg.max_epochs = 1; cfg.train_dataloader.batch_size = 2
    cfg.work_dir = "/tmp/accum_grad_v2"; cfg.launcher = "none"
    cfg.default_hooks.checkpoint = dict(type="CheckpointHook", interval=999)
    runner = Runner.from_cfg(cfg); runner.load_or_resume()
    model = runner.model.module if hasattr(runner.model, "module") else runner.model
    model.train()

    ri = model.residual_inject
    ri_params = [p for p in ri.net.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in ri_params)
    print("R(M) params: %d" % n_params)

    head = model.bbox_head
    orig_loss = head.loss
    filter_batch_all = [False]  # mutable flag

    def hooked_loss(new_head_inputs_dict, old_head_inputs_dict,
                    ori_head_inputs_dict, batch_data_samples):
        if filter_batch_all[0]:
            # Replace batch_all_instances with GT-only
            # batch_all_instances is inside ori_head_inputs_dict
            from mmengine.structures import InstanceData
            new_all = []
            for ds in batch_data_samples:
                gt = ds.gt_instances
                all_inst = InstanceData()
                all_inst.bboxes = gt.bboxes
                all_inst.labels = gt.labels
                all_inst.positive_maps = gt.positive_maps
                # text_token_mask needs to match the expected shape
                ttm = new_head_inputs_dict.get('text_token_mask')
                if ttm is not None and ttm.dim() == 2:
                    all_inst.text_token_mask = ttm[:1].repeat(len(gt.labels), 1)
                elif ttm is not None:
                    all_inst.text_token_mask = ttm.repeat(len(gt.labels), 1)
                new_all.append(all_inst)
            ori_head_inputs_dict['batch_all_instances'] = new_all
            ori_head_inputs_dict['batch_pseudo_instances'] = [
                InstanceData(bboxes=p.bboxes[:0], labels=p.labels[:0],
                            positive_maps=p.positive_maps[:0])
                for p in ori_head_inputs_dict.get('batch_pseudo_instances', [])
            ]
        return orig_loss(new_head_inputs_dict, old_head_inputs_dict,
                        ori_head_inputs_dict, batch_data_samples)

    head.loss = hooked_loss

    N_STEPS = 50

    # Pass A: full gradient (new + old pseudo) accumulated
    print("Pass A: Full gradient (new + old pseudo), %d steps..." % N_STEPS)
    filter_batch_all[0] = False
    accum_full = torch.zeros(n_params, device='cuda')
    dl = iter(runner.train_dataloader)
    for step in range(N_STEPS):
        data = next(dl)
        data = model.data_preprocessor(data, True)
        model.zero_grad()
        losses = model(**data, mode="loss")
        L = sum(v for v in losses.values()
                if isinstance(v, torch.Tensor) and v.requires_grad)
        grads = torch.autograd.grad(L, ri_params, retain_graph=False, allow_unused=True)
        g = flat_params([x for x in grads if x is not None])
        accum_full += g
        if step % 20 == 0:
            print("  step %d: ||accum||=%.6f ||g||=%.6f" % (step, accum_full.norm().item(), g.norm().item()))
    E_full = accum_full / N_STEPS
    print("  Done. ||E[g_full]|| = %.6f" % E_full.norm().item())

    # Pass B: new-only gradient (suppress pseudo-labels from batch_all_instances)
    print("Pass B: New-only gradient (remove pseudo), %d steps..." % N_STEPS)
    filter_batch_all[0] = True
    accum_new = torch.zeros(n_params, device='cuda')
    dl2 = iter(runner.train_dataloader)
    n_errors = 0
    for step in range(N_STEPS):
        data = next(dl2)
        data = model.data_preprocessor(data, True)
        model.zero_grad()
        try:
            losses = model(**data, mode="loss")
            L = sum(v for v in losses.values()
                    if isinstance(v, torch.Tensor) and v.requires_grad)
            grads = torch.autograd.grad(L, ri_params, retain_graph=False, allow_unused=True)
            g = flat_params([x for x in grads if x is not None])
            if g is None or g.norm().item() == 0:
                n_errors += 1
                continue
            accum_new += g
        except Exception as e:
            n_errors += 1
            if n_errors <= 3:
                print("  step %d: ERROR %s" % (step, str(e)[:120]))
            continue
        if step % 20 == 0:
            print("  step %d: ||accum||=%.6f ||g||=%.6f" % (step, accum_new.norm().item(), g.norm().item()))
    E_new = accum_new / N_STEPS
    print("  Done. ||E[g_new]|| = %.6f  errors=%d" % (E_new.norm().item(), n_errors))

    head.loss = orig_loss

    # Compute E[g_old] by subtraction
    E_old = E_full - E_new

    # Compare directions
    cos_fn = torch.nn.functional.cosine_similarity

    cos_full_new = float(cos_fn(E_full.unsqueeze(0), E_new.unsqueeze(0)))
    cos_new_old = float(cos_fn(E_new.unsqueeze(0), E_old.unsqueeze(0)))
    cos_full_old = float(cos_fn(E_full.unsqueeze(0), E_old.unsqueeze(0)))

    print()
    print("=" * 60)
    print("ACCUMULATED GRADIENT DIRECTION (50 steps each)")
    print("=" * 60)
    print("||E[g_full]|| = %.6f" % E_full.norm().item())
    print("||E[g_new]||  = %.6f" % E_new.norm().item())
    print("||E[g_old]||  = %.6f (by subtraction)" % E_old.norm().item())
    print()
    print("cos(E[g_new], E[g_full]) = %.4f" % cos_full_new)
    print("cos(E[g_new], E[g_old])  = %.4f" % cos_new_old)
    print("cos(E[g_old], E[g_full]) = %.4f" % cos_full_old)
    print()
    norm_ratio = E_new.norm().item() / max(E_old.norm().item(), 1e-10)
    print("Norm ratio: ||E[g_new]|| / ||E[g_old]|| = %.2f" % norm_ratio)
    print()

    m = cos_new_old
    if m > 0.7:
        print("CONCLUSION: E[g_new] and E[g_old] ALIGN (cos=%.2f)" % m)
        print("  => Removing old gradient won't change R(M) direction much")
        print("  => Splitting has LIMITED value")
    elif m > 0.3:
        print("CONCLUSION: E[g_new] and E[g_old] PARTIALLY aligned (cos=%.2f)" % m)
        print("  => Splitting has MODERATE value")
    elif m > -0.1:
        print("CONCLUSION: E[g_new] and E[g_old] ORTHOGONAL (cos=%.2f)" % m)
        print("  => Old gradient is noise relative to new-class signal")
        print("  => Splitting IS valuable")
    else:
        print("CONCLUSION: E[g_new] and E[g_old] CONFLICT (cos=%.2f)" % m)
        print("  => Old gradient ACTIVELY FIGHTS new-class optimization")
        print("  => Splitting is CRITICAL")

    # Also compute per-step gradient stability for Pass A
    print()
    print("=" * 60)
    print("PER-STEP GRADIENT DIRECTION STABILITY (Pass A)")
    print("=" * 60)
    filter_batch_all[0] = False
    dl3 = iter(runner.train_dataloader)
    per_step_grads = []
    for step in range(20):
        data = next(dl3)
        data = model.data_preprocessor(data, True)
        model.zero_grad()
        losses = model(**data, mode="loss")
        L = sum(v for v in losses.values()
                if isinstance(v, torch.Tensor) and v.requires_grad)
        grads = torch.autograd.grad(L, ri_params, retain_graph=False, allow_unused=True)
        g = flat_params([x for x in grads if x is not None])
        per_step_grads.append(g.detach().cpu())

    cos_pairs = []
    for i in range(len(per_step_grads)):
        for j in range(i+1, len(per_step_grads)):
            c = float(cos_fn(per_step_grads[i].unsqueeze(0), per_step_grads[j].unsqueeze(0)))
            cos_pairs.append(c)
    print("Per-step cos(g_i, g_j):  mean=%.4f std=%.4f" % (np.mean(cos_pairs), np.std(cos_pairs)))
    # Compare per-step with accumulated
    ref = per_step_grads[0]
    cos_to_ref = [float(cos_fn(g.unsqueeze(0), ref.unsqueeze(0))) for g in per_step_grads[1:]]
    print("Per-step cos(g_i, g_0):  mean=%.4f" % np.mean(cos_to_ref))

    head.loss = orig_loss
    print()
    print("DONE.")


if __name__ == "__main__":
    main()
