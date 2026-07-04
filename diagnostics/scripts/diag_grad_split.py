"""Test: cos(grad_new_detect, grad_old_detect) on R(M) params.
Uses separate forwards with new-only GT vs all GT (including pseudo-labels).
The DIFFERENCE = old-class pseudo-label gradient contribution.
"""
import mmdet.apis, mmdet.engine.hooks  # noqa
import torch, numpy as np, copy
from mmengine.config import Config
from mmengine.runner import Runner

def flat_grad(grads):
    parts = [g.flatten() for g in grads if g is not None]
    return torch.cat(parts) if parts else None

def cos_sim(a, b):
    if a is None or b is None: return None
    return float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)))

def main():
    cfg = Config.fromfile("configs/gdino_inc/70+10/grmi_t1_decouple_6e_coco.py")
    cfg.train_cfg.max_epochs = 1; cfg.train_dataloader.batch_size = 2
    cfg.work_dir = "/tmp/grad_split"; cfg.launcher = "none"
    cfg.default_hooks.checkpoint = dict(type="CheckpointHook", interval=999)
    runner = Runner.from_cfg(cfg); runner.load_or_resume()
    model = runner.model.module if hasattr(runner.model, "module") else runner.model
    model.train()

    ri = model.residual_inject
    ri_params = [p for p in ri.net.parameters() if p.requires_grad]

    # We can't easily separate new vs old detection in a single forward because
    # batch_all_instances merges them before matching. Instead:
    # Pass FULL: normal forward -> grad_full(R(M)) = grad_new + grad_old_pseudo + grad_distill
    # But distillation is decoupled (memory_raw), so grad_full = grad_detect = grad_new + grad_old_pseudo
    #
    # To isolate grad_new: modify batch_all_instances to only contain GT (new-class)
    # by removing pseudo-labels before matching.
    #
    # Strategy: hook generate_pseudo_label to return EMPTY pseudo-labels.
    head = model.bbox_head
    orig_gen_pseudo = head.generate_pseudo_label
    empty_mode = [False]

    def hook_gen_pseudo(*args, **kwargs):
        result = orig_gen_pseudo(*args, **kwargs)
        if empty_mode[0]:
            topk_query, batch_pseudo, batch_all = result
            # Replace batch_all with GT only (remove pseudo-labels)
            from mmengine.structures import InstanceData
            batch_data_samples = args[5] if len(args) > 5 else kwargs.get('batch_data_samples')
            new_batch_all = []
            for ds in batch_data_samples:
                gt = ds.gt_instances
                # GT instances need to have all fields that batch_all expects
                all_inst = InstanceData()
                all_inst.bboxes = gt.bboxes
                all_inst.labels = gt.labels
                all_inst.positive_maps = gt.positive_maps
                ttm = head.text_masks if hasattr(head, 'text_masks') else gt.text_token_mask
                if ttm.dim() == 2:
                    all_inst.text_token_mask = ttm[:1].repeat(len(gt.labels), 1)
                else:
                    all_inst.text_token_mask = ttm.repeat(len(gt.labels), 1)
                new_batch_all.append(all_inst)
            # Also empty pseudo-labels
            empty_pseudo = []
            for p in batch_pseudo:
                ep = InstanceData()
                ep.bboxes = p.bboxes[:0]
                ep.labels = p.labels[:0]
                ep.positive_maps = p.positive_maps[:0]
                ep.text_token_mask = p.text_token_mask[:0]
                empty_pseudo.append(ep)
            return topk_query, empty_pseudo, new_batch_all
        return result
    head.generate_pseudo_label = hook_gen_pseudo

    dl = iter(runner.train_dataloader)
    results = []

    for step in range(10):
        data_raw = next(dl)
        data_raw_copy = copy.deepcopy(data_raw)
        data_full = model.data_preprocessor(data_raw, True)

        # Pass FULL: normal (new GT + old pseudo-labels)
        empty_mode[0] = False
        model.zero_grad()
        losses_full = model(**data_full, mode="loss")
        L_full = sum(v for v in losses_full.values()
                     if isinstance(v, torch.Tensor) and v.requires_grad)
        gf = torch.autograd.grad(L_full, ri_params, retain_graph=False, allow_unused=True)
        g_full = flat_grad(gf)
        if g_full is None:
            print("iter%d: grad_full=None" % step)
            continue

        # Pass NEW-ONLY: remove pseudo-labels -> matching only on new-class GT
        data_new = model.data_preprocessor(data_raw_copy, True)
        empty_mode[0] = True
        model.zero_grad()
        try:
            losses_new = model(**data_new, mode="loss")
            L_new = sum(v for v in losses_new.values()
                        if isinstance(v, torch.Tensor) and v.requires_grad)
            gn = torch.autograd.grad(L_new, ri_params, retain_graph=False, allow_unused=True)
            g_new = flat_grad(gn)
        except Exception as e:
            print("iter%d: new-only forward failed: %s" % (step, str(e)[:100]))
            continue

        if g_new is None:
            print("iter%d: grad_new=None" % step)
            continue

        # g_old_pseudo = g_full - g_new (approximately, ignoring interaction effects)
        g_old_approx = g_full - g_new
        cos_no = cos_sim(g_new, g_old_approx)
        cos_nf = cos_sim(g_new, g_full)
        norm_new = float(g_new.norm())
        norm_full = float(g_full.norm())
        norm_old = float(g_old_approx.norm())
        results.append({
            'cos_new_old': cos_no, 'cos_new_full': cos_nf,
            'norm_new': norm_new, 'norm_full': norm_full, 'norm_old': norm_old
        })
        print("iter%d: cos(new,old)=%.4f cos(new,full)=%.4f ||new||=%.4f ||old||=%.4f ||full||=%.4f" % (
            step, cos_no if cos_no else 0, cos_nf if cos_nf else 0, norm_new, norm_old, norm_full))

    head.generate_pseudo_label = orig_gen_pseudo

    print("\n" + "=" * 60)
    print("NEW vs OLD-PSEUDO DETECTION GRADIENT ON R(M)")
    print("=" * 60)
    if results:
        cos_no_vals = [r['cos_new_old'] for r in results if r['cos_new_old'] is not None]
        cos_nf_vals = [r['cos_new_full'] for r in results if r['cos_new_full'] is not None]
        norm_new_vals = [r['norm_new'] for r in results]
        norm_old_vals = [r['norm_old'] for r in results]
        print("cos(g_new, g_old_pseudo):  mean=%.4f std=%.4f" % (np.mean(cos_no_vals), np.std(cos_no_vals)))
        print("cos(g_new, g_full):        mean=%.4f std=%.4f" % (np.mean(cos_nf_vals), np.std(cos_nf_vals)))
        print("||g_new||:                 mean=%.4f" % np.mean(norm_new_vals))
        print("||g_old_pseudo||:          mean=%.4f" % np.mean(norm_old_vals))
        print("||g_new|| / ||g_old_pseudo||: %.2f" % (np.mean(norm_new_vals)/max(np.mean(norm_old_vals),1e-10)))
        m = np.mean(cos_no_vals)
        print()
        if m > 0.5:
            print("CONCLUSION: New and old-pseudo gradients ALIGN (cos=%.2f)" % m)
            print("  => Removing old-pseudo gradient would REDUCE R(M) gradient magnitude")
            print("  => but not change DIRECTION much -> limited benefit from splitting")
        elif m > -0.1:
            print("CONCLUSION: New and old-pseudo gradients are ORTHOGONAL (cos=%.2f)" % m)
            print("  => Old-pseudo gradient is noise on R(M)")
            print("  => Removing it improves signal-to-noise ratio -> splitting IS valuable")
        else:
            print("CONCLUSION: New and old-pseudo gradients CONFLICT (cos=%.2f)" % m)
            print("  => Old-pseudo gradient ACTIVELY FIGHTS new-class learning on R(M)")
            print("  => Splitting is CRITICAL for improving new_ap")
    else:
        print("No valid results")


if __name__ == "__main__":
    main()
