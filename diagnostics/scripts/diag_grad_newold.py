"""New vs Old class detection gradient direction test on R(M).
Two-pass approach: for each batch, run forward twice:
  Pass A: only new-class GT -> get grad(L_new_detect, R(M))
  Pass B: only old-class GT -> get grad(L_old_detect, R(M))
Compare cos(g_new, g_old) to see if splitting matters.
"""
import mmdet.apis, mmdet.engine.hooks  # noqa
import torch, numpy as np, copy
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

def filter_gt(data, keep_new):
    """Deep-copy data and keep only new-class (>=70) or old-class (<70) GT."""
    d = copy.deepcopy(data)
    samples = d['data_samples'] if isinstance(d['data_samples'], list) else [d['data_samples']]
    for ds in samples:
        gt = ds.gt_instances
        if gt is None or len(gt.labels) == 0:
            continue
        if keep_new:
            mask = gt.labels >= 70
        else:
            mask = gt.labels < 70
        if mask.sum() == 0:
            gt.bboxes = gt.bboxes[:0]
            gt.labels = gt.labels[:0]
            if hasattr(gt, 'positive_maps'):
                gt.positive_maps = gt.positive_maps[:0]
            if hasattr(gt, 'text_token_mask'):
                gt.text_token_mask = gt.text_token_mask[:0]
        else:
            gt.bboxes = gt.bboxes[mask]
            gt.labels = gt.labels[mask]
            if hasattr(gt, 'positive_maps'):
                gt.positive_maps = gt.positive_maps[mask]
            if hasattr(gt, 'text_token_mask'):
                gt.text_token_mask = gt.text_token_mask[mask]
    return d


def flat_grad(grads):
    parts = [g.flatten() for g in grads if g is not None]
    return torch.cat(parts) if parts else None


def main():
    cfg = Config.fromfile("configs/gdino_inc/70+10/grmi_t1_decouple_6e_coco.py")
    cfg.train_cfg.max_epochs = 1; cfg.train_dataloader.batch_size = 2
    cfg.work_dir = "/tmp/grad_newold"; cfg.launcher = "none"
    cfg.default_hooks.checkpoint = dict(type="CheckpointHook", interval=999)
    runner = Runner.from_cfg(cfg); runner.load_or_resume()
    model = runner.model.module if hasattr(runner.model, "module") else runner.model
    model.train()

    ri = model.residual_inject
    ri_params = [p for p in ri.net.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in ri_params)
    print(f"R(M) MLP: {n_params} params")

    dl = iter(runner.train_dataloader)
    results_new_old = []
    results_new_total = []

    for step in range(10):
        data = next(dl)
        data = model.data_preprocessor(data, True)

        # Check if batch has both new and old class GT
        has_new = False; has_old = False
        samples = data['data_samples'] if isinstance(data['data_samples'], list) else [data['data_samples']]
        for ds in samples:
            gt = ds.gt_instances
            if gt is not None and len(gt.labels) > 0:
                if (gt.labels >= 70).any():
                    has_new = True
                if (gt.labels < 70).any():
                    has_old = True
        if not (has_new and has_old):
            print(f"  iter{step}: skip (has_new={has_new} has_old={has_old})")
            continue

        # Pass FULL: all GT
        model.zero_grad()
        losses_full = model(**data, mode="loss")
        L_full = sum(v for v in losses_full.values()
                     if isinstance(v, torch.Tensor) and v.requires_grad)
        gf = torch.autograd.grad(L_full, ri_params, retain_graph=False, allow_unused=True)
        g_full = flat_grad(gf)

        # Pass NEW: only new-class GT
        data_new = filter_gt(data, keep_new=True)
        model.zero_grad()
        try:
            losses_new = model(**data_new, mode="loss")
            L_new = sum(v for v in losses_new.values()
                        if isinstance(v, torch.Tensor) and v.requires_grad)
            gn = torch.autograd.grad(L_new, ri_params, retain_graph=False, allow_unused=True)
            g_new = flat_grad(gn)
        except Exception as e:
            print(f"  iter{step}: new-only forward failed: {e}")
            continue

        # Pass OLD: only old-class GT
        data_old = filter_gt(data, keep_new=False)
        model.zero_grad()
        try:
            losses_old = model(**data_old, mode="loss")
            L_old = sum(v for v in losses_old.values()
                        if isinstance(v, torch.Tensor) and v.requires_grad)
            go = torch.autograd.grad(L_old, ri_params, retain_graph=False, allow_unused=True)
            g_old = flat_grad(go)
        except Exception as e:
            print(f"  iter{step}: old-only forward failed: {e}")
            continue

        if g_new is None or g_old is None or g_full is None:
            print(f"  iter{step}: gradient None")
            continue

        cos_no = float(torch.nn.functional.cosine_similarity(
            g_new.unsqueeze(0), g_old.unsqueeze(0)))
        cos_nf = float(torch.nn.functional.cosine_similarity(
            g_new.unsqueeze(0), g_full.unsqueeze(0)))
        results_new_old.append(cos_no)
        results_new_total.append(cos_nf)
        print(f"  iter{step}: cos(new,old)={cos_no:.4f}  cos(new,full)={cos_nf:.4f}  "
              f"||new||={g_new.norm():.4f} ||old||={g_old.norm():.4f} ||full||={g_full.norm():.4f}")

    print()
    print("=" * 60)
    print("NEW vs OLD DETECTION GRADIENT ON R(M)")
    print("=" * 60)
    if results_new_old:
        print(f"cos(g_new, g_old):  mean={np.mean(results_new_old):.4f} "
              f"std={np.std(results_new_old):.4f}")
        print(f"cos(g_new, g_full): mean={np.mean(results_new_total):.4f} "
              f"std={np.std(results_new_total):.4f}")
        m = np.mean(results_new_old)
        print()
        if m > 0.7:
            print("CONCLUSION: g_new and g_old ALIGN (cos > 0.7)")
            print("  => Splitting has LIMITED value.")
            print("  => Both new/old detection push R(M) the same way.")
            print("  => Keeping old-class gradient doesn't hurt new-class direction.")
        elif m > 0.3:
            print("CONCLUSION: g_new and g_old PARTIALLY aligned (0.3 < cos < 0.7)")
            print("  => Splitting has MODERATE value.")
            print("  => Removing old gradient will shift R(M) direction somewhat.")
        elif m > -0.3:
            print("CONCLUSION: g_new and g_old are NEAR-ORTHOGONAL (cos ~ 0)")
            print("  => Splitting IS valuable.")
            print("  => Old gradient is noise relative to new-class signal.")
            print("  => Removing it improves signal-to-noise ratio on R(M).")
        else:
            print("CONCLUSION: g_new and g_old CONFLICT (cos < -0.3)")
            print("  => Splitting is CRITICAL.")
            print("  => Old gradient actively fights new-class learning on R(M).")
    else:
        print("No valid gradient pairs collected.")


if __name__ == "__main__":
    main()
