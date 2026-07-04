"""
GS-GRMI Part 3 FIX: gradient direction test.
Problem: autograd.grad returns None for distillation because decouple config
routes distillation through memory_raw (bypassing R(M)).
Fix: compute grad(L_detect, R(M)) directly, then compare detection-only
gradient direction across different training batches to check stability.

Also: compute grad(L_total, R(M)) to see if distillation gradient reaches R(M) at all.
If grad(L_total, R(M)) == grad(L_detect, R(M)), distillation is fully decoupled.
"""
import mmdet.apis, mmdet.engine.hooks  # noqa
import torch, numpy as np
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


def flat_grad(grads):
    parts = [g.flatten() for g in grads if g is not None]
    if not parts:
        return None
    return torch.cat(parts)


def cos_sim(a, b):
    if a is None or b is None:
        return None
    return float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)))


def main():
    cfg = Config.fromfile("/home/yelingfei/projects/GCD/configs/gdino_inc/70+10/grmi_t1_decouple_6e_coco.py")
    cfg.train_cfg.max_epochs = 1
    cfg.train_dataloader.batch_size = 2
    cfg.work_dir = "/tmp/gsv2_grad"
    cfg.launcher = "none"
    cfg.default_hooks.checkpoint = dict(type="CheckpointHook", interval=999)
    runner = Runner.from_cfg(cfg); runner.load_or_resume()
    model = runner.model.module if hasattr(runner.model, "module") else runner.model
    model.train()

    ri = model.residual_inject
    ri_params = [p for p in ri.net.parameters() if p.requires_grad]
    print(f"R(M) MLP params: {sum(p.numel() for p in ri_params)} total")

    dl = iter(runner.train_dataloader)
    grad_history = []

    for step in range(10):
        data = next(dl)
        data = model.data_preprocessor(data, True)
        model.zero_grad()
        losses = model(**data, mode="loss")

        detect_keys = [k for k in losses if any(k.startswith(p) for p in
                       ('loss_cls', 'loss_bbox', 'loss_iou',
                        'enc_loss_cls', 'enc_loss_bbox', 'enc_loss_iou'))
                       or (len(k) > 2 and k[0] == 'd' and k[1].isdigit() and '.loss_' in k)]
        distill_keys = [k for k in losses if 'ld_' in k or 'inter_' in k]

        L_detect = sum(losses[k] for k in detect_keys
                       if isinstance(losses[k], torch.Tensor) and losses[k].requires_grad)
        L_distill_parts = [losses[k] for k in distill_keys
                           if isinstance(losses[k], torch.Tensor) and losses[k].requires_grad]
        L_distill = sum(L_distill_parts) if L_distill_parts else None
        L_total = sum(v for v in losses.values()
                      if isinstance(v, torch.Tensor) and v.requires_grad)

        # Gradient of detection loss w.r.t. R(M) params
        try:
            gd = torch.autograd.grad(L_detect, ri_params, retain_graph=True,
                                     allow_unused=True, create_graph=False)
            g_detect = flat_grad(gd)
        except Exception as e:
            print(f"  iter{step}: detect grad failed: {e}")
            g_detect = None

        # Gradient of total loss w.r.t. R(M) params
        try:
            gt = torch.autograd.grad(L_total, ri_params, retain_graph=True,
                                     allow_unused=True, create_graph=False)
            g_total = flat_grad(gt)
        except Exception as e:
            print(f"  iter{step}: total grad failed: {e}")
            g_total = None

        # Gradient of distillation loss w.r.t. R(M) params
        g_distill = None
        if L_distill is not None:
            try:
                gdi = torch.autograd.grad(L_distill, ri_params, retain_graph=False,
                                          allow_unused=True, create_graph=False)
                g_distill = flat_grad(gdi)
            except Exception:
                pass

        if g_detect is not None and g_total is not None:
            cos_dt = cos_sim(g_detect, g_total)
            diff_norm = float((g_detect - g_total).norm()) if g_total is not None else None
            detect_norm = float(g_detect.norm())
            total_norm = float(g_total.norm())
            distill_on_rm = "NO" if (diff_norm is not None and diff_norm < 1e-6) else "YES"
            if g_distill is not None:
                distill_norm = float(g_distill.norm())
                distill_on_rm = f"YES (norm={distill_norm:.6f})"
            else:
                distill_norm = 0.0
                distill_on_rm = f"NO (None or not connected)"

            print(f"  iter{step}: cos(detect,total)={cos_dt:.6f}  "
                  f"||detect||={detect_norm:.4f}  ||total||={total_norm:.4f}  "
                  f"||detect-total||={diff_norm:.6f}  "
                  f"distill->R(M)? {distill_on_rm}")

            grad_history.append({
                'g_detect': g_detect.detach().cpu(),
                'cos_dt': cos_dt,
                'detect_norm': detect_norm,
                'total_norm': total_norm,
                'diff_norm': diff_norm,
                'distill_norm': distill_norm,
            })
        else:
            print(f"  iter{step}: grad computation failed")

    # Cross-iter gradient direction stability
    print(f"\n{'='*60}")
    print(f"CROSS-ITER GRADIENT DIRECTION ANALYSIS")
    print(f"{'='*60}")
    if len(grad_history) >= 3:
        cos_pairs = []
        for i in range(len(grad_history)):
            for j in range(i+1, len(grad_history)):
                c = cos_sim(grad_history[i]['g_detect'], grad_history[j]['g_detect'])
                cos_pairs.append(c)
        print(f"cos(detect_grad_i, detect_grad_j) across all iter pairs:")
        print(f"  mean={np.mean(cos_pairs):.4f}  std={np.std(cos_pairs):.4f}  "
              f"min={np.min(cos_pairs):.4f}  max={np.max(cos_pairs):.4f}")

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    if grad_history:
        mean_cos_dt = np.mean([h['cos_dt'] for h in grad_history])
        mean_diff = np.mean([h['diff_norm'] for h in grad_history])
        mean_distill = np.mean([h['distill_norm'] for h in grad_history])
        print(f"cos(detect_grad, total_grad) on R(M): {mean_cos_dt:.6f}")
        print(f"||detect_grad - total_grad||: {mean_diff:.6f}")
        print(f"||distill_grad|| on R(M): {mean_distill:.6f}")
        if mean_diff < 1e-4:
            print(f"=> Distillation gradient does NOT reach R(M) (decouple working)")
            print(f"=> Current gs_ratio mechanism is REDUNDANT with decouple")
            print(f"=> R(M) already receives only detection gradient")
        else:
            print(f"=> Distillation gradient DOES reach R(M) (decouple incomplete)")
            print(f"=> gs_ratio is needed to scale down distillation contribution")


if __name__ == "__main__":
    main()
