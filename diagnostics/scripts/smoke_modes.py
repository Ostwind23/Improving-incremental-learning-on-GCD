"""Smoke test for each R(M) mode."""
import mmdet.apis, mmdet.engine.hooks, torch
from mmengine.config import Config
from mmengine.runner import Runner

MODES = ['mlp', 'tc_mlp', 'crossattn']

for mode in MODES:
    print(f"\n=== Testing {mode} ===")
    cfg = Config.fromfile("configs/gdino_inc/70+10/gdino_inc_70+10_70-79_gcd_scratch_coco.py")
    cfg.train_cfg.max_epochs = 1; cfg.train_dataloader.batch_size = 1
    cfg.work_dir = f"/tmp/smoke_{mode}"; cfg.launcher = "none"
    cfg.default_hooks.checkpoint = dict(type="CheckpointHook", interval=999)
    cfg.model.residual_inject_cfg = dict(
        enable=True, mode=mode, gamma_init=0.5, freeze_gamma=False,
        bilinear_bottleneck=64, hidden_dim=128)

    try:
        runner = Runner.from_cfg(cfg); runner.load_or_resume()
        model = runner.model.module if hasattr(runner.model, "module") else runner.model
        ri = model.residual_inject
        print(f"  Mode={ri.mode} gamma={ri.gamma.item():.4f} params={sum(p.numel() for p in ri.parameters())}")
        dl = iter(runner.val_dataloader); batch = next(dl)
        batch = model.data_preprocessor(batch, False)
        with torch.no_grad():
            _ = model(**batch, mode="loss")
        print(f"  OK: rm_norm={ri._cached_rm_norm:.3f} mem_norm={ri._cached_mem_norm:.3f}")
        del model, runner; torch.cuda.empty_cache()
    except Exception as e:
        print(f"  CRASH: {str(e)[:200]}")
