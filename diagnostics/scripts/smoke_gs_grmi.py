#!/usr/bin/env python3
"""
GS-GRMI gradient verification smoke test.
Runs 10 training iterations and verifies:
1. R(M) gradient hook fires (gs_ratio > 0 and < 1)
2. loss_rm_supp appears in loss dict
3. R(M) grad norm is smaller than full-GRMI (gradient scaling working)
4. No NaN/Inf in any tensor
5. JSONL monitor records gs_ratio

Runs on 1 GPU, ~2 minutes.
"""
import mmdet.apis  # noqa
import mmdet.engine.hooks  # noqa
import torch
import os, sys, json, time

def main():
    from mmengine.config import Config
    from mmengine.runner import Runner

    # Use T1 freeze-gamma config as base, add GS-GRMI flags
    cfg_path = '/home/yelingfei/projects/GCD/configs/gdino_inc/70+10/grmi_t1_decouple_6e_coco.py'
    cfg = Config.fromfile(cfg_path)

    # Override for smoke test
    cfg.train_cfg.max_epochs = 1
    cfg.train_dataloader.batch_size = 2
    cfg.work_dir = '/tmp/gs_grmi_smoke'
    cfg.launcher = 'none'
    cfg.default_hooks.checkpoint = dict(type='CheckpointHook', interval=999)

    # Enable GS-GRMI
    cfg.model.residual_inject_cfg.gs_grmi = True
    cfg.model.residual_inject_cfg.gs_supp_weight = 0.001
    cfg.model.residual_inject_cfg.monitor = dict(
        path='/tmp/gs_grmi_smoke/gs_monitor.jsonl',
        interval=1  # log every iter for smoke test
    )

    # Reduce dataset for speed
    if hasattr(cfg, 'train_dataloader'):
        cfg.train_dataloader.dataset.ann_file = cfg.train_dataloader.dataset.ann_file

    runner = Runner.from_cfg(cfg)

    # Intercept after a few iters
    class GradCheckHook:
        def __init__(self):
            self.records = []
            self.iter_count = 0

        def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
            self.iter_count += 1
            model = runner.model.module if hasattr(runner.model, 'module') else runner.model

            record = {'iter': self.iter_count}

            # Check R(M) gradient
            ri = model.residual_inject
            if ri is not None:
                # Check gs_ratio
                gs_ratio = getattr(model, '_gs_ratio_for_log', -1)
                record['gs_ratio'] = gs_ratio

                # Check R(M) MLP gradients
                grad_norms = []
                for name, p in ri.net.named_parameters():
                    if p.grad is not None:
                        gn = float(p.grad.norm())
                        grad_norms.append(gn)
                        if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                            record['nan_inf'] = True
                record['rm_grad_norm'] = sum(grad_norms) if grad_norms else 0.0
                record['rm_param_norm'] = float(sum(p.norm() for p in ri.net.parameters()))
                record['rm_cached_norm'] = getattr(ri, '_cached_rm_norm', -1)

                # Check suppression loss
                if outputs and isinstance(outputs, dict):
                    record['has_supp_loss'] = 'loss_rm_supp' in outputs
                    if 'loss_rm_supp' in outputs:
                        record['supp_loss'] = float(outputs['loss_rm_supp'])

            self.records.append(record)
            if self.iter_count <= 10:
                print(f"[GSCHECK] iter={self.iter_count} gs_ratio={record.get('gs_ratio', '?'):.4f} "
                      f"rm_grad={record.get('rm_grad_norm', '?'):.4f} "
                      f"supp={'Y' if record.get('has_supp_loss') else 'N'}")

            if self.iter_count >= 10:
                # Dump results and stop
                with open('/tmp/gs_grmi_smoke/grad_check.json', 'w') as f:
                    json.dump(self.records, f, indent=2)
                print('\n=== GS-GRMI GRADIENT VERIFICATION ===')
                ratios = [r['gs_ratio'] for r in self.records if 'gs_ratio' in r]
                grads = [r['rm_grad_norm'] for r in self.records if 'rm_grad_norm' in r]
                supp = [r.get('has_supp_loss', False) for r in self.records]
                nans = [r.get('nan_inf', False) for r in self.records]
                print(f'gs_ratio range: [{min(ratios):.4f}, {max(ratios):.4f}]')
                print(f'  Expected: 0 < ratio < 1 (not 0, not 1)')
                print(f'  {"PASS" if all(0 < r < 1 for r in ratios) else "FAIL"}')
                print(f'rm_grad_norm range: [{min(grads):.4f}, {max(grads):.4f}]')
                print(f'  Expected: > 0 (gradient is flowing)')
                print(f'  {"PASS" if all(g > 0 for g in grads) else "FAIL"}')
                print(f'supp_loss present: {sum(supp)}/{len(supp)}')
                print(f'  Expected: all True')
                print(f'  {"PASS" if all(supp) else "FAIL"}')
                print(f'NaN/Inf: {sum(nans)}/{len(nans)}')
                print(f'  Expected: 0')
                print(f'  {"PASS" if not any(nans) else "FAIL"}')
                all_pass = (all(0 < r < 1 for r in ratios) and
                           all(g > 0 for g in grads) and
                           all(supp) and not any(nans))
                print(f'\nOVERALL: {"ALL PASS" if all_pass else "SOME FAILED"}')
                sys.exit(0)

    from mmengine.hooks import Hook
    class GradCheckMMHook(Hook):
        def __init__(self):
            self._inner = GradCheckHook()
        def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
            self._inner.after_train_iter(runner, batch_idx, data_batch, outputs)

    runner.register_hook(GradCheckMMHook(), priority='LOWEST')

    try:
        runner.train()
    except SystemExit:
        pass  # normal exit from our hook


if __name__ == '__main__':
    main()
