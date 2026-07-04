#!/usr/bin/env python3
"""
GS-GRMI (Gradient-Selective GRMI) patcher.
Minimal, precise patches to two files.

Mechanism:
  In forward: residual.register_hook(lambda g: g * ratio)
  ratio = L_detect / L_total (computed after loss, stored before backward)
  This scales R(M)'s gradient so distillation loss component is removed.

Also adds R(M) norm suppression: L_supp = λ * mean(||R(M)||²)
"""
import py_compile, sys

RI = '/home/yelingfei/projects/GCD/mmdet/models/utils/residual_inject.py'
DET = '/home/yelingfei/projects/GCD/mmdet/models/detectors/gdino_inc_gcd.py'


def patch_ri():
    """Add gradient hook and residual caching to ResidualInject.forward()."""
    code = open(RI, encoding='utf-8').read()

    OLD = """        residual = self.net(memory)
        # Cache norms for monitoring (no grad, cheap)
        with torch.no_grad():
            self._cached_mem_norm = float(memory.detach().norm(dim=-1).mean())
            self._cached_rm_norm = float(residual.detach().norm(dim=-1).mean())
        return memory + self.gamma * residual"""

    NEW = """        residual = self.net(memory)
        # Cache norms for monitoring (no grad, cheap)
        with torch.no_grad():
            self._cached_mem_norm = float(memory.detach().norm(dim=-1).mean())
            self._cached_rm_norm = float(residual.detach().norm(dim=-1).mean())
        # GS-GRMI: backward hook scales gradient by ratio (set externally)
        if self.training and getattr(self, '_gs_ratio', None) is not None:
            r = self._gs_ratio
            residual = residual * 1.0  # create leaf for hook
            residual.register_hook(lambda g, _r=r: g * _r)
        # Cache for norm suppression loss
        if self.training:
            self._cached_residual = residual
        return memory + self.gamma * residual"""

    if OLD not in code:
        print(f'[FAIL] forward pattern not found in {RI}')
        return False
    code = code.replace(OLD, NEW)
    open(RI, 'w', encoding='utf-8').write(code)
    py_compile.compile(RI, doraise=True)
    print(f'[OK] {RI}')
    return True


def patch_det():
    """Add GS ratio computation and suppression loss to detector.loss()."""
    code = open(DET, encoding='utf-8').read()

    # PATCH 1: Add gs_supp_weight after R(M) init
    OLD_INIT = """            self._grmi_prev_perturb = 0.0"""
    NEW_INIT = """            self._grmi_prev_perturb = 0.0
            self._gs_supp_weight = float(self.residual_inject_cfg.get('gs_supp_weight', 0.001))
            self._gs_ratio_for_log = 0.0
            self._gs_grmi_enabled = bool(self.residual_inject_cfg.get('gs_grmi', False))"""

    if OLD_INIT not in code:
        print('[FAIL] _grmi_prev_perturb init not found')
        return False
    code = code.replace(OLD_INIT, NEW_INIT, 1)

    # PATCH 2: After bbox_head.loss(), compute ratio and supp loss
    # Find: "if hasattr(self, \"_grmi_mon_path\") and self._grmi_mon_path:"
    OLD_MON = '        if hasattr(self, "_grmi_mon_path") and self._grmi_mon_path:'
    GS_BLOCK = """        # === GS-GRMI: gradient scaling ratio + suppression loss ===
        if self.residual_inject is not None and getattr(self, '_gs_grmi_enabled', False):
            detect_keys = [k for k in losses if any(k.startswith(p) for p in
                           ('loss_cls', 'loss_bbox', 'loss_iou',
                            'enc_loss_cls', 'enc_loss_bbox', 'enc_loss_iou'))
                           or (len(k) > 2 and k[0] == 'd' and k[1].isdigit() and '.loss_' in k)]
            L_detect = sum(losses[k] for k in detect_keys if k in losses
                           and isinstance(losses[k], torch.Tensor) and losses[k].requires_grad)
            L_total = sum(v for v in losses.values()
                          if isinstance(v, torch.Tensor) and v.requires_grad)
            with torch.no_grad():
                if isinstance(L_total, torch.Tensor) and L_total.abs() > 1e-8:
                    ratio = (L_detect / L_total).clamp(0.0, 1.0).item()
                else:
                    ratio = 0.0
                self.residual_inject._gs_ratio = ratio
                self._gs_ratio_for_log = ratio
            # R(M) norm suppression
            if hasattr(self.residual_inject, '_cached_residual'):
                rm = self.residual_inject._cached_residual
                losses['loss_rm_supp'] = self._gs_supp_weight * (rm ** 2).mean()

"""

    if OLD_MON not in code:
        print('[FAIL] _grmi_mon_path check not found')
        return False
    code = code.replace(OLD_MON, GS_BLOCK + '        ' + OLD_MON.lstrip(), 1)

    # PATCH 3: Add gs_ratio to JSONL monitor record
    # Find the line with 'perturb_pct' in _grmi_monitor_log
    if "'perturb_pct'" in code:
        # Add gs_ratio field to the record
        code = code.replace(
            "'perturb_pct': perturb_pct,",
            "'perturb_pct': perturb_pct,\n"
            "                'gs_ratio': getattr(self, '_gs_ratio_for_log', 0.0),",
            1)

    open(DET, 'w', encoding='utf-8').write(code)
    py_compile.compile(DET, doraise=True)
    print(f'[OK] {DET}')
    return True


if __name__ == '__main__':
    ok1 = patch_ri()
    ok2 = patch_det()
    if ok1 and ok2:
        print('\n=== GS-GRMI patches applied ===')
    else:
        print('\n=== SOME PATCHES FAILED ===')
        sys.exit(1)
