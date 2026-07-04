#!/usr/bin/env python3
"""
Patch: Add R(M) norm monitoring to GRMI + upgrade to T1 capacity tier.
Changes:
1. In forward_encoder: cache memory and R(M) output for monitor
2. In _grmi_monitor_log: add rm_norm, mem_norm, perturb_ratio, delta_perturb
3. Config uses hidden_dim=256 (T1 tier)
"""
import py_compile

DET = '/home/yelingfei/projects/GCD/mmdet/models/detectors/gdino_inc_gcd.py'
src = open(DET, encoding='utf-8').read()

# === 1. Cache R(M) output in forward_encoder ===
# Find the _sel_grmi_apply call (current GRMI injection point)
old_inject = "encoder_outputs_dict['memory'] = self._sel_grmi_apply("
if old_inject in src and '_grmi_cached_rm' not in src:
    # We need to cache both memory (before R(M)) and the R(M) output
    # Replace the injection with a version that caches
    new_inject = """# Cache for R(M) norm monitoring
                if hasattr(self, 'residual_inject') and self.residual_inject is not None:
                    _mem_pre = encoder_outputs_dict['memory'].detach()
                    _rm_out = self.residual_inject.net(_mem_pre)
                    self._grmi_cached_rm = {
                        'mem_norm': float(_mem_pre.norm(dim=-1).mean()),
                        'rm_norm': float(_rm_out.norm(dim=-1).mean()),
                    }
                encoder_outputs_dict['memory'] = self._sel_grmi_apply("""
    src = src.replace(old_inject, new_inject, 1)
    print('DET: R(M) norm caching added in forward_encoder')
else:
    if '_grmi_cached_rm' in src:
        print('DET: R(M) caching already present')
    else:
        print('DET: WARNING - injection point not found')

# === 2. Upgrade _grmi_monitor_log to include norm metrics ===
old_monitor_rec = ('        rec = {"step": self._grmi_mon_step, "epoch": epoch, "iter": it,\n'
                   '               "gamma": round(float(ri.gamma.detach().item()), 6),\n'
                   '               "freeze_gamma": ri.freeze_gamma}')

new_monitor_rec = ('        # R(M) norm metrics\n'
                   '        _cached = getattr(self, "_grmi_cached_rm", {})\n'
                   '        _mem_n = _cached.get("mem_norm", 0)\n'
                   '        _rm_n = _cached.get("rm_norm", 0)\n'
                   '        _gamma_val = float(ri.gamma.detach().item())\n'
                   '        _perturb = _gamma_val * _rm_n / max(_mem_n, 1e-8)\n'
                   '        # Track delta_perturb for convergence detection\n'
                   '        _prev = getattr(self, "_grmi_prev_perturb", _perturb)\n'
                   '        _delta = abs(_perturb - _prev) / max(abs(_prev), 1e-8)\n'
                   '        self._grmi_prev_perturb = _perturb\n'
                   '        rec = {"step": self._grmi_mon_step, "epoch": epoch, "iter": it,\n'
                   '               "gamma": round(_gamma_val, 6),\n'
                   '               "freeze_gamma": ri.freeze_gamma,\n'
                   '               "rm_norm": round(_rm_n, 4),\n'
                   '               "mem_norm": round(_mem_n, 4),\n'
                   '               "rm_ratio": round(_rm_n / max(_mem_n, 1e-8), 6),\n'
                   '               "perturb_pct": round(_perturb * 100, 6),\n'
                   '               "delta_perturb": round(_delta, 6),\n'
                   '               "hidden_dim": ri.net[0].out_features,\n'
                   '               "n_params": sum(p.numel() for p in ri.parameters())}')

if old_monitor_rec in src:
    src = src.replace(old_monitor_rec, new_monitor_rec, 1)
    print('DET: monitor upgraded with R(M) norm metrics')
else:
    print('DET: WARNING - monitor rec pattern not found, trying flexible match')
    # Try line by line
    if '"freeze_gamma": ri.freeze_gamma}' in src and 'rm_norm' not in src:
        src = src.replace(
            '"freeze_gamma": ri.freeze_gamma}',
            '"freeze_gamma": ri.freeze_gamma,\n'
            '               "rm_norm": 0, "mem_norm": 0, "perturb_pct": 0,\n'
            '               "delta_perturb": 0, "hidden_dim": ri.net[0].out_features,\n'
            '               "n_params": sum(p.numel() for p in ri.parameters())}',
            1)
        print('DET: fallback monitor upgrade applied')

open(DET, 'w', encoding='utf-8').write(src)
py_compile.compile(DET, doraise=True)
print('DET: SYNTAX OK')
