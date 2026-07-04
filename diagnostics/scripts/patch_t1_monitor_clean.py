#!/usr/bin/env python3
"""
Clean patch: R(M) norm monitoring + freeze_gamma + T1 capacity support.
Applied to clean detector file (no D2 sel_grmi patch).
"""
import py_compile

# === 1. Update ResidualInject to cache norm in forward ===
RI = '/home/yelingfei/projects/GCD/mmdet/models/utils/residual_inject.py'
src_ri = open(RI, encoding='utf-8').read()

old_fwd = '''    def forward(self, memory: Tensor) -> Tensor:
        """memory: (bs, N_pos, C) -> (bs, N_pos, C)."""
        # residual is computed from and added to memory in-place-style;
        # returns a new tensor, autograd handles backward.
        residual = self.net(memory)
        return memory + self.gamma * residual'''

new_fwd = '''    def forward(self, memory: Tensor) -> Tensor:
        """memory: (bs, N_pos, C) -> (bs, N_pos, C)."""
        residual = self.net(memory)
        # Cache norms for monitoring (no grad, cheap)
        with torch.no_grad():
            self._cached_mem_norm = float(memory.detach().norm(dim=-1).mean())
            self._cached_rm_norm = float(residual.detach().norm(dim=-1).mean())
        return memory + self.gamma * residual'''

if old_fwd in src_ri:
    src_ri = src_ri.replace(old_fwd, new_fwd, 1)
    print('RI: norm caching added to forward')
elif '_cached_rm_norm' in src_ri:
    print('RI: already has norm caching')
else:
    print('RI: WARNING forward pattern not found')

open(RI, 'w', encoding='utf-8').write(src_ri)
py_compile.compile(RI, doraise=True)
print('RI: SYNTAX OK')

# === 2. Patch detector: add freeze_gamma + enhanced monitor ===
DET = '/home/yelingfei/projects/GCD/mmdet/models/detectors/gdino_inc_gcd.py'
src = open(DET, encoding='utf-8').read()

# 2a. Pass freeze_gamma from config
old_build = "act_inference=bool(self.residual_inject_cfg.get('act_inference', True)))"
new_build = ("act_inference=bool(self.residual_inject_cfg.get('act_inference', True)),\n"
             "                freeze_gamma=bool(self.residual_inject_cfg.get('freeze_gamma', False)))")
if 'freeze_gamma' not in src and old_build in src:
    src = src.replace(old_build, new_build, 1)
    print('DET: freeze_gamma passthrough added')

    old_print = 'f"act_inference={_ri.act_inference}")'
    new_print = 'f"act_inference={_ri.act_inference} freeze_gamma={_ri.freeze_gamma}")'
    if old_print in src:
        src = src.replace(old_print, new_print, 1)

# 2b. Add monitor init after GRMI print
if '_grmi_mon_path' not in src:
    grmi_marker = 'freeze_gamma={_ri.freeze_gamma}")'
    if grmi_marker in src:
        idx = src.index(grmi_marker) + len(grmi_marker)
        monitor_init = ("\n"
            "            _grmi_mon = self.residual_inject_cfg.get('monitor', {}) or {}\n"
            "            self._grmi_mon_path = str(_grmi_mon.get('path', ''))\n"
            "            self._grmi_mon_interval = int(_grmi_mon.get('interval', 250))\n"
            "            self._grmi_mon_step = 0\n"
            "            self._grmi_prev_perturb = 0.0\n")
        src = src[:idx] + monitor_init + src[idx:]
        print('DET: monitor init added')

# 2c. Add enhanced monitor method before loss()
if '_grmi_monitor_log' not in src:
    loss_def = '    def loss('
    if loss_def in src:
        idx = src.index(loss_def)
        method = (
            '    def _grmi_monitor_log(self, losses):\n'
            '        import json as _jg\n'
            '        if not hasattr(self, "_grmi_mon_path") or not self._grmi_mon_path:\n'
            '            return\n'
            '        self._grmi_mon_step += 1\n'
            '        if self._grmi_mon_step % self._grmi_mon_interval != 0:\n'
            '            return\n'
            '        try:\n'
            '            from mmengine.logging import MessageHub\n'
            '            hub = MessageHub.get_current_instance()\n'
            '            epoch = hub.get_info("epoch"); it = hub.get_info("iter")\n'
            '        except Exception:\n'
            '            epoch = -1; it = -1\n'
            '        ri = self.residual_inject\n'
            '        _gamma = float(ri.gamma.detach().item())\n'
            '        _rm_n = getattr(ri, "_cached_rm_norm", 0.0)\n'
            '        _mem_n = getattr(ri, "_cached_mem_norm", 1.0)\n'
            '        _ratio = _rm_n / max(_mem_n, 1e-8)\n'
            '        _perturb = _gamma * _ratio\n'
            '        _prev = getattr(self, "_grmi_prev_perturb", _perturb)\n'
            '        _delta = abs(_perturb - _prev) / max(abs(_prev), 1e-8) if abs(_prev) > 1e-10 else 0.0\n'
            '        self._grmi_prev_perturb = _perturb\n'
            '        rec = {"step": self._grmi_mon_step, "epoch": epoch, "iter": it,\n'
            '               "gamma": round(_gamma, 6),\n'
            '               "freeze_gamma": ri.freeze_gamma,\n'
            '               "rm_norm": round(_rm_n, 4),\n'
            '               "mem_norm": round(_mem_n, 4),\n'
            '               "rm_ratio": round(_ratio, 6),\n'
            '               "perturb_pct": round(_perturb * 100, 6),\n'
            '               "delta_perturb": round(_delta, 6),\n'
            '               "hidden_dim": ri.net[0].out_features,\n'
            '               "n_params": sum(p.numel() for p in ri.parameters())}\n'
            '        for k, v in losses.items():\n'
            '            if hasattr(v, "item"):\n'
            '                rec[k] = round(float(v.detach().item()), 6)\n'
            '        try:\n'
            '            import os as _osg\n'
            '            _osg.makedirs(_osg.path.dirname(self._grmi_mon_path), exist_ok=True)\n'
            '            with open(self._grmi_mon_path, "a") as f:\n'
            '                f.write(_jg.dumps(rec) + chr(10))\n'
            '        except Exception:\n'
            '            pass\n\n'
        )
        src = src[:idx] + method + src[idx:]
        print('DET: enhanced monitor method added')

    # Wire into loss() before return
    lines = src.split('\n')
    loss_start = None
    for i, line in enumerate(lines):
        if '    def loss(' in line and 'def loss_by' not in line:
            loss_start = i
            break
    if loss_start is not None:
        next_def = None
        for i in range(loss_start + 1, len(lines)):
            if lines[i].startswith('    def ') and 'def loss(' not in lines[i]:
                next_def = i
                break
        ret_line = None
        for i in range(next_def - 1 if next_def else len(lines) - 1, loss_start, -1):
            if 'return losses' in lines[i] and 'return losses_' not in lines[i]:
                ret_line = i
                break
        if ret_line and '_grmi_monitor_log' not in '\n'.join(lines[loss_start:ret_line+1]):
            indent = len(lines[ret_line]) - len(lines[ret_line].lstrip())
            sp = ' ' * indent
            wire = (f'{sp}if hasattr(self, "_grmi_mon_path") and self._grmi_mon_path:\n'
                    f'{sp}    self._grmi_monitor_log(losses)')
            lines.insert(ret_line, wire)
            src = '\n'.join(lines)
            print('DET: monitor wired at line', ret_line)

open(DET, 'w', encoding='utf-8').write(src)
py_compile.compile(DET, doraise=True)
print('DET: SYNTAX OK')
