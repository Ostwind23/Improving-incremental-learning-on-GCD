#!/usr/bin/env python3
"""Patch: add freeze_gamma + JSONL monitor to GRMI."""
import py_compile

# === 1. ResidualInject: add freeze_gamma ===
RI = '/home/yelingfei/projects/GCD/mmdet/models/utils/residual_inject.py'
src = open(RI, encoding='utf-8').read()
if 'freeze_gamma' not in src:
    src = src.replace(
        '                 act_inference: bool = True):',
        '                 act_inference: bool = True,\n'
        '                 freeze_gamma: bool = False):')
    src = src.replace(
        '        self.act_inference = bool(act_inference)',
        '        self.act_inference = bool(act_inference)\n'
        '        self.freeze_gamma = bool(freeze_gamma)')
    src = src.replace(
        '        return memory + self.gamma * residual',
        '        g = self.gamma.detach() if self.freeze_gamma else self.gamma\n'
        '        return memory + g * residual')
    open(RI, 'w', encoding='utf-8').write(src)
    print('RI: freeze_gamma added')
else:
    print('RI: already patched')
py_compile.compile(RI, doraise=True)
print('RI: SYNTAX OK')

# === 2. Detector: pass freeze_gamma from config + add monitor ===
DET = '/home/yelingfei/projects/GCD/mmdet/models/detectors/gdino_inc_gcd.py'
src2 = open(DET, encoding='utf-8').read()
changed = False

# 2a. Pass freeze_gamma
old_ri_build = "act_inference=bool(self.residual_inject_cfg.get('act_inference', True)))"
new_ri_build = ("act_inference=bool(self.residual_inject_cfg.get('act_inference', True)),\n"
                "                freeze_gamma=bool(self.residual_inject_cfg.get('freeze_gamma', False)))")
if 'freeze_gamma' not in src2 and old_ri_build in src2:
    src2 = src2.replace(old_ri_build, new_ri_build, 1)
    changed = True
    print('DET: freeze_gamma passthrough added')

    # Update print
    old_print = 'f"act_inference={_ri.act_inference}")'
    new_print = 'f"act_inference={_ri.act_inference} freeze_gamma={_ri.freeze_gamma}")'
    if old_print in src2:
        src2 = src2.replace(old_print, new_print, 1)

# 2b. Add monitor init after GRMI print
if '_grmi_mon_path' not in src2:
    # Find the GRMI print line end
    grmi_marker = 'freeze_gamma={_ri.freeze_gamma}")'
    if grmi_marker in src2:
        idx = src2.index(grmi_marker) + len(grmi_marker)
        monitor_init = ("\n"
            "            _grmi_mon = self.residual_inject_cfg.get('monitor', {}) or {}\n"
            "            self._grmi_mon_path = str(_grmi_mon.get('path', ''))\n"
            "            self._grmi_mon_interval = int(_grmi_mon.get('interval', 250))\n"
            "            self._grmi_mon_step = 0\n")
        src2 = src2[:idx] + monitor_init + src2[idx:]
        changed = True
        print('DET: monitor init added')

# 2c. Add monitor method before loss()
if '_grmi_monitor_log' not in src2:
    loss_def = '    def loss('
    if loss_def in src2:
        idx = src2.index(loss_def)
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
            '        rec = {"step": self._grmi_mon_step, "epoch": epoch, "iter": it,\n'
            '               "gamma": round(float(ri.gamma.detach().item()), 6),\n'
            '               "freeze_gamma": ri.freeze_gamma}\n'
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
        src2 = src2[:idx] + method + src2[idx:]
        changed = True
        print('DET: monitor method added')

    # Wire into loss() before return
    lines = src2.split('\n')
    loss_start = None
    for i, line in enumerate(lines):
        if '    def loss(' in line and 'def loss_by' not in line:
            loss_start = i
    if loss_start is not None:
        # Find next top-level def after loss()
        next_def = None
        for i in range(loss_start + 1, len(lines)):
            if lines[i].startswith('    def ') and 'def loss(' not in lines[i]:
                next_def = i
                break
        # Find last "return losses" in this method
        ret_line = None
        for i in range(next_def - 1 if next_def else len(lines) - 1, loss_start, -1):
            if 'return losses' in lines[i] and 'return losses_' not in lines[i]:
                ret_line = i
                break
        if ret_line is not None and '_grmi_monitor_log' not in '\n'.join(lines[loss_start:ret_line+1]):
            indent = len(lines[ret_line]) - len(lines[ret_line].lstrip())
            sp = ' ' * indent
            wire = (f'{sp}if hasattr(self, "_grmi_mon_path") and self._grmi_mon_path:\n'
                    f'{sp}    self._grmi_monitor_log(losses)')
            lines.insert(ret_line, wire)
            src2 = '\n'.join(lines)
            changed = True
            print('DET: monitor wired at line', ret_line)

if changed:
    open(DET, 'w', encoding='utf-8').write(src2)
py_compile.compile(DET, doraise=True)
print('DET: SYNTAX OK')
