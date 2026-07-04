#!/usr/bin/env python3
"""Apply D2 selective GRMI patch directly via line manipulation."""
import datetime, sys
TARGET = "/home/yelingfei/projects/GCD/mmdet/models/detectors/gdino_inc_gcd.py"
src = open(TARGET, encoding='utf-8').read()
backup = TARGET + '.bak_d2_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
open(backup, 'w', encoding='utf-8').write(src)
print(f"backup -> {backup}")
lines = src.split('\n')
changes = 0

# 1. Add param to __init__
for i, line in enumerate(lines):
    if 'residual_inject_cfg: OptConfigType = None,' in line and 'gt_selective' not in lines[i]:
        lines[i] = line + '\n                 gt_selective_grmi_cfg: OptConfigType = None,'
        changes += 1; print(f'[1] init param at line {i+1}'); break

# 2. Add config parsing before TATRI section
for i, line in enumerate(lines):
    if '# --- TATRI: text-side gated residual injection ---' in line:
        block = [
            '',
            '        # --- GT-Guided Selective GRMI ---',
            '        from mmengine.config import Config as _SelCfg2',
            '        _sel_cfg = _SelCfg2._dict_to_config_dict_lazy(',
            '            gt_selective_grmi_cfg or dict(enable=False))',
            '        self._sel_grmi_enable = bool(_sel_cfg.get("enable", False))',
            '        self._sel_grmi_alpha = float(_sel_cfg.get("alpha", 1.0))',
            '        self._sel_grmi_beta = float(_sel_cfg.get("beta", 0.1))',
            '        self._sel_grmi_ns = int(_sel_cfg.get("ns", 70))',
            '        self._sel_grmi_ne = int(_sel_cfg.get("ne", 80))',
            '        self._sel_grmi_gamma_boost = float(_sel_cfg.get("gamma_boost", 1.0))',
            '        self._sel_grmi_monitor_path = str(_sel_cfg.get("monitor_path", ""))',
            '        self._sel_grmi_monitor_interval = int(_sel_cfg.get("monitor_interval", 250))',
            '        self._sel_grmi_step = 0',
            '        self._sel_grmi_batch_gt = None',
            '        if self._sel_grmi_enable:',
            '            print("[SelGRMI] enabled: alpha=%s beta=%s boost=%s" % (',
            '                self._sel_grmi_alpha, self._sel_grmi_beta, self._sel_grmi_gamma_boost))',
            '',
        ]
        for j, bl in enumerate(reversed(block)):
            lines.insert(i, bl)
        changes += 1; print(f'[2] config parsing before line {i+1}'); break

# 3. Replace GRMI injection
for i, line in enumerate(lines):
    if "encoder_outputs_dict['memory'] = self.residual_inject(memory)" in line:
        indent = len(line) - len(line.lstrip())
        sp = ' ' * indent
        lines[i] = (sp + "encoder_outputs_dict['memory'] = self._sel_grmi_apply(\n"
                    + sp + "    self.residual_inject, memory, encoder_outputs_dict.get('spatial_shapes'))")
        changes += 1; print(f'[3] replaced GRMI injection at line {i+1}'); break

# 4. Add methods before loss()
method_lines = """
    def _sel_grmi_apply(self, residual_inject, memory, spatial_shapes):
        import json as _json
        import torch
        residual = residual_inject.transform(memory)
        gamma = residual_inject.gamma
        if not self.training or not self._sel_grmi_enable or self._sel_grmi_batch_gt is None:
            return memory + gamma * residual
        B, N, C = memory.shape
        weight = torch.full((B, N, 1), self._sel_grmi_beta, device=memory.device, dtype=memory.dtype)
        if spatial_shapes is None:
            return memory + gamma * self._sel_grmi_gamma_boost * residual
        ssl = spatial_shapes.cpu().long().tolist()
        li = max(range(len(ssl)), key=lambda k: ssl[k][0] * ssl[k][1])
        H0, W0 = ssl[li]
        off = sum(ssl[k][0] * ssl[k][1] for k in range(li))
        n_boosted = 0
        for bi in range(min(B, len(self._sel_grmi_batch_gt))):
            gt_inst = self._sel_grmi_batch_gt[bi]
            gt_labels = gt_inst.labels
            gt_bboxes = gt_inst.bboxes
            if hasattr(gt_bboxes, 'tensor'): gt_bboxes = gt_bboxes.tensor
            ns, ne = self._sel_grmi_ns, self._sel_grmi_ne
            new_mask = (gt_labels >= ns) & (gt_labels < ne)
            if not new_mask.any(): continue
            meta = getattr(gt_inst, '_sel_meta', {})
            ih, iw = meta.get('img_shape', (800, 800))
            for j in range(len(gt_labels)):
                if not (ns <= int(gt_labels[j]) < ne): continue
                bx = gt_bboxes[j]
                gx1 = int(max(0, min(W0-1, bx[0].item()/max(float(iw), 1)*W0)))
                gx2 = int(max(1, min(W0, bx[2].item()/max(float(iw), 1)*W0)))
                gy1 = int(max(0, min(H0-1, bx[1].item()/max(float(ih), 1)*H0)))
                gy2 = int(max(1, min(H0, bx[3].item()/max(float(ih), 1)*H0)))
                for y in range(gy1, gy2):
                    for x in range(gx1, gx2):
                        idx = off + y * W0 + x
                        if idx < N:
                            weight[bi, idx, 0] = self._sel_grmi_alpha
                            n_boosted += 1
        out = memory + gamma * self._sel_grmi_gamma_boost * residual * weight
        self._sel_grmi_step += 1
        if self._sel_grmi_monitor_path and self._sel_grmi_step % self._sel_grmi_monitor_interval == 0:
            try:
                from mmengine.logging import MessageHub
                hub = MessageHub.get_current_instance()
                epoch = hub.get_info('epoch'); it = hub.get_info('iter')
            except Exception:
                epoch = -1; it = -1
            rec = {'step': self._sel_grmi_step, 'epoch': epoch, 'iter': it,
                   'gamma': round(float(gamma.detach().item()), 6),
                   'n_boosted': n_boosted, 'n_total': B * N,
                   'residual_norm': round(float(residual.detach().norm(dim=-1).mean().item()), 4)}
            try:
                with open(self._sel_grmi_monitor_path, 'a') as f:
                    f.write(_json.dumps(rec) + chr(10))
            except Exception: pass
        return out

    def _sel_grmi_stash_gt(self, batch_data_samples):
        if not self._sel_grmi_enable: return
        gt_list = []
        for ds in batch_data_samples:
            gt = ds.gt_instances
            gt._sel_meta = {'img_shape': ds.metainfo.get('img_shape', (800, 800))}
            gt_list.append(gt)
        self._sel_grmi_batch_gt = gt_list
""".split('\n')

for i, line in enumerate(lines):
    if line.strip().startswith('def loss(') and '    def loss(' in line:
        for j, ml in enumerate(reversed(method_lines)):
            lines.insert(i, ml)
        changes += 1; print(f'[4] methods before line {i+1}'); break

# 5. Wire GT stash before forward_transformer
for i, line in enumerate(lines):
    if 'new_head_inputs_dict, old_head_inputs_dict = self.forward_transformer' in line:
        indent = len(line) - len(line.lstrip())
        lines.insert(i, ' ' * indent + 'self._sel_grmi_stash_gt(batch_data_samples)')
        changes += 1; print(f'[5] stash wired at line {i+1}'); break

open(TARGET, 'w', encoding='utf-8').write('\n'.join(lines))
print(f'\nTotal: {changes} changes')

import py_compile
py_compile.compile(TARGET, doraise=True)
print('SYNTAX OK')
