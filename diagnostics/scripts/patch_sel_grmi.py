#!/usr/bin/env python3
"""
Patch: GT-Guided Selective GRMI (D2).

During training only, apply R(M) residual with higher weight at positions
near new-class GT boxes and lower weight at old-class/background positions.

This addresses the R2 risk finding: R(M) is uniform (new/old ratio=0.9989),
so increasing γ globally perturbs old-class features equally.
GT-guided selection directs the gradient signal specifically to new-class
regions without disturbing old-class features.

Changes to gdino_inc_gcd.py:
  1. In forward_encoder (after GRMI residual), if training and gt_selective_grmi
     is enabled, build a per-position weight map from new-class GT boxes and
     multiply the residual by this weight before adding to memory.

The selective weight is: w(pos) = α if pos is inside any new-class GT box,
                                   β otherwise (β < α, default α=1.0, β=0.1)
"""
import re, sys, datetime

TARGET = "/home/yelingfei/projects/GCD/mmdet/models/detectors/gdino_inc_gcd.py"

def main():
    src = open(TARGET, encoding="utf-8").read()
    backup = TARGET + ".bak_selgrmi_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    open(backup, "w", encoding="utf-8").write(src)
    print(f"[patch] backup -> {backup}")
    changes = 0

    # ─── 1. Add gt_selective_grmi_cfg to __init__ ───
    # Find existing residual_inject_cfg line
    init_marker = "residual_inject_cfg: OptConfigType = None,"
    if init_marker in src and 'gt_selective_grmi_cfg' not in src:
        new_init = (init_marker + "\n"
                    "                 gt_selective_grmi_cfg: OptConfigType = None,")
        src = src.replace(init_marker, new_init, 1)
        changes += 1
        print("[patch] 1/3: added gt_selective_grmi_cfg to __init__ signature")

        # Add config parsing after GRMI init block
        grmi_print = 'print(f"[GRMI] enabled:'
        if grmi_print in src:
            grmi_line_end = src.index(grmi_print)
            nl_after = src.index('\n', grmi_line_end)
            insert_pos = nl_after + 1
            sel_init = '''
        # --- GT-Guided Selective GRMI ---
        from mmengine.config import Config as _SelCfg
        _sel_cfg = _SelCfg._dict_to_config_dict_lazy(
            gt_selective_grmi_cfg or dict(enable=False))
        self._sel_grmi_enable = bool(_sel_cfg.get('enable', False))
        self._sel_grmi_alpha = float(_sel_cfg.get('alpha', 1.0))
        self._sel_grmi_beta = float(_sel_cfg.get('beta', 0.1))
        self._sel_grmi_ns = int(_sel_cfg.get('ns', 70))
        self._sel_grmi_ne = int(_sel_cfg.get('ne', 80))
        self._sel_grmi_gamma_boost = float(_sel_cfg.get('gamma_boost', 1.0))
        self._sel_grmi_monitor_path = str(_sel_cfg.get('monitor_path', ''))
        self._sel_grmi_monitor_interval = int(_sel_cfg.get('monitor_interval', 250))
        self._sel_grmi_step = 0
        self._sel_grmi_batch_gt = None
        if self._sel_grmi_enable:
            print(f"[SelGRMI] enabled: alpha={self._sel_grmi_alpha} beta={self._sel_grmi_beta} "
                  f"gamma_boost={self._sel_grmi_gamma_boost}")

'''
            src = src[:insert_pos] + sel_init + src[insert_pos:]
            changes += 1
            print("[patch] 2/3: added selective GRMI config parsing")
    else:
        if 'gt_selective_grmi_cfg' in src:
            print("[patch] 1-2/3: SKIP (already patched)")
        else:
            print("[patch] 1-2/3: SKIP (init_marker not found)")

    # ─── 2. Modify forward_encoder to apply selective weight ───
    # Find the GRMI injection point
    grmi_inject = "encoder_outputs_dict['memory'] = self.residual_inject(memory)"
    if grmi_inject in src and '_sel_grmi_apply' not in src:
        sel_forward = '''encoder_outputs_dict['memory'] = self._sel_grmi_apply(
                    self.residual_inject, memory, encoder_outputs_dict.get('spatial_shapes'))'''
        src = src.replace(grmi_inject, sel_forward, 1)

        # Add the _sel_grmi_apply method
        # Find a good place to insert — after forward_encoder method
        loss_method = "    def loss("
        if loss_method in src:
            insert_pos = src.index(loss_method)
            sel_method = '''
    def _sel_grmi_apply(self, residual_inject, memory, spatial_shapes):
        """Apply GRMI residual with optional GT-guided spatial selection."""
        import json as _json
        residual = residual_inject.transform(memory)
        gamma = residual_inject.gamma

        if not self.training or not self._sel_grmi_enable or self._sel_grmi_batch_gt is None:
            return memory + gamma * residual

        # Build per-position weight map from new-class GT
        B, N, C = memory.shape
        weight = torch.full((B, N, 1), self._sel_grmi_beta,
                           device=memory.device, dtype=memory.dtype)

        if spatial_shapes is not None:
            ssl = spatial_shapes.cpu().long().tolist()
            # Find finest level
            li = max(range(len(ssl)), key=lambda k: ssl[k][0] * ssl[k][1])
            H0, W0 = ssl[li]
            off = sum(ssl[k][0] * ssl[k][1] for k in range(li))
        else:
            weight[:] = 1.0
            return memory + gamma * self._sel_grmi_gamma_boost * residual * weight

        n_boosted = 0
        for bi in range(min(B, len(self._sel_grmi_batch_gt))):
            gt_inst = self._sel_grmi_batch_gt[bi]
            gt_labels = gt_inst.labels
            gt_bboxes = gt_inst.bboxes
            if hasattr(gt_bboxes, 'tensor'):
                gt_bboxes = gt_bboxes.tensor
            ns, ne = self._sel_grmi_ns, self._sel_grmi_ne
            new_mask = (gt_labels >= ns) & (gt_labels < ne)
            if not new_mask.any():
                continue
            ih = float(gt_inst.metainfo.get('img_shape', [1, 1])[0]) if hasattr(gt_inst, 'metainfo') else 1.0
            iw = float(gt_inst.metainfo.get('img_shape', [1, 1])[1]) if hasattr(gt_inst, 'metainfo') else 1.0
            # Fallback: use the image shape from the first data_sample
            # The gt_bboxes are in pixel coords, map to finest-level grid
            for i in range(len(gt_labels)):
                if not (ns <= int(gt_labels[i]) < ne):
                    continue
                bx = gt_bboxes[i]
                # Map bbox to finest-level grid coords
                # gt_bboxes are in pixel coords of preprocessed image
                gx1 = int(max(0, min(W0-1, bx[0].item()/max(iw, 1)*W0)))
                gx2 = int(max(1, min(W0, bx[2].item()/max(iw, 1)*W0)))
                gy1 = int(max(0, min(H0-1, bx[1].item()/max(ih, 1)*H0)))
                gy2 = int(max(1, min(H0, bx[3].item()/max(ih, 1)*H0)))
                for y in range(gy1, gy2):
                    for x in range(gx1, gx2):
                        idx = off + y * W0 + x
                        if idx < N:
                            weight[bi, idx, 0] = self._sel_grmi_alpha
                            n_boosted += 1

        out = memory + gamma * self._sel_grmi_gamma_boost * residual * weight

        # Monitor
        self._sel_grmi_step += 1
        if (self._sel_grmi_monitor_path and
                self._sel_grmi_step % self._sel_grmi_monitor_interval == 0):
            try:
                from mmengine.logging import MessageHub
                hub = MessageHub.get_current_instance()
                epoch = hub.get_info('epoch')
                iter_in_epoch = hub.get_info('iter')
            except Exception:
                epoch = -1; iter_in_epoch = -1
            record = {
                'step': self._sel_grmi_step,
                'epoch': epoch, 'iter': iter_in_epoch,
                'gamma': round(float(gamma.detach().item()), 6),
                'n_boosted_positions': n_boosted,
                'n_total_positions': B * N,
                'boost_ratio': round(n_boosted / max(B * N, 1), 6),
                'residual_norm_mean': round(float(residual.detach().norm(dim=-1).mean().item()), 4),
                'weight_mean': round(float(weight.mean().item()), 4),
            }
            try:
                with open(self._sel_grmi_monitor_path, 'a') as f:
                    f.write(_json.dumps(record) + '\\n')
            except Exception:
                pass

        return out

    def _sel_grmi_stash_gt(self, batch_data_samples):
        """Stash GT instances for selective GRMI. Called from loss()."""
        if not self._sel_grmi_enable:
            return
        gt_list = []
        for ds in batch_data_samples:
            gt = ds.gt_instances
            # Attach img_shape for coordinate mapping
            if not hasattr(gt, 'metainfo'):
                gt.metainfo = {}
            gt.metainfo['img_shape'] = ds.metainfo.get('img_shape', (1, 1))
            gt_list.append(gt)
        self._sel_grmi_batch_gt = gt_list

'''
            src = src[:insert_pos] + sel_method + src[insert_pos:]
            changes += 1
            print("[patch] 3a/3: added _sel_grmi_apply and _sel_grmi_stash_gt methods")

    # ─── 3. Wire GT stashing in loss() ───
    # Find where batch_data_samples is first used in loss()
    loss_start = "    def loss(self, batch_inputs"
    if loss_start in src and '_sel_grmi_stash_gt' not in src:
        # Find the forward_transformer call (which is where we need GT stashed BEFORE)
        fwd_trans = "head_inputs_dict = self.forward_transformer("
        if fwd_trans in src:
            fwd_pos = src.index(fwd_trans)
            # Insert stash call right before forward_transformer
            stash_call = "        self._sel_grmi_stash_gt(batch_data_samples)\n        "
            src = src[:fwd_pos] + stash_call + src[fwd_pos:]
            changes += 1
            print("[patch] 3b/3: wired _sel_grmi_stash_gt before forward_transformer")

    if 'gt_selective_grmi_cfg' not in src:
        print("[patch] FAILED: no changes applied")
    else:
        open(TARGET, "w", encoding="utf-8").write(src)
        print(f"\n[patch] applied {changes} changes to {TARGET}")


if __name__ == '__main__':
    main()
