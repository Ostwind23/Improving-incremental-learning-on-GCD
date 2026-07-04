#!/usr/bin/env python3
"""
Combined patcher for D1 (distill truncation) + D3 (GT duplication) on
gdino_head_inc_gcd.py. Single-pass, robust string matching.
"""
import datetime, sys, os

TARGET = "/home/yelingfei/projects/GCD/mmdet/models/dense_heads/gdino_head_inc_gcd.py"

def main():
    src = open(TARGET, encoding="utf-8").read()
    backup = TARGET + ".bak_d1d3_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    open(backup, "w", encoding="utf-8").write(src)
    print(f"[patch] backup -> {backup}")
    changes = 0

    # ══════════════════════════════════════════════════════
    # CHANGE 1: Add config parsing after super().__init__
    # ══════════════════════════════════════════════════════
    anchor1 = "super().__init__(**kwargs)"
    if anchor1 in src and '_distill_trunc_enable' not in src:
        idx = src.index(anchor1)
        nl = src.index('\n', idx)
        block = '''

        # === D1: Distillation Truncation ===
        _dt_cfg = getattr(self, 'distill_trunc_cfg', None) or {}
        self._distill_trunc_enable = bool(_dt_cfg.get('enable', False))
        self._distill_trunc_iou_thr = float(_dt_cfg.get('iou_thr', 0.1))
        self._distill_trunc_weight = float(_dt_cfg.get('weight', 0.0))
        self._distill_trunc_ns = int(_dt_cfg.get('ns', 70))
        self._distill_trunc_ne = int(_dt_cfg.get('ne', 80))
        _dt_mon = _dt_cfg.get('monitor', {}) if isinstance(_dt_cfg, dict) else {}
        self._dt_monitor_path = str(_dt_mon.get('path', ''))
        self._dt_monitor_interval = int(_dt_mon.get('interval', 250))
        self._dt_step = 0
        self._dt_n_truncated = 0
        if self._distill_trunc_enable:
            print(f"[DT] distill truncation: iou={self._distill_trunc_iou_thr} w={self._distill_trunc_weight}")

        # === D3: GT Duplication ===
        _gd_cfg = getattr(self, 'gt_dup_cfg', None) or {}
        self._gt_dup_enable = bool(_gd_cfg.get('enable', False))
        self._gt_dup_factor = int(_gd_cfg.get('dup_factor', 2))
        self._gt_dup_iou_thr = float(_gd_cfg.get('iou_thr', 0.1))
        self._gt_dup_ns = int(_gd_cfg.get('ns', 70))
        self._gt_dup_ne = int(_gd_cfg.get('ne', 80))
        _gd_mon = _gd_cfg.get('monitor', {}) if isinstance(_gd_cfg, dict) else {}
        self._gd_monitor_path = str(_gd_mon.get('path', ''))
        self._gd_monitor_interval = int(_gd_mon.get('interval', 250))
        self._gd_step = 0
        self._gd_stats = {}
        if self._gt_dup_enable:
            print(f"[GTDup] factor={self._gt_dup_factor} iou_thr={self._gt_dup_iou_thr}")
'''
        src = src[:nl+1] + block + src[nl+1:]
        changes += 1
        print("[patch] 1: init config parsing for D1+D3")

    # ══════════════════════════════════════════════════════
    # CHANGE 2: Stash batch_gt_instances in head's loss()
    # ══════════════════════════════════════════════════════
    # Find: batch_gt_instances = [...]  (around line 489-493)
    anchor2 = "batch_gt_instances ="
    if anchor2 in src and '_batch_gt_instances' not in src:
        idx = src.index(anchor2)
        # Find end of this statement (next line that doesn't continue)
        nl = src.index('\n', idx)
        # Check if it's a multi-line list comprehension
        while src[nl+1:nl+10].strip().startswith('for ') or src[nl+1:nl+10].strip() == '':
            nl = src.index('\n', nl+1)
        src = src[:nl+1] + "        self._batch_gt_instances = batch_gt_instances\n" + src[nl+1:]
        changes += 1
        print("[patch] 2: stash _batch_gt_instances")

    # ══════════════════════════════════════════════════════
    # CHANGE 3: Add D1 truncation in loss_by_feat_ld_distn_single
    # ══════════════════════════════════════════════════════
    anchor3 = "        ori_text_masks = self.ori_text_masks"
    if anchor3 in src and '_distill_trunc_apply' not in src:
        trunc_call = '''        # === D1: truncation ===
        if getattr(self, '_distill_trunc_enable', False) and hasattr(self, '_batch_gt_instances'):
            self._dt_n_truncated = self._distill_trunc_apply(
                bbox_preds, label_weights, bbox_weights, valid_mask_list)

'''
        src = src.replace(anchor3, trunc_call + anchor3, 1)
        changes += 1
        print("[patch] 3: D1 truncation call inserted")

    # ══════════════════════════════════════════════════════
    # CHANGE 4: Add D3 GT duplication in loss_by_feat_new
    # ══════════════════════════════════════════════════════
    # Insert after the first "loss_dict = dict()" that appears inside loss_by_feat_new
    anchor4_method = "    def loss_by_feat_new("
    anchor4_body = "        loss_dict = dict()"
    if anchor4_method in src and '_gt_dup_augment' not in src:
        method_start = src.index(anchor4_method)
        # Find "loss_dict = dict()" AFTER this method start
        body_start = src.index(anchor4_body, method_start)
        nl = src.index('\n', body_start)
        dup_call = '''
        # === D3: GT Duplication ===
        if getattr(self, '_gt_dup_enable', False):
            batch_gt_instances = self._gt_dup_augment_instances(batch_gt_instances)
            batch_all_instances = self._gt_dup_augment_instances(batch_all_instances)
'''
        src = src[:nl+1] + dup_call + src[nl+1:]
        changes += 1
        print("[patch] 4: D3 GT duplication call inserted")

    # ══════════════════════════════════════════════════════
    # CHANGE 5: Add method bodies BEFORE loss_by_feat_ld_distn_single
    # ══════════════════════════════════════════════════════
    anchor5 = "    def loss_by_feat_ld_distn_single("
    if anchor5 in src and 'def _distill_trunc_apply(' not in src:
        idx = src.index(anchor5)
        methods = '''    # ============================================================
    # D1: Distillation Truncation methods
    # ============================================================
    def _distill_trunc_apply(self, bbox_preds, label_weights, bbox_weights, valid_mask_list):
        from mmdet.structures.bbox import bbox_overlaps, bbox_cxcywh_to_xyxy
        ns, ne = self._distill_trunc_ns, self._distill_trunc_ne
        trunc_iou = self._distill_trunc_iou_thr
        trunc_w = self._distill_trunc_weight
        n_truncated = 0
        B = bbox_preds.shape[0]
        for bi in range(B):
            if bi >= len(self._batch_gt_instances): continue
            gt_inst = self._batch_gt_instances[bi]
            gt_labels = gt_inst.labels
            gt_bboxes = gt_inst.bboxes
            if hasattr(gt_bboxes, 'tensor'): gt_bboxes = gt_bboxes.tensor
            new_mask = (gt_labels >= ns) & (gt_labels < ne)
            if not new_mask.any(): continue
            new_gt = gt_bboxes[new_mask].to(bbox_preds.device)
            pred_xyxy = bbox_cxcywh_to_xyxy(bbox_preds[bi].detach())
            ious = bbox_overlaps(pred_xyxy, new_gt)
            near_new = (ious >= trunc_iou).any(dim=1)
            n_near = int(near_new.sum())
            if n_near > 0:
                import torch
                damp = torch.where(near_new,
                    torch.tensor(trunc_w, device=near_new.device, dtype=label_weights.dtype),
                    torch.tensor(1.0, device=near_new.device, dtype=label_weights.dtype))
                label_weights[bi] = label_weights[bi] * damp
                bbox_weights[bi] = bbox_weights[bi] * damp
                valid_mask_list[bi] = valid_mask_list[bi] * damp
                n_truncated += n_near
        return n_truncated

    def _dt_monitor_log(self, loss_dict):
        import json as _json
        self._dt_step += 1
        if not self._dt_monitor_path or self._dt_step % self._dt_monitor_interval != 0:
            return
        try:
            from mmengine.logging import MessageHub
            hub = MessageHub.get_current_instance()
            epoch = hub.get_info('epoch'); it = hub.get_info('iter')
        except Exception:
            epoch = -1; it = -1
        rec = {'step': self._dt_step, 'epoch': epoch, 'iter': it,
               'n_truncated': self._dt_n_truncated}
        for k, v in loss_dict.items():
            if hasattr(v, 'item'): rec[k] = round(float(v.detach().item()), 6)
        try:
            with open(self._dt_monitor_path, 'a') as f:
                f.write(_json.dumps(rec) + '\\n')
        except Exception: pass

    # ============================================================
    # D3: GT Duplication methods
    # ============================================================
    def _gt_dup_augment_instances(self, batch_gt_instances):
        import torch
        from mmengine.structures import InstanceData
        ns, ne = self._gt_dup_ns, self._gt_dup_ne
        dup = self._gt_dup_factor
        augmented = []
        n_orig = 0; n_extra = 0
        for gt_inst in batch_gt_instances:
            labels = gt_inst.labels
            bboxes = gt_inst.bboxes
            if hasattr(bboxes, 'tensor'): bboxes = bboxes.tensor
            new_mask = (labels >= ns) & (labels < ne)
            if not new_mask.any() or dup <= 1:
                augmented.append(gt_inst)
                continue
            n_orig += int(new_mask.sum())
            new_bboxes = bboxes[new_mask]
            new_labels = labels[new_mask]
            extra_b = new_bboxes.repeat(dup - 1, 1)
            extra_l = new_labels.repeat(dup - 1)
            n_extra += len(extra_l)
            new_inst = InstanceData()
            new_inst.bboxes = torch.cat([bboxes, extra_b], dim=0)
            new_inst.labels = torch.cat([labels, extra_l], dim=0)
            augmented.append(new_inst)
        self._gd_stats = {'n_orig_new': n_orig, 'n_extra': n_extra}
        return augmented

    def _gd_monitor_log(self, loss_dict):
        import json as _json
        self._gd_step += 1
        if not self._gd_monitor_path or self._gd_step % self._gd_monitor_interval != 0:
            return
        try:
            from mmengine.logging import MessageHub
            hub = MessageHub.get_current_instance()
            epoch = hub.get_info('epoch'); it = hub.get_info('iter')
        except Exception:
            epoch = -1; it = -1
        rec = {'step': self._gd_step, 'epoch': epoch, 'iter': it}
        rec.update(self._gd_stats)
        for k, v in loss_dict.items():
            if hasattr(v, 'item'): rec[k] = round(float(v.detach().item()), 6)
        try:
            with open(self._gd_monitor_path, 'a') as f:
                f.write(_json.dumps(rec) + '\\n')
        except Exception: pass

'''
        src = src[:idx] + methods + src[idx:]
        changes += 1
        print("[patch] 5: D1+D3 method bodies inserted")

    # ══════════════════════════════════════════════════════
    # CHANGE 6: Wire D1 monitor into loss_by_feat_old return
    # ══════════════════════════════════════════════════════
    anchor6 = "    def loss_by_feat_old("
    if anchor6 in src and '_dt_monitor_log' not in src:
        old_start = src.index(anchor6)
        # Find "return loss_dict" in this method
        next_def = src.find("\n    def ", old_start + 10)
        return_keyword = "return loss_dict"
        ret_pos = src.rfind(return_keyword, old_start, next_def if next_def > 0 else len(src))
        if ret_pos > 0:
            monitor = "        if getattr(self, '_distill_trunc_enable', False):\n            self._dt_monitor_log(loss_dict)\n        "
            src = src[:ret_pos] + monitor + src[ret_pos:]
            changes += 1
            print("[patch] 6: D1 monitor wired in loss_by_feat_old")

    # ══════════════════════════════════════════════════════
    # CHANGE 7: Wire D3 monitor in loss_by_feat_new return
    # ══════════════════════════════════════════════════════
    anchor7 = "    def loss_by_feat_new("
    if anchor7 in src and '_gd_monitor_log' not in src:
        new_start = src.index(anchor7)
        next_def = src.find("\n    def ", new_start + 10)
        return_keyword = "return loss_dict"
        ret_pos = src.rfind(return_keyword, new_start, next_def if next_def > 0 else len(src))
        if ret_pos > 0:
            monitor = "        if getattr(self, '_gt_dup_enable', False):\n            self._gd_monitor_log(loss_dict)\n        "
            src = src[:ret_pos] + monitor + src[ret_pos:]
            changes += 1
            print("[patch] 7: D3 monitor wired in loss_by_feat_new")

    open(TARGET, "w", encoding="utf-8").write(src)
    print(f"\n[patch] total {changes} changes applied")
    if changes < 7:
        print(f"[patch] WARNING: expected 7 changes, got {changes}")
        # List what's missing
        for check, name in [
            ('_distill_trunc_enable', 'D1 init'),
            ('_batch_gt_instances', 'GT stash'),
            ('_distill_trunc_apply', 'D1 call+method'),
            ('_gt_dup_augment', 'D3 call'),
            ('_gt_dup_augment_instances', 'D3 method'),
            ('_dt_monitor_log', 'D1 monitor'),
            ('_gd_monitor_log', 'D3 monitor'),
        ]:
            if check not in src:
                print(f"  MISSING: {name} ({check})")

if __name__ == '__main__':
    main()
