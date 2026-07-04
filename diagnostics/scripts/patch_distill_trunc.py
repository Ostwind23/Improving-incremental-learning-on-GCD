#!/usr/bin/env python3
"""Patcher: Distillation Truncation at New-Class GT regions (COCO 70+10).

Background
----------
GCD (AAAI 2025) trains a student Grounding-DINO to incrementally learn new
classes 70-79 while preserving old classes 0-69 via teacher-student
distillation. The teacher was only trained on classes 0-69, so at new-class
GT positions the teacher emits strong "this is an old class" logits that
actively pull the student AWAY from the correct new-class prediction.

D1 (this patch) zeroes out the per-query distillation weights for student
queries whose predicted boxes overlap any new-class GT box above a small
IoU threshold (default 0.1). A `weight` knob (default 0.0 = full zero)
allows soft dampening instead of hard truncation if desired.

Target file
-----------
/home/yelingfei/projects/GCD/mmdet/models/dense_heads/gdino_head_inc_gcd.py

Idempotency
-----------
A single timestamped backup is written the first time this patcher modifies
the file. Re-runs detect the sentinel comment `# === D1: distill_trunc ===`
(or any pre-existing `_distill_trunc_enable` attribute from a prior patch)
and exit as a clean no-op. The patcher never duplicates edits.

Run on PolyU
------------
    cd /home/yelingfei/projects/GCD
    python /path/to/patch_distill_trunc.py
"""
import datetime
import os
import sys

TARGET = "/home/yelingfei/projects/GCD/mmdet/models/dense_heads/gdino_head_inc_gcd.py"
SENTINEL = "# === D1: distill_trunc ==="


def _die(msg, code):
    print(f"[patch] FAIL: {msg}")
    sys.exit(code)


def main():
    if not os.path.exists(TARGET):
        _die(f"target not found: {TARGET}", 1)
    src = open(TARGET, encoding="utf-8").read()

    # ─── Idempotency ───
    # Either our own sentinel, or an equivalent prior patch that already
    # defines `_distill_trunc_enable`, means we should not edit again.
    if SENTINEL in src or "_distill_trunc_enable" in src:
        print(f"[patch] target already provides distill_trunc; no-op")
        sys.exit(0)

    # ─── Timestamped backup (only on first successful patch) ───
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = f"{TARGET}.bak_dtrunc_{ts}"
    open(backup, "w", encoding="utf-8").write(src)
    print(f"[patch] backup -> {backup}")

    changes = 0

    # ─── 1. __init__: pop distill_trunc_cfg before super().__init__(**kwargs),
    #         then parse all knobs immediately after the super() call. ───
    super_line = "        super().__init__(**kwargs)\n"
    if super_line not in src:
        _die("cannot find `super().__init__(**kwargs)`", 2)
    super_pos = src.index(super_line)

    pop_block = (
        "        # === D1: distill_trunc === pop knob before forwarding kwargs\n"
        "        _distill_trunc_cfg = kwargs.pop('distill_trunc_cfg', None) or {}\n"
    )
    src = src[:super_pos] + pop_block + src[super_pos:]
    # super_pos shifted by len(pop_block); recompute the line end.
    super_pos = src.index(super_line)
    super_end = super_pos + len(super_line)

    init_block = (
        "\n"
        "        # === D1: distill_trunc ===\n"
        "        self.distill_trunc_cfg = _distill_trunc_cfg\n"
        "        self._distill_trunc_enable = bool(_distill_trunc_cfg.get('enable', False))\n"
        "        self._distill_trunc_iou_thr = float(_distill_trunc_cfg.get('iou_thr', 0.1))\n"
        "        self._distill_trunc_weight = float(_distill_trunc_cfg.get('weight', 0.0))\n"
        "        self._distill_trunc_ns = int(_distill_trunc_cfg.get('ns', 70))\n"
        "        self._distill_trunc_ne = int(_distill_trunc_cfg.get('ne', 80))\n"
        "        _dt_mon = _distill_trunc_cfg.get('monitor', {}) or {}\n"
        "        self._dt_monitor_path = str(_dt_mon.get('path', ''))\n"
        "        self._dt_monitor_interval = int(_dt_mon.get('interval', 250))\n"
        "        self._dt_step = 0\n"
        "        self._dt_n_truncated = 0\n"
        "        self._dt_mean_distill_w = 1.0\n"
        "        if self._distill_trunc_enable:\n"
        "            print(f\"[D1] distill_trunc enabled: iou_thr={self._distill_trunc_iou_thr}, \"\n"
        "                  f\"weight={self._distill_trunc_weight}, ns={self._distill_trunc_ns}, \"\n"
        "                  f\"ne={self._distill_trunc_ne}\")\n"
    )
    src = src[:super_end] + init_block + src[super_end:]
    changes += 1
    print("[patch] 1/5: added distill_trunc_cfg parsing in __init__")

    # ─── 2. Stash batch_gt_instances inside loss() ───
    # Two methods in this file iterate `for data_sample in batch_data_samples`,
    # so the bare loop is NOT unique. We exploit the fact that ONLY loss()
    # builds the two lists in this exact order:
    #     batch_gt_instances = []
    #     batch_img_metas = []
    # (generate_pseudo_label uses the swapped order: img_metas first).
    collect_anchor = (
        "        batch_gt_instances = []\n"
        "        batch_img_metas = []\n"
        "        for data_sample in batch_data_samples:\n"
        "            batch_img_metas.append(data_sample.metainfo)\n"
        "            batch_gt_instances.append(data_sample.gt_instances)\n"
    )
    if collect_anchor not in src:
        _die("cannot find loss()-specific batch_gt_instances collection block", 3)
    stash_line = (
        "        # === D1: distill_trunc === stash real GT for the distillation loss\n"
        "        self._batch_gt_instances = batch_gt_instances\n"
    )
    src = src.replace(collect_anchor, collect_anchor + stash_line, 1)
    changes += 1
    print("[patch] 2/5: stashed self._batch_gt_instances in loss()")

    # ─── 3. Insert helper methods (_apply + _monitor_log) ───
    # Place them immediately before loss_by_feat_ld_distn_single so they sit
    # in the same class body and reach self attributes naturally.
    method_block = '''    # === D1: distill_trunc === helpers
    def _distill_trunc_apply(self, bbox_preds, label_weights, bbox_weights,
                             valid_mask_list):
        """Dampen distillation weights for queries near new-class GT.

        Modifies `label_weights`, `bbox_weights`, `valid_mask_list` in place.
        Returns a tuple (n_truncated, mean_distill_w) for monitoring, where
        `mean_distill_w` is the per-batch mean of the multiplicative weight
        applied to label_weights (1.0 = no truncation, <1.0 = active).
        """
        from mmdet.structures.bbox import bbox_overlaps, bbox_cxcywh_to_xyxy
        ns = self._distill_trunc_ns
        ne = self._distill_trunc_ne
        trunc_iou = self._distill_trunc_iou_thr
        trunc_w = self._distill_trunc_weight
        n_truncated = 0
        if not hasattr(self, '_batch_gt_instances'):
            return 0, 1.0
        weight_sum = 0.0
        weight_count = 0
        B = bbox_preds.shape[0]
        for bi in range(B):
            if bi >= len(self._batch_gt_instances):
                continue
            gt_inst = self._batch_gt_instances[bi]
            gt_labels = gt_inst.labels
            gt_bboxes = gt_inst.bboxes
            if hasattr(gt_bboxes, 'tensor'):
                gt_bboxes = gt_bboxes.tensor
            new_mask = (gt_labels >= ns) & (gt_labels < ne)
            if not bool(new_mask.any()):
                continue
            new_gt = gt_bboxes[new_mask].to(bbox_preds.device)
            # bbox_preds[bi] is cxcywh in normalized image coordinates.
            pred_xyxy = bbox_cxcywh_to_xyxy(bbox_preds[bi].detach())
            ious = bbox_overlaps(pred_xyxy, new_gt)  # (n_queries, n_new_gt)
            near_new = (ious >= trunc_iou).any(dim=1)  # (n_queries,)
            n_near = int(near_new.sum().item())
            if n_near == 0:
                continue
            # Use torch.full_like to inherit label_weights' dtype + device
            # (avoids AMP half/full mismatches inside torch.where).
            damp = torch.where(
                near_new,
                torch.full_like(label_weights[bi], trunc_w),
                torch.full_like(label_weights[bi], 1.0),
            )
            label_weights[bi] = label_weights[bi] * damp
            bbox_weights[bi] = bbox_weights[bi] * damp
            valid_mask_list[bi] = valid_mask_list[bi] * damp
            n_truncated += n_near
            weight_sum += float(damp.sum().item())
            weight_count += int(damp.numel())
        mean_w = (weight_sum / weight_count) if weight_count > 0 else 1.0
        return n_truncated, mean_w

    def _dt_monitor_log(self, loss_dict):
        """Append a JSONL record every `monitor_interval` iterations.

        Records step/epoch/iter, all scalar losses, the per-batch truncation
        count, the configured IoU threshold and truncation weight, and the
        mean distillation weight for this step. Only rank 0 writes when DDP
        is initialized.
        """
        import json as _json
        import torch.distributed as dist
        self._dt_step += 1
        if (not self._dt_monitor_path
                or self._dt_step % self._dt_monitor_interval != 0):
            return
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        try:
            from mmengine.logging import MessageHub
            hub = MessageHub.get_current_instance()
            epoch = hub.get_info('epoch')
            iter_val = hub.get_info('iter')
        except Exception:
            epoch = -1
            iter_val = -1
        record = {
            'step': self._dt_step,
            'epoch': epoch,
            'iter': iter_val,
            'n_truncated': int(self._dt_n_truncated),
            'mean_distill_weight': round(float(self._dt_mean_distill_w), 6),
            'iou_thr': self._distill_trunc_iou_thr,
            'trunc_weight': self._distill_trunc_weight,
        }
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                record[k] = round(float(v.detach().item()), 6)
        try:
            os.makedirs(os.path.dirname(self._dt_monitor_path), exist_ok=True)
            with open(self._dt_monitor_path, 'a') as f:
                f.write(_json.dumps(record) + '\\n')
        except Exception:
            pass

'''
    insert_anchor = "    def loss_by_feat_ld_distn_single("
    if insert_anchor not in src:
        _die("cannot find loss_by_feat_ld_distn_single definition", 4)
    insert_pos = src.index(insert_anchor)
    src = src[:insert_pos] + method_block + src[insert_pos:]
    changes += 1
    print("[patch] 3/5: inserted _distill_trunc_apply + _dt_monitor_log methods")

    # ─── 4. Call the truncation helper inside loss_by_feat_ld_distn_single ───
    # The weighted=True and weighted=False branches both end at:
    #     valid_mask_list[overlap_list] = 0.0
    # and then converge at the shared line:
    #     ori_text_masks = self.ori_text_masks
    # We insert the call immediately BEFORE that shared anchor so it runs
    # regardless of which branch executed.
    call_block = (
        "        # === D1: distill_trunc === dampen distillation near new-class GT\n"
        "        if getattr(self, '_distill_trunc_enable', False):\n"
        "            self._dt_n_truncated, self._dt_mean_distill_w = \\\n"
        "                self._distill_trunc_apply(\n"
        "                    bbox_preds, label_weights, bbox_weights, valid_mask_list)\n"
        "\n"
    )
    ld_anchor = "        ori_text_masks = self.ori_text_masks\n"
    if ld_anchor not in src:
        _die("cannot find `ori_text_masks = self.ori_text_masks` anchor", 5)
    src = src.replace(ld_anchor, call_block + ld_anchor, 1)
    changes += 1
    print("[patch] 4/5: wired _distill_trunc_apply into loss_by_feat_ld_distn_single")

    # ─── 5. Wire monitor into loss() before its return ───
    # The head's loss() ends with:
    #     loss_dict_new.update(loss_dict_old)
    #     return loss_dict_new
    # NOTE: the source line ends with 4 trailing spaces ("return loss_dict_new    "),
    # so we include them in the anchor to make the match exact.
    ret_anchor = (
        "        loss_dict_new.update(loss_dict_old)\n"
        "        return loss_dict_new    \n"
    )
    # Fall back to the no-trailing-whitespace form in case the file changes.
    if ret_anchor not in src:
        ret_alt = (
            "        loss_dict_new.update(loss_dict_old)\n"
            "        return loss_dict_new\n"
        )
        if ret_alt not in src:
            _die("cannot find `return loss_dict_new` anchor in loss()", 6)
        ret_anchor = ret_alt
    monitor_call = (
        "        # === D1: distill_trunc === write JSONL monitor record\n"
        "        if getattr(self, '_distill_trunc_enable', False):\n"
        "            self._dt_monitor_log(loss_dict_new)\n"
    )
    src = src.replace(
        ret_anchor,
        "        loss_dict_new.update(loss_dict_old)\n" + monitor_call + ret_anchor.split("\n", 1)[1],
        1,
    )
    changes += 1
    print("[patch] 5/5: wired _dt_monitor_log into loss() before return")

    # ─── Write file back ───
    open(TARGET, "w", encoding="utf-8").write(src)
    print(f"\n[patch] applied {changes}/5 changes to {TARGET}")
    print(f"[patch] backup retained at {backup}")
    print(f"[patch] idempotency sentinel: {SENTINEL!r}")


if __name__ == "__main__":
    main()
