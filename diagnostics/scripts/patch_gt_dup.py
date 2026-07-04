#!/usr/bin/env python3
"""
Surgical patcher: upgrade the existing GT-duplication scaffolding in
GCD's incremental detection head to add IoU-based filtering and a richer
per-query JSONL training monitor.

Target file (on PolyU):
    /home/yelingfei/projects/GCD/mmdet/models/dense_heads/gdino_head_inc_gcd.py

STATE OF THE TARGET FILE
------------------------
The remote file already contains a previous, simpler GT-duplication
implementation:
  - __init__ pops `gt_dup_cfg` from kwargs and sets self._gt_dup_enable,
    self._gt_dup_factor, self._gt_dup_iou_thr, self._gt_dup_ns/ne,
    self._gd_monitor_path, self._gd_monitor_interval, self._gd_step,
    self._gd_stats.
  - loss_by_feat_new calls self._gt_dup_augment_instances on both
    batch_gt_instances and batch_all_instances BEFORE the multi_apply that
    runs self.loss_by_feat_single (i.e. the Hungarian matcher sees the
    duplicated GT set).
  - _gt_dup_augment_instances builds new InstanceData with only bboxes
    and labels (drops positive_maps / text_token_mask).
  - _gd_monitor_log writes a JSONL line every _gd_monitor_interval iters
    with all losses plus _gd_stats (n_orig_new, n_extra).

WHAT IS MISSING (and what this patch adds)
------------------------------------------
1. The Hungarian matcher is allowed to match queries to duplicated GTs
   with arbitrarily low IoU. Those noise matches then receive full
   classification + regression gradients, which is exactly the failure
   mode described in the design (extra matches with IoU < 0.1 are noise).
2. The monitor only logs how many duplicated GTs were *created*, not how
   many were *matched* nor how many were *filtered*.

This patch adds the missing pieces by introducing a new per-layer loss
method `_loss_by_feat_single_gtdup` that:
  * runs Hungarian matching against the duplicated GT set,
  * demotes any positive query whose matched GT is a duplicated instance
    (index >= dup_start) and whose IoU with that GT is below
    `self._gt_dup_iou_thr`,
  * builds the cls / bbox / iou targets via the same token-masked path
    as the original loss_by_feat_single,
  * returns a stat dict with monitor counters.
And it rewires `loss_by_feat_new` so that, when GT-dup is enabled, it
calls this new method instead of `self.loss_by_feat_single`. The monitor
is enriched with `gd_n_orig_new`, `gd_n_dup_pos`, `gd_n_kept`,
`gd_n_filtered`, `gd_dup_iou_sum`.

The patch is IDEMPOTENT: a sentinel marker `GT-DUP2-MARKER` is left in
the file. Re-running detects it and exits cleanly.

The patcher:
  * reads the file from PolyU via `ssh polyu "cat <path>"`,
  * makes a timestamped backup on PolyU (`cp -n` to avoid overwriting),
  * applies the patch on PolyU via `scp` + `mv`,
  * performs a local `python -m py_compile` and a remote
    `python3 -c "import ast; ast.parse(...)"` before the final move,
  * leaves the previous __init__ / augment / monitor_log helpers intact.
"""
from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys
import tempfile

# ---------------------------------------------------------------------------
# Paths / SSH config
# ---------------------------------------------------------------------------
SSH_HOST = "polyu"                       # configured in ~/.ssh/config
REMOTE_TARGET = (
    "/home/yelingfei/projects/GCD/"
    "mmdet/models/dense_heads/gdino_head_inc_gcd.py"
)

SENTINEL = "# GT-DUP2-MARKER"

# ---------------------------------------------------------------------------
# Code blocks to inject
# ---------------------------------------------------------------------------

# Inserted immediately BEFORE the existing `def loss_by_feat_new(`.
METHODS_BLOCK = f'''    {SENTINEL}
    def _gt_dup2_augment_instances(self, batch_gt_instances):
        """GT-duplication v2: copies every tensor field, not just bboxes/labels.

        Returns:
            augmented (list[InstanceData]): originals followed by duplicated
                new-class GTs at the end.
            meta (list[dict]): per-image book-keeping with keys
                n_orig, n_new_orig, n_dup, dup_start.
        """
        from mmengine.structures import InstanceData
        ns, ne = self._gt_dup_ns, self._gt_dup_ne
        dup = max(1, self._gt_dup_factor)
        augmented = []
        meta = []
        for gt_inst in batch_gt_instances:
            labels = gt_inst.labels
            n_orig = int(labels.numel())
            new_mask = (labels >= ns) & (labels < ne)
            n_new_orig = int(new_mask.sum().item())
            if not new_mask.any() or dup <= 1:
                augmented.append(gt_inst)
                meta.append({{
                    'n_orig': n_orig,
                    'n_new_orig': n_new_orig,
                    'n_dup': 0,
                    'dup_start': n_orig,
                }})
                continue

            new_inst = InstanceData()
            for field in gt_inst.keys():
                val = getattr(gt_inst, field)
                if (isinstance(val, torch.Tensor)
                        and val.dim() >= 1
                        and val.size(0) == n_orig):
                    sub = val[new_mask]
                    extra = sub.repeat(
                        dup - 1, *([1] * (val.dim() - 1)))
                    new_inst.set_field(
                        torch.cat([val, extra], dim=0), field)
                else:
                    new_inst.set_field(val, field)
            n_dup = int(new_mask.sum().item()) * (dup - 1)
            augmented.append(new_inst)
            meta.append({{
                'n_orig': n_orig,
                'n_new_orig': n_new_orig,
                'n_dup': n_dup,
                'dup_start': n_orig,
            }})
        self._gt_dup2_last_meta = meta
        self._gt_dup2_last_augmented = augmented
        return augmented

    def _gt_dup2_filter_assign(self, cls_scores, bbox_preds,
                               batch_gt_instances, batch_img_metas):
        """Per-image Hungarian assignment with new-class GT duplicated,
        then demotion of duplicated matches whose IoU < threshold.

        Returns:
            pos_inds_list, pos_assigned_gt_inds_list, pos_gt_labels_list,
            pos_gt_bboxes_list, stat (dict with monitor stats as tensors).
        """
        ns, ne = self._gt_dup_ns, self._gt_dup_ne
        iou_thr = self._gt_dup_iou_thr

        num_imgs = cls_scores.size(0)
        stat = dict(
            gd_n_orig_new=0.0,
            gd_n_dup_pos=0.0,
            gd_n_kept=0.0,
            gd_n_filtered=0.0,
            gd_dup_iou_sum=0.0,
        )

        augmented = getattr(self, '_gt_dup2_last_augmented', None)
        if augmented is None or len(augmented) != num_imgs:
            augmented = self._gt_dup2_augment_instances(batch_gt_instances)
        meta = self._gt_dup2_last_meta

        pos_inds_list = []
        pos_assigned_gt_inds_list = []
        pos_gt_labels_list = []
        pos_gt_bboxes_list = []

        for i in range(num_imgs):
            cls_i = cls_scores[i]
            bbox_i = bbox_preds[i]
            gt_inst = augmented[i]
            img_meta = batch_img_metas[i]
            dup_start = meta[i]['dup_start']

            img_h, img_w = img_meta['img_shape']
            factor = bbox_i.new_tensor(
                [img_w, img_h, img_w, img_h]).unsqueeze(0)
            pred_xyxy = bbox_cxcywh_to_xyxy(bbox_i) * factor
            pred_xyxy[:, 0::2].clamp_(min=0, max=img_w)
            pred_xyxy[:, 1::2].clamp_(min=0, max=img_h)
            pred_instances = InstanceData(scores=cls_i, bboxes=pred_xyxy)

            with torch.no_grad():
                assign_result = self.assigner.assign(
                    pred_instances=pred_instances,
                    gt_instances=gt_inst,
                    img_meta=img_meta)
            gt_inds = assign_result.gt_inds.clone()  # 0=bg, 1-indexed FG

            pos_mask = gt_inds > 0
            pos_q_inds = torch.nonzero(
                pos_mask, as_tuple=False).squeeze(-1)
            if pos_q_inds.numel() == 0:
                pos_inds_list.append(pos_q_inds)
                pos_assigned_gt_inds_list.append(
                    pos_q_inds.new_zeros((0,)).long())
                pos_gt_labels_list.append(
                    gt_inst.labels.new_zeros((0,)).long())
                pos_gt_bboxes_list.append(
                    gt_inst.bboxes.new_zeros((0, 4)))
                continue

            pos_assigned = (gt_inds[pos_q_inds] - 1).long()  # 0-indexed GT

            # Vectorized IoU filter for matches to duplicated GTs.
            dup_match_mask = pos_assigned >= dup_start
            if dup_match_mask.any():
                dup_q_inds = pos_q_inds[dup_match_mask]
                dup_gt_inds = pos_assigned[dup_match_mask]
                dup_pred_boxes = pred_xyxy[dup_q_inds]
                dup_gt_boxes = gt_inst.bboxes[dup_gt_inds]
                dup_ious = bbox_overlaps(
                    dup_pred_boxes, dup_gt_boxes, is_aligned=True)
                stat['gd_n_dup_pos'] += float(dup_match_mask.sum().item())
                stat['gd_dup_iou_sum'] += float(dup_ious.sum().item())
                bad_local = dup_ious < iou_thr
                if bad_local.any():
                    bad_q_inds = dup_q_inds[bad_local]
                    gt_inds[bad_q_inds] = 0  # demote to background
                    stat['gd_n_filtered'] += float(bad_local.sum().item())
                stat['gd_n_kept'] += float(
                    (dup_ious >= iou_thr).sum().item())

            # Original (non-duplicated) new-class positives (monitor only)
            kept_pos = gt_inds > 0
            kept_inds_local = (gt_inds[kept_pos] - 1).long()
            if kept_inds_local.numel() > 0:
                kept_labels = gt_inst.labels[kept_inds_local]
                is_new_orig = ((kept_labels >= ns)
                               & (kept_labels < ne)
                               & (kept_inds_local < dup_start))
                stat['gd_n_orig_new'] += float(is_new_orig.sum().item())

            # Re-extract positives after demotion
            pos_inds = torch.nonzero(
                gt_inds > 0, as_tuple=False).squeeze(-1).unique()
            pos_assigned_gt_inds = (gt_inds[pos_inds] - 1).long()
            pos_inds_list.append(pos_inds)
            pos_assigned_gt_inds_list.append(pos_assigned_gt_inds)
            pos_gt_labels_list.append(
                gt_inst.labels[pos_assigned_gt_inds])
            pos_gt_bboxes_list.append(
                gt_inst.bboxes[pos_assigned_gt_inds])

        # Convert stat floats to tensors on the right device for aggregation
        zero = cls_scores.new_zeros(())
        for k in stat:
            stat[k] = (stat[k] if isinstance(stat[k], torch.Tensor)
                       else zero.detach().new_tensor(float(stat[k])))
        return (pos_inds_list, pos_assigned_gt_inds_list,
                pos_gt_labels_list, pos_gt_bboxes_list, stat)

    def _loss_by_feat_single_gtdup(self, cls_scores, bbox_preds,
                                   batch_gt_instances, batch_img_metas):
        """Per-decoder-layer DETR loss with GT duplication + IoU filter.

        Mirrors self.loss_by_feat_single but builds targets through
        _gt_dup2_filter_assign so duplicated matches with IoU < threshold
        are demoted before classification / regression target construction.
        """
        num_imgs = cls_scores.size(0)

        (pos_inds_list, pos_assigned_gt_inds_list, _pos_gt_labels_list,
         _pos_gt_bboxes_list, stat) = self._gt_dup2_filter_assign(
            cls_scores, bbox_preds, batch_gt_instances, batch_img_metas)

        augmented = self._gt_dup2_last_augmented

        num_bboxes = cls_scores.size(1)
        labels_list = []
        label_weights_list = []
        bbox_targets_list = []
        bbox_weights_list = []
        num_total_pos = 0
        num_total_neg = 0
        for i in range(num_imgs):
            pos_inds = pos_inds_list[i]
            pos_assigned_gt_inds_i = pos_assigned_gt_inds_list[i]
            aug_inst = augmented[i]

            labels_i = aug_inst.bboxes.new_full(
                (num_bboxes, self.max_text_len), 0, dtype=torch.float32)
            label_weights_i = aug_inst.bboxes.new_ones(num_bboxes)
            bbox_targets_i = torch.zeros_like(
                bbox_preds[i], dtype=aug_inst.bboxes.dtype)
            bbox_weights_i = torch.zeros_like(
                bbox_preds[i], dtype=aug_inst.bboxes.dtype)

            if pos_inds.numel() > 0:
                # positive_maps carries the per-GT text-token label mask.
                if hasattr(aug_inst, 'positive_maps'):
                    labels_i[pos_inds] = aug_inst.positive_maps[
                        pos_assigned_gt_inds_i]
                bbox_weights_i[pos_inds] = 1.0
                img_h, img_w = batch_img_metas[i]['img_shape']
                factor_i = bbox_preds[i].new_tensor(
                    [img_w, img_h, img_w, img_h]).unsqueeze(0)
                pos_gt_bboxes = aug_inst.bboxes[pos_assigned_gt_inds_i]
                pos_gt_norm = pos_gt_bboxes / factor_i
                pos_gt_cxcywh = bbox_xyxy_to_cxcywh(pos_gt_norm)
                bbox_targets_i[pos_inds] = pos_gt_cxcywh

            labels_list.append(labels_i)
            label_weights_list.append(label_weights_i)
            bbox_targets_list.append(bbox_targets_i)
            bbox_weights_list.append(bbox_weights_i)
            num_total_pos += pos_inds.numel()
            num_total_neg += (num_bboxes - pos_inds.numel())

        labels = torch.stack(labels_list, 0)
        label_weights = torch.stack(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # Token-mask the same way as the base head
        assert self.text_masks.dim() == 2
        text_masks = self.text_masks.new_zeros(
            (self.text_masks.size(0), self.max_text_len))
        text_masks[:, :self.text_masks.size(1)] = self.text_masks
        text_mask = (text_masks > 0).unsqueeze(1).repeat(
            1, cls_scores.size(1), 1)
        cls_scores_masked = torch.masked_select(
            cls_scores, text_mask).contiguous()
        labels_masked = torch.masked_select(labels, text_mask)
        label_weights_exp = label_weights[..., None].repeat(
            1, 1, text_mask.size(-1))
        label_weights_masked = torch.masked_select(
            label_weights_exp, text_mask)

        cls_avg_factor = (num_total_pos * 1.0
                          + num_total_neg * self.bg_cls_weight)
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores_masked.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        if isinstance(self.loss_cls, QualityFocalLoss):
            raise NotImplementedError(
                'QualityFocalLoss for GroundingDINOHead is not supported.')
        loss_cls = self.loss_cls(
            cls_scores_masked, labels_masked,
            label_weights_masked, avg_factor=cls_avg_factor)

        num_total_pos_t = loss_cls.new_tensor([num_total_pos])
        num_total_pos_t = torch.clamp(
            reduce_mean(num_total_pos_t), min=1).item()

        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, bbox_preds):
            img_h, img_w = img_meta['img_shape']
            factor = bbox_pred.new_tensor(
                [img_w, img_h, img_w, img_h]).unsqueeze(0).repeat(
                    bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        bbox_preds_flat = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds_flat) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos_t)
        loss_bbox = self.loss_bbox(
            bbox_preds_flat, bbox_targets, bbox_weights,
            avg_factor=num_total_pos_t)

        return loss_cls, loss_bbox, loss_iou, stat

    def _gd2_monitor_log(self, loss_dict, stat):
        """Enhanced monitor: also logs dup-match stats (kept/filtered/iou)."""
        import json as _json
        import os as _os
        self._gd_step += 1
        if not self._gd_monitor_path:
            return
        if (self._gd_step % self._gd_monitor_interval) != 0:
            return
        epoch_val = -1
        iter_val = -1
        try:
            from mmengine import MessageHub
            hub = MessageHub.get_current_instance()
            epoch_val = int(hub.get_info('epoch', -1))
            iter_val = int(hub.get_info('iter', -1))
        except Exception:
            pass
        rec = {{
            'step': self._gd_step,
            'epoch': epoch_val,
            'iter': iter_val,
        }}
        rec.update(self._gd_stats)
        for k, v in stat.items():
            try:
                rec[k] = round(float(v.detach().item()
                                      if hasattr(v, 'detach') else v), 6)
            except Exception:
                pass
        for k, v in loss_dict.items():
            if hasattr(v, 'detach'):
                try:
                    rec[k] = round(float(v.detach().item()), 6)
                except Exception:
                    pass
        try:
            _dir = _os.path.dirname(self._gd_monitor_path)
            if _dir:
                _os.makedirs(_dir, exist_ok=True)
            with open(self._gd_monitor_path, 'a') as f:
                f.write(_json.dumps(rec) + '\\n')
        except Exception:
            pass

'''


# ---------------------------------------------------------------------------
# Edit to apply INSIDE loss_by_feat_new: replace the bare multi_apply +
# monitor call with a GT-dup-aware dispatch when enabled.
# ---------------------------------------------------------------------------
# The remote source has this exact block at the top of loss_by_feat_new
# (right after loss_dict = dict()). It does a naive augment that drops
# positive_maps / text_token_mask — we neutralise it because our new
# dispatch inside multi_apply calls _gt_dup2_augment_instances (which
# preserves all fields) itself.
OLD_LEGACY_AUGMENT = (
    "        loss_dict = dict()\n"
    "\n"
    "        # === D3: GT Duplication ===\n"
    "        if getattr(self, '_gt_dup_enable', False):\n"
    "            batch_gt_instances = self._gt_dup_augment_instances(batch_gt_instances)\n"
    "            batch_all_instances = self._gt_dup_augment_instances(batch_all_instances)\n"
    "        # extract denoising and matching part of outputs\n"
)
NEW_LEGACY_AUGMENT = (
    "        loss_dict = dict()\n"
    "        # NOTE: legacy _gt_dup_augment_instances call removed by GT-dup2\n"
    "        # patcher; the v2 helper inside _loss_by_feat_single_gtdup\n"
    "        # preserves all InstanceData fields (positive_maps, etc.).\n"
    "        # extract denoising and matching part of outputs\n"
)

# The remote source has this exact block later inside loss_by_feat_new:
OLD_MULTI_APPLY = (
    "        # ===== detr loss ===== \n"
    "        losses_cls, losses_bbox, losses_iou = multi_apply(\n"
    "            self.loss_by_feat_single,\n"
    "            all_layers_matching_cls_scores,\n"
    "            all_layers_matching_bbox_preds,\n"
    "            batch_gt_instances=batch_all_instances,\n"
    "            batch_img_metas=batch_img_metas)\n"
)

NEW_MULTI_APPLY = (
    "        # ===== detr loss ===== \n"
    "        if getattr(self, '_gt_dup_enable', False):\n"
    "            # v2 augment preserves ALL InstanceData fields\n"
    "            " + SENTINEL + " dispatch.\n"
    "            batch_gt_instances = self._gt_dup2_augment_instances(\n"
    "                batch_gt_instances)\n"
    "            batch_all_instances = self._gt_dup2_augment_instances(\n"
    "                batch_all_instances)\n"
    "            losses_cls, losses_bbox, losses_iou, _stat = multi_apply(\n"
    "                self._loss_by_feat_single_gtdup,\n"
    "                all_layers_matching_cls_scores,\n"
    "                all_layers_matching_bbox_preds,\n"
    "                batch_gt_instances=batch_all_instances,\n"
    "                batch_img_metas=batch_img_metas)\n"
    "            # Aggregate per-layer stats and log them.\n"
    "            _agg = dict()\n"
    "            for _k in ('gd_n_orig_new', 'gd_n_dup_pos',\n"
    "                       'gd_n_kept', 'gd_n_filtered',\n"
    "                       'gd_dup_iou_sum'):\n"
    "                _agg[_k] = sum(\n"
    "                    float(_s[_k].detach().item()\n"
    "                          if hasattr(_s[_k], 'detach') else _s[_k])\n"
    "                    for _s in _stat)\n"
    "            self._gd2_monitor_log(dict(), _agg)\n"
    "        else:\n"
    "            losses_cls, losses_bbox, losses_iou = multi_apply(\n"
    "                self.loss_by_feat_single,\n"
    "                all_layers_matching_cls_scores,\n"
    "                all_layers_matching_bbox_preds,\n"
    "                batch_gt_instances=batch_all_instances,\n"
    "                batch_img_metas=batch_img_metas)\n"
)

OLD_MONITOR_CALL = "            self._gd_monitor_log(loss_dict)\n"
NEW_MONITOR_CALL = (
    "            # _gd2_monitor_log is called per-layer above when GT-dup is\n"
    "            # enabled; this fallback only fires for the legacy path.\n"
    "            if not getattr(self, '_gt_dup_enable', False):\n"
    "                self._gd_monitor_log(loss_dict)\n"
)


def run(cmd, check=True):
    print(f"[cmd] {cmd}")
    subprocess.run(cmd, check=check)


def remote_read(path):
    r = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=30", SSH_HOST, "cat", path],
        capture_output=True, text=True, check=True)
    return r.stdout


def remote_backup(path):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = f"{path}.bak_gtdup2_{ts}"
    # `cp -n` avoids overwriting an existing backup with the same name
    subprocess.run(
        ["ssh", "-o", "ConnectTimeout=30", SSH_HOST,
         "cp", "-n", path, backup],
        check=False, capture_output=True)
    print(f"[patch] remote backup -> {backup}")
    return backup


def remote_write(path, content):
    with tempfile.NamedTemporaryFile(
            "w", suffix=".py", delete=False, encoding="utf-8") as tf:
        tf.write(content)
        tmp_path = tf.name
    try:
        remote_tmp = f"/tmp/gtdup2_patch_{os.getpid()}.py"
        subprocess.run(
            ["scp", "-q", tmp_path, f"{SSH_HOST}:{remote_tmp}"],
            check=True)
        subprocess.run(
            ["ssh", "-o", "ConnectTimeout=30", SSH_HOST,
             "mv", remote_tmp, path],
            check=True)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def apply_patch(src: str) -> str:
    """Apply the patch to local copy of `src`; return patched text."""
    if SENTINEL in src:
        print("[patch] GT-dup2 already applied; no-op")
        return src

    changes = 0

    # --- 1) Insert new methods just before def loss_by_feat_new( -----------
    method_anchor = "    def loss_by_feat_new(\n"
    if method_anchor not in src:
        raise RuntimeError(
            "anchor 'def loss_by_feat_new(' not found; "
            "is the target file the GCD head?")
    idx = src.index(method_anchor)
    src = src[:idx] + METHODS_BLOCK + src[idx:]
    changes += 1
    print("[patch] 1/4: inserted v2 augment / filter_assign / "
          "single_gtdup / gd2_monitor_log methods")

    # --- 2) Rewire multi_apply inside loss_by_feat_new --------------------
    if OLD_MULTI_APPLY not in src:
        raise RuntimeError(
            "anchor multi_apply block not found in loss_by_feat_new; "
            "remote file structure changed")
    src = src.replace(OLD_MULTI_APPLY, NEW_MULTI_APPLY, 1)
    changes += 1
    print("[patch] 2/4: rewired multi_apply to use _loss_by_feat_single_gtdup "
          "when GT-dup is enabled")

    # --- 2b) Remove legacy augment call at top of loss_by_feat_new ---------
    if OLD_LEGACY_AUGMENT in src:
        src = src.replace(OLD_LEGACY_AUGMENT, NEW_LEGACY_AUGMENT, 1)
        changes += 1
        print("[patch] 3/4: removed legacy _gt_dup_augment_instances call "
              "(v2 dispatch handles augment itself)")
    else:
        print("[patch] 3/4: SKIP (legacy augment call not found)")

    # --- 3) Avoid double monitor logging ----------------------------------
    if OLD_MONITOR_CALL in src:
        src = src.replace(OLD_MONITOR_CALL, NEW_MONITOR_CALL, 1)
        changes += 1
        print("[patch] 4/4: gated legacy _gd_monitor_log to skip when "
              "GT-dup is enabled")
    else:
        print("[patch] 4/4: SKIP (legacy monitor call not found — "
              "already patched or removed)")

    print(f"[patch] applied {changes} changes")
    return src


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target", default=REMOTE_TARGET,
        help="Remote target file path on PolyU")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Apply patch locally only; do not write to PolyU")
    parser.add_argument(
        "--from-local", default=None,
        help="Read source from a local path instead of SSH")
    args = parser.parse_args()

    print(f"[patch] target: {args.target}")

    # Step 1: read source
    if args.from_local:
        src = open(args.from_local, encoding="utf-8").read()
        print(f"[patch] read {len(src)} bytes from {args.from_local}")
    else:
        src = remote_read(args.target)
        print(f"[patch] read {len(src)} bytes via ssh {SSH_HOST}")

    # Step 2: apply patch
    patched = apply_patch(src)

    if patched == src:
        return 0

    # Step 3: write back
    if args.dry_run:
        out = os.path.join(tempfile.gettempdir(), "gtdup2_patched.py")
        open(out, "w", encoding="utf-8").write(patched)
        print(f"[patch] dry-run wrote patched file to {out}")
        return 0

    # Step 3a: backup on remote
    remote_backup(args.target)

    # Step 3b: local syntax check before pushing
    with tempfile.NamedTemporaryFile(
            "w", suffix=".py", delete=False, encoding="utf-8") as tf:
        tf.write(patched)
        tmp_check = tf.name
    syntax_check = subprocess.run(
        ["python", "-m", "py_compile", tmp_check],
        capture_output=True, text=True)
    os.unlink(tmp_check)
    if syntax_check.returncode != 0:
        print("[patch] LOCAL SYNTAX CHECK FAILED:")
        print(syntax_check.stderr, file=sys.stderr)
        return 1
    print("[patch] local syntax check passed")

    # Step 3c: push to remote
    remote_write(args.target, patched)
    print(f"[patch] wrote patched file to {args.target}")

    # Step 3d: remote AST check (use python3 — always available on Ubuntu)
    remote_check = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=30", SSH_HOST, "python3", "-c",
         f"import ast; ast.parse(open('{args.target}').read()); "
         f"print('REMOTE_AST_OK')"],
        capture_output=True, text=True)
    if (remote_check.returncode != 0
            or "REMOTE_AST_OK" not in remote_check.stdout):
        print("[patch] REMOTE AST CHECK FAILED:")
        print(remote_check.stdout)
        print(remote_check.stderr, file=sys.stderr)
        return 1
    print(f"[patch] remote ast check: {remote_check.stdout.strip()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
