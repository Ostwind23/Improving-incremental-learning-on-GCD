#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
patch_d1d3_v3.py

Applies TWO independent modifications to GCD's incremental detection head:
  D1: Distillation Truncation (BUG-FIXED version)
       - Stashes both batch_gt_instances and batch_img_metas in self.loss()
       - In loss_by_feat_ld_distn_single, after BOTH branches that set
         valid_mask_list[overlap_list] = 0.0, calls _distill_trunc_apply
       - _distill_trunc_apply normalizes GT from pixel xyxy -> [0,1] xyxy
         before computing IoU against bbox_preds (which are [0,1] cxcywh).
  D3: GT Duplication
       - In loss_by_feat_new, after loss_dict = dict(), augments
         batch_gt_instances and batch_all_instances with duplicated new-class GT.
       - Generic InstanceData duplication covering ALL fields (not just
         bboxes/labels) so positive_maps/text_token_mask propagate.

Idempotent: detects the marker comment "# D1D3-V3-APPLIED" and refuses to
re-apply. Detects individual markers ("# D1-V3-APPLIED", "# D3-V3-APPLIED")
to support partial re-application.

Usage:
    python patch_d1d3_v3.py                 # apply D1 + D3
    python patch_d1d3_v3.py --check         # only report current status
    python patch_d1d3_v3.py --revert        # NOT SUPPORTED; use PolyU backup

The script:
  1. Reads the file from PolyU via SSH
  2. Creates a timestamped backup on PolyU
  3. Applies all changes locally
  4. Verifies syntax with py_compile
  5. Writes the result back to PolyU
  6. Prints clear success/failure messages
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from textwrap import dedent

# ===========================================================================
# Configuration
# ===========================================================================

SSH_TARGET = "polyu"
REMOTE_FILE = "/home/yelingfei/projects/GCD/mmdet/models/dense_heads/gdino_head_inc_gcd.py"
REMOTE_DIR = os.path.dirname(REMOTE_FILE)
REMOTE_BACKUP_DIR = "/home/yelingfei/projects/GCD/.patch_backups"

MARKER_D1 = "# D1-V3-APPLIED"
MARKER_D3 = "# D3-V3-APPLIED"
MARKER_BOTH = "# D1D3-V3-APPLIED"

# ===========================================================================
# Helper: run shell command
# ===========================================================================


def run(cmd: str, capture: bool = True, check: bool = True, timeout: int = 300):
    """Run a shell command, return CompletedProcess.

    Uses utf-8 decoding (with errors='replace' fallback) because the
    target file contains non-ASCII characters and Windows defaults to
    GBK on the cp936 console.
    """
    sys.stderr.write(f"[cmd] {cmd}\n")
    return subprocess.run(
        cmd,
        shell=True,
        capture_output=capture,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=check,
    )


def ssh(cmd: str, check: bool = True, timeout: int = 300):
    """Run a command on PolyU via SSH.

    `polyu` is an ~/.ssh/config alias, so we must invoke it through the
    `ssh` binary (Windows doesn't treat ssh-config Host aliases as
    executables). We also set BatchMode + a ConnectTimeout so the
    process fails fast instead of hanging on a missing relay VM.
    """
    # Replace any embedded " in cmd with \\" to survive the outer quote.
    quoted = cmd.replace('"', '\\"')
    ssh_cmd = (
        f'ssh -o BatchMode=yes -o ConnectTimeout=10 '
        f'{SSH_TARGET} "{quoted}"'
    )
    return run(ssh_cmd, capture=True, check=check, timeout=timeout)


def scp_to_remote(local_path: str, remote_path: str, check: bool = True):
    """Copy a local file to PolyU via scp."""
    cmd = (
        f'scp -o BatchMode=yes -o ConnectTimeout=10 '
        f'"{local_path}" {SSH_TARGET}:"{remote_path}"'
    )
    return run(cmd, capture=True, check=check)


# ===========================================================================
# Code blocks to insert
# ===========================================================================

# --- D1 __init__ pre-super pop (BEFORE super().__init__(**kwargs)) ---
D1_INIT_PRE_BLOCK = (
    "        # >>> D1: Distillation Truncation config pop (pre-super)\n"
    "        self._d1_distill_trunc_cfg = kwargs.pop('distill_trunc_cfg', {})\n"
    "        # >>> D3: GT Duplication config pop (pre-super)\n"
    "        self._d3_gt_dup_cfg = kwargs.pop('gt_dup_cfg', {})\n"
)

# --- D1 __init__ post-super parse ---
D1_INIT_POST_BLOCK = (
    "        # <<< D1-V3-APPLIED: distill trunc config parse >>>\n"
    "        _d1 = self._d1_distill_trunc_cfg\n"
    "        self._d1_enable = bool(_d1.get('enable', False))\n"
    "        self._distill_trunc_iou_thr = float(_d1.get('iou_thr', 0.1))\n"
    "        self._distill_trunc_weight = float(_d1.get('weight', 0.0))\n"
    "        self._distill_trunc_ns = int(_d1.get('ns', 70))\n"
    "        self._distill_trunc_ne = int(_d1.get('ne', 80))\n"
    "        _d1_mon = _d1.get('monitor', {}) or {}\n"
    "        self._d1_mon_enable = bool(_d1_mon.get('enable', True))\n"
    "        self._d1_mon_path = str(_d1_mon.get('path', '/home/yelingfei/logs/d1_distill_trunc.jsonl'))\n"
    "        self._d1_mon_every = int(_d1_mon.get('every', 250))\n"
    "        self._d1_mon_counter = 0\n"
    "        self._d1_mon_acc_trunc = 0\n"
    "        self._d1_mon_acc_calls = 0\n"
    "        # <<< D3-V3-APPLIED: gt-dup config parse >>>\n"
    "        _d3 = self._d3_gt_dup_cfg\n"
    "        self._gt_dup_enable = bool(_d3.get('enable', False))\n"
    "        self._gt_dup_factor = int(_d3.get('dup_factor', 2))\n"
    "        self._gt_dup_iou_thr = float(_d3.get('iou_thr', 0.1))\n"
    "        self._gt_dup_ns = int(_d3.get('ns', 70))\n"
    "        self._gt_dup_ne = int(_d3.get('ne', 80))\n"
    "        _d3_mon = _d3.get('monitor', {}) or {}\n"
    "        self._d3_mon_enable = bool(_d3_mon.get('enable', True))\n"
    "        self._d3_mon_path = str(_d3_mon.get('path', '/home/yelingfei/logs/d3_gt_dup.jsonl'))\n"
    "        self._d3_mon_every = int(_d3_mon.get('every', 250))\n"
    "        self._d3_mon_counter = 0\n"
    "        self._d3_mon_acc_augmented = 0\n"
    "        self._d3_mon_acc_calls = 0\n"
)

# --- D1 loss() stash (after batch_gt_instances loop) ---
D1_LOSS_STASH_BLOCK = (
    "        # <<< D1-V3-APPLIED: stash GT + metas for distill trunc >>>\n"
    "        self._batch_gt_instances = batch_gt_instances\n"
    "        self._batch_img_metas = batch_img_metas\n"
)

# --- D1 distn apply call (after the second branch) ---
# Inserted AFTER the if/else that built valid_mask_list, BEFORE ori_text_masks = self.ori_text_masks.
D1_DISTN_CALL_BLOCK = (
    "        # <<< D1-V3-APPLIED: distill trunc apply >>>\n"
    "        if getattr(self, '_d1_enable', False) and hasattr(self, '_batch_gt_instances'):\n"
    "            try:\n"
    "                self._distill_trunc_apply(bbox_preds, label_weights,\n"
    "                                           bbox_weights, valid_mask_list)\n"
    "            except Exception as _e_dt:\n"
    "                import warnings as _w_dt\n"
    "                _w_dt.warn(f'[D1] distill trunc skipped: {_e_dt}')\n"
)

# --- D1 method definitions (appended BEFORE def loss_by_feat_new) ---
# Note: methods must be indented 4 spaces to live inside the class body.
D1_METHODS_BLOCK = (
    "    # ===================== D1-V3-APPLIED: methods =====================\n"
    "    def _distill_trunc_apply(self, bbox_preds, label_weights,\n"
    "                             bbox_weights, valid_mask_list):\n"
    "        \"\"\"D1: zero distillation weight for queries whose predicted box\n"
    "        overlaps new-class GT. BUG-FIXED: normalizes GT pixel-xyxy to\n"
    "        [0,1] xyxy before IoU (bbox_preds is [0,1] cxcywh).\n"
    "        \"\"\"\n"
    "        from mmdet.structures.bbox import bbox_overlaps, bbox_cxcywh_to_xyxy\n"
    "        ns, ne = self._distill_trunc_ns, self._distill_trunc_ne\n"
    "        trunc_w = self._distill_trunc_weight\n"
    "        trunc_iou = self._distill_trunc_iou_thr\n"
    "        n_truncated = 0\n"
    "        B = bbox_preds.shape[0]\n"
    "        for bi in range(B):\n"
    "            if bi >= len(self._batch_gt_instances):\n"
    "                continue\n"
    "            gt_inst = self._batch_gt_instances[bi]\n"
    "            gt_labels = gt_inst.labels\n"
    "            gt_bboxes = gt_inst.bboxes\n"
    "            if hasattr(gt_bboxes, 'tensor'):\n"
    "                gt_bboxes = gt_bboxes.tensor\n"
    "            new_mask = (gt_labels >= ns) & (gt_labels < ne)\n"
    "            if not new_mask.any():\n"
    "                continue\n"
    "            new_gt = gt_bboxes[new_mask].to(bbox_preds.device)\n"
    "\n"
    "            # FIX: normalize GT pixel xyxy -> [0,1] xyxy\n"
    "            img_meta = self._batch_img_metas[bi]\n"
    "            h, w = img_meta['img_shape'][:2]\n"
    "            factor = new_gt.new_tensor([w, h, w, h]).unsqueeze(0)\n"
    "            new_gt_norm = new_gt / factor  # [0,1] xyxy\n"
    "\n"
    "            pred_xyxy = bbox_cxcywh_to_xyxy(bbox_preds[bi].detach())  # [0,1] xyxy\n"
    "            ious = bbox_overlaps(pred_xyxy, new_gt_norm)\n"
    "            near_new = (ious >= trunc_iou).any(dim=1)\n"
    "            n_near = int(near_new.sum())\n"
    "            if n_near > 0:\n"
    "                import torch as _t_dt\n"
    "                damp = _t_dt.full_like(label_weights[bi], trunc_w)\n"
    "                keep = _t_dt.ones_like(label_weights[bi])\n"
    "                mask = _t_dt.where(near_new, damp, keep)\n"
    "                label_weights[bi] = label_weights[bi] * mask\n"
    "                bbox_weights[bi] = bbox_weights[bi] * mask\n"
    "                valid_mask_list[bi] = valid_mask_list[bi] * mask\n"
    "                n_truncated += n_near\n"
    "        # monitor\n"
    "        self._d1_mon_acc_trunc += int(n_truncated)\n"
    "        self._d1_mon_acc_calls += 1\n"
    "        self._dt_monitor_log()\n"
    "        return n_truncated\n"
    "\n"
    "    def _dt_monitor_log(self):\n"
    "        \"\"\"D1 monitor: append one JSONL line every N calls.\"\"\"\n"
    "        if not getattr(self, '_d1_mon_enable', False):\n"
    "            return\n"
    "        self._d1_mon_counter += 1\n"
    "        if self._d1_mon_counter < self._d1_mon_every:\n"
    "            return\n"
    "        try:\n"
    "            import json as _j_dt\n"
    "            import os as _os_dt\n"
    "            _os_dt.makedirs(_os_dt.dirname(self._d1_mon_path), exist_ok=True)\n"
    "            avg = (self._d1_mon_acc_trunc /\n"
    "                   max(1, self._d1_mon_acc_calls))\n"
    "            rec = {\n"
    "                't': time.time(),\n"
    "                'iter_group': int(self._d1_mon_counter),\n"
    "                'calls': int(self._d1_mon_acc_calls),\n"
    "                'truncated_total': int(self._d1_mon_acc_trunc),\n"
    "                'truncated_avg': float(avg),\n"
    "            }\n"
    "            with open(self._d1_mon_path, 'a') as fh:\n"
    "                fh.write(_j_dt.dumps(rec) + '\\\\n')\n"
    "            self._d1_mon_counter = 0\n"
    "            self._d1_mon_acc_trunc = 0\n"
    "            self._d1_mon_acc_calls = 0\n"
    "        except Exception:\n"
    "            pass\n"
    "\n"
    "    # ===================== D1-V3-APPLIED: end methods =====================\n"
    "\n"
)

# --- D3 augment call (after loss_dict = dict() in loss_by_feat_new) ---
D3_AUGMENT_CALL_BLOCK = (
    "        # <<< D3-V3-APPLIED: GT duplication augmentation >>>\n"
    "        if getattr(self, '_gt_dup_enable', False):\n"
    "            try:\n"
    "                batch_gt_instances = self._gt_dup_augment(batch_gt_instances)\n"
    "                batch_all_instances = self._gt_dup_augment(batch_all_instances)\n"
    "            except Exception as _e_d3:\n"
    "                import warnings as _w_d3\n"
    "                _w_d3.warn(f'[D3] gt duplication skipped: {_e_d3}')\n"
)

# --- D3 monitor call (before final return of loss_by_feat_new) ---
D3_MONITOR_CALL_BLOCK = (
    "        # <<< D3-V3-APPLIED: monitor >>>\n"
    "        if getattr(self, '_d3_mon_enable', False):\n"
    "            self._gd_monitor_log()\n"
)

# --- D3 method definitions (inserted BEFORE def get_b1_margin_weight_scale) ---
D3_METHODS_BLOCK = (
    "    # ===================== D3-V3-APPLIED: methods =====================\n"
    "    def _gt_dup_augment(self, batch_gt_instances):\n"
    "        \"\"\"D3: duplicate new-class GT instances dup_factor times.\n"
    "        Uses a generic InstanceData field walk so ALL fields propagate\n"
    "        (bboxes, labels, positive_maps, text_token_mask, scores, ...).\n"
    "        \"\"\"\n"
    "        if not getattr(self, '_gt_dup_enable', False):\n"
    "            return batch_gt_instances\n"
    "        dup = max(2, int(self._gt_dup_factor))\n"
    "        ns = self._gt_dup_ns\n"
    "        ne = self._gt_dup_ne\n"
    "        n_extra_total = 0\n"
    "        out = []\n"
    "        for gt_inst in batch_gt_instances:\n"
    "            n_orig = int(gt_inst.labels.shape[0])\n"
    "            if n_orig == 0:\n"
    "                out.append(gt_inst)\n"
    "                continue\n"
    "            new_mask = ((gt_inst.labels >= ns) &\n"
    "                        (gt_inst.labels < ne))\n"
    "            n_new_sum = new_mask.sum()\n"
    "            n_new = int(n_new_sum.item()) if hasattr(n_new_sum, 'item') else int(n_new_sum)\n"
    "            if n_new == 0:\n"
    "                out.append(gt_inst)\n"
    "                continue\n"
    "            new_inst = InstanceData()\n"
    "            for field in gt_inst.keys():\n"
    "                val = getattr(gt_inst, field)\n"
    "                try:\n"
    "                    import torch as _t_d3\n"
    "                    if (isinstance(val, _t_d3.Tensor) and val.dim() >= 1\n"
    "                            and val.size(0) == n_orig):\n"
    "                        sub = val[new_mask]\n"
    "                        rep_shape = [dup - 1] + [1] * (val.dim() - 1)\n"
    "                        extra = sub.repeat(*rep_shape)\n"
    "                        merged = _t_d3.cat([val, extra], dim=0)\n"
    "                        new_inst.set_field(merged, field)\n"
    "                    else:\n"
    "                        new_inst.set_field(val, field)\n"
    "                except Exception:\n"
    "                    new_inst.set_field(val, field)\n"
    "            out.append(new_inst)\n"
    "            n_extra_total += n_new * (dup - 1)\n"
    "        # monitor accumulation\n"
    "        self._d3_mon_acc_augmented += int(n_extra_total)\n"
    "        self._d3_mon_acc_calls += 1\n"
    "        return out\n"
    "\n"
    "    def _gd_monitor_log(self):\n"
    "        \"\"\"D3 monitor: append one JSONL line every N calls.\"\"\"\n"
    "        self._d3_mon_counter += 1\n"
    "        if self._d3_mon_counter < self._d3_mon_every:\n"
    "            return\n"
    "        try:\n"
    "            import json as _j_d3\n"
    "            import os as _os_d3\n"
    "            _os_d3.makedirs(_os_d3.dirname(self._d3_mon_path), exist_ok=True)\n"
    "            avg = (self._d3_mon_acc_augmented /\n"
    "                   max(1, self._d3_mon_acc_calls))\n"
    "            rec = {\n"
    "                't': time.time(),\n"
    "                'iter_group': int(self._d3_mon_counter),\n"
    "                'calls': int(self._d3_mon_acc_calls),\n"
    "                'extra_gt_total': int(self._d3_mon_acc_augmented),\n"
    "                'extra_gt_avg': float(avg),\n"
    "                'dup_factor': int(self._gt_dup_factor),\n"
    "            }\n"
    "            with open(self._d3_mon_path, 'a') as fh:\n"
    "                fh.write(_j_d3.dumps(rec) + '\\\\n')\n"
    "            self._d3_mon_counter = 0\n"
    "            self._d3_mon_acc_augmented = 0\n"
    "            self._d3_mon_acc_calls = 0\n"
    "        except Exception:\n"
    "            pass\n"
    "\n"
    "    # ===================== D3-V3-APPLIED: end methods =====================\n"
    "\n"
)


# ===========================================================================
# Anchor strings (exact match, including indentation/whitespace)
# ===========================================================================

ANCHOR_INIT_PRE = (
    "        self.d2_sa_margin_cfg = kwargs.pop('d2_sa_margin_cfg', {})\n"
)

ANCHOR_INIT_POST = (
    "        self.gm_giou_scale = self.geometric_matching_cfg.get(\"giou_scale\", 2.5)\n"
    "        super().__init__(**kwargs)\n"
    "        self.distn_cfg = distn_cfg\n"
)

ANCHOR_LOSS_STASH = (
    "        batch_gt_instances = []\n"
    "        batch_img_metas = []\n"
    "        for data_sample in batch_data_samples:\n"
    "            batch_img_metas.append(data_sample.metainfo)\n"
    "            batch_gt_instances.append(data_sample.gt_instances)\n"
)

# This matches the ELSE branch (the second occurrence with trailing whitespace).
# Replacement target: insert D1 distn call AFTER the else-branch's
# valid_mask_list[overlap_list] = 0.0 line and BEFORE the blank line + ori_text_masks.
ANCHOR_DISTN_ELSE = (
    "            label_weights[overlap_list] = 0.0  \n"
    "            bbox_weights[overlap_list] = 0.0\n"
    "            valid_mask_list[overlap_list] = 0.0   \n"
    "        \n"
    "        ori_text_masks = self.ori_text_masks\n"
)

# Anchor for inserting D1 method definitions BEFORE def loss_by_feat_new
ANCHOR_D1_METHODS = (
    "                self._last_inter_query_loss = inter_query_loss.detach()\n"
    "\n"
    "        return loss_dict\n"
    "\n"
    "    def loss_by_feat_new(\n"
)

# Anchor for D3 augment call: the entire multi-line destructuring
# statement that opens loss_by_feat_new's body. The augmentation must
# be inserted AFTER this statement, not in the middle of it.
# Note: the denoising continuation line has 9 leading spaces (not 8) in the
# original file, so we build the anchor with an explicit " " * 9 to be safe.
_NINE = " " * 9
ANCHOR_D3_AUGMENT = (
    "        loss_dict = dict()\n"
    "        # extract denoising and matching part of outputs\n"
    "        (all_layers_matching_cls_scores, all_layers_matching_bbox_preds,\n"
    + _NINE + "all_layers_denoising_cls_scores, all_layers_denoising_bbox_preds) = \\\n"
    "            self.split_outputs(all_layers_cls_scores, all_layers_bbox_preds, dn_meta)   \n"
)

# Anchor for D3 monitor call + D3 method definitions.
# The D3 monitor call is inserted right before `return loss_dict` at end of
# loss_by_feat_new, and the D3 method block is inserted right after
# (before def get_b1_margin_weight_scale).
ANCHOR_D3_RETURN = (
    "                loss_dict[f'd{num_dec_layer}.dn_loss_iou'] = loss_iou_i\n"
    "        return loss_dict\n"
    "\n"
    "    def get_b1_margin_weight_scale(self, ref_tensor: Tensor) -> Tensor:\n"
)


# ===========================================================================
# Patcher logic
# ===========================================================================


def fetch_remote_file() -> str:
    """Read the file from PolyU."""
    proc = ssh(f"cat {REMOTE_FILE}", check=True)
    return proc.stdout


def push_remote_file(content: str) -> None:
    """Write the file back to PolyU via a temp local file + scp."""
    fd, tmp_path = tempfile.mkstemp(suffix=".py", prefix="patched_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(content)
        scp_to_remote(tmp_path, REMOTE_FILE, check=True)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def create_remote_backup() -> str:
    """Make a timestamped backup on PolyU."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    ssh(f"mkdir -p {REMOTE_BACKUP_DIR}")
    backup_name = f"gdino_head_inc_gcd.py.bak_{ts}"
    backup_path = f"{REMOTE_BACKUP_DIR}/{backup_name}"
    ssh(f"cp -p {REMOTE_FILE} {backup_path}", check=True)
    # Also keep a sibling .bak_next_to_original for easy diff
    ssh(f"cp -p {REMOTE_FILE} {REMOTE_FILE}.bak_{ts}", check=False)
    return backup_path


def verify_syntax_remote() -> bool:
    """Run py_compile on PolyU using the GCD conda env Python.

    The GCD env was installed at /home/yelingfei/conda_envs/gcd/ (Python 3.8.20)
    — invoke its python directly so we don't depend on conda activation.
    """
    remote_python = "/home/yelingfei/conda_envs/gcd/bin/python"
    py_compile_cmd = (
        'import py_compile; py_compile.compile("'
        + REMOTE_FILE
        + '", doraise=True); print("PYCOMPILE_OK")'
    )
    cmd = (
        f'cd {REMOTE_DIR} && '
        f'{remote_python} -c \'{py_compile_cmd}\''
    )
    proc = ssh(cmd, check=False)
    if proc.stdout and "PYCOMPILE_OK" in proc.stdout:
        return True
    # fallback: system python3
    proc = ssh(
        f'python3 -c \'{py_compile_cmd}\'',
        check=False,
    )
    if proc.stdout and "PYCOMPILE_OK" in proc.stdout:
        return True
    sys.stderr.write(
        "[verify_syntax_remote] gcd-env output: "
        + (proc.stdout or "").strip()[:200]
        + "\n"
    )
    return False


def apply_d1(text: str) -> str:
    """Apply D1 modifications to the file content."""
    if MARKER_D1 in text:
        sys.stderr.write("[D1] marker already present, skipping D1\n")
        return text
    # sanity: ensure all anchors are uniquely present
    for name, anc in [
        ("D1_INIT_PRE", ANCHOR_INIT_PRE),
        ("D1_INIT_POST", ANCHOR_INIT_POST),
        ("D1_LOSS_STASH", ANCHOR_LOSS_STASH),
        ("D1_DISTN_ELSE", ANCHOR_DISTN_ELSE),
        ("D1_METHODS", ANCHOR_D1_METHODS),
    ]:
        if text.count(anc) != 1:
            raise RuntimeError(
                f"[D1] anchor {name} has unexpected count "
                f"{text.count(anc)} (expected 1). Aborting."
            )

    # 1. Init pre block: insert right after ANCHOR_INIT_PRE
    text = text.replace(
        ANCHOR_INIT_PRE,
        ANCHOR_INIT_PRE + D1_INIT_PRE_BLOCK,
        1,
    )

    # 2. Init post block: insert right after ANCHOR_INIT_POST
    text = text.replace(
        ANCHOR_INIT_POST,
        ANCHOR_INIT_POST + D1_INIT_POST_BLOCK,
        1,
    )

    # 3. loss() stash block: insert right after the GT loop
    text = text.replace(
        ANCHOR_LOSS_STASH,
        ANCHOR_LOSS_STASH + D1_LOSS_STASH_BLOCK,
        1,
    )

    # 4. distn apply call: insert right after the else-branch block,
    #    keeping the trailing "ori_text_masks = self.ori_text_masks" line.
    text = text.replace(
        ANCHOR_DISTN_ELSE,
        (
            "            label_weights[overlap_list] = 0.0  \n"
            "            bbox_weights[overlap_list] = 0.0\n"
            "            valid_mask_list[overlap_list] = 0.0   \n"
            + D1_DISTN_CALL_BLOCK
            + "        \n"
            + "        ori_text_masks = self.ori_text_masks\n"
        ),
        1,
    )

    # 5. D1 method definitions: insert right BEFORE def loss_by_feat_new
    text = text.replace(
        ANCHOR_D1_METHODS,
        (
            "                self._last_inter_query_loss = inter_query_loss.detach()\n"
            "\n"
            "        return loss_dict\n"
            "\n"
            + D1_METHODS_BLOCK
            + "    def loss_by_feat_new(\n"
        ),
        1,
    )

    # Mark D1 applied (we'll write combined marker at the end if both succeed)
    return text


def apply_d3(text: str) -> str:
    """Apply D3 modifications to the file content."""
    if MARKER_D3 in text:
        sys.stderr.write("[D3] marker already present, skipping D3\n")
        return text
    for name, anc in [
        ("D3_AUGMENT", ANCHOR_D3_AUGMENT),
        ("D3_RETURN", ANCHOR_D3_RETURN),
    ]:
        if text.count(anc) != 1:
            raise RuntimeError(
                f"[D3] anchor {name} has unexpected count "
                f"{text.count(anc)} (expected 1). Aborting."
            )

    # 1. Augment call: insert right after first 3 lines of loss_by_feat_new body
    text = text.replace(
        ANCHOR_D3_AUGMENT,
        ANCHOR_D3_AUGMENT + D3_AUGMENT_CALL_BLOCK,
        1,
    )

    # 2. Monitor call + method definitions: split the return anchor.
    text = text.replace(
        ANCHOR_D3_RETURN,
        (
            "                loss_dict[f'd{num_dec_layer}.dn_loss_iou'] = loss_iou_i\n"
            + D3_MONITOR_CALL_BLOCK
            + "        return loss_dict\n"
            + "\n"
            + D3_METHODS_BLOCK
            + "    def get_b1_margin_weight_scale(self, ref_tensor: Tensor) -> Tensor:\n"
        ),
        1,
    )

    return text


def add_combined_marker(text: str) -> str:
    """Insert a small marker comment after the module docstring/imports
    so the patch is detectable on re-runs. Only adds marker if absent."""
    if MARKER_BOTH in text:
        return text
    marker_line = (
        f"# {MARKER_BOTH}  (D1 distill-trunc + D3 gt-dup; applied by patch_d1d3_v3.py)\n"
    )
    # Insert after the first line of the file (after copyright line)
    lines = text.split("\n")
    # find first non-blank line and insert after it
    insert_at = 0
    for i, ln in enumerate(lines[:5]):
        if ln.strip():
            insert_at = i + 1
            break
    lines.insert(insert_at, marker_line.rstrip("\n"))
    return "\n".join(lines)


def check_status(text: str) -> None:
    """Print current patch status."""
    has_d1 = MARKER_D1 in text
    has_d3 = MARKER_D3 in text
    has_both = MARKER_BOTH in text
    print(f"file size: {len(text)} chars")
    print(f"has D1 marker ({MARKER_D1}): {has_d1}")
    print(f"has D3 marker ({MARKER_D3}): {has_d3}")
    print(f"has combined marker ({MARKER_BOTH}): {has_both}")
    # also probe for characteristic strings
    d1_method = "_distill_trunc_apply" in text
    d3_method = "_gt_dup_augment" in text
    print(f"has D1 method (_distill_trunc_apply): {d1_method}")
    print(f"has D3 method (_gt_dup_augment): {d3_method}")


# ===========================================================================
# Main
# ===========================================================================


def main():
    parser = argparse.ArgumentParser(description="D1+D3 patcher for GCD head")
    parser.add_argument(
        "--check", action="store_true",
        help="only report current status, do not modify",
    )
    parser.add_argument(
        "--no-d1", action="store_true",
        help="skip D1 patch",
    )
    parser.add_argument(
        "--no-d3", action="store_true",
        help="skip D3 patch",
    )
    parser.add_argument(
        "--skip-syntax", action="store_true",
        help="skip remote py_compile (not recommended)",
    )
    args = parser.parse_args()

    print(f"[1/6] Reading {REMOTE_FILE} from {SSH_TARGET} ...")
    text = fetch_remote_file()
    print(f"      fetched {len(text)} chars")
    check_status(text)

    if args.check:
        return 0

    if MARKER_BOTH in text:
        print(f"[ok] combined marker {MARKER_BOTH} already present; nothing to do.")
        return 0

    print("[2/6] Creating timestamped backup on PolyU ...")
    backup_path = create_remote_backup()
    print(f"      backup: {backup_path}")

    # Work on a local copy of the text. If anything fails, we abort before push.
    new_text = text
    try:
        if not args.no_d1:
            print("[3/6] Applying D1 (Distillation Truncation, bug-fixed) ...")
            new_text = apply_d1(new_text)
        if not args.no_d3:
            print("[4/6] Applying D3 (GT Duplication) ...")
            new_text = apply_d3(new_text)
    except RuntimeError as e:
        print(f"[FAIL] patch aborted: {e}", file=sys.stderr)
        print(f"       remote file is unchanged. backup at {backup_path}",
              file=sys.stderr)
        return 2

    # Insert per-feature markers as comments near top if not present.
    if MARKER_D1 not in new_text:
        new_text = insert_feature_marker(new_text, MARKER_D1)
    if MARKER_D3 not in new_text:
        new_text = insert_feature_marker(new_text, MARKER_D3)
    # Insert combined marker
    new_text = add_combined_marker(new_text)

    # Quick sanity: ensure text still contains essential anchors (no overlap)
    for essential in (
        "def __init__(self, distn_cfg",
        "def loss_by_feat_old(",
        "def loss_by_feat_new(",
        "def loss_by_feat_ld_distn_single(",
        "def get_b1_margin_weight_scale(self, ref_tensor: Tensor) -> Tensor:",
        "ori_text_masks = self.ori_text_masks",
        "return loss_dict",
    ):
        if essential not in new_text:
            print(f"[FAIL] post-patch sanity: missing '{essential}'",
                  file=sys.stderr)
            return 3

    # Write to a local temp and run py_compile locally first (cheap check)
    print("[5/6] Local py_compile pre-check ...")
    fd, tmp_path = tempfile.mkstemp(suffix=".py", prefix="patched_check_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(new_text)
        # Write a tiny wrapper script that imports py_compile and compiles
        # the patched file. Doing it this way (rather than `python -c`)
        # sidesteps Windows backslash escaping in the temp path.
        fd2, wrapper_path = tempfile.mkstemp(
            suffix=".py", prefix="pyc_wrapper_")
        try:
            with os.fdopen(fd2, "w", encoding="utf-8") as fh:
                fh.write(
                    "import py_compile, sys\n"
                    f"py_compile.compile({tmp_path!r}, doraise=True)\n"
                    "print('OK')\n"
                )
            local_proc = run(
                f'python "{wrapper_path}"',
                check=False,
            )
        finally:
            try:
                os.unlink(wrapper_path)
            except OSError:
                pass
        if "OK" not in (local_proc.stdout or ""):
            print("[FAIL] local py_compile failed:", file=sys.stderr)
            print(local_proc.stdout, file=sys.stderr)
            print(local_proc.stderr, file=sys.stderr)
            return 4
        print("      local py_compile OK")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # Push
    print("[6/6] Writing patched file back to PolyU ...")
    push_remote_file(new_text)
    print("      pushed.")

    if not args.skip_syntax:
        print("[verify] Remote py_compile ...")
        ok = verify_syntax_remote()
        if ok:
            print("        remote py_compile OK")
        else:
            print("[WARN] remote py_compile did not report OK; check env.",
                  file=sys.stderr)
            print("       The file was still written. Inspect manually.",
                  file=sys.stderr)

    print()
    print("=================== PATCH SUMMARY ===================")
    print(f"backup : {backup_path}")
    print(f"target : {REMOTE_FILE}")
    print(f"D1     : {'applied' if (not args.no_d1) else 'skipped'}")
    print(f"D3     : {'applied' if (not args.no_d3) else 'skipped'}")
    print(f"marker : {MARKER_BOTH}")
    print("=====================================================")
    return 0


def insert_feature_marker(text: str, marker: str) -> str:
    """Insert a feature-level marker comment near the top of the file."""
    if marker in text:
        return text
    marker_line = f"# {marker}\n"
    lines = text.split("\n")
    insert_at = 0
    for i, ln in enumerate(lines[:5]):
        if ln.strip():
            insert_at = i + 1
            break
    lines.insert(insert_at, marker_line.rstrip("\n"))
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
