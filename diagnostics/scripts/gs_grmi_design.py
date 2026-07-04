#!/usr/bin/env python3
"""
Gradient-Selective GRMI (GS-GRMI) patcher.

Patches gdino_inc_gcd.py to implement:
  - R(M) only receives new-class detection loss gradients
  - Old-class detection + distillation gradients do NOT flow through R(M)
  - R(M) norm suppression loss to prevent unconstrained growth
  - JSONL gradient monitoring

Implementation: override loss() to do gradient surgery after
bbox_head.loss() returns. Specifically:
  1. Separate loss_dict into new-class-detect vs old-detect+distill
  2. Compute sum of old-detect+distill losses
  3. Use torch.autograd.grad on old+distill sum w.r.t. R(M) params
  4. After full backward, subtract the old+distill grad from R(M) params
     (equivalent to: R(M) only gets new-class detect gradient)

Alternative (simpler): stop_gradient on R(M) output for the old branch.
Since forward_transformer already splits into new_branch (with R(M)) and
old_branch (with memory_raw in decouple mode), we can:
  - Keep decouple: old branch uses memory_raw (no R(M) gradient from distillation)
  - For detection loss: need to separate new-class vs old-class

But detection loss goes through new_branch decoder which uses M' = M + γR(M)
for ALL queries (new and old class queries share the same M').
So we can't trivially split at the forward level.

ACTUAL approach: hook-based gradient zeroing.
Register a backward hook on R(M).net output that zeros gradient
components attributable to old-class losses.

SIMPLEST correct approach:
  After full backward(), manually set R(M) params grad to only
  the new-class-detect component, computed via separate autograd.grad.
"""
import py_compile
import textwrap

DET = '/home/yelingfei/projects/GCD/mmdet/models/detectors/gdino_inc_gcd.py'


def patch():
    code = open(DET, encoding='utf-8').read()
    lines = code.split('\n')

    # ================================================================
    # PATCH 1: Add config option for GS-GRMI
    # Find __init__ where residual_inject_cfg is handled
    # ================================================================

    # ================================================================
    # PATCH 2: Override loss() to do gradient surgery
    # The cleanest approach: after all losses are computed, before returning,
    # tag which loss keys are new-class-detect vs old-detect+distill.
    #
    # Then in train_step (or via a hook), do:
    #   1. L_total.backward()
    #   2. Compute L_old_distill = sum of old+distill losses
    #   3. grad_old = autograd.grad(L_old_distill, R(M).parameters(), retain_graph=True)
    #   4. For each R(M) param: p.grad -= grad_old  (remove old+distill component)
    #
    # But this requires retain_graph=True which is expensive.
    #
    # BETTER: use a gradient scaling hook on R(M) output.
    # In forward: compute scale = L_new / L_total for this iter.
    # Register backward hook on R(M) output: multiply gradient by scale.
    # Effect: R(M) gradient ≈ proportional to new-class contribution.
    #
    # BEST (exact, cheap):
    # In forward_encoder, where M' = M + γ*R(M):
    #   residual = self.residual_inject.net(memory)
    #   M' = memory + gamma * residual
    #
    # Register a BACKWARD HOOK on `residual` that scales gradient.
    # The hook runs during backward and can zero out gradient from old losses.
    #
    # But we don't know old/new loss ratio at forward time.
    # Solution: use loss ratio from PREVIOUS iteration as proxy.
    # ================================================================

    # Actually, the simplest correct approach:
    # Two-step backward.
    # Step 1: compute R(M) grad from new-class detection loss only
    # Step 2: full backward for all other params
    # Step 3: overwrite R(M) grad with step 1 result

    # This is implemented by modifying the detector's loss() to return
    # an annotated losses dict, and adding a custom train_step or hook.

    # Let me implement it as a method in the detector that gets called
    # after mmengine's parse_losses.

    print("=== GS-GRMI Implementation Plan ===")
    print()
    print("Approach: register_full_backward_hook on residual_inject.net")
    print()
    print("In forward_encoder:")
    print("  residual = self.residual_inject.net(memory)")
    print("  # register hook that will scale gradient")
    print("  if self.training and self._gs_grmi_enabled:")
    print("    residual.register_hook(self._gs_grmi_grad_hook)")
    print("  M' = memory + gamma * residual")
    print()
    print("_gs_grmi_grad_hook(grad):")
    print("  # grad flows from ALL losses through M'-> residual")
    print("  # We want to keep only the new-class detection portion")
    print("  # Use the ratio new_detect_loss / total_loss from current iter")
    print("  # This ratio is computed in loss() and stored as self._gs_ratio")
    print("  return grad * self._gs_ratio")
    print()
    print("In loss():")
    print("  losses = bbox_head.loss(...)")
    print("  # Identify new-class detection losses vs old+distill")
    print("  # detection: loss_cls, loss_bbox, loss_iou, d0-d4 variants, enc_loss_*")
    print("  # distillation: loss_ld_cls, loss_ld_bbox, loss_ld_iou, inter_*")
    print("  # new_detect_loss = sum of detection losses")
    print("  # total_loss = sum of all losses")
    print("  # self._gs_ratio = new_detect_loss.detach() / total_loss.detach()")
    print()
    print("PROBLEM with ratio approach:")
    print("  Detection loss includes BOTH new-class and old-class queries")
    print("  loss_cls/loss_bbox/loss_iou are computed on ALL matched queries")
    print("  We need to know what fraction of detection loss is from new-class GT")
    print("  This requires splitting loss_by_feat_single by class")
    print()
    print("=== REVISED APPROACH: Split detection loss by class ===")
    print("In loss_by_feat_new, after Hungarian matching:")
    print("  For each matched query, check if its GT label >= 70")
    print("  Compute separate loss_cls_new, loss_cls_old")
    print("  Return both in the loss dict")
    print("  Then _gs_ratio = L_detect_new / (L_detect_new + L_detect_old + L_distill)")


patch()
