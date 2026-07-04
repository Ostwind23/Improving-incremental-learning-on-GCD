#!/usr/bin/env python3
"""Surgical patcher: add TATRI (text-side residual injection) to
GroundingDINO_inc_gcd on PolyU. Idempotent: skips if already patched.

PREREQUISITE: GRMI patch must already be applied (forward_encoder override exists).

Changes:
  1. Copy text_residual_inject.py to mmdet/models/utils/
  2. Add import TextResidualInject
  3. __init__ signature += tatri_cfg
  4. __init__ body: build self.text_residual_inject + compute new_text_token_mask
  5. Extend forward_encoder to apply text-side injection on memory_text
  6. Add TATRI metrics to existing _training_monitor_log (if it exists)
"""
import os, sys, datetime, shutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GCD_ROOT = "/home/yelingfei/projects/GCD"
DETECTOR_PATH = os.path.join(GCD_ROOT, "mmdet/models/detectors/gdino_inc_gcd.py")
UTILS_DIR = os.path.join(GCD_ROOT, "mmdet/models/utils")

# --- Step 0: Copy module file ---
module_src = os.path.join(SCRIPT_DIR, "text_residual_inject.py")
module_dst = os.path.join(UTILS_DIR, "text_residual_inject.py")
if os.path.exists(module_src):
    shutil.copy2(module_src, module_dst)
    print(f"[patch] copied {module_src} -> {module_dst}")
elif os.path.exists(module_dst):
    print(f"[patch] module already at {module_dst}")
else:
    print(f"[patch] ERROR: cannot find text_residual_inject.py")
    sys.exit(1)

src = open(DETECTOR_PATH, encoding="utf-8").read()

if "text_residual_inject" in src and "TextResidualInject" in src:
    print("[patch] TATRI already applied; no-op")
    sys.exit(0)

backup = DETECTOR_PATH + ".bak_tatri_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
open(backup, "w", encoding="utf-8").write(src)
print(f"[patch] backup -> {backup}")

changes = 0

# --- 1. Import ---
anchor_imp = "from ..utils.residual_inject import ResidualInject"
if anchor_imp in src:
    src = src.replace(
        anchor_imp,
        anchor_imp + "\nfrom ..utils.text_residual_inject import TextResidualInject")
    changes += 1
    print("[patch] 1/5: added TextResidualInject import")
else:
    print("[patch] 1/5: SKIP (GRMI import anchor not found — is GRMI patched?)")

# --- 2. __init__ signature: add tatri_cfg ---
# Find the residual_inject_cfg line and add tatri_cfg after it
old_sig = "                 residual_inject_cfg: OptConfigType = None, **kwargs) -> None:"
new_sig = ("                 residual_inject_cfg: OptConfigType = None,\n"
           "                 tatri_cfg: OptConfigType = None, **kwargs) -> None:")
if "tatri_cfg" not in src and old_sig in src:
    src = src.replace(old_sig, new_sig, 1)
    changes += 1
    print("[patch] 2/5: added tatri_cfg to __init__ signature")
else:
    # Try alternate signature forms
    old_sig2 = "                 residual_inject_cfg: OptConfigType = None,\n                 dec_aux_cfg: OptConfigType = None, **kwargs) -> None:"
    new_sig2 = ("                 residual_inject_cfg: OptConfigType = None,\n"
                "                 dec_aux_cfg: OptConfigType = None,\n"
                "                 tatri_cfg: OptConfigType = None, **kwargs) -> None:")
    if "tatri_cfg" not in src and old_sig2 in src:
        src = src.replace(old_sig2, new_sig2, 1)
        changes += 1
        print("[patch] 2/5: added tatri_cfg to __init__ signature (after dec_aux_cfg)")
    else:
        print("[patch] 2/5: SKIP (signature pattern not found or already present)")

# --- 3. Build text_residual_inject in __init__ ---
# Insert right after the GRMI print line
grmi_print = '            print(f"[GRMI] enabled:'
if grmi_print in src and 'text_residual_inject' not in src:
    # Find end of GRMI print block
    idx = src.index(grmi_print)
    nl = src.index('\n', idx)
    insert_pos = nl + 1

    tatri_build = '''
        # --- TATRI: text-side gated residual injection ---
        self.tatri_cfg_obj = Config._dict_to_config_dict_lazy(
            tatri_cfg or dict(enable=False))
        self.text_residual_inject = None
        self._new_text_token_mask = None
        if bool(self.tatri_cfg_obj.get('enable', False)):
            self.text_residual_inject = TextResidualInject(
                in_dim=int(self.tatri_cfg_obj.get('in_dim', 256)),
                hidden_dim=int(self.tatri_cfg_obj.get('hidden_dim', 64)),
                gamma_init=float(self.tatri_cfg_obj.get('gamma_init', 0.05)),
                mode=str(self.tatri_cfg_obj.get('mode', 'fixed_gamma')),
                gate_hidden=int(self.tatri_cfg_obj.get('gate_hidden', 32)))
'''
    src = src[:insert_pos] + tatri_build + src[insert_pos:]
    changes += 1
    print("[patch] 3/5: added TATRI build block in __init__")
else:
    if 'text_residual_inject' in src:
        print("[patch] 3/5: SKIP (already present)")
    else:
        print("[patch] 3/5: SKIP (GRMI print anchor not found)")

# --- 4. Extend forward_encoder to inject text-side residual ---
# The existing GRMI forward_encoder looks like:
#   def forward_encoder(self, *args, **kwargs):
#       encoder_outputs_dict = super().forward_encoder(*args, **kwargs)
#       if self.residual_inject is not None and ...
#           memory = encoder_outputs_dict['memory']
#           encoder_outputs_dict['memory'] = self.residual_inject(memory)
#       return encoder_outputs_dict

# We need to add TATRI before the return:
old_return = "        return encoder_outputs_dict\n\n    def forward_transformer("
tatri_inject = """        # TATRI: text-side gated residual injection
        if self.text_residual_inject is not None and \\
                (self.training or True):
            mt = encoder_outputs_dict['memory_text']
            # Build new-class token mask on first call
            if self._new_text_token_mask is None and hasattr(self, 'token_positive_maps'):
                tpm = self.token_positive_maps
                T_len = mt.shape[1]
                mask = torch.zeros(T_len, device=mt.device)
                for k, positions in tpm.items():
                    cls_id = k - 1
                    if 70 <= cls_id < 80:
                        for p in positions:
                            if p < T_len:
                                mask[p] = 1.0
                self._new_text_token_mask = mask
                print(f"[TATRI] new-class mask: {int(mask.sum())}/{T_len} tokens")
            encoder_outputs_dict['memory_text'] = self.text_residual_inject(
                mt, self._new_text_token_mask)
        return encoder_outputs_dict

    def forward_transformer("""

if "text_residual_inject" not in src.split("forward_encoder")[1].split("forward_transformer")[0] \
   and old_return in src:
    src = src.replace(old_return, tatri_inject, 1)
    changes += 1
    print("[patch] 4/5: extended forward_encoder with TATRI injection")
else:
    print("[patch] 4/5: SKIP (TATRI injection already present or anchor not found)")

# --- 5. Add TATRI metrics to monitor ---
# Find the monitor JSONL write section and add tatri gamma / gate stats
monitor_anchor = "        # GRMI gamma\n"
if monitor_anchor in src and 'tatri_gamma' not in src:
    tatri_monitor = """        # TATRI metrics
        if self.text_residual_inject is not None:
            tri = self.text_residual_inject
            if hasattr(tri, 'gamma'):
                record['tatri_gamma'] = round(float(tri.gamma.detach().item()), 6)
            if hasattr(tri, 'gate') and self._new_text_token_mask is not None:
                # Log mean gate value at new-class tokens (training snapshot)
                try:
                    with torch.no_grad():
                        mt = new_head_inputs_dict.get('memory_text', None)
                        if mt is not None:
                            g = tri.gate(mt)  # (bs, T, 1)
                            mask = self._new_text_token_mask
                            new_g = g[0, mask > 0.5, 0]
                            if len(new_g) > 0:
                                record['tatri_gate_mean'] = round(float(new_g.mean()), 4)
                                record['tatri_gate_min'] = round(float(new_g.min()), 4)
                                record['tatri_gate_max'] = round(float(new_g.max()), 4)
                except Exception:
                    pass

"""
    src = src.replace(monitor_anchor, monitor_anchor + tatri_monitor, 1)
    changes += 1
    print("[patch] 5/5: added TATRI metrics to monitor")
else:
    print("[patch] 5/5: SKIP (monitor anchor not found or already patched)")

open(DETECTOR_PATH, "w", encoding="utf-8").write(src)
print(f"\n[patch] applied {changes}/5 changes")
