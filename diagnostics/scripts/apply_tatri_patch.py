#!/usr/bin/env python3
"""Apply TATRI patch to gdino_inc_gcd.py properly (multiline safe)."""
import sys

p = "/home/yelingfei/projects/GCD/mmdet/models/detectors/gdino_inc_gcd.py"

# Restore from backup first
import shutil
backup = p + ".bak_tatri_20260627_225941"
shutil.copy2(backup, p)
print("Restored from backup")

src = open(p, encoding="utf-8").read()
changes = 0

# 1. Import
anchor = "from ..utils.residual_inject import ResidualInject"
if "TextResidualInject" not in src and anchor in src:
    src = src.replace(anchor, anchor + "\nfrom ..utils.text_residual_inject import TextResidualInject")
    changes += 1
    print("1. import added")

# 2. Signature
old_sig = "                 dec_aux_cfg: OptConfigType = None, **kwargs) -> None:"
new_sig = ("                 dec_aux_cfg: OptConfigType = None,\n"
           "                 tatri_cfg: OptConfigType = None, **kwargs) -> None:")
if "tatri_cfg" not in src and old_sig in src:
    src = src.replace(old_sig, new_sig, 1)
    changes += 1
    print("2. signature added")

# 3. Build block after GRMI
grmi_end = 'f"act_inference={_ri.act_inference}")'
if grmi_end in src and "tatri_cfg_obj" not in src:
    idx = src.index(grmi_end) + len(grmi_end)
    nl = src.index("\n", idx)
    block = (
        "\n"
        "\n"
        "        # --- TATRI: text-side gated residual injection ---\n"
        "        self.tatri_cfg_obj = Config._dict_to_config_dict_lazy(\n"
        "            tatri_cfg or dict(enable=False))\n"
        "        self.text_residual_inject = None\n"
        "        self._new_text_token_mask = None\n"
        "        if bool(self.tatri_cfg_obj.get('enable', False)):\n"
        "            self.text_residual_inject = TextResidualInject(\n"
        "                in_dim=int(self.tatri_cfg_obj.get('in_dim', 256)),\n"
        "                hidden_dim=int(self.tatri_cfg_obj.get('hidden_dim', 64)),\n"
        "                gamma_init=float(self.tatri_cfg_obj.get('gamma_init', 0.05)),\n"
        "                mode=str(self.tatri_cfg_obj.get('mode', 'fixed_gamma')),\n"
        "                gate_hidden=int(self.tatri_cfg_obj.get('gate_hidden', 32)))\n"
    )
    src = src[:nl+1] + block + src[nl+1:]
    changes += 1
    print("3. build block added")

# 4. forward_encoder extension
old_ret = "        return encoder_outputs_dict\n\n    def forward_transformer("
tatri_fe = (
    "        # TATRI: text-side gated residual injection\n"
    "        if self.text_residual_inject is not None and \\\n"
    "                (self.training or True):\n"
    "            mt = encoder_outputs_dict['memory_text']\n"
    "            if self._new_text_token_mask is None and hasattr(self, 'token_positive_maps'):\n"
    "                tpm = self.token_positive_maps\n"
    "                T_len = mt.shape[1]\n"
    "                mask = mt.new_zeros(T_len)\n"
    "                for k, positions in tpm.items():\n"
    "                    cls_id = k - 1\n"
    "                    if 70 <= cls_id < 80:\n"
    "                        for pos in positions:\n"
    "                            if pos < T_len:\n"
    "                                mask[pos] = 1.0\n"
    "                self._new_text_token_mask = mask\n"
    "                print(f'[TATRI] new-class mask: {int(mask.sum())}/{T_len} tokens')\n"
    "            encoder_outputs_dict['memory_text'] = self.text_residual_inject(\n"
    "                mt, self._new_text_token_mask)\n"
    "        return encoder_outputs_dict\n"
    "\n"
    "    def forward_transformer(\n"
)
fe_section = src.split("def forward_encoder")[1].split("def forward_transformer")[0]
if "text_residual_inject" not in fe_section and old_ret in src:
    src = src.replace(old_ret, tatri_fe, 1)
    changes += 1
    print("4. forward_encoder extended")

# 5. Monitor
monitor_anchor = "            record['grmi_gamma'] = round(float(self.residual_inject.gamma.detach().item()), 6)\n"
if monitor_anchor in src and "tatri_gamma" not in src:
    tatri_mon = (
        "\n"
        "            # TATRI metrics\n"
        "            if self.text_residual_inject is not None:\n"
        "                tri = self.text_residual_inject\n"
        "                if hasattr(tri, 'gamma'):\n"
        "                    record['tatri_gamma'] = round(float(tri.gamma.detach().item()), 6)\n"
        "                if hasattr(tri, 'gate') and self._new_text_token_mask is not None:\n"
        "                    try:\n"
        "                        with torch.no_grad():\n"
        "                            mt_snap = new_head_inputs_dict.get('memory_text', None)\n"
        "                            if mt_snap is not None:\n"
        "                                g = tri.gate(mt_snap)\n"
        "                                msk = self._new_text_token_mask\n"
        "                                new_g = g[0, msk > 0.5, 0]\n"
        "                                if len(new_g) > 0:\n"
        "                                    record['tatri_gate_mean'] = round(float(new_g.mean()), 4)\n"
        "                                    record['tatri_gate_min'] = round(float(new_g.min()), 4)\n"
        "                                    record['tatri_gate_max'] = round(float(new_g.max()), 4)\n"
        "                    except Exception:\n"
        "                        pass\n"
    )
    src = src.replace(monitor_anchor, monitor_anchor + tatri_mon, 1)
    changes += 1
    print("5. monitor added")

open(p, "w", encoding="utf-8").write(src)
print(f"\nApplied {changes}/5 changes")
