#!/usr/bin/env python3
"""
Patch: decouple R(M) from distillation branch.
Detection branch uses memory_enhanced = M + gamma*R(M).
Distillation branch uses memory_raw = M (no R(M)).
This prevents R(M) from destabilizing topology distillation.
"""
import py_compile

DET = '/home/yelingfei/projects/GCD/mmdet/models/detectors/gdino_inc_gcd.py'
src = open(DET, encoding='utf-8').read()

if 'memory_raw' in src:
    print('Already patched, skip')
    exit(0)

# === Change 1: forward_encoder saves memory_raw ===
old_inject = """        if self.residual_inject is not None and \\
                (self.training or self.residual_inject.act_inference):
            memory = encoder_outputs_dict['memory']
            encoder_outputs_dict['memory'] = self.residual_inject(memory)"""

new_inject = """        if self.residual_inject is not None and \\
                (self.training or self.residual_inject.act_inference):
            memory = encoder_outputs_dict['memory']
            encoder_outputs_dict['memory_raw'] = memory
            encoder_outputs_dict['memory'] = self.residual_inject(memory)
        else:
            encoder_outputs_dict['memory_raw'] = encoder_outputs_dict['memory']"""

if old_inject in src:
    src = src.replace(old_inject, new_inject, 1)
    print('Change 1: memory_raw saved in forward_encoder')
else:
    print('ERROR: forward_encoder injection pattern not found')
    exit(1)

# === Change 2: old (distillation) branch uses memory_raw ===
old_old_branch = """            # forward on oldtext
            encoder_outputs_dict['text_token_mask'] = self.ori_text_masks

            old_tmp_dec_in, old_head_inputs_dict = self.pre_decoder_old(
                **encoder_outputs_dict, aux_dict=aux_dict, batch_data_samples=batch_data_samples)"""

new_old_branch = """            # forward on oldtext — use memory_raw to decouple R(M) from distillation
            encoder_outputs_for_old = encoder_outputs_dict.copy()
            encoder_outputs_for_old['memory'] = encoder_outputs_dict['memory_raw']
            encoder_outputs_for_old['text_token_mask'] = self.ori_text_masks

            old_tmp_dec_in, old_head_inputs_dict = self.pre_decoder_old(
                **encoder_outputs_for_old, aux_dict=aux_dict, batch_data_samples=batch_data_samples)"""

if old_old_branch in src:
    src = src.replace(old_old_branch, new_old_branch, 1)
    print('Change 2: distillation branch uses memory_raw')
else:
    print('ERROR: old branch pattern not found')
    exit(1)

open(DET, 'w', encoding='utf-8').write(src)
py_compile.compile(DET, doraise=True)
print('SYNTAX OK')
