#!/usr/bin/env python3
"""Fix: apply Change 2 (decouple old branch) by line number."""
DET = '/home/yelingfei/projects/GCD/mmdet/models/detectors/gdino_inc_gcd.py'
lines = open(DET, encoding='utf-8').read().split('\n')

# Find "# forward on oldtext"
target = None
for i, line in enumerate(lines):
    if '# forward on oldtext' in line:
        target = i
        break

if target is None:
    print('ERROR: marker not found')
    exit(1)

print('Found old branch at line', target + 1)

# Replace lines target to target+4:
# Original:
#   # forward on oldtext
#   encoder_outputs_dict['text_token_mask'] = self.ori_text_masks
#
#   old_tmp_dec_in, old_head_inputs_dict = self.pre_decoder_old(
#       **encoder_outputs_dict, ...)
# New:
#   # forward on oldtext — use memory_raw to decouple R(M) from distillation
#   encoder_outputs_for_old = encoder_outputs_dict.copy()
#   encoder_outputs_for_old['memory'] = encoder_outputs_dict['memory_raw']
#   encoder_outputs_for_old['text_token_mask'] = self.ori_text_masks
#
#   old_tmp_dec_in, old_head_inputs_dict = self.pre_decoder_old(
#       **encoder_outputs_for_old, ...)

new_lines = [
    '            # forward on oldtext -- use memory_raw to decouple R(M) from distillation',
    '            encoder_outputs_for_old = encoder_outputs_dict.copy()',
    "            encoder_outputs_for_old['memory'] = encoder_outputs_dict['memory_raw']",
    "            encoder_outputs_for_old['text_token_mask'] = self.ori_text_masks",
    '',
    '            old_tmp_dec_in, old_head_inputs_dict = self.pre_decoder_old(',
    '                **encoder_outputs_for_old, aux_dict=aux_dict, batch_data_samples=batch_data_samples)',
]

# Find how many lines to replace (from target to the pre_decoder_old call)
end = target
for i in range(target, min(target + 10, len(lines))):
    if 'pre_decoder_old(' in lines[i]:
        end = i + 1
        break

print('Replacing lines %d-%d (%d lines) with %d new lines' % (target+1, end+1, end-target, len(new_lines)))
lines[target:end] = new_lines

open(DET, 'w', encoding='utf-8').write('\n'.join(lines))

import py_compile
py_compile.compile(DET, doraise=True)
print('SYNTAX OK')
