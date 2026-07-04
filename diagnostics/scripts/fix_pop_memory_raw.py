#!/usr/bin/env python3
"""Fix: pop memory_raw before passing encoder_outputs_dict to pre_decoder_new."""
import py_compile
DET = '/home/yelingfei/projects/GCD/mmdet/models/detectors/gdino_inc_gcd.py'
lines = open(DET, encoding='utf-8').read().split('\n')

# Find: new_tmp_dec_in, new_head_inputs_dict = self.pre_decoder_new(
# Insert before it: save and pop memory_raw
for i, line in enumerate(lines):
    if 'new_tmp_dec_in, new_head_inputs_dict = self.pre_decoder_new(' in line:
        indent = '            '
        lines.insert(i, indent + "_memory_raw = encoder_outputs_dict.pop('memory_raw', None)")
        print('Inserted memory_raw pop before pre_decoder_new at line', i+1)
        break

# Find: encoder_outputs_for_old['memory'] = encoder_outputs_dict['memory_raw']
# Change to use _memory_raw
for i, line in enumerate(lines):
    if "encoder_outputs_for_old['memory'] = encoder_outputs_dict['memory_raw']" in line:
        lines[i] = line.replace("encoder_outputs_dict['memory_raw']", "_memory_raw")
        print('Fixed memory_raw reference in old branch at line', i+1)
        break

# Also need to restore memory_raw after pre_decoder_new for the old branch
# Actually _memory_raw is already saved, just use it directly. No restore needed.

open(DET, 'w', encoding='utf-8').write('\n'.join(lines))
py_compile.compile(DET, doraise=True)
print('SYNTAX OK')
