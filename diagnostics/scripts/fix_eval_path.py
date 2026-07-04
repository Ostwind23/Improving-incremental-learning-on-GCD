#!/usr/bin/env python3
"""Fix: pop memory_raw in eval path too."""
import py_compile
DET = '/home/yelingfei/projects/GCD/mmdet/models/detectors/gdino_inc_gcd.py'
lines = open(DET, encoding='utf-8').read().split('\n')

for i, line in enumerate(lines):
    if 'tmp_dec_in, head_inputs_dict = self.pre_decoder(' in line:
        indent = '            '
        lines.insert(i, indent + "encoder_outputs_dict.pop('memory_raw', None)")
        print('Inserted memory_raw pop before pre_decoder (eval) at line', i+1)
        break

open(DET, 'w', encoding='utf-8').write('\n'.join(lines))
py_compile.compile(DET, doraise=True)
print('SYNTAX OK')
