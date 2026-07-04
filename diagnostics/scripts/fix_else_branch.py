#!/usr/bin/env python3
"""Add else branch for memory_raw when residual_inject is None."""
import py_compile
DET = '/home/yelingfei/projects/GCD/mmdet/models/detectors/gdino_inc_gcd.py'
lines = open(DET, encoding='utf-8').read().split('\n')

for i, line in enumerate(lines):
    if '# TATRI: text-side' in line:
        lines.insert(i, "        else:")
        lines.insert(i+1, "            encoder_outputs_dict['memory_raw'] = encoder_outputs_dict['memory']")
        print('Inserted else branch at line', i+1)
        break

open(DET, 'w', encoding='utf-8').write('\n'.join(lines))
py_compile.compile(DET, doraise=True)
print('SYNTAX OK')
