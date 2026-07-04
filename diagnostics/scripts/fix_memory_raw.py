#!/usr/bin/env python3
"""Add memory_raw to forward_encoder."""
import py_compile
DET = '/home/yelingfei/projects/GCD/mmdet/models/detectors/gdino_inc_gcd.py'
lines = open(DET, encoding='utf-8').read().split('\n')

# Find: encoder_outputs_dict['memory'] = self.residual_inject(memory)
# Insert before it: encoder_outputs_dict['memory_raw'] = memory
inserted_raw = False
for i, line in enumerate(lines):
    if "encoder_outputs_dict['memory'] = self.residual_inject(memory)" in line:
        indent = '            '
        lines.insert(i, indent + "encoder_outputs_dict['memory_raw'] = memory")
        inserted_raw = True
        print('Inserted memory_raw before residual_inject at line', i+1)
        break

if not inserted_raw:
    print('ERROR: residual_inject line not found')
    exit(1)

# Find "# TATRI:" and insert else branch before it
for i, line in enumerate(lines):
    if '# TATRI: text-side' in line:
        # Check previous lines for else
        if 'memory_raw' not in lines[i-1] and 'memory_raw' not in lines[i-2]:
            indent = '        '
            lines.insert(i, indent + "else:")
            lines.insert(i+1, indent + "    encoder_outputs_dict['memory_raw'] = encoder_outputs_dict['memory']")
            print('Inserted else branch at line', i+1)
        break

open(DET, 'w', encoding='utf-8').write('\n'.join(lines))
py_compile.compile(DET, doraise=True)
print('SYNTAX OK')
