#!/usr/bin/env python3
"""Fix: use register_buffer for gamma when freeze_gamma=True."""
import py_compile
RI = '/home/yelingfei/projects/GCD/mmdet/models/utils/residual_inject.py'
src = open(RI, encoding='utf-8').read()

# Replace nn.Parameter with register_buffer when freeze_gamma=True
old = "        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))"
new = ("        if freeze_gamma:\n"
       "            self.register_buffer('gamma', torch.tensor(float(gamma_init)))\n"
       "        else:\n"
       "            self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))")
src = src.replace(old, new, 1)

# Remove the detach logic in forward (buffer doesn't need detach)
old_fwd = ("        g = self.gamma.detach() if self.freeze_gamma else self.gamma\n"
           "        return memory + g * residual")
new_fwd = "        return memory + self.gamma * residual"
src = src.replace(old_fwd, new_fwd, 1)

open(RI, 'w', encoding='utf-8').write(src)
py_compile.compile(RI, doraise=True)
print('FIXED: gamma uses register_buffer when freeze_gamma=True')
print('This prevents both gradient flow AND weight decay')
