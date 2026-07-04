import sys
p = sys.argv[1]
with open(p) as f:
    code = f.read()

old = 'bilinear_bottleneck=_bb, norm_mode'
new = 'bilinear_bottleneck=_bb, use_old_gate=bool(self.residual_inject_cfg.get("use_old_gate", False)), norm_mode'

if old in code:
    code = code.replace(old, new)
    with open(p, 'w') as f:
        f.write(code)
    print("PATCHED use_old_gate")
else:
    print("Already patched or pattern not found")
    for i, line in enumerate(code.split('\n'), 1):
        if 'bilinear_bottleneck' in line:
            print(f"  L{i}: {line.strip()[:100]}")
