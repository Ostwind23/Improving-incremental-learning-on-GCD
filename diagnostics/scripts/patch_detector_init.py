"""Patch detector init to pass use_old_gate to ResidualInject."""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else 'mmdet/models/detectors/gdino_inc_gcd.py'

with open(path) as f:
    code = f.read()

# Find the ResidualInject constructor call and add use_old_gate
old = """                freeze_gamma=bool(self.residual_inject_cfg.get('freeze_gamma', False)),
                mode=_mode,
                bilinear_bottleneck=_bb)"""

new = """                freeze_gamma=bool(self.residual_inject_cfg.get('freeze_gamma', False)),
                mode=_mode,
                bilinear_bottleneck=_bb,
                use_old_gate=bool(self.residual_inject_cfg.get('use_old_gate', False)))"""

if old in code:
    code = code.replace(old, new)
    with open(path, 'w') as f:
        f.write(code)
    print("DETECTOR_INIT_PATCHED: use_old_gate added")
else:
    print("WARNING: pattern not found. Searching for alternative...")
    idx = code.find('bilinear_bottleneck=_bb')
    if idx >= 0:
        print("Found at:", code[idx-50:idx+50])
    else:
        print("bilinear_bottleneck not found — init may already be patched")
