"""Fix TC-MLP shape and detector init for multi-mode support."""
import sys

# Fix 1: residual_inject.py - TC-MLP batch dim
ri_path = sys.argv[1]
with open(ri_path) as f:
    code = f.read()

old_tc = "            T_pool = T_new.mean(dim=0, keepdim=True)\n            T_bcast = T_pool.expand(memory.shape[0], -1)\n            M_aug = torch.cat([memory, T_bcast], dim=-1)\n            residual = self.net(M_aug)"
new_tc = "            T_pool = T_new.mean(dim=0, keepdim=True)\n            T_bcast = T_pool.unsqueeze(0).expand(memory.shape[0], memory.shape[1], -1)\n            M_aug = torch.cat([memory, T_bcast], dim=-1)\n            residual = self.net(M_aug)"

if old_tc in code:
    code = code.replace(old_tc, new_tc)
    with open(ri_path, 'w') as f: f.write(code)
    print("FIX_TCMLP_OK")
else:
    print("TCMLP_PATTERN_NOT_FOUND")

# Fix 2: detector init print
det_path = sys.argv[2]
with open(det_path) as f:
    dcode = f.read()

old_print = '            if _mode == \'bilinear\':\n                print(f"[GRMI] bilinear enabled: bottleneck={_bb} "\n                      f"gamma_init={float(_ri.gamma):.4f}")\n            else:\n                print(f"[GRMI] mlp enabled: hidden={int(_ri.net[0].out_features)} "\n                      f"gamma_init={float(_ri.gamma):.4f}")'
new_print = '            _params = sum(p.numel() for p in _ri.parameters())\n            print(f"[GRMI] {_mode} enabled: gamma_init={float(_ri.gamma):.4f} params={_params}")'

if old_print in dcode:
    dcode = dcode.replace(old_print, new_print)
    with open(det_path, 'w') as f: f.write(dcode)
    print("FIX_PRINT_OK")
else:
    # Try with escaped quotes
    old_print2 = old_print.replace("'", "'").replace('"', '"')
    if old_print2 in dcode:
        dcode = dcode.replace(old_print2, new_print)
        with open(det_path, 'w') as f: f.write(dcode)
        print("FIX_PRINT_OK_V2")
    else:
        print("PRINT_NOT_FOUND, searching...")
        idx = dcode.find('_mode ==')
        if idx >= 0: print(repr(dcode[idx:idx+300]))
