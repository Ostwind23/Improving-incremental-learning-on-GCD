"""Patch detector init for bilinear mode."""
import sys
path = '/root/autodl-tmp/gcd_work/mmdet/models/detectors/gdino_inc_gcd.py'
with open(path) as f:
    code = f.read()

old = """            self.residual_inject = ResidualInject(
                in_dim=int(self.residual_inject_cfg.get('in_dim', 256)),
                hidden_dim=int(self.residual_inject_cfg.get('hidden_dim', 128)),
                gamma_init=float(self.residual_inject_cfg.get('gamma_init', 1e-2)),
                dropout=float(self.residual_inject_cfg.get('dropout', 0.0)),
                act_inference=bool(self.residual_inject_cfg.get('act_inference', True)))
            _ri = self.residual_inject
            print(f"[GRMI] enabled: hidden={int(_ri.net[0].out_features)} "
                  f"gamma_init={float(_ri.gamma):.4f} "
                  f"act_inference={_ri.act_inference}")"""

new = """            _mode = str(self.residual_inject_cfg.get('mode', 'mlp'))
            _bb = int(self.residual_inject_cfg.get('bilinear_bottleneck', 64))
            self.residual_inject = ResidualInject(
                in_dim=int(self.residual_inject_cfg.get('in_dim', 256)),
                hidden_dim=int(self.residual_inject_cfg.get('hidden_dim', 128)),
                gamma_init=float(self.residual_inject_cfg.get('gamma_init', 1e-2)),
                dropout=float(self.residual_inject_cfg.get('dropout', 0.0)),
                act_inference=bool(self.residual_inject_cfg.get('act_inference', True)),
                freeze_gamma=bool(self.residual_inject_cfg.get('freeze_gamma', False)),
                mode=_mode,
                bilinear_bottleneck=_bb)
            _ri = self.residual_inject
            if _mode == 'bilinear':
                print(f"[GRMI] bilinear enabled: bottleneck={_bb} "
                      f"gamma_init={float(_ri.gamma):.4f}")
            else:
                print(f"[GRMI] mlp enabled: hidden={int(_ri.net[0].out_features)} "
                      f"gamma_init={float(_ri.gamma):.4f}")"""

if old in code:
    code = code.replace(old, new)
    with open(path, 'w') as f:
        f.write(code)
    print("INIT_PATCH_OK")
else:
    print("INIT_NOT_FOUND")
    # Show surrounding context
    idx = code.find('ResidualInject(')
    if idx >= 0:
        print(code[max(0,idx-50):idx+300])
