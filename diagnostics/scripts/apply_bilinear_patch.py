"""Apply bilinear residual_inject patch to PolyU GCD code."""
import sys

# 1. Update residual_inject.py
ri_path = sys.argv[1] + '/mmdet/models/utils/residual_inject.py'
with open(ri_path, 'r') as f:
    ri_code = f.read()

# Replace the class definition
new_ri = '''"""
GRMI: Gated Residual injection on encoder Memory.

Architectures:
  - 'mlp' (default): M' = M + gamma * MLP(M)  
  - 'bilinear': M' = M + gamma * LN(Bilinear(M, T_new))

Bilinear: R = LN(W_out(W_m(M) * W_t(T_pool)))
LayerNorm on output bounds ||R|| naturally, preventing explosion.
"""
import torch, torch.nn as nn
from torch import Tensor


class ResidualInject(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=128, gamma_init=1e-2,
                 dropout=0.0, act_inference=True, freeze_gamma=False,
                 mode='mlp', bilinear_bottleneck=64):
        super().__init__()
        self.mode = mode
        self.in_dim = in_dim

        if mode == 'bilinear':
            self.W_m = nn.Linear(in_dim, bilinear_bottleneck, bias=False)
            self.W_t = nn.Linear(in_dim, bilinear_bottleneck, bias=False)
            self.W_out = nn.Linear(bilinear_bottleneck, in_dim, bias=False)
            self.output_norm = nn.LayerNorm(in_dim, elementwise_affine=False)
            for m in [self.W_m, self.W_t, self.W_out]:
                nn.init.normal_(m.weight, std=0.1)
        else:
            layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True)]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, in_dim))
            self.net = nn.Sequential(*layers)

        if freeze_gamma:
            self.register_buffer('gamma', torch.tensor(float(gamma_init)))
        else:
            self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.act_inference = bool(act_inference)
        self.freeze_gamma = bool(freeze_gamma)

    def forward(self, memory: Tensor, T_new: Tensor = None) -> Tensor:
        if self.mode == 'bilinear':
            assert T_new is not None, "Bilinear mode requires T_new"
            T_pool = T_new.mean(dim=0, keepdim=True)
            h_m = self.W_m(memory)
            h_t = self.W_t(T_pool)
            h = h_m * h_t.squeeze(0)
            residual = self.W_out(h)
            residual = self.output_norm(residual)
        else:
            residual = self.net(memory)

        with torch.no_grad():
            self._cached_mem_norm = float(memory.detach().norm(dim=-1).mean())
            self._cached_rm_norm = float(residual.detach().norm(dim=-1).mean())

        if self.training and getattr(self, '_gs_ratio', None) is not None:
            r = self._gs_ratio
            residual = residual * 1.0
            residual.register_hook(lambda g, _r=r: g * _r)

        if self.training:
            self._cached_residual = residual

        return memory + self.gamma * residual

    def extra_repr(self) -> str:
        return (f'mode={self.mode}, gamma={float(self.gamma.detach()):.4f}, '
                f'act_inference={self.act_inference}')
'''

with open(ri_path, 'w') as f:
    f.write(new_ri)
print(f"[1/2] Updated {ri_path}")

# 2. Patch detector forward_encoder to pass T_new
det_path = sys.argv[1] + '/mmdet/models/detectors/gdino_inc_gcd.py'
with open(det_path, 'r') as f:
    det_code = f.read()

# Find the residual_inject call and add T_new extraction
old_call = "encoder_outputs_dict['memory'] = self.residual_inject(memory)"
new_call = (
    "            _mt = encoder_outputs_dict.get('memory_text')\n"
    "            _T_new = _mt[:, 169:189, :] if _mt is not None else None\n"
    "            encoder_outputs_dict['memory'] = self.residual_inject(memory, _T_new)"
)

if old_call in det_code:
    det_code = det_code.replace(old_call, new_call)
    print(f"[2/2] Patched forward_encoder in {det_path}")
else:
    # Try alternative patterns
    if "self.residual_inject(memory)" in det_code:
        print(f"[2/2] Found residual_inject(memory), but exact pattern differs. Manual check needed.")
        # Show context
        idx = det_code.find("self.residual_inject(memory)")
        print(det_code[max(0,idx-100):idx+50])
    else:
        print(f"[2/2] WARNING: residual_inject(memory) not found in detector!")

with open(det_path, 'w') as f:
    f.write(det_code)
