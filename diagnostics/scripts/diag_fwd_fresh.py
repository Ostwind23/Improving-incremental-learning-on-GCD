"""Forward diagnostic: FRESH init Bilinear, test all variants"""
import torch, numpy as np
from mmengine.registry import DefaultScope
DefaultScope.get_instance('mmdet', scope_name='mmdet')
from mmengine.config import Config
from mmdet.registry import MODELS

cfg = Config.fromfile('configs/gdino_inc/70+10/grmi_bilinear_12e_clean.py')
cfg.default_scope = 'mmdet'
m = MODELS.build(cfg.model)
ri = m.residual_inject  # FRESH init: normal_(std=0.1)
print("Fresh weights loaded")

torch.manual_seed(42)
N = 200
T_pool = torch.randn(1, 256)

def forward(M, Tp):
    h = M @ ri.W_m.weight.T * (Tp @ ri.W_t.weight.T)
    return h @ ri.W_out.weight.T

# A: No LN
norms_A = [forward(torch.randn(900,256)*15, T_pool).norm(dim=-1).mean().item() for _ in range(N)]
print(f"\nA (no LN, fresh init): mean={np.mean(norms_A):.2f} std={np.std(norms_A):.2f} "
      f"[{np.min(norms_A):.2f}, {np.max(norms_A):.2f}] trend={np.polyfit(range(N),norms_A,1)[0]:.4f}")

# B: RMS target=2.0
norms_B = []
for _ in range(N):
    R = forward(torch.randn(900,256)*15, T_pool)
    R = R * (2.0 / (R.norm(dim=-1).mean() + 1e-8))
    norms_B.append(R.norm(dim=-1).mean().item())
print(f"B (RMS t=2, fresh init): mean={np.mean(norms_B):.2f} std={np.std(norms_B):.2f} trend={np.polyfit(range(N),norms_B,1)[0]:.4f}")

# B2: RMS target=1.0 (closer to actual signal)
norms_B2 = []
for _ in range(N):
    R = forward(torch.randn(900,256)*15, T_pool)
    R = R * (1.0 / (R.norm(dim=-1).mean() + 1e-8))
    norms_B2.append(R.norm(dim=-1).mean().item())
print(f"B2 (RMS t=1, fresh init): mean={np.mean(norms_B2):.2f} std={np.std(norms_B2):.2f} trend={np.polyfit(range(N),norms_B2,1)[0]:.4f}")

# D: soft cap at 3x mean
cap = np.mean(norms_A) * 3
norms_D = []
for _ in range(N):
    R = forward(torch.randn(900,256)*15, T_pool)
    rms = R.norm(dim=-1).mean()
    if rms > cap: R = R * (cap / rms)
    norms_D.append(R.norm(dim=-1).mean().item())
print(f"D (cap={cap:.1f}): mean={np.mean(norms_D):.2f} std={np.std(norms_D):.2f} trend={np.polyfit(range(N),norms_D,1)[0]:.4f}")

# E: L2 reparam (same as A since fresh init ~ same dist)
norms_E = [forward(torch.randn(900,256)*15, T_pool).norm(dim=-1).mean().item() for _ in range(N)]
print(f"E (L2 reparam, =A fresh): mean={np.mean(norms_E):.2f} std={np.std(norms_E):.2f}")

print("\n=== VERDICT (fresh init) ===")
for name, n in [('A_noLN', np.mean(norms_A)), ('B_RMS_t2', np.mean(norms_B)), 
                 ('B2_RMS_t1', np.mean(norms_B2)), ('D_cap', np.mean(norms_D))]:
    v = "VIABLE" if 0.3 < n < 8 else "REJECTED"
    print(f"  {name}: {v} (mean norm={n:.2f})")
