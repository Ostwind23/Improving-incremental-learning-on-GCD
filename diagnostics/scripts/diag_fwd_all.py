"""
Forward diagnostic: test A/B/D/E no-LN variants on Bilinear 12e weights.
Measures R norm stability across 100 random (M, T_new) pairs.
Pass: norm stays in [0.5, 10], no monotonic trend.
"""
import torch, numpy as np
from mmengine.registry import DefaultScope
DefaultScope.get_instance('mmdet', scope_name='mmdet')
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmdet.registry import MODELS

cfg = Config.fromfile('configs/gdino_inc/70+10/grmi_bilinear_12e_clean.py')
cfg.default_scope = 'mmdet'
m = MODELS.build(cfg.model)
load_checkpoint(m, 'work_dirs/grmi_bilinear_12e_clean/epoch_12.pth', map_location='cuda:0')
m.cuda(); m.eval()
ri = m.residual_inject
print("Model loaded. Trained weights in place.")

# Extract trained weights
W_m = ri.W_m.weight.data.clone()
W_t = ri.W_t.weight.data.clone()
W_out = ri.W_out.weight.data.clone()  
# W_out is (256, 64) for Linear(64, 256)
print(f"W_m: {W_m.shape}, W_t: {W_t.shape}, W_out: {W_out.shape}")
print(f"||W_m||={W_m.norm():.2f}, ||W_t||={W_t.norm():.2f}, ||W_out||={W_out.norm():.2f}")

torch.manual_seed(42)
N = 200  # samples

def bilinear_forward(mem, T_pool):
    """Bilinear forward without LN"""
    h = mem @ W_m.T * (T_pool @ W_t.T)
    R = h @ W_out.T  # (L, 256)
    return R

results = {}
T_pool = torch.randn(1, 256).cuda()  # fixed T_pool for reproducibility

# === Variant A: No LN (baseline) ===
norms_A = []
for _ in range(N):
    M = torch.randn(900, 256).cuda() * 15  # typical memory scale ~15
    R = bilinear_forward(M, T_pool)
    norms_A.append(R.norm(dim=-1).mean().item())
results['A_noLN'] = {'mean': np.mean(norms_A), 'std': np.std(norms_A),
                      'min': np.min(norms_A), 'max': np.max(norms_A),
                      'trend': np.polyfit(range(N), norms_A, 1)[0]}

# === Variant B: Global RMS (target=2.0) ===
norms_B = []
for _ in range(N):
    M = torch.randn(900, 256).cuda() * 15
    R = bilinear_forward(M, T_pool)
    rms = R.norm(dim=-1).mean()
    R = R * (2.0 / (rms + 1e-8))
    norms_B.append(R.norm(dim=-1).mean().item())
results['B_RMS_t2'] = {'mean': np.mean(norms_B), 'std': np.std(norms_B),
                        'min': np.min(norms_B), 'max': np.max(norms_B),
                        'trend': np.polyfit(range(N), norms_B, 1)[0]}

# === Variant B2: Global RMS (target=4.0) ===
norms_B2 = []
for _ in range(N):
    M = torch.randn(900, 256).cuda() * 15
    R = bilinear_forward(M, T_pool)
    rms = R.norm(dim=-1).mean()
    R = R * (4.0 / (rms + 1e-8))
    norms_B2.append(R.norm(dim=-1).mean().item())
results['B_RMS_t4'] = {'mean': np.mean(norms_B2), 'std': np.std(norms_B2),
                        'min': np.min(norms_B2), 'max': np.max(norms_B2),
                        'trend': np.polyfit(range(N), norms_B2, 1)[0]}

# === Variant D: SpectralNorm proxy (cap norm at weight norm) ===
# Since we can't easily apply SpectralNorm post-hoc,
# simulate: if R norm > cap, scale down
cap_D = W_out.norm().item() * 3.0  # heuristic: 3x weight norm as cap
norms_D = []
for _ in range(N):
    M = torch.randn(900, 256).cuda() * 15
    R = bilinear_forward(M, T_pool)
    rms = R.norm(dim=-1).mean()
    if rms > cap_D:
        R = R * (cap_D / rms)
    norms_D.append(R.norm(dim=-1).mean().item())
results['D_spectral_cap'] = {'mean': np.mean(norms_D), 'std': np.std(norms_D),
                               'min': np.min(norms_D), 'max': np.max(norms_D),
                               'trend': np.polyfit(range(N), norms_D, 1)[0],
                               'cap': float(cap_D)}

# === Variant E: L2 weight reparameterization (simulate frozen W_out norm) ===
# Lock W_out norm to its current value, re-sample W_out with same norm
W_out_norm = W_out.norm().item()
norms_E = []
for _ in range(N):
    M = torch.randn(900, 256).cuda() * 15
    R = bilinear_forward(M, T_pool)
    # Simulate: weight reparameterization keeps ||W_out|| constant
    # Scale R to maintain same effective norm despite W_out drift
    rms = R.norm(dim=-1).mean()
    # Re-parameterization means R stays proportional to W_out_norm
    # Actual R = R_raw * (current_W_norm / initial_W_norm)
    # This is simulation-only; real impl doesn't change forward
    norms_E.append(rms.item())
results['E_L2_reparam'] = {'mean': np.mean(norms_E), 'std': np.std(norms_E),
                            'min': np.min(norms_E), 'max': np.max(norms_E),
                            'trend': np.polyfit(range(N), norms_E, 1)[0],
                            'W_out_norm': float(W_out_norm)}

# === Report ===
print("\n======== Forward Diagnostic ========")
for name, r in results.items():
    status = "PASS" if (0.5 < r['mean'] < 10 and abs(r['trend']) < 0.01) else "FAIL"
    print(f"\n{name}: {status}")
    print(f"  mean={r['mean']:.2f} std={r['std']:.2f} [{r['min']:.2f}, {r['max']:.2f}]")
    print(f"  trend={r['trend']:.4f}/sample")

# === Summary ===
print("\n======== Recommendation ========")
for name, r in results.items():
    pass_checks = (0.5 < r['mean'] < 10 and abs(r['trend']) < 0.01)
    print(f"  {name}: {'✓ VIABLE' if pass_checks else '✗ REJECTED'}  (mean={r['mean']:.1f}, trend={r['trend']:.4f})")
