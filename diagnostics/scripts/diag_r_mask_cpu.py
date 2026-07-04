"""Evaluate R-norm-based gate using extracted weights (CPU, no model load)"""
import torch, numpy as np

# Load Bilinear weights from checkpoint
ckpt = torch.load('/home/yelingfei/projects/GCD/work_dirs/grmi_rms_t1_3e/epoch_3.pth', map_location='cpu')
sd = ckpt['state_dict']
W_m = sd['residual_inject.W_m.weight']   # (64, 256)
W_t = sd['residual_inject.W_t.weight']   # (64, 256) 
W_out = sd['residual_inject.W_out.weight'] # (256, 64)
print(f"Weights: W_m {W_m.norm():.1f}, W_t {W_t.norm():.1f}, W_out {W_out.norm():.1f}")

# Simulate: random memory with varying alignment to T_new
torch.manual_seed(42)
N = 1000
results = []

# Case 1: High alignment (new-class-like) — M aligns with T_new
# Case 2: Low alignment (old-class-like) — M is orthogonal to T_new
# Case 3: Background — random M

for _ in range(N):
    # Generate T_pool
    T_pool = torch.randn(1, 256)
    T_pool = T_pool / T_pool.norm() * 16  # typical norm
    
    # New-class-like: M = T_pool * scale + noise
    aligned_scale = torch.rand(1).item() * 2.0  # 0 to 2x alignment
    M_new = T_pool * aligned_scale + torch.randn(1, 256) * 5
    
    # Old-class-like: M = T_orthogonal * scale + noise
    T_orth = torch.randn(1, 256)
    T_orth = T_orth - (T_orth @ T_pool.T) * T_pool / (T_pool @ T_pool.T + 1e-8)
    old_scale = torch.rand(1).item() * 2.0
    M_old = T_orth * old_scale + torch.randn(1, 256) * 5
    
    # Background: random
    M_bg = torch.randn(1, 256) * 15
    
    for label, M in [('new', M_new), ('old', M_old), ('bg', M_bg)]:
        h = (M @ W_m.T) * (T_pool @ W_t.T)
        R = h @ W_out.T  # (1, 256)
        r_norm = R.norm().item()
        cos_R_M = (R @ M.T).item() / (r_norm * M.norm().item() + 1e-8)
        cos_R_T = (R @ T_pool.T).item() / (r_norm * T_pool.norm().item() + 1e-8)
        results.append({'label': label, 'r_norm': r_norm, 'cos_R_M': cos_R_M, 'cos_R_T': cos_R_T})

# Aggregate
for label in ['new', 'old', 'bg']:
    vals = [r['r_norm'] for r in results if r['label'] == label]
    cos_m = [r['cos_R_M'] for r in results if r['label'] == label]
    cos_t = [r['cos_R_T'] for r in results if r['label'] == label]
    arr = np.array(vals)
    print(f"  {label:5s}: |R|={arr.mean():.2f}±{arr.std():.2f} median={np.median(arr):.2f} "
          f"cos(R,M)={np.mean(cos_m):.3f} cos(R,T)={np.mean(cos_t):.3f}")

# Threshold analysis
all_vals = {l: np.array([r['r_norm'] for r in results if r['label'] == l]) for l in ['new', 'old', 'bg']}
global_median = np.median(np.concatenate(list(all_vals.values())))
print(f"\nGlobal median |R| = {global_median:.2f}")

for thr in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
    passes = {}
    for l in ['new', 'old', 'bg']:
        passes[l] = (all_vals[l] > thr).mean() * 100
    new_to_old = passes['new'] / max(passes['old'], 0.01)
    print(f"  thr={thr:.1f}: new={passes['new']:.0f}% old={passes['old']:.0f}% bg={passes['bg']:.0f}% | new/old ratio={new_to_old:.1f}x")
