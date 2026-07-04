#!/usr/bin/env python3
"""
Q1: Measure R(M) output norm trajectory across training steps.
Q2: Benchmark R(M) capacity tiers (forward time + param count).

Uses GRMI 12e checkpoint for static norm measurement,
and synthetic benchmarks for capacity tiers.
"""
import os, sys, time, json
import torch
import torch.nn as nn

GCD_ROOT = '/home/yelingfei/projects/GCD'
os.chdir(GCD_ROOT); sys.path.insert(0, GCD_ROOT)

# === Q2: R(M) CAPACITY TIER BENCHMARK ===
print("=" * 70)
print("Q2: R(M) CAPACITY TIER BENCHMARK")
print("=" * 70)

# Current: 256->128->256 (66k params)
# Tiers to test:
tiers = [
    ("T0-current",  128, 1),   # 256->128->256, 1 block
    ("T1-wider",    256, 1),   # 256->256->256
    ("T2-deep2",    128, 2),   # 256->128->256 x2 (residual chain)
    ("T3-wide-deep", 256, 2),  # 256->256->256 x2
    ("T4-512",      512, 1),   # 256->512->256
]

in_dim = 256
device = torch.device('cuda:0')
batch = 4
seq_len = 12000  # typical multi-scale feature map size

print()
print("  %-15s | params  | fwd_ms | overhead | output_norm | expressiveness" % "Tier")
print("  " + "-" * 85)

baseline_time = None
x = torch.randn(batch, seq_len, in_dim, device=device)

for name, hidden, n_blocks in tiers:
    layers = []
    for b in range(n_blocks):
        layers.extend([
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_dim),
        ])
        if b < n_blocks - 1:
            layers.append(nn.ReLU(inplace=True))
    net = nn.Sequential(*layers).to(device)
    n_params = sum(p.numel() for p in net.parameters())

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = net(x)
    torch.cuda.synchronize()

    # Benchmark forward time
    t0 = time.time()
    N = 50
    with torch.no_grad():
        for _ in range(N):
            out = net(x)
    torch.cuda.synchronize()
    fwd_ms = (time.time() - t0) / N * 1000

    if baseline_time is None:
        baseline_time = fwd_ms
    overhead = (fwd_ms - baseline_time) / baseline_time * 100 if baseline_time > 0 else 0

    out_norm = out.norm(dim=-1).mean().item()

    # Expressiveness: rank of output (how many independent directions?)
    # Use a smaller batch for SVD
    with torch.no_grad():
        small_x = torch.randn(1, 500, in_dim, device=device)
        small_out = net(small_x)[0]  # (500, 256)
        try:
            s = torch.linalg.svdvals(small_out)
            rank_90 = int((s.cumsum(0) / s.sum() < 0.9).sum()) + 1
        except:
            rank_90 = -1

    print("  %-15s | %6dk | %5.1f  |  %+4.0f%%   |   %6.2f     | rank90=%d/256" % (
        name, n_params // 1000, fwd_ms, overhead, out_norm, rank_90))

# Also benchmark full GCD forward to get relative overhead
print()
print("  Reference: full GCD training step ~1200ms")
print("  R(M) forward is called ONCE per step on encoder output")
print("  Overhead = fwd_ms / 1200 * 100")
for name, hidden, n_blocks in tiers:
    layers = []
    for b in range(n_blocks):
        layers.extend([nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, in_dim)])
        if b < n_blocks - 1: layers.append(nn.ReLU(inplace=True))
    net = nn.Sequential(*layers).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    with torch.no_grad():
        for _ in range(5): _ = net(x)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(50): _ = net(x)
    torch.cuda.synchronize()
    fwd_ms = (time.time() - t0) / 50 * 1000
    pct = fwd_ms / 1200 * 100
    print("  %-15s: %5.1fms = %.1f%% of training step" % (name, fwd_ms, pct))

# === Q1: R(M) OUTPUT NORM FROM GRMI CHECKPOINT ===
print()
print("=" * 70)
print("Q1: R(M) OUTPUT NORM MEASUREMENT (GRMI 12e checkpoint)")
print("=" * 70)

from mmengine.config import Config
from mmengine.runner import Runner, load_checkpoint

cfg = Config.fromfile('configs/gdino_inc/70+10/gdino_inc_70+10_70-79_gcd_grmi_12e_coco.py')
cfg.work_dir = '/tmp/rm_diag'; cfg.launcher = 'none'
cfg.val_dataloader['batch_size'] = 1
vd = cfg.val_dataloader
if 'dataset' in vd and isinstance(vd['dataset'], dict):
    vd['dataset'].pop('_delete_', None)
runner = Runner.from_cfg(cfg); runner.load_or_resume()
model = runner.model.module if hasattr(runner.model, 'module') else runner.model
load_checkpoint(model, 'work_dirs/_preserved/grmi_first_best_ep12.pth', map_location='cpu')
model.to(device).eval()
if runner.model is not model: runner.model.to(device).eval()
for p in model.parameters(): p.requires_grad_(False)

ri = model.residual_inject
gamma = float(ri.gamma)
print()
print("  R(M) gamma (trained 12e): %.6f" % gamma)
print("  R(M) net:", ri.net)

# Hook encoder to capture memory before and after R(M)
cap = {}
orig_fe = model.forward_encoder
def fe_hook(*a, **k):
    out = orig_fe(*a, **k)
    cap['memory'] = out.get('memory', None)
    if cap['memory'] is not None:
        cap['memory'] = cap['memory'].detach()
    return out
model.forward_encoder = fe_hook

# Measure R(M) output norm on val images
norms_ratio = []
norms_absolute = []
gamma_x_norm = []
seen = 0
for data in runner.val_dataloader:
    if seen >= 100: break
    cap.clear()
    with torch.no_grad():
        _ = runner.model.val_step(data)
    mem = cap.get('memory')
    if mem is None: seen += 1; continue

    rm_out = ri.net(mem)
    mem_norm = mem.norm(dim=-1).mean().item()
    rm_norm = rm_out.norm(dim=-1).mean().item()
    ratio = rm_norm / max(mem_norm, 1e-8)
    eff = gamma * rm_norm

    norms_ratio.append(ratio)
    norms_absolute.append(rm_norm)
    gamma_x_norm.append(eff)
    seen += 1

import numpy as np
nr = np.array(norms_ratio)
na = np.array(norms_absolute)
gn = np.array(gamma_x_norm)

print()
print("  100 val images:")
print("  ||R(M)||/||M||:     mean=%.4f std=%.4f" % (nr.mean(), nr.std()))
print("  ||R(M)||:           mean=%.2f std=%.2f" % (na.mean(), na.std()))
print("  gamma*||R(M)||:     mean=%.4f std=%.4f" % (gn.mean(), gn.std()))
print("  ||M||:              mean=%.2f" % (na.mean() / max(nr.mean(), 1e-8)))
print("  Effective perturbation: %.3f%% of memory norm" % (nr.mean() * gamma * 100))
print()
print("  If gamma were fixed at 0.01:")
print("    perturbation = %.3f%% of memory norm" % (nr.mean() * 0.01 * 100))
print("    vs current:    %.3f%%" % (nr.mean() * gamma * 100))
print("    ratio: %.0fx stronger" % (0.01 / max(gamma, 1e-8)))

# Save
result = {
    'gamma_trained': gamma,
    'rm_norm_ratio_mean': round(float(nr.mean()), 4),
    'rm_norm_ratio_std': round(float(nr.std()), 4),
    'rm_absolute_norm_mean': round(float(na.mean()), 2),
    'memory_norm_mean': round(float(na.mean() / max(nr.mean(), 1e-8)), 2),
    'effective_perturbation_pct': round(float(nr.mean() * gamma * 100), 4),
    'perturbation_if_gamma_001': round(float(nr.mean() * 0.01 * 100), 4),
}
json.dump(result, open('/home/yelingfei/logs/tatri/rm_capacity_benchmark.json', 'w'), indent=2)
print("\nSaved: rm_capacity_benchmark.json")
