"""Analyze rm_norm vs detection loss relationship."""
import json, numpy as np, sys

fpath = sys.argv[1] if len(sys.argv) > 1 else '/home/yelingfei/logs/tatri/grmi_filter_old_6e_monitor.jsonl'

with open(fpath) as f:
    data = [json.loads(l) for l in f]

# Find step resets (new run appended)
steps = [d['step'] for d in data]
reset_points = []
for i in range(1, len(steps)):
    if steps[i] < steps[i-1]:
        reset_points.append(i)

print(f"Entries: {len(data)}, step resets: {len(reset_points)}")

# Use last contiguous segment (current run)
if reset_points:
    data = data[reset_points[-1]:]

# Deduplicate by step
seen = set()
unique = []
for d in data:
    s = d['step']
    if s not in seen:
        seen.add(s)
        unique.append(d)
unique.sort(key=lambda d: d['step'])

print(f"Unique steps in current run: {len(unique)}")
if len(unique) < 3:
    print("Not enough data")
    sys.exit(0)

# Growth rate analysis
print(f'\n{"step":>6s} {"rm":>7s} {"mem":>7s} {"ratio":>7s} {"rate/100":>10s} {"loss_cls":>10s} {"inter_q":>10s}')
for i in range(1, len(unique)):
    drm = unique[i]['rm_norm'] - unique[i-1]['rm_norm']
    ds = unique[i]['step'] - unique[i-1]['step']
    rate = drm / ds * 100
    lc = unique[i].get('loss_cls', 0)
    iq = unique[i].get('inter_query_loss', 0)
    if ds >= 200:
        print(f'{unique[i]["step"]:6d} {unique[i]["rm_norm"]:7.3f} {unique[i]["mem_norm"]:7.3f} '
              f'{unique[i]["rm_norm"]/unique[i]["mem_norm"]:7.4f} {rate:10.4f} {lc:10.4f} {iq:10.4f}')

# Correlation analysis
rm_vals = [d['rm_norm'] for d in unique]
lc_vals = [d.get('loss_cls', 0) for d in unique]
iq_vals = [d.get('inter_query_loss', 0) for d in unique]

corr_rm_lc = np.corrcoef(rm_vals, lc_vals)[0,1] if len(rm_vals) > 2 else 0
corr_rm_iq = np.corrcoef(rm_vals, iq_vals)[0,1] if len(rm_vals) > 2 else 0

# Split into early (first 40%) and late (last 40%) for trend analysis
n = len(unique)
split = max(n // 3, 2)
early = unique[:split]
late = unique[-split:]

avg_rm_early = np.mean([d['rm_norm'] for d in early])
avg_rm_late = np.mean([d['rm_norm'] for d in late])
avg_lc_early = np.mean([d.get('loss_cls', 0) for d in early])
avg_lc_late = np.mean([d.get('loss_cls', 0) for d in late])
avg_iq_early = np.mean([d.get('inter_query_loss', 0) for d in early])
avg_iq_late = np.mean([d.get('inter_query_loss', 0) for d in late])

d_rm = avg_rm_late - avg_rm_early
d_lc = avg_lc_late - avg_lc_early
d_iq = avg_iq_late - avg_iq_early

print(f'\n{"="*60}')
print(f'TREND ANALYSIS (early {split} vs late {split} entries)')
print(f'{"="*60}')
print(f'  rm_norm:      {avg_rm_early:.3f} -> {avg_rm_late:.3f}  (delta={d_rm:+.3f})')
print(f'  loss_cls:     {avg_lc_early:.4f} -> {avg_lc_late:.4f}  (delta={d_lc:+.4f})')
print(f'  inter_query:  {avg_iq_early:.4f} -> {avg_iq_late:.4f}  (delta={d_iq:+.4f})')
print(f'  corr(rm, loss_cls):    {corr_rm_lc:+.4f}')
print(f'  corr(rm, inter_query): {corr_rm_iq:+.4f}')
if abs(d_rm) > 1e-6:
    print(f'  marginal benefit (Δloss_cls/Δrm): {d_lc/d_rm:+.4f}')
    print(f'  marginal cost   (Δinter_q/Δrm):   {d_iq/d_rm:+.4f}')

# Stability analysis: at what rm_ratio does growth stop being beneficial?
print(f'\n{"="*60}')
print(f'EQUILIBRIUM CHECK')
print(f'{"="*60}')
# In the STABLE phase (bug run, step 250-1250): rm oscillated at ~3.7, mem ~15.5
print(f'Stable phase rm_ratio: ~0.237 (from bug run step 250-1250)')
print(f'Current rm_ratio range: {np.min([d["rm_norm"]/d["mem_norm"] for d in unique]):.3f} - '
      f'{np.max([d["rm_norm"]/d["mem_norm"] for d in unique]):.3f}')
# If loss_cls keeps decreasing with rm, the equilibrium hasn't been reached
if d_lc < -0.01:
    print(f'Detection loss STILL DECREASING with rm -> equilibrium NOT reached')
    print(f'R(M) will continue growing until:')
    print(f'  (a) detection loss stops decreasing, OR')
    print(f'  (b) inter_query_loss grows enough to cause training instability')
else:
    print(f'Detection loss NOT decreasing -> may be near equilibrium or oscillating')
