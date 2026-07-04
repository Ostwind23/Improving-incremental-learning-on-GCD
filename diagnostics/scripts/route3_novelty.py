#!/usr/bin/env python3
"""
Route 3 Novelty Weight Diagnostic.
Can `1 - sigmoid(max_score)` from enc_cls serve as a new-class region mask?
200 val images, GCD 12e checkpoint.
"""
import os, sys, json, time
import numpy as np
import torch
GCD_ROOT = '/home/yelingfei/projects/GCD'
os.chdir(GCD_ROOT); sys.path.insert(0, GCD_ROOT)
NS, NE = 70, 80

from mmengine.config import Config
from mmengine.runner import Runner, load_checkpoint

cfg = Config.fromfile('configs/gdino_inc/70+10/gdino_inc_70+10_70-79_gcd_scratch_coco.py')
cfg.work_dir = '/tmp/r3nov'; cfg.launcher = 'none'
cfg.val_dataloader['batch_size'] = 1
vd = cfg.val_dataloader
if 'dataset' in vd and isinstance(vd['dataset'], dict):
    vd['dataset'].pop('_delete_', None)
runner = Runner.from_cfg(cfg); runner.load_or_resume()
model = runner.model.module if hasattr(runner.model, 'module') else runner.model
load_checkpoint(model, 'work_dirs/gcd_70plus10_2gpu_20260426_223507/epoch_12.pth', map_location='cpu')
dev = torch.device('cuda:0')
model.to(dev).eval()
if runner.model is not model: runner.model.to(dev).eval()
for p in model.parameters(): p.requires_grad_(False)

cap = {}
lgqs_cls = model.bbox_head.cls_branches[model.decoder.num_layers]
def lgqs_hook(module, inputs, output):
    cap['enc_cls'] = output.detach()
lgqs_cls.register_forward_hook(lgqs_hook)
orig_fe = model.forward_encoder
def fe_hook(*a, **k):
    out = orig_fe(*a, **k)
    for key in ('spatial_shapes',):
        v = out.get(key)
        if v is not None: cap[key] = v.detach()
    return out
model.forward_encoder = fe_hook

nw_new, nw_old, nw_bg = [], [], []
# Per new-GT: what percentile is its novelty_weight?
gt_novelty_percentiles = []

seen = 0; t0 = time.time()
for data in runner.val_dataloader:
    if seen >= 200: break
    sl = data['data_samples']
    s = sl[0] if isinstance(sl, (list, tuple)) else sl
    gt = s.gt_instances
    if gt is None or len(gt.bboxes) == 0: seen += 1; continue
    gt_labels = gt.labels
    gt_bboxes = gt.bboxes.tensor if hasattr(gt.bboxes, 'tensor') else gt.bboxes
    new_mask = (gt_labels >= NS) & (gt_labels < NE)
    if not new_mask.any(): seen += 1; continue
    old_mask = gt_labels < NS

    cap.clear()
    with torch.no_grad():
        _ = runner.model.val_step(data)
    enc = cap.get('enc_cls'); ss = cap.get('spatial_shapes')
    if enc is None or ss is None: seen += 1; continue

    ih, iw = s.metainfo['img_shape']
    ssl = ss.cpu().long().tolist()
    li = max(range(len(ssl)), key=lambda k: ssl[k][0]*ssl[k][1])
    H0, W0 = ssl[li]
    off = sum(ssl[k][0]*ssl[k][1] for k in range(li))

    enc_f = enc[0, off:off+H0*W0, :]  # (H*W, max_text_len)
    max_score = enc_f.max(dim=-1)[0]   # (H*W,)
    novelty = 1.0 - max_score.sigmoid()  # high = likely NOT old class

    # Build grid masks
    new_grid = torch.zeros(H0, W0, device=dev)
    old_grid = torch.zeros(H0, W0, device=dev)
    for i in range(len(gt_labels)):
        lab = int(gt_labels[i])
        bx = gt_bboxes[i]
        gx1 = int(max(0, min(W0-1, bx[0].item()/iw*W0)))
        gx2 = int(max(1, min(W0, bx[2].item()/iw*W0)))
        gy1 = int(max(0, min(H0-1, bx[1].item()/ih*H0)))
        gy2 = int(max(1, min(H0, bx[3].item()/ih*H0)))
        if NS <= lab < NE:
            new_grid[gy1:gy2, gx1:gx2] = 1
        elif lab < NS:
            old_grid[gy1:gy2, gx1:gx2] = 1

    nf = new_grid.reshape(-1) > 0
    of = old_grid.reshape(-1) > 0
    bf = (~nf) & (~of)
    nv = novelty.cpu().numpy()
    if nf.any(): nw_new.extend(nv[nf.cpu().numpy()].tolist())
    if of.any(): nw_old.extend(nv[of.cpu().numpy()].tolist())
    if bf.any():
        bidx = torch.where(bf)[0]
        if len(bidx) > 100: bidx = bidx[torch.randperm(len(bidx))[:100]]
        nw_bg.extend(nv[bidx.cpu().numpy()].tolist())

    # Per new-GT: percentile of its average novelty_weight
    for i in range(len(gt_labels)):
        lab = int(gt_labels[i])
        if lab < NS or lab >= NE: continue
        bx = gt_bboxes[i]
        gx1 = int(max(0, min(W0-1, bx[0].item()/iw*W0)))
        gx2 = int(max(1, min(W0, bx[2].item()/iw*W0)))
        gy1 = int(max(0, min(H0-1, bx[1].item()/ih*H0)))
        gy2 = int(max(1, min(H0, bx[3].item()/ih*H0)))
        if gx2 <= gx1 or gy2 <= gy1: continue
        region_nv = novelty[gy1*W0+gx1:gy1*W0+gx1+1].item()  # approx
        # Actually compute mean over the box region
        region_vals = []
        for y in range(gy1, gy2):
            for x in range(gx1, gx2):
                region_vals.append(nv[y*W0+x])
        if region_vals:
            gt_mean_nv = np.mean(region_vals)
            pct = np.searchsorted(np.sort(nv), gt_mean_nv) / len(nv) * 100
            gt_novelty_percentiles.append(pct)

    seen += 1
    if seen % 50 == 0:
        print(f"  [{seen}/200] {time.time()-t0:.0f}s", flush=True)

nn = np.array(nw_new); on = np.array(nw_old); bn = np.array(nw_bg)
print("\n" + "=" * 70)
print("ROUTE 3: NOVELTY WEIGHT DIAGNOSTIC")
print("=" * 70)
print(f"\n  novelty_weight = 1 - sigmoid(max_enc_cls_score)")
print(f"\n  Region       | n positions | mean   | median | std    | p10    | p90")
print(f"  -------------|-------------|--------|--------|--------|--------|-------")
print(f"  New-class GT | {len(nn):>11} | {nn.mean():.4f} | {np.median(nn):.4f} | {nn.std():.4f} | {np.percentile(nn,10):.4f} | {np.percentile(nn,90):.4f}")
print(f"  Old-class GT | {len(on):>11} | {on.mean():.4f} | {np.median(on):.4f} | {on.std():.4f} | {np.percentile(on,10):.4f} | {np.percentile(on,90):.4f}")
print(f"  Background   | {len(bn):>11} | {bn.mean():.4f} | {np.median(bn):.4f} | {bn.std():.4f} | {np.percentile(bn,10):.4f} | {np.percentile(bn,90):.4f}")

print(f"\n  Separation metrics:")
print(f"    New vs Old gap: {nn.mean()-on.mean():+.4f}")
print(f"    New vs BG gap:  {nn.mean()-bn.mean():+.4f}")

# Can we threshold to capture new-class?
print(f"\n  Threshold analysis (capture rate at different novelty thresholds):")
for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    new_cap = np.mean(nn >= thr)
    old_cap = np.mean(on >= thr)
    bg_cap = np.mean(bn >= thr)
    selectivity = new_cap / max(old_cap, 0.001)
    print(f"    thr≥{thr:.1f}: new={new_cap:.1%} old={old_cap:.1%} bg={bg_cap:.1%} selectivity={selectivity:.1f}x")

# What threshold captures ≥80% of new-class positions?
sorted_new = np.sort(nn)
thr_80 = sorted_new[int(len(sorted_new)*0.2)] if len(sorted_new) > 0 else 0
old_at_thr80 = np.mean(on >= thr_80)
print(f"\n  To capture ≥80% new-class: threshold={thr_80:.4f}, old-class false positive={old_at_thr80:.1%}")

# Per-GT percentile
if gt_novelty_percentiles:
    gp = np.array(gt_novelty_percentiles)
    print(f"\n  Per new-GT: mean percentile={gp.mean():.1f}% median={np.median(gp):.1f}%")
    print(f"    > 50th percentile: {np.mean(gp > 50):.1%}")
    print(f"    > 70th percentile: {np.mean(gp > 70):.1%}")
    print(f"    > 90th percentile: {np.mean(gp > 90):.1%}")

# Verdict
print(f"\n  ═══ VERDICT ═══")
gap = nn.mean() - on.mean()
if gap > 0.05:
    print(f"  VIABLE: new-class novelty is {gap:.4f} higher than old-class")
    print(f"  Can use as region-selective mask for GRMI enhancement")
elif gap > 0.01:
    print(f"  MARGINAL: gap={gap:.4f}, weak but nonzero discrimination")
    print(f"  May work with careful threshold, but risk of high false positive")
else:
    print(f"  NOT VIABLE: gap={gap:.4f}, no discrimination")
    print(f"  Cannot distinguish new-class from old-class regions")

result = {
    'new_mean': round(float(nn.mean()), 4),
    'old_mean': round(float(on.mean()), 4),
    'bg_mean': round(float(bn.mean()), 4),
    'gap_new_old': round(float(gap), 4),
    'thr_80pct_capture': round(float(thr_80), 4),
    'old_fp_at_80pct': round(float(old_at_thr80), 4),
    'n_images': seen,
}
json.dump(result, open('/home/yelingfei/logs/tatri/route3_novelty.json', 'w'), indent=2)
print(f"\nSaved: route3_novelty.json")
