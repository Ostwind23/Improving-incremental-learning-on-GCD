#!/usr/bin/env python3
"""Analyze fixed-gamma TATRI console log: per-epoch loss curves, grad spikes, compare to baseline."""
import re, sys
from collections import defaultdict
import numpy as np

LOG = "/home/yelingfei/logs/tatri_3e_20260627_230607/fixed_console.log"
text = open(LOG).read()
lines = [l for l in text.split("\n") if "Epoch(train)" in l]

keys = ["loss","loss_cls","loss_bbox","loss_iou","enc_loss_cls","enc_loss_bbox","enc_loss_iou",
        "loss_ld_cls","loss_ld_bbox","loss_ld_iou","inter_text_loss","inter_query_loss",
        "loss_vlm_aux_cls","grad_norm"]

def parse_kv(line, k):
    # match "key: value" but skip keys that are prefixes of others by requiring word boundary
    m = re.search(r"(?<![a-z_])" + re.escape(k) + r":\s*(inf|[0-9.\-e]+)", line)
    if not m: return None
    v = m.group(1)
    if v == "inf": return float("inf")
    try: return float(v)
    except: return None

by_ep = defaultdict(lambda: defaultdict(list))
inf_count_by_ep = defaultdict(int)
total_inf = 0
for line in lines:
    m = re.search(r"Epoch\(train\)\s*\[(\d+)\]", line)
    if not m: continue
    ep = int(m.group(1))
    for k in keys:
        v = parse_kv(line, k)
        if v is not None:
            by_ep[ep][k].append(v)
            if np.isinf(v) and k == "grad_norm":
                inf_count_by_ep[ep] += 1
                total_inf += 1

print("="*100)
print("FIXED-gamma TATRI per-epoch loss summary")
print("="*100)
header = f"{'ep':>3} {'n':>4} {'loss':>7} {'cls':>6} {'bbox':>6} {'iou':>6} {'enc_cls':>7} {'ld_cls':>7} {'ld_iou':>7} {'int_txt':>7} {'int_qry':>7} {'vlm_aux':>7} {'grad':>7} {'inf#':>4}"
print(header)
for ep in sorted(by_ep):
    r = by_ep[ep]
    cells = []
    for k in keys:
        v = r[k]
        finite = [x for x in v if not np.isinf(x)]
        if finite:
            cells.append(f"{np.mean(finite):.4f}")
        else:
            cells.append("  NA  ")
    print(f"{ep:>3} {len(r['loss']):>4} " + " ".join(f"{c:>7}" for c in cells) + f" {inf_count_by_ep[ep]:>4}")

print()
print("="*100)
print("GRAD_NORM detail (where are the inf spikes?)")
print("="*100)
for ep in sorted(by_ep):
    grads = by_ep[ep]["grad_norm"]
    finite = [x for x in grads if not np.isinf(x)]
    if finite:
        g = np.array(finite)
        print(f"  ep{ep}: mean={g.mean():.2f} max={g.max():.2f} min={g.min():.2f} p99={np.percentile(g,99):.2f} inf_count={inf_count_by_ep[ep]}")

# inter_text_loss is the KEY signal: if TATRI is perturbing text, CTD should react
print()
print("="*100)
print("CRITICAL: inter_text_loss trajectory (CTD reaction to text perturbation)")
print("  If TATRI works, text is being changed -> inter_text_loss should RISE (teacher detects mismatch)")
print("="*100)
for ep in sorted(by_ep):
    itl = by_ep[ep]["inter_text_loss"]
    finite = [x for x in itl if not np.isinf(x)]
    if finite:
        v = np.array(finite)
        print(f"  ep{ep}: inter_text mean={v.mean():.4f} start={v[:5].mean():.4f} end={v[-5:].mean():.4f}")

# Compare first 100 iters of ep1 vs last 100 of ep3 (trainability)
print()
print("="*100)
print("LEARNING DYNAMICS: first-100 vs last-100 iters")
print("="*100)
all_ep1 = []
all_ep3 = []
for line in lines:
    m = re.search(r"Epoch\(train\)\s*\[(\d+)\]\s*\[\s*(\d+)/", line)
    if not m: continue
    ep, it = int(m.group(1)), int(m.group(2))
    if ep == 1 and it <= 100:
        l = parse_kv(line, "loss")
        if l: all_ep1.append(l)
    if ep == 3 and it >= 2473:
        l = parse_kv(line, "loss")
        if l: all_ep3.append(l)
if all_ep1 and all_ep3:
    print(f"  ep1 first-100 loss mean: {np.mean(all_ep1):.4f}")
    print(f"  ep3 last-100 loss mean:  {np.mean(all_ep3):.4f}")
    print(f"  delta: {np.mean(all_ep3)-np.mean(all_ep1):+.4f} ({(np.mean(all_ep3)/np.mean(all_ep1)-1)*100:+.1f}%)")

# vlm_aux coverage
print()
print("="*100)
print("VLM aux coverage (Channel-1) per epoch")
print("="*100)
for ep in sorted(by_ep):
    lvc = by_ep[ep].get("loss_vlm_aux_cls", [])
    finite = [x for x in lvc if not np.isinf(x)]
    if finite:
        print(f"  ep{ep}: vlm_aux_cls mean={np.mean(finite):.4f}")
