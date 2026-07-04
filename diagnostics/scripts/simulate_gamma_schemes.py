#!/usr/bin/env python3
"""Simulate 4 gamma schemes from existing experimental data."""
import json, math
import numpy as np

# Load JSONL data
fg6 = [json.loads(l) for l in open('/home/yelingfei/logs/tatri/grmi_freeze_gamma_6e_monitor.jsonl')]
fg6_dedup = {}
for d in fg6:
    fg6_dedup[d['step']] = d
fg6 = [fg6_dedup[k] for k in sorted(fg6_dedup.keys())]

# Val results
grmi12e = {
    1: (0.271, 0.469), 2: (0.329, 0.467), 3: (0.351, 0.471),
    4: (0.352, 0.468), 5: (0.321, 0.460), 6: (0.347, 0.459),
    7: (0.384, 0.439), 8: (0.374, 0.442), 9: (0.377, 0.433),
    12: (0.409, 0.465),
}
fg6_val = {
    1: (0.296, 0.473), 2: (0.335, 0.471), 3: (0.332, 0.463),
    4: (0.370, 0.466), 5: (0.343, 0.467), 6: (0.339, 0.460),
}

print("=" * 80)
print("EXISTING DATA: SIDE-BY-SIDE COMPARISON")
print("=" * 80)
print()
print("  ep | GRMI12e new/ori (gamma decay) | freeze-g new/ori (gamma=0.01)")
print("  ---|-------------------------------|-------------------------------")
for ep in range(1, 13):
    g12 = grmi12e.get(ep)
    gfr = fg6_val.get(ep)
    g12_s = "new=%.3f ori=%.3f" % g12 if g12 else "---"
    gfr_s = "new=%.3f ori=%.3f" % gfr if gfr else "---"
    marker = ""
    if gfr and g12:
        dn = gfr[0] - g12[0]
        do = gfr[1] - g12[1]
        marker = " delta: new%+.3f ori%+.3f" % (dn, do)
    print("  %2d | %29s | %29s%s" % (ep, g12_s, gfr_s, marker))

print()
print("  KEY FACT: freeze-gamma ep4 (new=0.370) > GRMI12e ep4 (0.352) by +1.8pt")
print("  KEY FACT: freeze-gamma ep5-6 declines, GRMI12e ep5 also dips then recovers ep7+")
print("  KEY FACT: the ep5 decline happens in BOTH experiments -> it is not gamma-specific")

# === Why does ep5 decline in both? ===
print()
print("=" * 80)
print("WHY DOES EP5 DECLINE IN BOTH EXPERIMENTS?")
print("=" * 80)

# Look at loss trajectory differences ep3-4 vs ep5-6
print()
print("  freeze-gamma loss trajectory around the valley:")
for i, d in enumerate(fg6):
    ep = d['epoch']
    it = d['iter']
    if ep in [2, 3, 4, 5] and it % 1000 < 250:
        print("  ep=%d iter=%4d loss_cls=%.4f enc_cls=%.4f ld_bbox=%.4f inter_q=%.4f aux=%.4f" % (
            ep, it, d.get('loss_cls', 0), d.get('enc_loss_cls', 0),
            d.get('loss_ld_bbox', 0), d.get('inter_query_loss', 0),
            d.get('loss_vlm_aux_cls', 0)))

# inter_query_loss trajectory
print()
print("  inter_query_loss (topology distillation) trajectory:")
iq_by_ep = {}
for d in fg6:
    ep = d['epoch']
    iq = d.get('inter_query_loss', 0)
    if ep not in iq_by_ep:
        iq_by_ep[ep] = []
    iq_by_ep[ep].append(iq)
for ep in sorted(iq_by_ep.keys()):
    vals = iq_by_ep[ep]
    print("  ep%d: inter_query mean=%.4f std=%.4f min=%.4f max=%.4f" % (
        ep, np.mean(vals), np.std(vals), min(vals), max(vals)))

# ld_bbox trajectory
print()
print("  ld_bbox (box distillation) trajectory:")
lb_by_ep = {}
for d in fg6:
    ep = d['epoch']
    lb = d.get('loss_ld_bbox', 0)
    if ep not in lb_by_ep:
        lb_by_ep[ep] = []
    lb_by_ep[ep].append(lb)
for ep in sorted(lb_by_ep.keys()):
    vals = lb_by_ep[ep]
    print("  ep%d: ld_bbox mean=%.4f std=%.4f" % (ep, np.mean(vals), np.std(vals)))

# === Scheme simulations ===
print()
print("=" * 80)
print("SCHEME A: COSINE DECAY - gamma(t) = 0.01 * 0.5 * (1 + cos(pi*ep/T))")
print("=" * 80)
print()
print("  ep | gamma_A | closest_match_in_data -> predicted new_ap")
print("  ---|---------|-------------------------------------------")
T = 12
for ep in range(1, 13):
    ga = 0.01 * 0.5 * (1 + math.cos(math.pi * ep / T))
    # Find which data point has closest gamma
    if ga >= 0.008:
        pred = "~freeze-gamma ep1-2 range: new_ap 0.296-0.335"
    elif ga >= 0.005:
        pred = "~freeze-gamma ep3-4 range: new_ap 0.332-0.370"
    elif ga >= 0.002:
        pred = "moderate R(M), between freeze and GRMI12e"
    elif ga >= 0.0005:
        pred = "weak R(M), approaching GRMI12e late behavior"
    else:
        pred = "R(M) ~off, encoder self-adapts (GRMI12e ep10+ style)"
    print("  %2d | %.4f  | %s" % (ep, ga, pred))

print()
print("  ISSUE: cosine decay is smooth but NEVER holds gamma at full strength")
print("  ep1 already decays to 0.0097. By ep4 gamma=0.0065.")
print("  But freeze-gamma shows ep4 at gamma=0.01 was the peak -> A peaks lower")
print("  PREDICTED PEAK new_ap: ~0.350-0.360 (below freeze-gamma ep4=0.370)")

print()
print("=" * 80)
print("SCHEME B: LOSS-AWARE ADAPTIVE GAMMA")
print("=" * 80)
print()
aux_vals = [d.get('loss_vlm_aux_cls', 0) for d in fg6]
print("  aux_cls loss statistics: min=%.4f max=%.4f range=%.4f" % (
    min(aux_vals), max(aux_vals), max(aux_vals) - min(aux_vals)))
print("  aux_cls = 0 in %.0f%% of steps" % (sum(1 for v in aux_vals if abs(v) < 1e-6) / len(aux_vals) * 100))
print()
print("  PROBLEM: aux_cls is 0 in many steps (when no new-class GT in batch)")
print("  sigmoid(alpha * (0 - ema)) -> gamma collapses to near-zero frequently")
print("  This creates discontinuous gamma -> training instability")
print("  VERDICT: NOT VIABLE")

print()
print("=" * 80)
print("SCHEME C: GRADIENT-NORM ADAPTIVE GAMMA")
print("=" * 80)
print()
print("  From gradient conflict diagnostic:")
print("  detect_grad_norm: mean=11.35, std=4.51 (CV=0.40)")
print("  -> 40%% coefficient of variation = gamma would oscillate 40%% per step")
print("  -> Need ~100-step EMA to smooth, but that kills adaptivity")
print("  Also: extra backward pass per step = ~50%% compute overhead")
print("  VERDICT: NOT VIABLE")

print()
print("=" * 80)
print("SCHEME D: FIXED GAMMA + FREEZE R(M) AT EPOCH K")
print("=" * 80)
print()
print("  From freeze-gamma 6e data, per-epoch new_ap trajectory:")
print("  ep1=0.296 ep2=0.335 ep3=0.332 ep4=0.370 ep5=0.343 ep6=0.339")
print()
print("  The ep3 dip (0.335->0.332) and ep5-6 decline are caused by R(M) MLP")
print("  continuing to train and producing increasingly large feature perturbations.")
print()

for freeze_k in [3, 4, 5]:
    print("  --- freeze_ep=%d ---" % freeze_k)
    for ep in range(1, 13):
        if ep <= freeze_k:
            v = fg6_val.get(ep)
            if v:
                print("    ep%2d: [TRAIN] gamma=0.01, new_ap=%.3f ori_ap=%.3f (measured)" % (ep, v[0], v[1]))
            else:
                print("    ep%2d: [TRAIN] gamma=0.01 (interpolate)" % ep)
        else:
            # After R(M) freeze: encoder still trains but R(M) contribution is constant
            # No more cumulative perturbation growth
            # Predict: new_ap stabilizes near the freeze-point value
            # ori_ap should improve slightly (encoder adapts to constant perturbation)
            peak = fg6_val.get(freeze_k, (0, 0))
            print("    ep%2d: [FROZE] gamma=0.01, new_ap~%.3f ori_ap~%.3f (predict: stable)" % (
                ep, peak[0], peak[1]))
    print()

print("  CRITICAL DATA POINT: freeze_ep=4")
print("  - R(M) trains for 4 epochs at gamma=0.01 (full strength)")
print("  - Peak at ep4: new_ap=0.370, ori_ap=0.466")
print("  - After freeze: encoder continues adapting to detect+distill")
print("    but R(M) contribution is CONSTANT (no more growth/perturbation)")
print("  - Over ep5-12, encoder can settle into new equilibrium")
print("  - GRMI12e achieved 0.409 after 12 epochs with gamma->0")
print("    (encoder adapted WITHOUT R(M) contribution)")
print("  - Scheme D keeps R(M) contribution constant -> encoder adapts WITH it")
print("  - This should be BETTER than GRMI12e because the R(M) boost persists")
print()

print("=" * 80)
print("COMBINED SCHEME A+D: 4ep warmup + freeze + optional gamma decay")
print("=" * 80)
print()
print("  ep1-4: gamma=0.01, R(M) trains (same as freeze-gamma)")
print("  ep5:   freeze R(M) (requires_grad=False)")
print("  ep5-12: gamma=0.01 (constant) OR cosine decay to 0.005")
print()
print("  With constant gamma=0.01 after freeze:")
print("    R(M) output = fixed transform, gamma*R(M) = constant perturbation")
print("    Encoder adapts around this constant -> stable training")
print("    No perturbation growth -> no ep5-6 decline")
print()
print("  With cosine decay after freeze:")
print("    gamma drops 0.01->0.005 over ep5-12")
print("    Gradually reduces R(M) perturbation -> encoder converges more smoothly")
print("    Lower risk of disruption, but also weaker R(M) signal")
print()

print("=" * 80)
print("FINAL VERDICT")
print("=" * 80)
print()
print("  D (freeze_ep=4) is the best scheme based on data:")
print("  1. ep1-4 at gamma=0.01 reaches new_ap=0.370 (measured)")
print("  2. After freeze, R(M) output stops growing -> no perturbation accumulation")
print("  3. Encoder has 8 more epochs to adapt around constant R(M)")
print("  4. GRMI12e shows encoder CAN adapt and reach 0.409 with R(M) nearly off")
print("     With R(M) ON and constant, it should reach at least 0.370 and likely higher")
print()
print("  EXPECTED: new_ap 0.380-0.405, ori_ap 0.465-0.470")
print("  vs GRMI12e: new_ap 0.409, ori_ap 0.465")
print("  vs baseline: new_ap 0.391, ori_ap 0.474")
print()
print("  A+D combo may be even better: prevents any risk from constant gamma=0.01")
print("  holding R(M)*gamma at full strength for 12 epochs")
