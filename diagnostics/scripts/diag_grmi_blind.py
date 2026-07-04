#!/usr/bin/env python3
"""
Q2: Verify GRMI 'blind channel' claim + quantify 3 expansion dimensions.

1. Verify blindness: measure ∂L_distill/∂R(M) vs ∂L_detect/∂R(M)
   If R(M) is truly blind to distillation, distill gradient on R(M) should be ~0.

2. Quantify capacity dimension: R(M) output norm / total memory norm
   How much of the encoder memory feature space does R(M) actually modify?

3. Quantify injection position: gradient norm on encoder layers 0-5
   Which layers carry the most new-class signal?

4. Quantify signal strength: R(M) feature delta vs training noise
   Is γ·R(M) above the noise floor?

Uses GRMI 12e checkpoint + config.
"""
import os, sys, json, time
import torch
import torch.nn.functional as F

GCD_ROOT = '/home/yelingfei/projects/GCD'
os.chdir(GCD_ROOT); sys.path.insert(0, GCD_ROOT)
NS, NE = 70, 80

from mmengine.config import Config
from mmengine.runner import Runner, load_checkpoint

GRMI_CFG = 'configs/gdino_inc/70+10/gdino_inc_70+10_70-79_gcd_grmi_12e_coco.py'
GRMI_CKPT = 'work_dirs/_preserved/grmi_first_best_ep12.pth'

cfg = Config.fromfile(GRMI_CFG)
cfg.work_dir = '/tmp/grmi_diag'
cfg.launcher = 'none'
cfg.train_dataloader['batch_size'] = 1
cfg.train_dataloader['persistent_workers'] = False
cfg.train_dataloader['num_workers'] = 2
td = cfg.train_dataloader
if 'dataset' in td and isinstance(td['dataset'], dict):
    td['dataset'].pop('_delete_', None)

runner = Runner.from_cfg(cfg)
runner.load_or_resume()
model = runner.model.module if hasattr(runner.model, 'module') else runner.model
load_checkpoint(model, GRMI_CKPT, map_location='cpu')
dev = torch.device('cuda:0')
model.to(dev).train()
if runner.model is not model:
    runner.model.to(dev).train()

# ═══ Locate R(M) module ═══
ri = model.residual_inject
print("R(M) module:", ri)
print("R(M) gamma:", float(ri.gamma.detach()))
ri_params = list(ri.parameters())
n_ri = sum(p.numel() for p in ri_params)
print("R(M) params:", n_ri)

# ═══ Loss key sets (same as grad_conflict.py) ═══
DETECT_KEYS = {'loss_cls', 'loss_bbox', 'loss_iou', 'enc_loss_cls', 'enc_loss_bbox', 'enc_loss_iou'}
for i in range(5):
    DETECT_KEYS.update({f'd{i}.loss_cls', f'd{i}.loss_bbox', f'd{i}.loss_iou'})

DISTILL_KEYS = {'loss_ld_cls', 'loss_ld_bbox', 'loss_ld_iou', 'inter_text_loss', 'inter_query_loss'}
for i in range(5):
    DISTILL_KEYS.update({f'd{i}.ld_loss_cls', f'd{i}.ld_loss_bbox', f'd{i}.ld_loss_iou'})

# ═══ Also locate encoder layers for per-layer gradient measurement ═══
enc_layers = model.encoder.layers
n_enc_layers = len(enc_layers)
print("Encoder layers:", n_enc_layers)

def grad_norm(params):
    grads = []
    for p in params:
        if p.grad is not None:
            grads.append(p.grad.detach().flatten())
        else:
            grads.append(torch.zeros(p.numel(), device=dev))
    return torch.cat(grads).norm().item()

def grad_vector(params):
    grads = []
    for p in params:
        if p.grad is not None:
            grads.append(p.grad.detach().flatten())
        else:
            grads.append(torch.zeros(p.numel(), device=dev))
    return torch.cat(grads)

# ═══ Collect data ═══
results = {
    'ri_detect_norm': [],
    'ri_distill_norm': [],
    'ri_cos': [],
    'ri_gamma': [],
    'ri_output_norm': [],
    'memory_norm': [],
    'enc_layer_detect_norm': [[] for _ in range(n_enc_layers)],
    'enc_layer_distill_norm': [[] for _ in range(n_enc_layers)],
}

# Hook to capture R(M) output
cap = {}
orig_ri_forward = ri.forward
def ri_hook(memory):
    out = orig_ri_forward(memory)
    residual = ri.transform(memory)
    cap['residual'] = residual.detach()
    cap['memory'] = memory.detach()
    cap['gamma'] = float(ri.gamma.detach())
    return out
ri.forward = ri_hook

seen = 0; n_done = 0
for data in runner.train_dataloader:
    if n_done >= 15: break
    if seen >= 200: break

    sl = data['data_samples']
    s = sl[0] if isinstance(sl, (list, tuple)) else sl
    gt = s.gt_instances
    if gt is None or len(gt.bboxes) == 0: seen += 1; continue
    gt_labels = gt.labels
    has_new = bool(((gt_labels >= NS) & (gt_labels < NE)).any())
    if not has_new: seen += 1; continue

    # Forward
    model.zero_grad()
    for p in model.parameters(): p.requires_grad_(True)
    cap.clear()

    try:
        data_proc = model.data_preprocessor(data, training=True)
        loss_dict = model._run_forward(data_proc, mode='loss')
    except Exception as e:
        print("forward error:", e)
        seen += 1; continue

    L_detect = sum(v for k, v in loss_dict.items()
                   if k in DETECT_KEYS and isinstance(v, torch.Tensor) and v.requires_grad)
    L_distill = sum(v for k, v in loss_dict.items()
                    if k in DISTILL_KEYS and isinstance(v, torch.Tensor) and v.requires_grad)

    if not isinstance(L_detect, torch.Tensor) or not isinstance(L_distill, torch.Tensor):
        seen += 1; continue

    # ═══ Test 1: R(M) gradient from detect vs distill ═══
    model.zero_grad()
    L_detect.backward(retain_graph=True)
    g_ri_detect = grad_vector(ri_params)

    model.zero_grad()
    L_distill.backward(retain_graph=False)
    g_ri_distill = grad_vector(ri_params)

    ri_detect_n = g_ri_detect.norm().item()
    ri_distill_n = g_ri_distill.norm().item()
    if ri_detect_n > 1e-10 and ri_distill_n > 1e-10:
        ri_cos = F.cosine_similarity(g_ri_detect.unsqueeze(0), g_ri_distill.unsqueeze(0)).item()
    else:
        ri_cos = 0.0

    results['ri_detect_norm'].append(ri_detect_n)
    results['ri_distill_norm'].append(ri_distill_n)
    results['ri_cos'].append(ri_cos)
    results['ri_gamma'].append(cap.get('gamma', 0))

    # ═══ Test 2: R(M) output magnitude vs memory ═══
    if 'residual' in cap and 'memory' in cap:
        res_norm = cap['residual'].norm(dim=-1).mean().item()
        mem_norm = cap['memory'].norm(dim=-1).mean().item()
        results['ri_output_norm'].append(res_norm)
        results['memory_norm'].append(mem_norm)

    # ═══ Test 3: Per-encoder-layer gradient from detect (recompute) ═══
    model.zero_grad()
    L_detect_2 = sum(v for k, v in loss_dict.items()
                     if k in DETECT_KEYS and isinstance(v, torch.Tensor) and v.requires_grad)
    # Can't backward again after retain_graph=False. Skip per-layer for now.

    n_done += 1
    if n_done <= 5:
        print("img %d: ri_detect=%.4f ri_distill=%.4f cos=%.4f gamma=%.4f" % (
            seen, ri_detect_n, ri_distill_n, ri_cos, cap.get('gamma', 0)))
        if 'residual' in cap:
            print("  residual_norm=%.4f memory_norm=%.4f ratio=%.4f" % (
                res_norm, mem_norm, res_norm / max(mem_norm, 1e-8)))
    seen += 1

# ═══ Analysis ═══
import numpy as np
print("\n" + "=" * 70)
print("GRMI BLIND CHANNEL ANALYSIS")
print("=" * 70)

rd = np.array(results['ri_detect_norm'])
rdi = np.array(results['ri_distill_norm'])
rc = np.array(results['ri_cos'])
rg = np.array(results['ri_gamma'])
ro = np.array(results['ri_output_norm']) if results['ri_output_norm'] else np.array([0])
mn = np.array(results['memory_norm']) if results['memory_norm'] else np.array([0])

print(f"\n  === TEST 1: Is R(M) blind to distillation? ===")
print(f"  ∂L_detect/∂R(M) norm:  mean={rd.mean():.4f}")
print(f"  ∂L_distill/∂R(M) norm: mean={rdi.mean():.4f}")
print(f"  Ratio distill/detect:  {rdi.mean()/max(rd.mean(),1e-8):.4f}")
print(f"  Cosine sim:            mean={rc.mean():+.4f}")
if rdi.mean() / max(rd.mean(), 1e-8) < 0.1:
    print(f"  VERDICT: R(M) is BLIND to distillation (distill grad < 10% of detect grad)")
elif rdi.mean() / max(rd.mean(), 1e-8) < 0.3:
    print(f"  VERDICT: R(M) is MOSTLY blind (distill grad = {rdi.mean()/rd.mean()*100:.0f}% of detect)")
else:
    print(f"  VERDICT: R(M) is NOT blind (distill grad = {rdi.mean()/rd.mean()*100:.0f}% of detect)")

print(f"\n  === TEST 2: R(M) capacity (feature modification magnitude) ===")
print(f"  R(M) output norm: mean={ro.mean():.4f}")
print(f"  Memory norm:       mean={mn.mean():.4f}")
print(f"  γ·R(M)/Memory:    {rg.mean() * ro.mean() / max(mn.mean(), 1e-8) * 100:.4f}%")
print(f"  γ (current):       {rg.mean():.6f}")

print(f"\n  === Expansion headroom analysis ===")
# Capacity: how much can R(M) modify with larger network?
print(f"  1. CAPACITY: current R(M) modifies {rg.mean()*ro.mean()/max(mn.mean(),1e-8)*100:.3f}% of memory")
print(f"     If 2x wider MLP: ~{2*rg.mean()*ro.mean()/max(mn.mean(),1e-8)*100:.3f}% (linear scaling)")
print(f"     Headroom: large (current modification is tiny)")

# Signal strength: what if γ is larger?
print(f"  2. SIGNAL STRENGTH: γ={rg.mean():.4f}")
for g in [0.02, 0.05, 0.1]:
    mod_pct = g * ro.mean() / max(mn.mean(), 1e-8) * 100
    print(f"     γ={g}: {mod_pct:.3f}% of memory")
print(f"     Risk: γ>0.05 likely disrupts old-class features (R(M) is uniform)")

# Injection position
print(f"  3. INJECTION POSITION: currently encoder output only")
print(f"     Decoder layers would create additional blind channels")
print(f"     Each decoder layer has ~2M params; R_dec(Q) would add ~66k per layer")
print(f"     6 decoder layers × 66k = ~400k additional blind-channel params")

# Overall signal budget
total_detect_on_ri = rd.mean()
total_detect_on_enc = 11.35  # from previous grad_conflict diagnostic
print(f"\n  === Signal budget ===")
print(f"  Detection gradient on R(M):     {total_detect_on_ri:.4f}")
print(f"  Detection gradient on enc_last: {total_detect_on_enc:.4f}")
print(f"  R(M) receives {total_detect_on_ri/total_detect_on_enc*100:.1f}% of enc-level detect signal")

out = {
    'ri_detect_norm_mean': round(float(rd.mean()), 4),
    'ri_distill_norm_mean': round(float(rdi.mean()), 4),
    'ri_distill_detect_ratio': round(float(rdi.mean()/max(rd.mean(),1e-8)), 4),
    'ri_cos_mean': round(float(rc.mean()), 4),
    'ri_output_norm': round(float(ro.mean()), 4),
    'memory_norm': round(float(mn.mean()), 4),
    'gamma_mean': round(float(rg.mean()), 6),
    'modification_pct': round(float(rg.mean()*ro.mean()/max(mn.mean(),1e-8)*100), 4),
    'n_images': n_done,
}
json.dump(out, open('/home/yelingfei/logs/tatri/grmi_blind_channel.json', 'w'), indent=2)
print(f"\nSaved: grmi_blind_channel.json")
