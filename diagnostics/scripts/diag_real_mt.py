"""
R(M) Architecture Diagnostic v3 — REAL M from GCD checkpoint + REAL T_new.
"""
import mmdet.apis, mmdet.engine.hooks
import torch, torch.nn as nn, torch.nn.functional as F, pickle, numpy as np, os

D, DEV = 256, 'cuda'
torch.manual_seed(42)

# ── Load real M and T_new ──
from mmengine.config import Config
from mmengine.runner import Runner

cfg = Config.fromfile("configs/gdino_inc/70+10/gdino_inc_70+10_70-79_gcd_scratch_coco.py")
cfg.train_cfg.max_epochs = 1; cfg.train_dataloader.batch_size = 2
cfg.work_dir = "/tmp/extract_real"; cfg.launcher = "none"
cfg.default_hooks.checkpoint = dict(type="CheckpointHook", interval=999)

runner = Runner.from_cfg(cfg); runner.load_or_resume()
model = runner.model.module if hasattr(runner.model, "module") else runner.model
model.eval()

# Capture encoder memory
captured = []
orig = model.forward_encoder
def hook(*a, **kw):
    out = orig(*a, **kw)
    captured.append((out["memory"].detach(), out.get("memory_text")))
    return out
model.forward_encoder = hook

dl = iter(runner.train_dataloader)
data = next(dl); data = model.data_preprocessor(data, True)
with torch.no_grad(): _ = model(**data, mode="loss")
model.forward_encoder = orig

M_raw, MT = captured[0]
M_flat = M_raw.reshape(-1, D)[:5000, :]
T_new = MT[169:189, :].detach()

print(f"REAL M: {M_flat.shape}, norm={M_flat.norm(dim=-1).mean():.1f}")
print(f"REAL T_new: {T_new.shape}, norm={T_new.norm(dim=-1).mean():.1f}")
s_gt = (M_flat @ T_new.T).max(dim=-1).values
print(f"s_gt: mean={s_gt.mean():.3f} max={s_gt.max():.3f}")

del model, runner; torch.cuda.empty_cache()

# ── Architectures ──
def init_w(m, std=0.1):
    for mod in m.modules():
        if isinstance(mod, nn.Linear):
            mod.weight.data.normal_(std=std)
            if mod.bias is not None: mod.bias.data.zero_()

class MLP(nn.Module):
    def __init__(self): super().__init__(); self.net = nn.Sequential(nn.Linear(256,128), nn.ReLU(), nn.Linear(128,256)); init_w(self)
    def f(self, M, T=None): return self.net(M)

class Bilinear(nn.Module):
    def __init__(self): super().__init__(); self.Wm=nn.Linear(256,64,bias=False); self.Wt=nn.Linear(256,64,bias=False); self.Wo=nn.Linear(64,256,bias=False); init_w(self)
    def f(self, M, T): h = self.Wm(M) * self.Wt(T.mean(0,keepdim=True)).squeeze(0); return self.Wo(h)

class CrossAttn(nn.Module):
    def __init__(self): super().__init__(); self.Wq=nn.Linear(256,256,bias=False); self.Wk=nn.Linear(256,256,bias=False); init_w(self, 0.05)
    def f(self, M, T): Q=self.Wq(T); K=self.Wk(M); A=F.softmax(Q@K.T/16.0,dim=-1); return A.T@Q

def measure(a, M, T):
    with torch.no_grad():
        R = a.f(M) if isinstance(a, MLP) else a.f(M, T)
        rn = R.norm(dim=-1).mean().item()
        al = F.cosine_similarity(R, T.mean(0).unsqueeze(0).expand_as(R), dim=-1).mean().item()
        sg = ((M+R) @ T.T - M @ T.T).max(dim=-1).values.mean().item()
        k = max(1, M.shape[0] * 900 // 20000)
        _, to = torch.topk((M@T.T).max(-1).values, k)
        _, ta = torch.topk(((M+R)@T.T).max(-1).values, k)
        cg = len(set(ta.tolist()) - set(to.tolist())) / max(k, 1)
    return rn, al, sg, cg

# ── Run ──
N = M_flat.shape[0]
print(f"\n{'='*65}")
print(f"REAL M ({N} positions) + REAL T_new ({T_new.shape[0]} classes)")
print(f"{'='*65}")
print(f"{'Arch':>12s} | {'||R||':>8s} | {'cos':>8s} | {'dscore':>8s} | {'dcov%':>7s} | {'eff':>7s}")

for nm, cl in [('MLP', MLP), ('Bilinear', Bilinear), ('CrossAttn', CrossAttn)]:
    a = cl().to(DEV).eval()
    rn, al, sg, cg = measure(a, M_flat, T_new)
    eff = sg / max(rn, 1e-8)
    print(f"{nm:>12s} | {rn:8.3f} | {al:8.4f} | {sg:8.4f} | {cg*100:6.1f}% | {eff:7.1f}")

# ── Mini-training ──
print(f"\n{'='*65}")
print(f"MINI-TRAINING (200 steps, SGD lr=0.001, best of 3)")
print(f"{'='*65}")
print(f"{'Arch':>12s} | {'||R||':>8s} | {'cos':>8s} | {'dscore':>8s} | {'dcov%':>7s} | {'eff':>7s}")

for nm, cl in [('MLP', MLP), ('Bilinear', Bilinear), ('CrossAttn', CrossAttn)]:
    best = {'rn': 0, 'al': 0, 'sg': 0, 'cg': 0, 'eff': -999}
    for trial in range(3):
        a = cl().to(DEV).train(); opt = torch.optim.SGD(a.parameters(), lr=0.001)
        for _ in range(200):
            opt.zero_grad()
            R = a.f(M_flat, T_new) if not isinstance(a, MLP) else a.f(M_flat)
            L = -(M_flat + R) @ T_new.T
            L = -L.max(dim=-1).values.mean()
            L.backward(); opt.step()
        a.eval(); rn, al, sg, cg = measure(a, M_flat, T_new)
        eff = sg / max(rn, 1e-8)
        if eff > best['eff']: best = {'rn': rn, 'al': al, 'sg': sg, 'cg': cg, 'eff': eff}
    b = best
    print(f"{nm:>12s} | {b['rn']:8.1f} | {b['al']:8.4f} | {b['sg']:8.1f} | {b['cg']*100:6.1f}% | {b['eff']:7.1f}")
    torch.cuda.empty_cache()

print(f"\n{'='*65}")
print(f"ANALYSIS: Training curve for best architectures")
print(f"{'='*65}")
for nm, cl in [('MLP', MLP), ('Bilinear', Bilinear), ('CrossAttn', CrossAttn)]:
    a = cl().to(DEV).train(); opt = torch.optim.SGD(a.parameters(), lr=0.001)
    curve = []
    for step in range(200):
        opt.zero_grad()
        R = a.f(M_flat, T_new) if not isinstance(a, MLP) else a.f(M_flat)
        L = -(M_flat + R) @ T_new.T; L = -L.max(dim=-1).values.mean()
        L.backward(); opt.step()
        if step % 25 == 0:
            a.eval(); rn, al, sg, cg = measure(a, M_flat, T_new); a.train()
            curve.append((step, rn, sg, cg))
    a.eval(); rn, al, sg, cg = measure(a, M_flat, T_new); curve.append((200, rn, sg, cg))
    print(f"\n{nm}:")
    for s, rn, sg, cg in curve[::2]:
        print(f"  step{s:4d}: ||R||={rn:.1f} ds={sg:.2f} dcov={cg*100:.1f}%")
    torch.cuda.empty_cache()

print("\nDONE")
