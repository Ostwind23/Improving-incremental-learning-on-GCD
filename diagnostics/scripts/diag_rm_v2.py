"""
R(M) Architecture Diagnostic v2 — realistic init + mini-training.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

D, N, K = 256, 5000, 10
STEPS, LR, TRIALS = 200, 0.001, 3
device = 'cuda'
torch.manual_seed(42); np.random.seed(42)

# ── T_new: BERT-like embeddings, norm ≈ 11 ──
T_new = torch.randn(K, D, device=device)
T_new = T_new / T_new.norm(dim=-1, keepdim=True) * 11.0

# ── M: realistic encoder memory, norm ≈ 15.5, near-orthogonal to T ──
M = torch.randn(N, D, device=device)
M = M / M.norm(dim=-1, keepdim=True) * 15.5
M_mean = M.norm(dim=-1).mean()
T_mean = T_new.norm(dim=-1).mean()
s_gt = (M @ T_new.T).max(dim=-1).values
print(f"M_norm={M_mean:.1f} T_norm={T_mean:.1f} s_gt_mean={s_gt.mean():.1f}")

# ── Architectures ──
def init_weights(module, std1=0.1, std2=0.05):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(std=std1 if m.weight.shape[0] <= 256 else std2)
            if m.bias is not None: m.bias.data.zero_()

class MLPRes(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D,128), nn.ReLU(), nn.Linear(128,D))
        init_weights(self)
    def forward(self, M, T=None):
        return self.net(M)

class BilinearRes(nn.Module):
    def __init__(self, b=64):
        super().__init__()
        self.W_m = nn.Linear(D, b, bias=False)
        self.W_t = nn.Linear(D, b, bias=False)
        self.W_out = nn.Linear(b, D, bias=False)
        init_weights(self)
    def forward(self, M, T):
        Tp = T.mean(0, keepdim=True)
        h = self.W_m(M) * self.W_t(Tp).squeeze(0)
        return self.W_out(h)

class CrossAttnRes(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_q = nn.Linear(D, D, bias=False)
        self.W_k = nn.Linear(D, D, bias=False)
        init_weights(self, std1=0.05, std2=0.05)
    def forward(self, M, T):
        Q, K = self.W_q(T), self.W_k(M)
        A = F.softmax(Q @ K.T / 16.0, dim=-1)
        return A.T @ Q

def measure(arch, M, T):
    with torch.no_grad():
        R = arch(M, T) if not isinstance(arch, MLPRes) else arch(M)
        rn = R.norm(dim=-1).mean().item()
        al = F.cosine_similarity(R, T.mean(0).unsqueeze(0).expand_as(R), dim=-1).mean().item()
        Ma = M + R
        sg = (Ma @ T.T - M @ T.T).max(dim=-1).values.mean().item()
        # Top-900 coverage change
        k = max(1, N * 900 // 20000)
        _, topk_orig = torch.topk((M @ T.T).max(dim=-1).values, k)
        _, topk_aug = torch.topk((Ma @ T.T).max(dim=-1).values, k)
        cg = len(set(topk_aug.tolist()) - set(topk_orig.tolist())) / max(k, 1)
    return rn, al, sg, cg

print(f"\n{'='*72}")
print(f"INITIAL (random init)")
print(f"{'='*72}")
print(f"{'Arch':>12s} | {'||R||':>8s} | {'cos':>8s} | {'dscore':>8s} | {'Δcov':>7s}")
print("-" * 50)

for name, cls in [("MLP", MLPRes), ("Bilinear", BilinearRes), ("CrossAttn", CrossAttnRes)]:
    a = cls().to(device).eval()
    rn, al, sg, cg = measure(a, M, T_new)
    print(f"{name:>12s} | {rn:8.3f} | {al:8.4f} | {sg:8.4f} | {cg*100:6.1f}%")

print(f"\n{'='*72}")
print(f"MINI-TRAINING ({STEPS} steps, {TRIALS} trials)")
print(f"{'='*72}")
print(f"{'Arch':>12s} | {'||R||_end':>8s} | {'cos_end':>8s} | {'dscore':>8s} | {'Δcov':>7s} | {'eff':>7s}")
print("-" * 56)

for name, cls in [("MLP", MLPRes), ("Bilinear", BilinearRes), ("CrossAttn", CrossAttnRes)]:
    best_trial = {}
    for trial in range(TRIALS):
        a = cls().to(device).train()
        opt = torch.optim.SGD(a.parameters(), lr=LR)
        for step in range(STEPS):
            opt.zero_grad()
            R = a(M, T_new) if not isinstance(a, MLPRes) else a(M)
            loss = -(M + R) @ T_new.T  # (N, K)
            loss = -loss.max(dim=-1).values.mean()  # maximize max
            loss.backward()
            opt.step()
        a.eval()
        rn, al, sg, cg = measure(a, M, T_new)
        eff = sg / max(rn, 1e-6)
        if not best_trial or eff > best_trial.get("eff", -999):
            best_trial = {"rn": rn, "al": al, "sg": sg, "cg": cg, "eff": eff}
        torch.cuda.empty_cache()

    r = best_trial
    print(f"{name:>12s} | {r['rn']:8.3f} | {r['al']:8.4f} | {r['sg']:8.4f} | {r['cg']*100:6.1f}% | {r['eff']:7.2f}")

# ── Training curve for best trial ──
print(f"\n{'='*72}")
print(f"TRAINING CURVES (best trial each)")
print(f"{'='*72}")
for name, cls in [("MLP", MLPRes), ("Bilinear", BilinearRes), ("CrossAttn", CrossAttnRes)]:
    a = cls().to(device).train()
    opt = torch.optim.SGD(a.parameters(), lr=LR)
    curve = []
    for step in range(STEPS):
        opt.zero_grad()
        R = a(M, T_new) if not isinstance(a, MLPRes) else a(M)
        loss = -(M + R) @ T_new.T
        loss = -loss.max(dim=-1).values.mean()
        loss.backward()
        opt.step()
        if step % 25 == 0:
            a.eval()
            rn, al, sg, cg = measure(a, M, T_new)
            curve.append((step, rn, sg, cg))
            a.train()
    a.eval()
    rn, al, sg, cg = measure(a, M, T_new)
    curve.append((STEPS, rn, sg, cg))
    print(f"\n{name}:")
    for s, rn, sg, cg in curve:
        print(f"  step{s:4d}: ||R||={rn:.3f}  dscore={sg:.4f}  dcov={cg*100:.1f}%")
    torch.cuda.empty_cache()

print("\nDONE.")
