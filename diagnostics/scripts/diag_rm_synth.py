"""
Self-contained R(M) Architecture Efficiency Diagnostic.

Uses BERT for real T_new embeddings. Generates synthetic M with controlled
levels of "new-class signal". Measures each architecture's ability to:
  1. Align R with T_new (cosine similarity)
  2. Boost new-class LGQS scores efficiently (Δscore / ||R||)

No mmdet/mmcv needed. Only transformers + torch.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, json, time
from transformers import BertTokenizer, BertModel

torch.manual_seed(42)
np.random.seed(42)

ALL_CLASSES = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse",
    "sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie",
    "suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove",
    "skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon",
    "bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut",
    "cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book",
    "clock","vase","scissors","teddy bear","hair drier","toothbrush"]

D = 256  # Grounding DINO feature dim
NEW_START = 70
NEW_END = 80
NUM_NEW = NEW_END - NEW_START

# ── ARCHITECTURES ──────────────────────────────────────────
class MLPResidual(nn.Module):
    def __init__(self, d=256, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, d))
        self.gamma = nn.Parameter(torch.tensor(0.01))
    def forward(self, M, T_new=None):
        return self.gamma * self.net(M)

class BilinearResidual(nn.Module):
    def __init__(self, d=256, bottleneck=64):
        super().__init__()
        self.W_m = nn.Linear(d, bottleneck, bias=False)
        self.W_t = nn.Linear(d, bottleneck, bias=False)
        self.W_out = nn.Linear(bottleneck, d, bias=False)
        self.gamma = nn.Parameter(torch.tensor(0.01))
    def forward(self, M, T_new):
        T_pool = T_new.mean(dim=0, keepdim=True)
        h = self.W_m(M) * self.W_t(T_pool).squeeze(0)
        return self.gamma * self.W_out(h)

class CrossAttnResidual(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(d, d, bias=False)
        self.tau = d ** 0.5
        self.gamma = nn.Parameter(torch.tensor(0.01))
    def forward(self, M, T_new):
        Q = self.W_q(T_new)
        K = self.W_k(M)
        A = F.softmax(Q @ K.T / self.tau, dim=-1)
        return self.gamma * (A.T @ Q)

def count_params(m):
    return sum(p.numel() for p in m.parameters())

def reinit(module, seed):
    torch.manual_seed(seed)
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    module.gamma.data.fill_(0.01)

def measure(arch, M, T_new, T_new_mean):
    """Return {rm_norm, alignment, score_gain, efficiency, coverage_gain}."""
    R = arch(M, T_new)
    rm_norm = R.norm(dim=-1).mean().item()
    alignment = F.cosine_similarity(R, T_new_mean.unsqueeze(0).expand_as(R), dim=-1).mean().item()

    M_aug = M + R
    s_orig = M @ T_new.T
    s_aug = M_aug @ T_new.T
    score_gain = (s_aug - s_orig).mean(dim=1).mean().item()

    # Coverage gain: fraction of positions where max new-class score enters top-900
    # Simulate top-900: fraction of N positions
    k = max(1, M.shape[0] * 900 // 20000)  # approx top-900 out of ~20000 positions
    _, topk_orig = torch.topk(s_orig.max(dim=-1).values, k)
    _, topk_aug = torch.topk(s_aug.max(dim=-1).values, k)
    coverage_gain = len(set(topk_aug.tolist()) - set(topk_orig.tolist())) / max(k, 1)

    eff = score_gain / max(rm_norm, 1e-8)
    return {"rm_norm": rm_norm, "alignment": alignment, "score_gain": score_gain,
            "efficiency": eff, "coverage_gain": coverage_gain}

# ── MAIN ───────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("=" * 60)
    print("R(M) Architecture Efficiency Diagnostic")
    print("=" * 60)

    # ── 1. Get real T_new from BERT ──
    print("\n[1/4] Extracting BERT text embeddings for new classes...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert = BertModel.from_pretrained("bert-base-uncased").to(device).eval()

    # Build class prompt (same as GCD)
    cap = ". ".join(ALL_CLASSES) + "."
    tok = tokenizer(cap, return_tensors="pt", padding=True, truncation=True, max_length=256)
    tok = {k: v.to(device) for k, v in tok.items()}

    with torch.no_grad():
        out = bert(**tok)
    token_emb = out.last_hidden_state[0]  # (N_tok, 768)

    # Find new-class token positions
    offsets = tokenizer(cap, return_offsets_mapping=True).offset_mapping
    new_tokens = set()
    cursor = 0
    for ci, cname in enumerate(ALL_CLASSES):
        idx = cap.find(cname, cursor)
        if idx < 0: continue
        c0, c1 = idx, idx + len(cname)
        toks = [ti for ti, (s, e) in enumerate(offsets) if s < c1 and e > c0]
        if ci >= NEW_START and ci < NEW_END:
            new_tokens.update(toks)
        cursor = c1

    new_token_list = sorted(new_tokens)
    # Average token embeddings for each new class
    T_new_class = []
    cursor = 0
    for ci in range(NEW_START, NEW_END):
        idx = cap.find(ALL_CLASSES[ci], cursor)
        if idx < 0: continue
        c0, c1 = idx, idx + len(ALL_CLASSES[ci])
        toks = [ti for ti, (s, e) in enumerate(offsets) if s < c1 and e > c0]
        if toks:
            T_new_class.append(token_emb[toks].mean(dim=0))
        cursor = c1
    T_new_class = torch.stack(T_new_class)  # (10, 768)

    # Project to 256-dim (simulate Grounding DINO text projection)
    # Grounding DINO projects BERT 768 → 256 internally
    proj = nn.Linear(768, D, bias=False).to(device)
    nn.init.xavier_uniform_(proj.weight, gain=0.5)
    with torch.no_grad():
        T_new = proj(T_new_class)  # (10, 256)
    T_new = T_new.detach()
    T_new_mean = T_new.mean(dim=0)
    print(f"  T_new shape: {T_new.shape}, norm: {T_new.norm(dim=-1).mean():.2f}")

    # ── 2. Generate controlled M ──
    print("\n[2/4] Generating controlled encoder memory features...")
    N_pos = 5000  # simulate 5000 spatial positions
    # Three scenarios:
    scenarios = {
        "weak_signal":   0.03,  # s_gt ≈ 0.03 (worst new class)
        "medium_signal": 0.10,  # s_gt ≈ 0.10 (typical new class)
        "strong_signal": 0.25,  # s_gt ≈ 0.25 (best new class, like old)
    }

    all_scenarios = {}
    for name, signal in scenarios.items():
        # M has: (1-α) random + α aligned with T_new
        M_random = torch.randn(N_pos, D, device=device)
        M_random = M_random / M_random.norm(dim=-1, keepdim=True) * 15.0  # mem_norm ≈ 15

        # Aligned component: M_aligned has high dot product with T_new
        M_aligned_contrib = signal * T_new_mean.unsqueeze(0).expand(N_pos, -1)
        M_aligned_contrib = M_aligned_contrib + 0.3 * torch.randn(N_pos, D, device=device)

        # Mix: M = (1-s)*random + s*aligned
        M = (1 - signal) * M_random + signal * M_aligned_contrib
        M = M.detach()
        s_gt = (M @ T_new.T).max(dim=-1).values.mean().item()
        all_scenarios[name] = {"M": M, "s_gt": s_gt,
                               "label": f"signal={signal:.2f}, s_gt_mean={s_gt:.3f}"}
        print(f"  {name:15s}: s_gt_mean = {s_gt:.4f}")

    # ── 3. Compare architectures ──
    print("\n[3/4] Comparing architectures...")
    archs = {
        "MLP":        MLPResidual(d=D, hidden=128),
        "Bilinear":   BilinearResidual(d=D, bottleneck=64),
        "CrossAttn":  CrossAttnResidual(d=D),
    }

    for name, a in archs.items():
        print(f"  {name:12s}: {count_params(a):>6d} params")

    N_TRIALS = 10
    results = {}

    for arch_name, ArchClass in archs.items():
        arch_results = {}
        for sname, sdata in all_scenarios.items():
            M = sdata["M"]
            trial_metrics = {k: [] for k in ["rm_norm", "alignment", "score_gain", "efficiency", "coverage_gain"]}
            for trial in range(N_TRIALS):
                arch = ArchClass.to(device)
                reinit(arch, seed=1000 + trial)
                m = measure(arch, M, T_new, T_new_mean)
                for k in trial_metrics:
                    trial_metrics[k].append(m[k])
            arch_results[sname] = {k: (np.mean(v), np.std(v)) for k, v in trial_metrics.items()}
        results[arch_name] = arch_results

    # ── 4. Print comparison tables ──
    print("\n" + "=" * 70)
    print("RESULTS: Efficiency = Δscore / ||R|| (higher = more signal per perturbation)")
    print("=" * 70)

    for sname in scenarios:
        print(f"\n── Scenario: {all_scenarios[sname]['label']} ──")
        print(f"{'Arch':>12s} | {'||R||':>8s} | {'cos(R,T)':>10s} | {'Δscore':>10s} | {'efficiency':>10s} | {'Δcov%':>8s}")
        print("-" * 72)
        best_eff = -1; best_name = ""
        for arch_name in archs:
            r = results[arch_name][sname]
            rm = f"{r['rm_norm'][0]:.4f}"
            al = f"{r['alignment'][0]:.4f}"
            sg = f"{r['score_gain'][0]:.4f}"
            ef = f"{r['efficiency'][0]:.4f}"
            cv = f"{r['coverage_gain'][0]*100:.1f}%"
            print(f"{arch_name:>12s} | {rm:>8s} | {al:>10s} | {sg:>10s} | {ef:>10s} | {cv:>8s}")
            if r['efficiency'][0] > best_eff:
                best_eff = r['efficiency'][0]
                best_name = arch_name
        print(f"  → Best: {best_name} (efficiency = {best_eff:.4f})")

    # ── 5. Mini-training test ──
    print("\n" + "=" * 70)
    print("MINI-TRAINING: 100 SGD steps, new-class score maximization")
    print("=" * 70)

    M_train = all_scenarios["medium_signal"]["M"]
    n_steps = 100
    lr = 0.01

    training_results = {}
    for arch_name, ArchClass in archs.items():
        arch = ArchClass.to(device)
        reinit(arch, seed=42)
        opt = torch.optim.SGD(arch.parameters(), lr=lr)
        history = {"step": [], "rm_norm": [], "score_gain": [], "efficiency": []}

        for step in range(n_steps):
            opt.zero_grad()
            R = arch(M_train, T_new)
            M_aug = M_train + R
            # Objective: maximize new-class LGQS score
            s_aug = M_aug @ T_new.T
            loss = -s_aug.max(dim=-1).values.mean()  # maximize max new-class score
            loss.backward()
            opt.step()

            if step % 20 == 0:
                with torch.no_grad():
                    m = measure(arch, M_train, T_new, T_new_mean)
                history["step"].append(step)
                history["rm_norm"].append(m["rm_norm"])
                history["score_gain"].append(m["score_gain"])
                history["efficiency"].append(m["efficiency"])

        training_results[arch_name] = history
        final_eff = history["efficiency"][-1]
        final_rm = history["rm_norm"][-1]
        final_sg = history["score_gain"][-1]
        print(f"  {arch_name:12s}: final  ||R||={final_rm:.3f}  Δscore={final_sg:.4f}  eff={final_eff:.4f}")

    # Find best after training
    best_final = max(training_results, key=lambda k: training_results[k]["efficiency"][-1])
    print(f"\n  → Best after training: {best_final}")

    # ── SAVE ──
    out = {
        "scenarios": {k: {"s_gt": v["s_gt"]} for k, v in all_scenarios.items()},
        "results": {k: {sk: {mk: (mv[0], mv[1]) for mk, mv in sv.items()} for sk, sv in v.items()}
                    for k, v in results.items()},
        "training": {k: {kk: vv[-1] for kk, vv in v.items() if kk != "step"}
                     for k, v in training_results.items()},
        "best_initial": best_name,
        "best_trained": best_final,
    }
    with open("/root/autodl-tmp/rm_arch_diag.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nResults saved to /root/autodl-tmp/rm_arch_diag.json")


if __name__ == "__main__":
    main()
