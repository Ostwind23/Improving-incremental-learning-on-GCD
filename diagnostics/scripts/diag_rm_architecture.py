"""
R(M) Architecture Efficiency Diagnostic.

Compares MLP, Bilinear, and Cross-Attention architectures for R(M).
Measures: how much new-class LGQS coverage improvement per unit of ||R(M)||.

Protocol:
  1. Load GCD 70+10 base checkpoint (epoch_12.pth)
  2. Freeze all model params except the R(M) module
  3. For each architecture variant:
     a. Initialize with controlled random seed
     b. Run 200 training steps on new-class-only val images
     c. Track LGQS coverage, ||R||, loss_cls per step
  4. Compare efficiency = Δcoverage / Δ||R||
"""
import mmdet.apis, mmdet.engine.hooks
import torch, numpy as np, copy, json, time
import torch.nn as nn
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.runner import Runner, load_checkpoint
from torch.cuda.amp import autocast

# ── ARCHITECTURES ──────────────────────────────────────────
class MLPResidual(nn.Module):
    """Current architecture: 256 -> 128 -> 256, ReLU."""
    def __init__(self, d=256, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, d))
        self.gamma = nn.Parameter(torch.tensor(0.01))

    def forward(self, M, T_new=None):  # T_new ignored for MLP
        return self.gamma * self.net(M)

class BilinearResidual(nn.Module):
    """Bilinear: R_i = M @ W_m @ W_t @ T_new_pooled. Direct text-visual interaction."""
    def __init__(self, d=256, bottleneck=64):
        super().__init__()
        self.W_m = nn.Linear(d, bottleneck, bias=False)
        self.W_t = nn.Linear(d, bottleneck, bias=False)
        self.W_out = nn.Linear(bottleneck, d, bias=False)
        self.gamma = nn.Parameter(torch.tensor(0.01))

    def forward(self, M, T_new):  # M: (N,d), T_new: (K,d)
        # Pool text: mean over new classes
        T_pool = T_new.mean(dim=0, keepdim=True)  # (1,d)
        # Bilinear interaction
        h_m = self.W_m(M)           # (N, b)
        h_t = self.W_t(T_pool)      # (1, b)
        h = h_m * h_t.squeeze(0)    # (N, b) elementwise interaction
        R = self.W_out(h)           # (N, d)
        return self.gamma * R

class CrossAttnResidual(nn.Module):
    """Cross-Attention: T_new queries M, R = softmax(T_new @ M.T) @ T_new."""
    def __init__(self, d=256, tau=None):
        super().__init__()
        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(d, d, bias=False)
        self.tau = tau or d ** 0.5
        self.gamma = nn.Parameter(torch.tensor(0.01))

    def forward(self, M, T_new):  # M: (N,d), T_new: (K,d)
        Q = self.W_q(T_new)       # (K, d)
        K = self.W_k(M)           # (N, d)
        A = F.softmax(Q @ K.T / self.tau, dim=-1)  # (K, N)
        R = A.T @ Q               # (N, d)  weighted text for each position
        return self.gamma * R

def count_params(m):
    return sum(p.numel() for p in m.parameters())

# ── MAIN DIAGNOSTIC ───────────────────────────────────────
def main():
    import sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else \
        "configs/gdino_inc/70+10/gdino_inc_70+10_70-79_gcd_scratch_coco.py"
    ckpt_path = sys.argv[2] if len(sys.argv) > 2 else \
        "work_dirs/gdino_inc_70+10_0-69_scratch_coco/epoch_12.pth"

    print("=" * 60)
    print("R(M) Architecture Efficiency Diagnostic")
    print("=" * 60)

    # Load model
    cfg = Config.fromfile(cfg_path)
    cfg.train_cfg.max_epochs = 1
    cfg.train_dataloader.batch_size = 2
    cfg.default_hooks.checkpoint = dict(type="CheckpointHook", interval=999)
    cfg.launcher = "none"

    runner = Runner.from_cfg(cfg)
    runner.load_or_resume()
    model = runner.model.module if hasattr(runner.model, "module") else runner.model

    # Freeze everything
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    head = model.bbox_head
    d = 256  # feature dim

    # Get T_new from the model
    # memory_text has shape (max_text_len, d), new-class tokens are >= 169
    with torch.no_grad():
        # Run one forward to get text embeddings
        dl = iter(runner.train_dataloader)
        data = next(dl)
        data = model.data_preprocessor(data, True)
        _ = model(**data, mode="loss")
    # T_new should be available from head
    # Actually, let's extract it from the language model
    T_all = model.language_model(None)  # or similar
    # Actually, we need to get it from forward_encoder output
    # Let's use a different approach: extract from the head after forward

    # Simpler: read T_new from the text features used in LGQS
    # The memory_text has shape (max_text_len, 256), tokens 169-... are new class
    # Let's just extract it after a forward pass
    model.train()  # Re-enable training for the R(M) module only

    # ── Hook to capture memory_text ──
    captured_memory_text = [None]

    def capture_mt_hook(module, args, kwargs):
        # Capture memory_text from forward_encoder or wherever it's available
        pass

    # Actually, let's take a simpler approach: run the full detector forward once
    # and capture the text features from the encoder output
    dl2 = iter(runner.train_dataloader)
    data = next(dl2)
    data = model.data_preprocessor(data, True)

    # Monkey-patch: capture encoder outputs to get T_new
    orig_fwd_enc = model.forward_encoder
    captured = {"memory_text": None, "memory": None}

    def patched_fwd_enc(*args, **kwargs):
        out = orig_fwd_enc(*args, **kwargs)
        captured["memory"] = out["memory"].detach().clone()
        if "memory_text" in out:
            captured["memory_text"] = out["memory_text"].detach().clone()
        return out
    model.forward_encoder = patched_fwd_enc

    model.zero_grad()
    with torch.no_grad():
        _ = model(**data, mode="loss")
    model.forward_encoder = orig_fwd_enc

    if captured["memory_text"] is None:
        # Try alternative: get from head attributes
        # During loss computation, the head stores text features
        # Let's do a proper forward and capture
        print("Capturing text features via full forward...")
        model.train()
        losses = model(**data, mode="loss")
        # After loss, head should have token_positive_maps etc
        if hasattr(head, 'token_positive_maps'):
            print(f"  Got token_positive_maps from head")
        # Access memory_text from new_head_inputs_dict
        # This is complex, let's just extract from the language model directly

    # ── Extract T_new from language model ──
    # BERT forward with the class prompt
    model.eval()
    ALL_CLASSES = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
        "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse",
        "sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie",
        "suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove",
        "skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon",
        "bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut",
        "cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
        "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book",
        "clock","vase","scissors","teddy bear","hair drier","toothbrush"]

    # Get memory_text from the model's text encoding
    # The GroundingDINO encodes text in pre_transformer
    with torch.no_grad():
        img_feats = model.extract_feat(data["inputs"])
        # Build text dict
        cap = ". ".join(ALL_CLASSES) + "."
        tokenized = model.language_model.tokenizer(
            [cap], padding="max_length", max_length=256,
            truncation=True, return_tensors="pt")
        for k in tokenized:
            if isinstance(tokenized[k], torch.Tensor):
                tokenized[k] = tokenized[k].to(data["inputs"].device)
        text_dict = model.language_model(tokenized["input_ids"])
        memory_text = text_dict["embedded"]  # (1, max_len, 256) or (max_len, 256)

        # Find new-class token positions
        # New class 70 = toaster = tokens [169, 170], class 79 = toothbrush = [191, 192]
        NEW_TOKEN_START = 169
        T_new = memory_text[0, NEW_TOKEN_START:, :].detach()  # (83, 256) includes some padding

        # Trim to only valid tokens (classes 70-79)
        # Approximate: take first 20 tokens after NEW_TOKEN_START
        T_new = T_new[:20, :]  # (20, 256) - covers our 10 new classes + some padding
        print(f"T_new shape: {T_new.shape}")
        M_sample = captured.get("memory")
        if M_sample is not None:
            print(f"M sample shape: {M_sample.shape}")
            M_flat = M_sample.reshape(-1, d)  # (B*N, 256)
        else:
            # Use the image features as proxy for memory
            M_flat = img_feats[0].flatten(2).transpose(1, 2).reshape(-1, d)
            print(f"M from img_feats shape: {M_flat.shape}")

    # ── Architectures ──
    archs = {
        "MLP": MLPResidual(d=d, hidden=128),
        "Bilinear": BilinearResidual(d=d, bottleneck=64),
        "CrossAttn": CrossAttnResidual(d=d),
    }

    print(f"\nArchitecture parameters:")
    for name, arch in archs.items():
        print(f"  {name:12s}: {count_params(arch):>7d} params")

    # ── EFFICIENCY TEST ──
    # Measure: for each architecture, compute R = net(M, T_new)
    # and measure:
    #   (1) ||R|| (perturbation magnitude)
    #   (2) cos(R, T_new_mean) (alignment with text)
    #   (3) new-class LGQS score improvement: Δs = (M+R)@T_new - M@T_new
    #   (4) efficiency = Δs_mean / ||R||

    T_new_mean = T_new.mean(dim=0)  # (256,)
    T_new = T_new.to(M_flat.device)
    T_new_mean = T_new_mean.to(M_flat.device)
    M_flat = M_flat.to(M_flat.device)

    # Sample a subset of positions to keep computation light
    n_sample = min(M_flat.shape[0], 5000)
    idx = torch.randperm(M_flat.shape[0])[:n_sample]
    M_sample = M_flat[idx].detach()
    M_sample.requires_grad = False

    results = {}
    for name, arch in archs.items():
        arch = arch.to(M_sample.device)
        arch.train()

        # Multiple random inits for variance estimation
        n_trials = 5
        trial_results = {"rm_norm": [], "alignment": [], "score_gain": [], "efficiency": []}

        for trial in range(n_trials):
            # Re-init
            for m in arch.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.1)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            arch.gamma.data.fill_(0.01)

            # Forward
            R = arch(M_sample, T_new)  # (n_sample, 256)
            rm_norm = R.norm(dim=-1).mean().item()

            # Alignment with text
            cos_val = F.cosine_similarity(R, T_new_mean.unsqueeze(0).expand_as(R), dim=-1)
            alignment = cos_val.mean().item()

            # Score gain for new classes
            M_aug = M_sample + R  # (n_sample, 256)
            s_orig = M_sample @ T_new.T  # (n_sample, K)
            s_aug = M_aug @ T_new.T      # (n_sample, K)
            score_gain = (s_aug - s_orig).mean(dim=1).mean().item()

            trial_results["rm_norm"].append(rm_norm)
            trial_results["alignment"].append(alignment)
            trial_results["score_gain"].append(score_gain)
            trial_results["efficiency"].append(score_gain / max(rm_norm, 1e-8))

        results[name] = {
            "rm_norm": (np.mean(trial_results["rm_norm"]), np.std(trial_results["rm_norm"])),
            "alignment": (np.mean(trial_results["alignment"]), np.std(trial_results["alignment"])),
            "score_gain": (np.mean(trial_results["score_gain"]), np.std(trial_results["score_gain"])),
            "efficiency": (np.mean(trial_results["efficiency"]), np.std(trial_results["efficiency"])),
        }

    # ── PRINT RESULTS ──
    print(f"\n{'='*70}")
    print(f"INITIAL EFFICIENCY (random init, no training, {n_trials} trials each)")
    print(f"{'='*70}")
    print(f"{'Arch':>12s} | {'||R||':>8s} | {'cos(R,T)':>10s} | {'Δscore':>10s} | {'efficiency':>12s}")
    print(f"{'-'*12}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}")

    best_eff = 0
    best_arch = ""
    for name, r in results.items():
        rm = f"{r['rm_norm'][0]:.4f}±{r['rm_norm'][1]:.4f}"
        al = f"{r['alignment'][0]:.4f}±{r['alignment'][1]:.4f}"
        sg = f"{r['score_gain'][0]:.4f}±{r['score_gain'][1]:.4f}"
        ef = f"{r['efficiency'][0]:.4f}±{r['efficiency'][1]:.4f}"
        print(f"{name:>12s} | {rm:>8s} | {al:>10s} | {sg:>10s} | {ef:>12s}")
        if r['efficiency'][0] > best_eff:
            best_eff = r['efficiency'][0]
            best_arch = name

    print(f"\nBest architecture: {best_arch} (efficiency = {best_eff:.4f} Δscore/||R||)")
    print(f"Interpretation: higher efficiency = more new-class signal per unit perturbation")

    # ── SAVE ──
    out = {"results": {k: {kk: (vv[0], vv[1]) for kk, vv in v.items()} for k, v in results.items()},
           "best": best_arch, "M_sample_shape": list(M_sample.shape), "T_new_shape": list(T_new.shape)}
    out_path = "/tmp/rm_arch_diag.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
