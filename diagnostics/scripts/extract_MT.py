"""Extract M (encoder memory) and T_new from GCD checkpoint for diagnostic."""
import mmdet.apis, mmdet.engine.hooks
import torch, pickle, os
from mmengine.config import Config
from mmengine.runner import Runner

cfg = Config.fromfile("configs/gdino_inc/70+10/grmi_t1_decouple_6e_coco.py")
cfg.train_cfg.max_epochs = 1; cfg.train_dataloader.batch_size = 4
cfg.work_dir = "/tmp/extract_m"; cfg.launcher = "none"
cfg.default_hooks.checkpoint = dict(type="CheckpointHook", interval=999)
runner = Runner.from_cfg(cfg); runner.load_or_resume()
model = runner.model.module if hasattr(runner.model, "module") else runner.model
model.eval()

# Hook to capture encoder memory and text
captured = {"memory": [], "memory_text": []}

orig_fwd_enc = model.forward_encoder
def patched_enc(*args, **kwargs):
    out = orig_fwd_enc(*args, **kwargs)
    captured["memory"].append(out["memory"].detach().cpu())
    if "memory_text" in out:
        captured["memory_text"].append(out["memory_text"].detach().cpu())
    return out
model.forward_encoder = patched_enc

# Run 10 batches
dl = iter(runner.train_dataloader)
for step in range(10):
    data = next(dl)
    data = model.data_preprocessor(data, True)
    with torch.no_grad():
        _ = model(**data, mode="loss")

model.forward_encoder = orig_fwd_enc

# Stack
M_all = torch.cat(captured["memory"], dim=0) if captured["memory"] else None
MT_all = captured["memory_text"][0] if captured["memory_text"] else None

print(f"M shape: {M_all.shape if M_all is not None else 'NONE'}")
print(f"memory_text shape: {MT_all.shape if MT_all is not None else 'NONE'}")
if M_all is not None:
    print(f"M norm: {M_all.norm(dim=-1).mean():.3f}")
    # Only keep new-class tokens: >= 169
    NEW_TOKEN_START = 169
    T_new = MT_all[NEW_TOKEN_START:, :].cpu()
    print(f"T_new shape: {T_new.shape}, norm: {T_new.norm(dim=-1).mean():.3f}")
    # Trim to 20 tokens (covers 10 new classes + padding)
    T_new = T_new[:20, :]
    # Save
    out = {"M": M_all[:5000, :], "T_new": T_new, 
           "M_norm": float(M_all.norm(dim=-1).mean()),
           "T_norm": float(T_new.norm(dim=-1).mean())}
    path = "/home/yelingfei/logs/tatri/extracted_MT.pkl"
    with open(path, "wb") as f:
        pickle.dump(out, f)
    print(f"Saved to {path} ({os.path.getsize(path)/1024:.0f} KB)")
