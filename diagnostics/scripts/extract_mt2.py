"""Extract M + T_new from GCD checkpoint (lightweight, batch_size=1)."""
import mmdet.apis, mmdet.engine.hooks
import torch, pickle, os
from mmengine.config import Config
from mmengine.runner import Runner

cfg = Config.fromfile("configs/gdino_inc/70+10/grmi_t1_decouple_6e_coco.py")
cfg.train_cfg.max_epochs = 1; cfg.train_dataloader.batch_size = 1
cfg.work_dir = "/tmp/extract_m3"; cfg.launcher = "none"
cfg.default_hooks.checkpoint = dict(type="CheckpointHook", interval=999)
runner = Runner.from_cfg(cfg); runner.load_or_resume()
model = runner.model.module if hasattr(runner.model, "module") else runner.model
model.eval()

captured = []
orig = model.forward_encoder
def hook(*a, **kw):
    out = orig(*a, **kw)
    captured.append((out["memory"].detach().cpu(), out.get("memory_text")))
    return out
model.forward_encoder = hook

dl = iter(runner.train_dataloader)
data = next(dl); data = model.data_preprocessor(data, True)
with torch.no_grad():
    _ = model(**data, mode="loss")
model.forward_encoder = orig

M, MT = captured[0]
MT = MT.cpu()
M_flat = M.reshape(-1, 256)[:5000, :]
T_new = MT[169:189, :].cpu()

print(f"M: {M_flat.shape}, norm={M_flat.norm(dim=-1).mean():.3f}")
print(f"T_new: {T_new.shape}, norm={T_new.norm(dim=-1).mean():.3f}")

# Also compute s_gt stats for reference
s_gt = (M_flat @ T_new.T).max(dim=-1).values
print(f"s_gt: mean={s_gt.mean():.3f} max={s_gt.max():.3f}")

out = {"M": M_flat, "T_new": T_new}
path = "/home/yelingfei/logs/tatri/extracted_MT.pkl"
with open(path, "wb") as f:
    pickle.dump(out, f)
print(f"Saved: {os.path.getsize(path)/1024:.0f} KB")
del model, runner
torch.cuda.empty_cache()
print("Done")
