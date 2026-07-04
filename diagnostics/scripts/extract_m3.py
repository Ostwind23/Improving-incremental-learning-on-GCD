"""Minimal M extraction from val loader."""
import mmdet.apis, mmdet.engine.hooks
import torch
from mmengine.config import Config
from mmengine.runner import Runner

cfg = Config.fromfile("configs/gdino_inc/70+10/gdino_inc_70+10_70-79_gcd_scratch_coco.py")
cfg.train_cfg.max_epochs = 1; cfg.work_dir = "/tmp/ex6"; cfg.launcher = "none"
cfg.default_hooks.checkpoint = dict(type="CheckpointHook", interval=999)
runner = Runner.from_cfg(cfg); runner.load_or_resume()
model = runner.model.module if hasattr(runner.model, "module") else runner.model
model.eval()

captured = []
orig = model.forward_encoder
def hook(*a, **kw):
    out = orig(*a, **kw)
    captured.append((out["memory"].detach(), out.get("memory_text")))
    return out
model.forward_encoder = hook

# Val loop: iterate once
dl = iter(runner.val_dataloader)
batch = next(dl)
model.data_preprocessor.to("cuda")
data = model.data_preprocessor(batch, False)
with torch.no_grad():
    _ = model(**data, mode="loss")
model.forward_encoder = orig

M, MT = captured[0]
Mf = M.reshape(-1, 256)[:5000, :]
Tn = MT[169:189, :]
print(f"M: {Mf.shape}, norm={Mf.norm(dim=-1).mean():.1f}")
print(f"T: {Tn.shape}, norm={Tn.norm(dim=-1).mean():.1f}")
s = (Mf @ Tn.T).max(dim=-1).values
print(f"s_gt: mean={s.mean():.3f} max={s.max():.3f}")

import pickle
with open("/root/autodl-tmp/real_MT.pkl", "wb") as f:
    pickle.dump({"M": Mf.cpu(), "T_new": Tn.cpu()}, f)
print("Saved real_MT.pkl")
