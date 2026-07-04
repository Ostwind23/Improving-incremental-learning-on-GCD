"""Final M+T_new extraction. Saves before potential crash."""
import mmdet.apis, mmdet.engine.hooks
import torch, pickle, os
from mmengine.config import Config
from mmengine.runner import Runner

cfg = Config.fromfile("configs/gdino_inc/70+10/gdino_inc_70+10_70-79_gcd_scratch_coco.py")
cfg.train_cfg.max_epochs = 1; cfg.train_dataloader.batch_size = 1
cfg.work_dir = "/tmp/ex_final"; cfg.launcher = "none"
cfg.default_hooks.checkpoint = dict(type="CheckpointHook", interval=999)
runner = Runner.from_cfg(cfg); runner.load_or_resume()
model = runner.model.module if hasattr(runner.model, "module") else runner.model

# Get T_new
ALL80 = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse",
    "sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie",
    "suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove",
    "skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon",
    "bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut",
    "cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book",
    "clock","vase","scissors","teddy bear","hair drier","toothbrush"]
cap = ". ".join(ALL80) + "."
text_dict = model.language_model([cap])
emb = text_dict["embedded"][0]
with torch.no_grad():
    T_all = model.bbox_head.text_feat_map(emb)
Tn = T_all[169:189, :].cpu()
print(f"T_new: {Tn.shape}, norm={Tn.norm(dim=-1).mean():.1f}")

# Extract M
model.eval()
captured = {}
orig = model.forward_encoder
def hook(*a, **kw):
    out = orig(*a, **kw)
    captured["M"] = out["memory"].detach().cpu()
    return out
model.forward_encoder = hook

dl = iter(runner.val_dataloader)
batch = next(dl)
try:
    batch = model.data_preprocessor(batch, False)
    with torch.no_grad():
        _ = model(**batch, mode="loss")
except:
    pass
model.forward_encoder = orig

M = captured["M"].reshape(-1, 256)[:5000, :]
print(f"M: {M.shape}, norm={M.norm(dim=-1).mean():.1f}")
s = (M @ Tn.T).max(-1).values
print(f"s_gt: mean={s.mean():.3f} max={s.max():.3f}")

with open("/root/autodl-tmp/real_MT.pkl", "wb") as f:
    pickle.dump({"M": M, "T_new": Tn}, f)
print(f"Saved {os.path.getsize('/root/autodl-tmp/real_MT.pkl')//1024}KB")
