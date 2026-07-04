"""Scale up to realistic loss magnitude and measure gate gradient."""
import torch, numpy as np
from mmengine.registry import DefaultScope
DefaultScope.get_instance('mmdet', scope_name='mmdet')
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmdet.registry import MODELS
from transformers import BertModel, BertTokenizer
from pycocotools.coco import COCO

cfg = Config.fromfile('configs/gdino_inc/70+10/grmi_rms_gate_3e.py')
cfg.default_scope = 'mmdet'
m = MODELS.build(cfg.model)
load_checkpoint(m, 'work_dirs/grmi_rms_gate_3e_v2/epoch_3.pth', map_location='cuda:0')
m.cuda(); ri = m.residual_inject; ri.train()

coco = COCO('data/coco/annotations/instances_val2017.json')
bert = BertModel.from_pretrained('bert-base-uncased').cuda().eval()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
names = [coco.loadCats([c])[0]['name'] for c in sorted(coco.getCatIds())[70:80]]
tok = tokenizer(' . '.join(names)+' .', return_tensors='pt', padding='max_length', max_length=256, truncation=True)
out = bert(input_ids=tok['input_ids'].cuda(), attention_mask=tok['attention_mask'].cuda())
T_new = m.text_feat_map(out.last_hidden_state).squeeze(0).detach()
T_pool = T_new.mean(0, keepdim=True)

torch.manual_seed(42)
M = torch.randn(1, 900, 256).cuda() * 15
h = ri.W_m(M) * ri.W_t(T_pool).squeeze(0)
R_raw = ri.W_out(h)
rms = R_raw.norm(dim=-1).mean()
R = R_raw * (1.0 / (rms + 1e-8))
gate = ri.gate_net(M)
R_gated = R * gate
Mp = M + ri.gamma * R_gated

# Scale loss to ~10 to match real training scale
target = M + torch.randn_like(M) * 0.5
loss_raw = torch.nn.functional.mse_loss(Mp, target)
scale = 10.0 / (loss_raw.item() + 1e-8)
loss = scale * loss_raw
print(f"Loss: {loss_raw.item():.4f} → scaled to {loss.item():.2f}")
loss.backward()

gb = ri.gate_net[2].bias.grad.abs().item()
wo = ri.W_out.weight.grad.abs().mean().item()
wm = ri.W_m.weight.grad.abs().mean().item()

print(f"\nGate bias grad (loss≈10): {gb:.2e}")
print(f"W_out grad:               {wo:.2e}")
print(f"W_m grad:                 {wm:.2e}")
print(f"Gate/W_out = {gb/wo:.1f}x")
print(f"Gate/W_m   = {gb/wm:.1f}x")

lr = 5e-5
obs_move = 0.0004 / 7500  # per-step observed bias move
print(f"\nAt LR={lr}:")
print(f"  Expected per-step bias move: {gb*lr:.2e}")
print(f"  Over 7500 steps: {gb*lr*7500:.4f}")
print(f"  Observed per-step total: {obs_move:.2e}")
print(f"  Ratio (expected/observed): {gb*lr/obs_move:.1f}x")
print(f"  → Decoder attenuation factor: {obs_move/(gb*lr):.2e}×")
