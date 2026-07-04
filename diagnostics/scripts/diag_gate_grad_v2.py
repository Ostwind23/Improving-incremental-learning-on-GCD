"""Measure gate gradient at encoder output level (bypass decoder complexity).
Creates a simple MSE loss on M' = M + gamma * R_gated, measures dL/d(gate_bias)."""
import torch, numpy as np
from mmengine.registry import DefaultScope
DefaultScope.get_instance('mmdet', scope_name='mmdet')
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmdet.registry import MODELS

cfg = Config.fromfile('configs/gdino_inc/70+10/grmi_rms_gate_3e.py')
cfg.default_scope = 'mmdet'
m = MODELS.build(cfg.model)
load_checkpoint(m, 'work_dirs/grmi_rms_gate_3e_v2/epoch_3.pth', map_location='cuda:0')
m.cuda()

ri = m.residual_inject
ri.train()

# Get T_new via standalone BERT
from transformers import BertModel, BertTokenizer
from pycocotools.coco import COCO
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained('bert-base-uncased').cuda().eval()
coco = COCO('data/coco/annotations/instances_val2017.json')
new_names = [coco.loadCats([c])[0]['name'] for c in sorted(coco.getCatIds())[70:80]]
prompt = ' . '.join(new_names) + ' .'
tok = tokenizer(prompt, return_tensors='pt', padding='max_length', max_length=256, truncation=True)
with torch.no_grad():
    out = bert(input_ids=tok['input_ids'].cuda(), attention_mask=tok['attention_mask'].cuda())
    T_all = m.text_feat_map(out.last_hidden_state)
    T_new = T_all.squeeze(0).detach()

torch.manual_seed(42)
N = 50
results = {'gate_bias_grad': [], 'W_out_grad': [], 'W_m_grad': []}

for _ in range(N):
    # Simulate encoder memory — random scale ~15
    memory = torch.randn(1, 900, 256).cuda() * 15
    memory.requires_grad = False  # detach: like in pre_decoder
    
    # Forward through Bilinear + RMS + Gate
    T_pool = T_new.mean(dim=0, keepdim=True)
    h = ri.W_m(memory) * ri.W_t(T_pool).squeeze(0)
    R = ri.W_out(h)
    rms = R.norm(dim=-1).mean()
    R = R * (1.0 / (rms + 1e-8))  # RMS
    gate = ri.gate_net(memory)  # no detach here — gradient test
    R_gated = R * gate
    M_prime = memory + ri.gamma * R_gated
    
    # Simple supervision: minimize MSE between M' and a target
    # Target: scale M slightly (simulates encoder adaptation)
    target = memory * 1.01  
    loss = torch.nn.functional.mse_loss(M_prime, target)
    loss.backward()
    
    # Record gate gradient  
    gb = ri.gate_net[2].bias.grad
    if gb is not None:
        results['gate_bias_grad'].append(gb.abs().item())
    else:
        results['gate_bias_grad'].append(0)
    
    wo = ri.W_out.weight.grad
    if wo is not None:
        results['W_out_grad'].append(wo.abs().mean().item())
    else:
        results['W_out_grad'].append(0)
    
    wm = ri.W_m.weight.grad
    if wm is not None:
        results['W_m_grad'].append(wm.abs().mean().item())
    else:
        results['W_m_grad'].append(0)
    
    ri.zero_grad()

print("\n===== Encoder-level gradient (simple MSE target) =====")
for k in ['gate_bias_grad', 'W_out_grad', 'W_m_grad']:
    v = [x for x in results[k] if x > 0]
    if v:
        a = np.array(v)
        print(f"  {k:20s}: mean={a.mean():.2e} median={np.median(a):.2e} std={a.std():.2e}")
    else:
        print(f"  {k:20s}: ALL ZERO")

gb = np.array([x for x in results['gate_bias_grad'] if x > 0])
wo = np.array([x for x in results['W_out_grad'] if x > 0])
wm = np.array([x for x in results['W_m_grad'] if x > 0])

if len(gb) > 0 and len(wo) > 0:
    print(f"\n  Gate/W_out ratio: {gb.mean()/wo.mean():.2e}")
    print(f"  Gate/W_m ratio:   {gb.mean()/wm.mean():.2e}")

# Now: what about gradient through the DETACHED memory path?
print("\n===== With memory.detach() (current code) =====")
results_d = {'gate_bias_grad': []}
for _ in range(N):
    memory = torch.randn(1, 900, 256).cuda() * 15
    T_pool = T_new.mean(dim=0, keepdim=True)
    h = ri.W_m(memory) * ri.W_t(T_pool).squeeze(0)
    R = ri.W_out(h)
    rms = R.norm(dim=-1).mean()
    R = R * (1.0 / (rms + 1e-8))
    gate = ri.gate_net(memory.detach())  # detach as in current code
    R_gated = R * gate
    M_prime = memory + ri.gamma * R_gated
    loss = torch.nn.functional.mse_loss(M_prime, memory * 1.01)
    loss.backward()
    gb = ri.gate_net[2].bias.grad
    results_d['gate_bias_grad'].append(gb.abs().item() if gb is not None else 0)
    ri.zero_grad()

v = [x for x in results_d['gate_bias_grad'] if x > 0]
if v:
    a = np.array(v)
    print(f"  gate_bias_grad (detached): mean={a.mean():.2e} median={np.median(a):.2e}")

# Ratio: with detach vs without
if len(gb) > 0 and len(v) > 0:
    print(f"  Detach effect: gradients {(np.array(v).mean()/gb.mean()):.2f}x with detach")
