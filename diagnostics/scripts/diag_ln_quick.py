"""Quick diagnosis: Bilinear pre-LN vs post-LN norm distribution"""
import torch, numpy as np
from mmengine.registry import DefaultScope
DefaultScope.get_instance('mmdet', scope_name='mmdet')

from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmdet.registry import MODELS

cfg = Config.fromfile('configs/gdino_inc/70+10/grmi_bilinear_12e_clean.py')
cfg.default_scope = 'mmdet'
cfg.model.train_cfg = None
m = MODELS.build(cfg.model)
load_checkpoint(m, 'work_dirs/grmi_bilinear_12e_clean/epoch_12.pth', map_location='cuda:0')
m = m.cuda().eval()
ri = m.residual_inject

torch.manual_seed(42)
B, L, C = 1, 900, 256
memory = torch.randn(B, L, C).cuda()
T_new = torch.randn(10, 20, 256).cuda().reshape(-1, 256)

with torch.no_grad():
    T_pool = T_new.mean(0, keepdim=True)
    h = ri.W_m(memory) * ri.W_t(T_pool).squeeze(0)
    pre_ln = ri.W_out(h)
    post_ln = ri.output_norm(pre_ln)
    pre_n = pre_ln.norm(dim=-1).cpu().numpy()
    post_n = post_ln.norm(dim=-1).cpu().numpy()

print("Synthetic test (random M, random T_new, trained W):")
print(f"  pre-LN:  mean={pre_n.mean():.2f} std={pre_n.std():.2f} [{pre_n.min():.2f}, {pre_n.max():.2f}]")
print(f"  post-LN: mean={post_n.mean():.2f} std={post_n.std():.2f} [{post_n.min():.2f}, {post_n.max():.2f}]")
print(f"  pre-LN  CV: {pre_n.std()/max(pre_n.mean(),1e-8):.3f}")
print(f"  post-LN CV: {post_n.std()/max(post_n.mean(),1e-8):.3f}")
print(f"  → LN {'PRESERVES' if pre_n.std()/max(pre_n.mean(),1e-8)>0.3 and post_n.std()/max(post_n.mean(),1e-8)>0.3 else 'DESTROYS'} variance")

# Now test with actual new-class text embeddings
cat_ids = [1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90]
# Build prompt and get real text embeddings
from pycocotools.coco import COCO
coco = COCO('data/coco/annotations/instances_val2017.json')
names = [coco.loadCats([c])[0]['name'] for c in cat_ids]
prompt = ' . '.join(names) + ' .'
tok = m.language_model.tokenizer([prompt], padding='max_length', max_length=256, truncation=True, return_tensors='pt')
ids = tok['input_ids'].cuda()
mask = tok['attention_mask'].cuda()
tf = m.language_model(ids, mask)
tf = tf['last_hidden_state'] if isinstance(tf, dict) else tf

# Get text_feat_map via bbox_head
mt = m.bbox_head._get_text_feat(prompt)
tfm = m.bbox_head._text_feat_map_  # (80, L_t, 256)

# Classes 70-79 are new
T_real = tfm[70:80].reshape(-1, 256).detach()

with torch.no_grad():
    T_pool_r = T_real.mean(0, keepdim=True)
    h_r = ri.W_m(memory) * ri.W_t(T_pool_r).squeeze(0)
    pre_ln_r = ri.W_out(h_r)
    post_ln_r = ri.output_norm(pre_ln_r)
    pre_n_r = pre_ln_r.norm(dim=-1).cpu().numpy()
    post_n_r = post_ln_r.norm(dim=-1).cpu().numpy()

print("\nReal T_new test (trained W, real BERT embeddings):")
print(f"  pre-LN:  mean={pre_n_r.mean():.2f} std={pre_n_r.std():.2f} [{pre_n_r.min():.2f}, {pre_n_r.max():.2f}]")
print(f"  post-LN: mean={post_n_r.mean():.2f} std={post_n_r.std():.2f} [{post_n_r.min():.2f}, {post_n_r.max():.2f}]")
print(f"  pre-LN  CV: {pre_n_r.std()/max(pre_n_r.mean(),1e-8):.3f}")
print(f"  post-LN CV: {post_n_r.std()/max(post_n_r.mean(),1e-8):.3f}")
