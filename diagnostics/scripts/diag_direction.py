"""Direction diagnosis: cos(R, M), cos(R, T) for old vs new class regions"""
import torch, numpy as np, json, os
from mmengine.registry import DefaultScope
DefaultScope.get_instance('mmdet', scope_name='mmdet')
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmdet.registry import MODELS
from pycocotools.coco import COCO
import mmcv
from transformers import BertModel, BertTokenizer

cfg = Config.fromfile('configs/gdino_inc/70+10/grmi_bilinear_12e_clean.py')
cfg.default_scope = 'mmdet'
m = MODELS.build(cfg.model)
load_checkpoint(m, 'work_dirs/grmi_bilinear_12e_clean/epoch_12.pth', map_location='cuda:0')
m.cuda(); m.eval()
ri = m.residual_inject
print("Model loaded")

# Get real T_new via standalone BERT
coco = COCO('data/coco/annotations/instances_val2017.json')
OLD, NEW = set(range(70)), set(range(70, 80))
cat_ids = sorted(coco.getCatIds())
names = [coco.loadCats([c])[0]['name'] for c in cat_ids]
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained('bert-base-uncased').cuda().eval()

# Encode new-class names (COCO classes 71-80, indices 70-79)
new_names = [coco.loadCats([c])[0]['name'] for c in cat_ids[70:80]]
new_prompt = ' . '.join(new_names) + ' .'
tok = tokenizer(new_prompt, return_tensors='pt', padding='max_length', 
                max_length=256, truncation=True)
with torch.no_grad():
    out = bert(input_ids=tok['input_ids'].cuda(), 
               attention_mask=tok['attention_mask'].cuda())
    text_emb = out.last_hidden_state  # (1, 256, 768)
    # Map to 256-dim via model's text_feat_map
    # text_feat_map is nn.Linear(768, 256) in GroundingDINO
    T_all = m.text_feat_map(text_emb)  # (1, 256, 256)
    # Take all tokens (pooling over the new-class segment)
    T_new = T_all.squeeze(0).detach()  # (256, 256)
    T_pool = T_new.mean(dim=0, keepdim=True)
print(f"T_pool ready, shape={T_pool.shape}")

# Image analysis
results = {'old': {'rmag':[], 'cos_R_M':[], 'cos_R_T':[]},
           'new': {'rmag':[], 'cos_R_M':[], 'cos_R_T':[]},
           'bg':  {'rmag':[], 'cos_R_M':[], 'cos_R_T':[]}}

img_ids = sorted(coco.getImgIds())[:150]
n_done = 0

for img_id in img_ids:
    if n_done >= 30: break
    info = coco.loadImgs([img_id])[0]
    anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
    boxes, labels = [], []
    for a in anns:
        if a.get('iscrowd', 0): continue
        x, y, w, h = a['bbox']; boxes.append([x,y,x+w,y+h]); labels.append(a['category_id']-1)
    if not boxes: continue

    boxes = torch.tensor(boxes).cuda()
    labels = torch.tensor(labels).cuda()
    img = mmcv.imread(f'data/coco/val2017/{info["file_name"]}')
    img_t = torch.from_numpy(img).float().permute(2,0,1).unsqueeze(0).cuda()
    mean = torch.tensor([123.675,116.28,103.53]).view(1,3,1,1).cuda()
    std = torch.tensor([58.395,57.12,57.375]).view(1,3,1,1).cuda()
    img_t = (img_t - mean) / std

    try:
        with torch.no_grad():
            feat = m.extract_feat(img_t)
            mem = feat[0]
            h = ri.W_m(mem) * ri.W_t(T_pool).squeeze(0)
            pre_ln = ri.W_out(h)
            mem = mem.squeeze(0); pre_ln = pre_ln.squeeze(0)
    except: continue

    L, C = mem.shape
    H = W = int(np.sqrt(L))
    if H*W != L:
        for h in range(H,0,-1):
            if L%h==0: H,W = h, L//h; break

    ih, iw = info['height'], info['width']
    r_norm = pre_ln.norm(dim=-1)
    m_norm = mem.norm(dim=-1)
    cos_R_M = (pre_ln * mem).sum(-1) / (r_norm * m_norm + 1e-8)
    Tp = T_pool.squeeze(0)
    cos_R_T = (pre_ln * Tp).sum(-1) / (r_norm * Tp.norm() + 1e-8)

    old_mask = torch.zeros(L, dtype=torch.bool, device='cuda:0')
    new_mask = torch.zeros(L, dtype=torch.bool, device='cuda:0')
    for ti in range(L):
        ty, tx = divmod(ti, W)
        cx, cy = (tx+0.5)/W*iw, (ty+0.5)/H*ih
        for bi, (x1, y1, x2, y2) in enumerate(boxes):
            if x1<=cx<=x2 and y1<=cy<=y2:
                if labels[bi].item() in OLD: old_mask[ti]=True
                elif labels[bi].item() in NEW: new_mask[ti]=True
                break
    bg_mask = ~(old_mask | new_mask)

    for mask, key in [(old_mask,'old'), (new_mask,'new'), (bg_mask,'bg')]:
        if mask.sum()>0:
            results[key]['rmag'].append(r_norm[mask].mean().item())
            results[key]['cos_R_M'].append(cos_R_M[mask].mean().item())
            results[key]['cos_R_T'].append(cos_R_T[mask].mean().item())
    n_done += 1
    if n_done%10==0: print(f"  {n_done}/30")

print("\n======== Direction Diagnosis ========")
for key in ['old', 'new', 'bg']:
    d = results[key]
    if d['rmag']:
        print(f"  {key:5s}: ||R||={np.mean(d['rmag']):.2f}±{np.std(d['rmag']):.2f}, "
              f"cos(R,M)={np.mean(d['cos_R_M']):.3f}±{np.std(d['cos_R_M']):.3f}, "
              f"cos(R,T)={np.mean(d['cos_R_T']):.3f}±{np.std(d['cos_R_T']):.3f}")

out = {'old':{}, 'new':{}, 'bg':{}}
for key in ['old', 'new', 'bg']:
    d = results[key]
    if d['rmag']:
        for k in d: out[key][k] = float(np.mean(d[k]))
os.makedirs('/home/yelingfei/logs/tatri/diag_ln_bilinear_12e', exist_ok=True)
with open('/home/yelingfei/logs/tatri/diag_ln_bilinear_12e/direction.json', 'w') as f:
    json.dump(out, f, indent=2)
print("Done")
