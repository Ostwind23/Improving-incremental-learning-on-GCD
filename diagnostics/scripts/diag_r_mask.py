"""Evaluate R-norm-based gate selectivity on real images."""
import torch, numpy as np, os
from mmengine.registry import DefaultScope
DefaultScope.get_instance('mmdet', scope_name='mmdet')
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmdet.registry import MODELS
from pycocotools.coco import COCO
import mmcv

cfg = Config.fromfile('configs/gdino_inc/70+10/grmi_bilinear_12e_clean.py')
cfg.default_scope = 'mmdet'
m = MODELS.build(cfg.model)
# Load Bilinear 12e checkpoint (trained with LN, but W_m/W_t/W_out encode the M×T selectivity)
load_checkpoint(m, 'work_dirs/grmi_bilinear_12e_clean/epoch_12.pth', map_location='cuda:0')
m.cuda(); m.eval()
ri = m.residual_inject

# Get T_new via standalone BERT
from transformers import BertModel, BertTokenizer
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
    T_pool = T_new.mean(dim=0, keepdim=True)

OLD, NEW = set(range(70)), set(range(70, 80))

all_r_norms = {'old': [], 'new': [], 'bg': []}
all_r_norms_per_token = {'old': [], 'new': [], 'bg': []}

img_ids = sorted(coco.getImgIds())[:100]
n_done = 0

for img_id in img_ids:
    if n_done >= 20: break
    info = coco.loadImgs([img_id])[0]
    anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
    boxes, labels = [], []
    for a in anns:
        if a.get('iscrowd', 0): continue
        x, y, w, h = a['bbox']; boxes.append([x,y,x+w,y+h]); labels.append(a['category_id']-1)
    if not boxes: continue

    boxes = torch.tensor(boxes).cuda(); labels = torch.tensor(labels).cuda()
    img = mmcv.imread(f'data/coco/val2017/{info["file_name"]}')
    img_t = torch.from_numpy(img).float().permute(2,0,1).unsqueeze(0).cuda()
    mean = torch.tensor([123.675,116.28,103.53]).view(1,3,1,1).cuda()
    std = torch.tensor([58.395,57.12,57.375]).view(1,3,1,1).cuda()
    img_t = (img_t - mean) / std

    try:
        with torch.no_grad():
            feat = m.backbone(img_t)
            mem = feat[0].flatten(2).permute(0,2,1)
            h = ri.W_m(mem) * ri.W_t(T_pool).squeeze(0)
            R = ri.W_out(h).squeeze(0)  # (L, 256) — pre-LN Bilinear output
            mem = mem.squeeze(0)
    except: continue

    L, C = mem.shape
    H = W = int(np.sqrt(L))
    if H*W != L:
        for h in range(H,0,-1):
            if L%h==0: H,W = h, L//h; break

    ih, iw = info['height'], info['width']
    r_norm = R.norm(dim=-1)

    old_mask = torch.zeros(L, dtype=torch.bool, device='cuda:0')
    new_mask = torch.zeros(L, dtype=torch.bool, device='cuda:0')
    for ti in range(L):
        ty, tx = divmod(ti, W)
        cx, cy = (tx+0.5)/W*iw, (ty+0.5)/H*ih
        for bi, (x1,y1,x2,y2) in enumerate(boxes):
            if x1<=cx<=x2 and y1<=cy<=y2:
                if labels[bi].item() in OLD: old_mask[ti]=True
                elif labels[bi].item() in NEW: new_mask[ti]=True
                break
    bg_mask = ~(old_mask | new_mask)

    for mask, key in [(old_mask,'old'), (new_mask,'new'), (bg_mask,'bg')]:
        if mask.sum() > 0:
            all_r_norms[key].append(r_norm[mask].mean().item())
            all_r_norms_per_token[key].extend(r_norm[mask].cpu().tolist())
    n_done += 1
    if n_done % 5 == 0: print(f"  {n_done}/20 images")

print("\n========= R-norm distribution =========")
for key in ['old', 'new', 'bg']:
    vals = np.array(all_r_norms_per_token[key])
    print(f"  {key:5s}: mean={vals.mean():.1f} median={np.median(vals):.1f} "
          f"std={vals.std():.1f} p25={np.percentile(vals,25):.1f} p75={np.percentile(vals,75):.1f}")

# Test thresholds
print("\n========= Threshold analysis =========")
for thr in [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]:
    old_pass = sum(1 for v in all_r_norms_per_token['old'] if v > thr) / max(len(all_r_norms_per_token['old']), 1)
    new_pass = sum(1 for v in all_r_norms_per_token['new'] if v > thr) / max(len(all_r_norms_per_token['new']), 1)
    bg_pass = sum(1 for v in all_r_norms_per_token['bg'] if v > thr) / max(len(all_r_norms_per_token['bg']), 1)
    old_mean_after = np.mean([v for v in all_r_norms_per_token['old'] if v > thr]) if old_pass > 0 else 0
    new_mean_after = np.mean([v for v in all_r_norms_per_token['new'] if v > thr]) if new_pass > 0 else 0
    print(f"  thr={thr:.1f}: old_pass={old_pass*100:.1f}% new_pass={new_pass*100:.1f}% bg_pass={bg_pass*100:.1f}% | after-thr old={old_mean_after:.1f} new={new_mean_after:.1f}")
