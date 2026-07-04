import torch, numpy as np, json, sys
from pycocotools.coco import COCO
import mmcv
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmengine.registry import DefaultScope

# Must set default_scope BEFORE importing mmdet models
DefaultScope.get_instance('mmdet', scope_name='mmdet')

from mmdet.registry import MODELS

CKPT = 'work_dirs/grmi_bilinear_12e_clean/epoch_12.pth'
CONFIG = 'configs/gdino_inc/70+10/grmi_bilinear_12e_clean.py'

cfg = Config.fromfile(CONFIG)
cfg.default_scope = 'mmdet'
cfg.model.train_cfg = None
model = MODELS.build(cfg.model)
load_checkpoint(model, CKPT, map_location='cuda:0')
model = model.cuda()
model.eval()
print("Model loaded")

ri = model.residual_inject
coco = COCO('data/coco/annotations/instances_val2017.json')
OLD = set(range(70))
NEW = set(range(70, 80))

pre_n = {'old': [], 'new': [], 'bg': []}
post_n = {'old': [], 'new': [], 'bg': []}

img_ids = sorted(coco.getImgIds())[:150]
n_done = 0

for i, img_id in enumerate(img_ids):
    if n_done >= 50:
        break
    info = coco.loadImgs([img_id])[0]
    anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
    boxes, labels = [], []
    for a in anns:
        if a.get('iscrowd', 0):
            continue
        x, y, w, h = a['bbox']
        boxes.append([x, y, x + w, y + h])
        labels.append(a['category_id'] - 1)
    if not boxes:
        continue

    boxes = torch.tensor(boxes).cuda()
    labels = torch.tensor(labels).cuda()

    img_path = f'data/coco/val2017/{info["file_name"]}'
    if not __import__('os').path.exists(img_path):
        continue

    img = mmcv.imread(img_path)
    img_t = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).cuda()
    mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1).cuda()
    std = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1).cuda()
    img_t = (img_t - mean) / std

    try:
        with torch.no_grad():
            feat = model.extract_feat(img_t)
            memory = feat[0]  # (1, L, 256)

            cat_ids = sorted(coco.getCatIds())
            names = [coco.loadCats([c])[0]['name'] for c in cat_ids]
            prompt = ' . '.join(names) + ' .'
            tokenized = model.language_model.tokenizer(
                [prompt], padding='max_length', max_length=256,
                truncation=True, return_tensors='pt')
            input_ids = tokenized['input_ids'].cuda()
            attention_mask = tokenized['attention_mask'].cuda()
            text_feat = model.language_model(input_ids, attention_mask)
            if isinstance(text_feat, dict):
                text_feat = text_feat['last_hidden_state']

            memory_text = model.bbox_head._get_text_feat(prompt)
            text_feat_map = model.bbox_head._text_feat_map_
            T_new = text_feat_map[70:80].reshape(-1, 256).detach()

            T_pool = T_new.mean(dim=0, keepdim=True)
            h = ri.W_m(memory) * ri.W_t(T_pool).squeeze(0)
            pre_ln = ri.W_out(h)
            post_ln = ri.output_norm(pre_ln)

            memory = memory.squeeze(0)
            pre_ln = pre_ln.squeeze(0)
            post_ln = post_ln.squeeze(0)
    except Exception as e:
        continue

    L, C = memory.shape
    H = W = int(np.sqrt(L))
    if H * W != L:
        for h in range(H, 0, -1):
            if L % h == 0:
                H, W = h, L // h
                break

    ih, iw = info['height'], info['width']
    pre_norm = pre_ln.norm(dim=-1)
    post_norm = post_ln.norm(dim=-1)

    old_mask = torch.zeros(L, dtype=torch.bool, device='cuda:0')
    new_mask = torch.zeros(L, dtype=torch.bool, device='cuda:0')

    for ti in range(L):
        ty, tx = divmod(ti, W)
        cx = (tx + 0.5) / W * iw
        cy = (ty + 0.5) / H * ih
        for bi, (x1, y1, x2, y2) in enumerate(boxes):
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                if labels[bi].item() in OLD:
                    old_mask[ti] = True
                elif labels[bi].item() in NEW:
                    new_mask[ti] = True
                break

    bg_mask = ~(old_mask | new_mask)

    for mask, key in [(old_mask, 'old'), (new_mask, 'new'), (bg_mask, 'bg')]:
        if mask.sum() > 0:
            pre_n[key].append(pre_norm[mask].mean().item())
            post_n[key].append(post_norm[mask].mean().item())

    n_done += 1
    if n_done % 10 == 0:
        print(f"  {n_done}/50 images done")

print("\n========== LN Diagnosis ==========")
for k in ['old', 'new', 'bg']:
    if pre_n[k]:
        p = np.mean(pre_n[k])
        po = np.mean(post_n[k])
        ps = np.std(pre_n[k])
        pos = np.std(post_n[k])
        print(f"  {k:5s}: pre-LN={p:.3f}±{ps:.3f}  →  "
              f"post-LN={po:.3f}±{pos:.3f}  "
              f"(×{po / max(p, 1e-8):.2f})")

out = {
    'old': {'pre': float(np.mean(pre_n['old'])), 'post': float(np.mean(post_n['old'])),
            'pre_std': float(np.std(pre_n['old'])), 'post_std': float(np.std(post_n['old']))},
    'new': {'pre': float(np.mean(pre_n['new'])), 'post': float(np.mean(post_n['new'])),
            'pre_std': float(np.std(pre_n['new'])), 'post_std': float(np.std(post_n['new']))},
    'bg': {'pre': float(np.mean(pre_n['bg'])), 'post': float(np.mean(post_n['bg'])),
           'pre_std': float(np.std(pre_n['bg'])), 'post_std': float(np.std(post_n['bg']))},
}
out['new_old_pre_ratio'] = out['new']['pre'] / max(out['old']['pre'], 1e-6)
out['new_old_post_ratio'] = out['new']['post'] / max(out['old']['post'], 1e-6)

os.makedirs('/home/yelingfei/logs/tatri/diag_ln_bilinear_12e', exist_ok=True)
with open('/home/yelingfei/logs/tatri/diag_ln_bilinear_12e/ln_diag.json', 'w') as f:
    json.dump(out, f, indent=2)

print(f"\n  New/Old pre-LN ratio:  {out['new_old_pre_ratio']:.2f}x")
print(f"  New/Old post-LN ratio: {out['new_old_post_ratio']:.2f}x")
print(f"  → LN {'preserved' if abs(out['new_old_pre_ratio']-out['new_old_post_ratio'])<0.2 else 'DESTROYED'} the new/old amplitude difference")
