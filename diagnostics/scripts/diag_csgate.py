"""
CS-GRMI gate quality precheck (no training).
Follows the proven mechB_oracle_probe pattern (hook on forward_encoder).

Question: is the text-guided spatial gate
    g_i = sigmoid( (s_new_i - s_old_i) / tau )
discriminative enough to separate new-class regions from old-class regions?

  s_new_i = max cos(m_i, t_c) over new-class text prototypes
  s_old_i = max cos(m_i, t_o) over old-class text prototypes

If gate AUC(new vs old) < 0.6, frozen-BERT text is too weak -> CS-GRMI not viable
with raw BERT text; would need learnable offset / VLM text.

Reports per-position gate values for NEW / OLD / BACKGROUND positions across val2017.
"""
import mmdet.apis  # noqa
import mmdet.engine.hooks  # noqa
import argparse, os, time, json
import numpy as np
import torch

NEW_NAMES = ["toothbrush","hair drier","scissors","teddy bear","toaster",
             "book","clock","vase","sink","refrigerator"]
ALL_CLASSES = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse",
    "sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie",
    "suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove",
    "skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon",
    "bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut",
    "cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book",
    "clock","vase","scissors","teddy bear","hair drier","toothbrush"]


def build_token_ranges(model):
    """list len 80: token-column indices per class."""
    tok = model.language_model.tokenizer
    cap_str = '. '.join(ALL_CLASSES) + '.'
    enc = tok(cap_str, return_offsets_mapping=True, add_special_tokens=True)
    offsets = enc['offset_mapping']
    ranges = []
    cursor = 0
    for ci, cname in enumerate(ALL_CLASSES):
        idx = cap_str.find(cname, cursor)
        if idx < 0:
            ranges.append([]); continue
        c0, c1 = idx, idx + len(cname)
        toks = [ti for ti, (s, e) in enumerate(offsets) if s < c1 and e > c0]
        ranges.append(toks); cursor = c1
    return ranges


def masks_from_sample(sample, H, W):
    """10 new-class masks (labels 70..79) at finest grid (H,W)."""
    masks = torch.zeros(10, H, W)
    gt = sample.gt_instances
    if gt is None or len(gt.bboxes) == 0:
        return masks
    boxes = gt.bboxes.tensor.cpu(); labels = gt.labels.cpu()
    H_img, W_img = sample.img_shape[:2]
    for box, lab in zip(boxes, labels):
        local = NEW_NAMES.index(ALL_CLASSES[int(lab)]) if ALL_CLASSES[int(lab)] in NEW_NAMES else -1
        if local < 0:
            continue
        x0, y0, x1, y1 = box.tolist()
        x0 /= W_img; x1 /= W_img; y0 /= H_img; y1 /= H_img
        c0 = int(x0 * W); c1 = int(x1 * W) + 1
        r0 = int(y0 * H); r1 = int(y1 * H) + 1
        c0 = max(0, min(c0, W)); c1 = max(0, min(c1, W))
        r0 = max(0, min(r0, H)); r1 = max(0, min(r1, H))
        if r1 > r0 and c1 > c0:
            masks[local, r0:r1, c0:c1] = 1.0
    return masks


def old_masks_from_sample(sample, H, W):
    """old-class mask (labels 0..69) at grid (H,W)."""
    mask = torch.zeros(H, W)
    gt = sample.gt_instances
    if gt is None or len(gt.bboxes) == 0:
        return mask
    boxes = gt.bboxes.tensor.cpu(); labels = gt.labels.cpu()
    H_img, W_img = sample.img_shape[:2]
    for box, lab in zip(boxes, labels):
        if int(lab) >= 70:
            continue
        x0, y0, x1, y1 = box.tolist()
        x0 /= W_img; x1 /= W_img; y0 /= H_img; y1 /= H_img
        c0 = int(x0 * W); c1 = int(x1 * W) + 1
        r0 = int(y0 * H); r1 = int(y1 * H) + 1
        c0 = max(0, min(c0, W)); c1 = max(0, min(c1, W))
        r0 = max(0, min(r0, H)); r1 = max(0, min(r1, H))
        if r1 > r0 and c1 > c0:
            mask[r0:r1, c0:c1] = 1.0
    return mask


def positions_to_grids(spatial_shapes):
    """Per-position (gx, gy) normalized [0,1] across ALL levels."""
    num_pos = int(spatial_shapes[:, 0].prod().sum())
    centers = torch.zeros(num_pos, 2)
    for li, (h, w) in enumerate(spatial_shapes.tolist()):
        s = int(sum(spatial_shapes[k][0]*spatial_shapes[k][1] for k in range(li)))
        e = s + int(h*w)
        xs = (torch.arange(w) + 0.5) / w
        ys = (torch.arange(h) + 0.5) / h
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        centers[s:e, 0] = gx.flatten()
        centers[s:e, 1] = gy.flatten()
    return centers


def auc_score(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return None
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(-scores)
    ls = labels[order]
    tp = np.cumsum(ls); fp = np.cumsum(1 - ls)
    tpr = tp / max(labels.sum(), 1); fpr = fp / max((1-labels).sum(), 1)
    return float(np.trapz(tpr, fpr))


def recall_at_precision(pos, neg, target_prec):
    if len(pos) == 0 or len(neg) == 0:
        return None
    thr = np.percentile(neg, target_prec * 100)
    return float(np.mean(pos >= thr))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default='/home/yelingfei/projects/GCD/configs/gdino_inc/70+10/gdino_inc_70+10_70-79_gcd_scratch_coco.py')
    ap.add_argument('--ckpt', default='/home/yelingfei/projects/GCD/work_dirs/gcd_70plus10_2gpu_20260426_223507/epoch_12.pth')
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--out', default='/home/yelingfei/logs/tatri/csgate_diag.json')
    args = ap.parse_args()

    from mmengine.config import Config
    from mmengine.runner import Runner, load_checkpoint
    cfg = Config.fromfile(args.cfg)
    cfg.work_dir = '/tmp/csgate_wd'; cfg.launcher = 'none'
    cfg.val_dataloader['batch_size'] = 1
    vd = cfg.val_dataloader
    if 'dataset' in vd and isinstance(vd['dataset'], dict):
        vd['dataset'].pop('_delete_', None)
    runner = Runner.from_cfg(cfg); runner.load_or_resume()
    model = runner.model.module if hasattr(runner.model, 'module') else runner.model
    load_checkpoint(model, args.ckpt, map_location='cpu')
    model.cuda().eval()
    if runner.model is not model:
        runner.model.cuda(); runner.model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    token_ranges = build_token_ranges(model)
    print(f'[CSGATE] new-class token cols: {token_ranges[70:80]}')

    # hook forward_encoder
    cap = {}
    orig_fe = model.forward_encoder
    def fe_hook(*a, **k):
        out = orig_fe(*a, **k)
        for key in ['memory', 'memory_mask', 'spatial_shapes', 'memory_text', 'text_token_mask']:
            v = out.get(key)
            if v is not None:
                cap[key] = v.detach()
        return out
    model.forward_encoder = fe_hook

    # accumulators: gate value (margin) for new/old/bg positions
    margin_new, margin_old, margin_bg = [], [], []
    snew_new, snew_old, snew_bg = [], [], []
    sold_new, sold_old, sold_bg = [], [], []
    n_imgs = 0; t0 = time.time()

    for data in runner.val_dataloader:
        if n_imgs >= args.n:
            break
        sl = data['data_samples']
        s = sl[0] if isinstance(sl, (list, tuple)) else sl
        if s.gt_instances is None or len(s.gt_instances.bboxes) == 0:
            continue
        # must have at least one new-class GT to be informative
        has_new = any(ALL_CLASSES[int(l.item())] in NEW_NAMES for l in s.gt_instances.labels)
        if not has_new:
            continue
        for k in list(cap.keys()):
            cap.pop(k, None)
        _ = runner.model.val_step(data)
        memory = cap.get('memory'); mmask = cap.get('memory_mask')
        ss = cap.get('spatial_shapes'); mt = cap.get('memory_text')
        if memory is None:
            continue
        if mmask is None:
            mmask = torch.zeros(memory.shape[:2], dtype=torch.bool, device=memory.device)

        # memory: (1, N, 256), mt: (1, N_tok, 256)
        mem = memory[0]  # (N, 256)
        text = mt[0]     # (N_tok, 256)
        N = mem.shape[0]
        ssl = ss.cpu().long().tolist()

        # build new/old prototypes (per-class mean of token columns)
        new_protos = []
        for c in range(70, 80):
            ids = [i for i in token_ranges[c] if i < text.shape[0]]
            if ids:
                new_protos.append(text[ids].mean(0))
        old_protos = []
        for c in range(0, 70):
            ids = [i for i in token_ranges[c] if i < text.shape[0]]
            if ids:
                old_protos.append(text[ids].mean(0))
        P_new = torch.stack(new_protos)  # (k_new, 256)
        P_old = torch.stack(old_protos)  # (k_old, 256)

        # cosine sim per position
        mem_n = mem / (mem.norm(dim=-1, keepdim=True) + 1e-6)
        pn_n = P_new / (P_new.norm(dim=-1, keepdim=True) + 1e-6)
        po_n = P_old / (P_old.norm(dim=-1, keepdim=True) + 1e-6)
        sim_new = mem_n @ pn_n.T  # (N, k_new)
        sim_old = mem_n @ po_n.T  # (N, k_old)
        s_new_pos = sim_new.max(-1)[0].cpu().numpy()  # (N,)
        s_old_pos = sim_old.max(-1)[0].cpu().numpy()
        margin_pos = (s_new_pos - s_old_pos)

        # per-position label: 0=bg, 1=old, 2=new
        labels = np.zeros(N, dtype=np.int8)
        # use finest level for box labeling (same as mechB)
        li = max(range(len(ssl)), key=lambda k: ssl[k][0]*ssl[k][1])
        H0, W0 = ssl[li]
        off = sum(ssl[k][0]*ssl[k][1] for k in range(li))
        m_new = masks_from_sample(s, H0, W0).numpy()  # (10, H0, W0)
        m_old = old_masks_from_sample(s, H0, W0).numpy()  # (H0, W0)
        new_union = (m_new.sum(0) > 0).astype(np.int8)
        old_mask = (m_old > 0).astype(np.int8)
        # finest-level labels
        lab_fine = np.zeros(H0*W0, dtype=np.int8)
        lab_fine[old_mask.flatten() > 0] = 1
        lab_fine[new_union.flatten() > 0] = 2
        labels[off:off+H0*W0] = lab_fine
        # also label coarser levels approximately (project boxes to each level)
        for k in range(len(ssl)):
            if k == li:
                continue
            h, w = ssl[k]
            o = int(sum(ssl[j][0]*ssl[j][1] for j in range(k)))
            mn = masks_from_sample(s, h, w).numpy().sum(0) > 0
            mo = old_masks_from_sample(s, h, w).numpy() > 0
            lab = np.zeros(h*w, dtype=np.int8)
            lab[mo.flatten()] = 1
            lab[mn.flatten()] = 2
            labels[o:o+h*w] = lab

        # valid mask: mmask True = padded/ignored
        valid = (~mmask[0].bool()).cpu().numpy()
        idx_new = (labels == 2) & valid
        idx_old = (labels == 1) & valid
        idx_bg = (labels == 0) & valid

        if idx_new.sum() > 0:
            margin_new.append(margin_pos[idx_new]); snew_new.append(s_new_pos[idx_new]); sold_new.append(s_old_pos[idx_new])
        if idx_old.sum() > 0:
            margin_old.append(margin_pos[idx_old]); snew_old.append(s_new_pos[idx_old]); sold_old.append(s_old_pos[idx_old])
        if idx_bg.sum() > 0:
            margin_bg.append(margin_pos[idx_bg]); snew_bg.append(s_new_pos[idx_bg]); sold_bg.append(s_old_pos[idx_bg])
        n_imgs += 1
        if n_imgs % 50 < 2:
            print(f'  [CSGATE] {n_imgs}/{args.n} imgs, {time.time()-t0:.0f}s')

    model.forward_encoder = orig_fe
    margin_new = np.concatenate(margin_new) if margin_new else np.array([])
    margin_old = np.concatenate(margin_old) if margin_old else np.array([])
    margin_bg = np.concatenate(margin_bg) if margin_bg else np.array([])
    sn_new = np.concatenate(snew_new) if snew_new else np.array([])
    sn_old = np.concatenate(snew_old) if snew_old else np.array([])
    sn_bg = np.concatenate(snew_bg) if snew_bg else np.array([])
    so_new = np.concatenate(sold_new) if sold_new else np.array([])
    so_old = np.concatenate(sold_old) if sold_old else np.array([])
    so_bg = np.concatenate(sold_bg) if sold_bg else np.array([])

    res = {
        'n_images': int(n_imgs),
        'counts': {'new': int(len(margin_new)), 'old': int(len(margin_old)), 'bg': int(len(margin_bg))},
        'margin_mean': {
            'new': float(np.mean(margin_new)) if len(margin_new) else None,
            'old': float(np.mean(margin_old)) if len(margin_old) else None,
            'bg': float(np.mean(margin_bg)) if len(margin_bg) else None,
        },
        's_new_mean': {
            'new': float(np.mean(sn_new)) if len(sn_new) else None,
            'old': float(np.mean(sn_old)) if len(sn_old) else None,
            'bg': float(np.mean(sn_bg)) if len(sn_bg) else None,
        },
        's_old_mean': {
            'new': float(np.mean(so_new)) if len(so_new) else None,
            'old': float(np.mean(so_old)) if len(so_old) else None,
            'bg': float(np.mean(so_bg)) if len(so_bg) else None,
        },
        'auc_margin_new_vs_old': auc_score(margin_new, margin_old),
        'auc_margin_new_vs_bg': auc_score(margin_new, margin_bg),
        'auc_margin_new_vs_notnew': auc_score(margin_new, np.concatenate([margin_old, margin_bg]) if len(margin_old)+len(margin_bg) else margin_bg),
        'auc_snew_new_vs_old': auc_score(sn_new, sn_old),
        'auc_snew_new_vs_bg': auc_score(sn_new, sn_bg),
        'tau_sweep': [],
    }
    for tau in [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
        g_new_t = 1/(1+np.exp(-margin_new/tau)) if len(margin_new) else np.array([])
        g_old_t = 1/(1+np.exp(-margin_old/tau)) if len(margin_old) else np.array([])
        g_bg_t = 1/(1+np.exp(-margin_bg/tau)) if len(margin_bg) else np.array([])
        res['tau_sweep'].append({
            'tau': tau,
            'gate_mean_new': float(np.mean(g_new_t)) if len(g_new_t) else None,
            'gate_mean_old': float(np.mean(g_old_t)) if len(g_old_t) else None,
            'gate_mean_bg': float(np.mean(g_bg_t)) if len(g_bg_t) else None,
            'gate_ratio_new_over_old': float(np.mean(g_new_t)/max(np.mean(g_old_t),1e-9)) if len(g_new_t) and len(g_old_t) else None,
            'recall_new_at_oldP90': recall_at_precision(g_new_t, g_old_t, 0.9) if len(g_new_t) and len(g_old_t) else None,
            'recall_new_at_oldP95': recall_at_precision(g_new_t, g_old_t, 0.95) if len(g_new_t) and len(g_old_t) else None,
        })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(res, f, indent=2)

    print('\n========== CS-GRMI GATE QUALITY DIAGNOSIS ==========')
    print(f'Images: {n_imgs}  Positions: new={len(margin_new)} old={len(margin_old)} bg={len(margin_bg)}')
    print(f'\nAUC (margin = s_new - s_old):')
    print(f'  new vs old:    {res["auc_margin_new_vs_old"]}')
    print(f'  new vs bg:     {res["auc_margin_new_vs_bg"]}')
    print(f'  new vs notnew: {res["auc_margin_new_vs_notnew"]}')
    print(f'  s_new alone (new vs old): {res["auc_snew_new_vs_old"]}')
    print(f'\nMean similarity:')
    print(f'  s_new: new={res["s_new_mean"]["new"]} old={res["s_new_mean"]["old"]} bg={res["s_new_mean"]["bg"]}')
    print(f'  s_old: new={res["s_old_mean"]["new"]} old={res["s_old_mean"]["old"]} bg={res["s_old_mean"]["bg"]}')
    print(f'  margin: new={res["margin_mean"]["new"]} old={res["margin_mean"]["old"]} bg={res["margin_mean"]["bg"]}')
    print(f'\nTau sweep (gate=sigmoid(margin/tau)):')
    for ts in res['tau_sweep']:
        print(f'  tau={ts["tau"]}: new={ts["gate_mean_new"]} old={ts["gate_mean_old"]} ratio={ts["gate_ratio_new_over_old"]} '
              f'recall@oldP90={ts["recall_new_at_oldP90"]} recall@oldP95={ts["recall_new_at_oldP95"]}')
    print(f'\nResults -> {args.out}')


if __name__ == '__main__':
    main()
