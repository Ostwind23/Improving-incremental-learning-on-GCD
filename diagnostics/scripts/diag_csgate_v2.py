"""
CS-GRMI gate quality precheck v2: use cls_branches (LGQS actual scoring) instead of raw cosine.
check1_hook showed cls_branches gives margin +2.19 at new-class GT positions with 98.9% argmax accuracy.
The v1 diagnostic used raw cosine (AUC 0.60) which is NOT what LGQS uses.

This v2 diagnostic:
1. Hooks forward_encoder to capture memory + memory_text + shapes
2. Runs gen_encoder_output_proposals -> output_memory (like LGQS does)
3. Runs cls_branches[6](output_memory, memory_text, mask) -> enc_cls scores
4. For each position, computes s_new = max(enc_cls[new_class_cols]) and s_old = max(enc_cls[old_class_cols])
5. Labels positions as NEW/OLD/BG based on GT, computes AUC
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
    ap.add_argument('--out', default='/home/yelingfei/logs/tatri/csgate_v2_diag.json')
    args = ap.parse_args()

    from mmengine.config import Config
    from mmengine.runner import Runner, load_checkpoint
    cfg = Config.fromfile(args.cfg)
    cfg.work_dir = '/tmp/csgate_v2_wd'; cfg.launcher = 'none'
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
    new_tok_flat = sorted(set(i for c in range(70, 80) for i in token_ranges[c]))
    old_tok_flat = sorted(set(i for c in range(0, 70) for i in token_ranges[c] if i not in new_tok_flat))
    print(f'[CSGATEv2] new tokens: {len(new_tok_flat)}  old tokens: {len(old_tok_flat)}')

    clsN = model.bbox_head.cls_branches[model.decoder.num_layers]

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
        has_new = any(ALL_CLASSES[int(l.item())] in NEW_NAMES for l in s.gt_instances.labels)
        if not has_new:
            continue
        for k in list(cap.keys()):
            cap.pop(k, None)
        _ = runner.model.val_step(data)
        memory = cap.get('memory'); mmask = cap.get('memory_mask')
        ss = cap.get('spatial_shapes'); mt = cap.get('memory_text')
        ttm = cap.get('text_token_mask')
        if memory is None:
            continue
        if mmask is None:
            mmask = torch.zeros(memory.shape[:2], dtype=torch.bool, device=memory.device)

        # gen_encoder_output_proposals like LGQS does
        output_memory, _ = model.gen_encoder_output_proposals(memory, mmask, ss)
        # cls_branches scoring (same as LGQS)
        enc_cls = clsN(output_memory, mt, ttm)  # (1, N_pos, max_text_len)
        enc_cls = enc_cls[0].cpu().numpy()  # (N_pos, max_text_len)
        N = enc_cls.shape[0]
        ssl = ss.cpu().long().tolist()

        # per-position new/old score (max over token columns)
        s_new_pos = enc_cls[:, new_tok_flat].max(axis=1)  # (N,)
        s_old_pos = enc_cls[:, old_tok_flat].max(axis=1)  # (N,)
        margin_pos = s_new_pos - s_old_pos  # (N,)

        # label positions
        labels = np.zeros(N, dtype=np.int8)
        boxes = s.gt_instances.bboxes.tensor.cpu()
        gt_labels = s.gt_instances.labels.cpu()
        H_img, W_img = s.img_shape[:2]
        for k in range(len(ssl)):
            h, w = ssl[k]
            o = int(sum(ssl[j][0]*ssl[j][1] for j in range(k)))
            lab_grid = np.zeros((h, w), dtype=np.int8)
            for box, gl in zip(boxes, gt_labels):
                x0, y0, x1, y1 = box.tolist()
                x0 /= W_img; x1 /= W_img; y0 /= H_img; y1 /= H_img
                c0 = max(0, int(x0*w)); c1 = min(w, int(x1*w)+1)
                r0 = max(0, int(y0*h)); r1 = min(h, int(y1*h)+1)
                if r1 <= r0 or c1 <= c0:
                    continue
                gl_int = int(gl)
                if gl_int < 70:
                    lab_grid[r0:r1, c0:c1] = np.maximum(lab_grid[r0:r1, c0:c1], 1)
                else:
                    lab_grid[r0:r1, c0:c1] = 2  # new takes precedence
            labels[o:o+h*w] = lab_grid.flatten()

        idx_new = labels == 2
        idx_old = labels == 1
        idx_bg = labels == 0

        if idx_new.sum() > 0:
            margin_new.append(margin_pos[idx_new])
            snew_new.append(s_new_pos[idx_new])
            sold_new.append(s_old_pos[idx_new])
        if idx_old.sum() > 0:
            margin_old.append(margin_pos[idx_old])
            snew_old.append(s_new_pos[idx_old])
            sold_old.append(s_old_pos[idx_old])
        if idx_bg.sum() > 0:
            margin_bg.append(margin_pos[idx_bg])
            snew_bg.append(s_new_pos[idx_bg])
            sold_bg.append(s_old_pos[idx_bg])
        n_imgs += 1
        if n_imgs % 50 < 2:
            print(f'  [CSGATEv2] {n_imgs}/{args.n} imgs, {time.time()-t0:.0f}s')

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
        'auc_margin_new_vs_notnew': auc_score(margin_new, np.concatenate([margin_old, margin_bg]) if len(margin_old)+len(margin_bg)>0 else margin_bg),
        'auc_snew_new_vs_old': auc_score(sn_new, sn_old),
        'auc_snew_new_vs_bg': auc_score(sn_new, sn_bg),
        'tau_sweep': [],
    }
    for tau in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        g_new_t = 1/(1+np.exp(-margin_new/tau)) if len(margin_new) else np.array([])
        g_old_t = 1/(1+np.exp(-margin_old/tau)) if len(margin_old) else np.array([])
        g_bg_t = 1/(1+np.exp(-margin_bg/tau)) if len(margin_bg) else np.array([])
        res['tau_sweep'].append({
            'tau': tau,
            'gate_mean_new': float(np.mean(g_new_t)) if len(g_new_t) else None,
            'gate_mean_old': float(np.mean(g_old_t)) if len(g_old_t) else None,
            'gate_mean_bg': float(np.mean(g_bg_t)) if len(g_bg_t) else None,
            'gate_ratio_new_over_old': float(np.mean(g_new_t)/max(np.mean(g_old_t),1e-9)) if len(g_new_t) and len(g_old_t) else None,
            'recall_new_at_oldP90': recall_at_precision(g_new_t, g_old_t, 0.9),
            'recall_new_at_oldP95': recall_at_precision(g_new_t, g_old_t, 0.95),
        })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(res, f, indent=2)

    print('\n========== CS-GRMI GATE v2 (cls_branches) DIAGNOSIS ==========')
    print(f'Images: {n_imgs}  Positions: new={len(margin_new)} old={len(margin_old)} bg={len(margin_bg)}')
    print(f'\nAUC (margin = s_new_cls - s_old_cls):')
    print(f'  new vs old:    {res["auc_margin_new_vs_old"]}')
    print(f'  new vs bg:     {res["auc_margin_new_vs_bg"]}')
    print(f'  new vs notnew: {res["auc_margin_new_vs_notnew"]}')
    print(f'  s_new alone (new vs old): {res["auc_snew_new_vs_old"]}')
    print(f'\nMean scores (cls_branches output):')
    print(f'  s_new: new={res["s_new_mean"]["new"]:.4f} old={res["s_new_mean"]["old"]:.4f} bg={res["s_new_mean"]["bg"]:.4f}')
    print(f'  s_old: new={res["s_old_mean"]["new"]:.4f} old={res["s_old_mean"]["old"]:.4f} bg={res["s_old_mean"]["bg"]:.4f}')
    print(f'  margin: new={res["margin_mean"]["new"]:.4f} old={res["margin_mean"]["old"]:.4f} bg={res["margin_mean"]["bg"]:.4f}')
    print(f'\nTau sweep (gate=sigmoid(margin/tau)):')
    for ts in res['tau_sweep']:
        print(f'  tau={ts["tau"]:.1f}: new={ts["gate_mean_new"]:.3f} old={ts["gate_mean_old"]:.3f} ratio={ts["gate_ratio_new_over_old"]:.2f} '
              f'recall@oldP90={ts["recall_new_at_oldP90"]} recall@oldP95={ts["recall_new_at_oldP95"]}')
    print(f'\nResults -> {args.out}')


if __name__ == '__main__':
    main()
