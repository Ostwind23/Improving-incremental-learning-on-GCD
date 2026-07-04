"""
Three-in-one diagnostic for CS-GRMI gate signal sources.
Runs on GCD 12e checkpoint + val2017, no training.

Diagnostic 1 (GT-gate upper bound):
  If we had a perfect oracle gate (g=1 at new-class GT positions, g=0 elsewhere),
  what is the theoretical selectivity? How many encoder positions fall inside
  new-class GT vs old-class GT vs background? This sets the upper bound for any
  gate-based approach.

Diagnostic 2 (EMA visual prototype feasibility):
  Extract decoder query features for matched new-class predictions.
  Compute class-mean prototypes. Then test: can cos(encoder_memory, prototype)
  separate new-class GT positions from old/bg? AUC measurement.

Diagnostic 3 (Decoder prediction as gate signal):
  After a full forward, decoder produces 900 predictions with boxes + class scores.
  Map new-class predictions back to encoder spatial positions via reference_points.
  Measure: does this signal have spatial selectivity for new-class GT regions?
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
    ranges = []; cursor = 0
    for ci, cname in enumerate(ALL_CLASSES):
        idx = cap_str.find(cname, cursor)
        if idx < 0: ranges.append([]); continue
        c0, c1 = idx, idx + len(cname)
        toks = [ti for ti, (s, e) in enumerate(offsets) if s < c1 and e > c0]
        ranges.append(toks); cursor = c1
    return ranges


def label_positions(s, ssl):
    """Label each encoder position as 0=bg, 1=old, 2=new based on GT."""
    N = sum(h*w for h, w in ssl)
    labels = np.zeros(N, dtype=np.int8)
    if s.gt_instances is None or len(s.gt_instances.bboxes) == 0:
        return labels
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
            if r1 <= r0 or c1 <= c0: continue
            if int(gl) < 70:
                lab_grid[r0:r1, c0:c1] = np.maximum(lab_grid[r0:r1, c0:c1], 1)
            else:
                lab_grid[r0:r1, c0:c1] = 2
        labels[o:o+h*w] = lab_grid.flatten()
    return labels


def auc_score(pos, neg):
    if len(pos) == 0 or len(neg) == 0: return None
    scores = np.concatenate([pos, neg])
    labs = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(-scores)
    ls = labs[order]
    tp = np.cumsum(ls); fp = np.cumsum(1 - ls)
    tpr = tp / max(labs.sum(), 1); fpr = fp / max((1-labs).sum(), 1)
    return float(np.trapz(tpr, fpr))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default='/home/yelingfei/projects/GCD/configs/gdino_inc/70+10/gdino_inc_70+10_70-79_gcd_scratch_coco.py')
    ap.add_argument('--ckpt', default='/home/yelingfei/projects/GCD/work_dirs/gcd_70plus10_2gpu_20260426_223507/epoch_12.pth')
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--out', default='/home/yelingfei/logs/tatri/csgate_threeway_diag.json')
    args = ap.parse_args()

    from mmengine.config import Config
    from mmengine.runner import Runner, load_checkpoint
    cfg = Config.fromfile(args.cfg)
    cfg.work_dir = '/tmp/csgate3_wd'; cfg.launcher = 'none'
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
    for p in model.parameters(): p.requires_grad_(False)

    token_ranges = build_token_ranges(model)
    new_tok_flat = sorted(set(i for c in range(70, 80) for i in token_ranges[c]))
    clsN = model.bbox_head.cls_branches[model.decoder.num_layers]

    # Hook encoder to capture memory + shapes
    cap = {}
    orig_fe = model.forward_encoder
    def fe_hook(*a, **k):
        out = orig_fe(*a, **k)
        for key in ['memory', 'memory_mask', 'spatial_shapes', 'level_start_index',
                     'memory_text', 'text_token_mask']:
            v = out.get(key)
            if v is not None: cap[key] = v.detach()
        return out
    model.forward_encoder = fe_hook

    # Hook decoder to capture hidden_states and references
    dec_cap = {}
    orig_fd = model.forward_decoder
    def fd_hook(*a, **k):
        out = orig_fd(*a, **k)
        for key in ['hidden_states', 'references']:
            v = out.get(key)
            if v is not None: dec_cap[key] = v.detach() if torch.is_tensor(v) else [x.detach() for x in v]
        return out
    model.forward_decoder = fd_hook

    # Accumulators
    # Diag 1: GT-gate stats
    gt_counts = {'new': 0, 'old': 0, 'bg': 0}
    gt_new_frac_per_img = []
    # Diag 2: EMA prototype - collect matched query features per new class
    proto_features = {c: [] for c in range(70, 80)}  # class -> list of feature vectors
    # Gate signals to test
    ema_margin_new, ema_margin_old, ema_margin_bg = [], [], []
    # Diag 3: decoder prediction gate
    dec_gate_new, dec_gate_old, dec_gate_bg = [], [], []

    n_imgs = 0; t0 = time.time()

    for data in runner.val_dataloader:
        if n_imgs >= args.n: break
        sl = data['data_samples']
        s = sl[0] if isinstance(sl, (list, tuple)) else sl
        if s.gt_instances is None or len(s.gt_instances.bboxes) == 0: continue
        has_new = any(ALL_CLASSES[int(l.item())] in NEW_NAMES for l in s.gt_instances.labels)
        if not has_new: continue
        for k in list(cap.keys()): cap.pop(k, None)
        for k in list(dec_cap.keys()): dec_cap.pop(k, None)

        _ = runner.model.val_step(data)

        memory = cap.get('memory')
        mmask = cap.get('memory_mask')
        ss = cap.get('spatial_shapes')
        mt = cap.get('memory_text')
        ttm = cap.get('text_token_mask')
        hs = dec_cap.get('hidden_states')
        refs = dec_cap.get('references')
        if memory is None: continue
        if mmask is None:
            mmask = torch.zeros(memory.shape[:2], dtype=torch.bool, device=memory.device)

        ssl = ss.cpu().long().tolist()
        N = memory.shape[1]
        labels = label_positions(s, ssl)

        # === Diag 1: GT gate statistics ===
        n_new = int((labels == 2).sum())
        n_old = int((labels == 1).sum())
        n_bg = int((labels == 0).sum())
        gt_counts['new'] += n_new
        gt_counts['old'] += n_old
        gt_counts['bg'] += n_bg
        if N > 0:
            gt_new_frac_per_img.append(n_new / N)

        # === Diag 3: Decoder prediction → spatial gate ===
        # Get final layer cls scores from decoder output
        if hs is not None:
            final_hs = hs[-1] if isinstance(hs, (list, tuple)) else hs[-1]  # (1, 900, 256)
            final_refs = refs[-1] if isinstance(refs, (list, tuple)) else refs[-1]  # (1, 900, 4)

            dec_cls = clsN(final_hs, mt, ttm)  # (1, 900, max_text_len)
            dec_cls_0 = dec_cls[0]  # (900, max_text_len)
            # new-class score per query
            new_scores = dec_cls_0[:, new_tok_flat].max(-1)[0]  # (900,)
            all_scores = dec_cls_0.max(-1)[0]  # (900,)

            # Reference points → spatial position mapping
            # refs are (1, 900, 4) in [0,1] cxcywh
            ref_pts = final_refs[0, :, :2].cpu().numpy()  # (900, 2) in [0,1] (cx, cy)

            # For each encoder position, find nearest decoder queries and aggregate their new-class score
            dec_gate = np.zeros(N, dtype=np.float32)
            for k in range(len(ssl)):
                h, w = ssl[k]
                o = int(sum(ssl[j][0]*ssl[j][1] for j in range(k)))
                for ri in range(h):
                    for ci in range(w):
                        cy = (ri + 0.5) / h
                        cx = (ci + 0.5) / w
                        # distance to all 900 reference points
                        dist = (ref_pts[:, 0] - cx)**2 + (ref_pts[:, 1] - cy)**2
                        # nearest 3 queries
                        nn_idx = np.argsort(dist)[:3]
                        dec_gate[o + ri*w + ci] = float(new_scores[nn_idx].max().cpu())

            if (labels == 2).sum() > 0:
                dec_gate_new.append(dec_gate[labels == 2])
            if (labels == 1).sum() > 0:
                dec_gate_old.append(dec_gate[labels == 1])
            if (labels == 0).sum() > 0:
                dec_gate_bg.append(dec_gate[labels == 0])

        # === Diag 2: collect matched query features for EMA prototype ===
        # Use GT matching: find queries whose reference points overlap new-class GT
        if hs is not None:
            final_hs_0 = final_hs[0].cpu()  # (900, 256)
            ref_pts_t = torch.from_numpy(ref_pts)  # (900, 2)
            boxes_t = s.gt_instances.bboxes.tensor.cpu()
            gt_labs = s.gt_instances.labels.cpu()
            H_img, W_img = s.img_shape[:2]
            for gi in range(len(boxes_t)):
                gl = int(gt_labs[gi])
                if gl < 70: continue
                bx = boxes_t[gi]
                x0, y0, x1, y1 = bx[0]/W_img, bx[1]/H_img, bx[2]/W_img, bx[3]/H_img
                # queries inside this box
                inside = (ref_pts_t[:, 0] >= x0) & (ref_pts_t[:, 0] <= x1) & \
                         (ref_pts_t[:, 1] >= y0) & (ref_pts_t[:, 1] <= y1)
                if inside.sum() > 0:
                    mean_feat = final_hs_0[inside].mean(0)  # (256,)
                    proto_features[gl].append(mean_feat)

        n_imgs += 1
        if n_imgs % 50 < 2:
            print(f'  [3WAY] {n_imgs}/{args.n} imgs, {time.time()-t0:.0f}s')

    model.forward_encoder = orig_fe
    model.forward_decoder = orig_fd

    # === Diag 2: compute EMA prototypes and test on encoder memory ===
    print('[3WAY] Computing EMA prototype AUC...')
    # Build prototypes from collected query features
    prototypes = {}
    for c in range(70, 80):
        if proto_features[c]:
            prototypes[c] = torch.stack(proto_features[c]).mean(0)  # (256,)
            print(f'  class {c} ({ALL_CLASSES[c]}): {len(proto_features[c])} features')
    if prototypes:
        P_new_ema = torch.stack(list(prototypes.values()))  # (k, 256)
        P_new_ema = P_new_ema / (P_new_ema.norm(dim=-1, keepdim=True) + 1e-6)
        # Re-iterate to compute per-position similarity to EMA prototypes
        ema_margin_new, ema_margin_old, ema_margin_bg = [], [], []
        n2 = 0
        for data in runner.val_dataloader:
            if n2 >= args.n: break
            sl = data['data_samples']
            s = sl[0] if isinstance(sl, (list, tuple)) else sl
            if s.gt_instances is None or len(s.gt_instances.bboxes) == 0: continue
            has_new = any(ALL_CLASSES[int(l.item())] in NEW_NAMES for l in s.gt_instances.labels)
            if not has_new: continue
            for k in list(cap.keys()): cap.pop(k, None)
            _ = runner.model.val_step(data)
            memory = cap.get('memory')
            if memory is None: continue
            ss = cap.get('spatial_shapes')
            ssl = ss.cpu().long().tolist()
            N = memory.shape[1]
            labels = label_positions(s, ssl)

            mem = memory[0].cpu()  # (N, 256)
            mem_n = mem / (mem.norm(dim=-1, keepdim=True) + 1e-6)
            sim = mem_n @ P_new_ema.T  # (N, k)
            s_proto = sim.max(-1)[0].numpy()  # (N,)

            if (labels == 2).sum() > 0: ema_margin_new.append(s_proto[labels == 2])
            if (labels == 1).sum() > 0: ema_margin_old.append(s_proto[labels == 1])
            if (labels == 0).sum() > 0: ema_margin_bg.append(s_proto[labels == 2 if False else 0])

            n2 += 1
            if n2 % 100 < 2: print(f'  [EMA pass2] {n2}/{args.n}')

    # Aggregate
    def cat(lst): return np.concatenate(lst) if lst else np.array([])
    dec_new = cat(dec_gate_new); dec_old = cat(dec_gate_old); dec_bg = cat(dec_gate_bg)
    ema_new = cat(ema_margin_new); ema_old = cat(ema_margin_old); ema_bg = cat(ema_margin_bg)

    res = {
        'n_images': n_imgs,
        'diag1_gt_gate': {
            'total_positions': gt_counts['new'] + gt_counts['old'] + gt_counts['bg'],
            'new_positions': gt_counts['new'],
            'old_positions': gt_counts['old'],
            'bg_positions': gt_counts['bg'],
            'new_fraction_mean': float(np.mean(gt_new_frac_per_img)) if gt_new_frac_per_img else 0,
            'new_fraction_median': float(np.median(gt_new_frac_per_img)) if gt_new_frac_per_img else 0,
        },
        'diag2_ema_prototype': {
            'n_classes_with_proto': len(prototypes),
            'auc_new_vs_old': auc_score(ema_new, ema_old),
            'auc_new_vs_bg': auc_score(ema_new, ema_bg),
            'mean_sim_new': float(np.mean(ema_new)) if len(ema_new) else None,
            'mean_sim_old': float(np.mean(ema_old)) if len(ema_old) else None,
            'mean_sim_bg': float(np.mean(ema_bg)) if len(ema_bg) else None,
        },
        'diag3_decoder_gate': {
            'auc_new_vs_old': auc_score(dec_new, dec_old),
            'auc_new_vs_bg': auc_score(dec_new, dec_bg),
            'auc_new_vs_notnew': auc_score(dec_new, np.concatenate([dec_old, dec_bg]) if len(dec_old)+len(dec_bg)>0 else dec_bg),
            'mean_score_new': float(np.mean(dec_new)) if len(dec_new) else None,
            'mean_score_old': float(np.mean(dec_old)) if len(dec_old) else None,
            'mean_score_bg': float(np.mean(dec_bg)) if len(dec_bg) else None,
        },
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(res, f, indent=2)

    print('\n========== CS-GRMI THREE-WAY GATE DIAGNOSIS ==========')
    print(f'Images: {n_imgs}')
    d1 = res['diag1_gt_gate']
    print(f'\n--- Diag 1: GT gate upper bound ---')
    print(f'  new={d1["new_positions"]}({d1["new_fraction_mean"]*100:.1f}%) old={d1["old_positions"]} bg={d1["bg_positions"]}')
    d2 = res['diag2_ema_prototype']
    print(f'\n--- Diag 2: EMA visual prototype ---')
    print(f'  {d2["n_classes_with_proto"]} classes with prototype')
    print(f'  AUC new vs old: {d2["auc_new_vs_old"]}')
    print(f'  AUC new vs bg:  {d2["auc_new_vs_bg"]}')
    print(f'  mean sim: new={d2["mean_sim_new"]} old={d2["mean_sim_old"]} bg={d2["mean_sim_bg"]}')
    d3 = res['diag3_decoder_gate']
    print(f'\n--- Diag 3: Decoder prediction gate ---')
    print(f'  AUC new vs old: {d3["auc_new_vs_old"]}')
    print(f'  AUC new vs bg:  {d3["auc_new_vs_bg"]}')
    print(f'  AUC new vs notnew: {d3["auc_new_vs_notnew"]}')
    print(f'  mean score: new={d3["mean_score_new"]} old={d3["mean_score_old"]} bg={d3["mean_score_bg"]}')
    print(f'\nResults -> {args.out}')


if __name__ == '__main__':
    main()
