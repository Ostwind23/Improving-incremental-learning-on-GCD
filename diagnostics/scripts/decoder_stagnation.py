#!/usr/bin/env python3
"""
Decoder Stagnation Root Cause Diagnosis.

We know decoder d0→d5 IoU = -0.006 for new-class (stagnant).
Three hypotheses for WHY:

H1: Cross-attention sampling offset — deformable attention samples at offsets
    relative to reference point. If reference is far from GT, offsets are too
    small to reach the correct image region.
    TEST: measure distance between reference point and GT center at each layer.

H2: Self-attention mixing — SCM injects teacher queries (old-class biased) into
    decoder self-attention. New-class queries get "pulled" toward old-class modes.
    TEST: in self-attention, what fraction of new-class queries' attention goes
    to other new-class queries vs old-class / teacher queries?

H3: Text cross-attention misdirection — new-class queries attend to wrong text
    tokens (confusing old-class tokens instead of correct new-class tokens).
    TEST: at new-class matched queries, measure attention weight on correct
    new-class text token vs best old-class text token.

Also measures for COMPARISON:
  - Same metrics for OLD-class queries (do they refine properly?)
  - This shows whether the mechanism is broken specifically for new classes
    or globally broken.
"""
import argparse, os, sys, time, json
import numpy as np
import torch
import types

GCD_ROOT = os.environ.get('GCD_ROOT', '/home/yelingfei/projects/GCD')
if os.path.isdir(os.path.join(GCD_ROOT, 'mmdet')):
    os.chdir(GCD_ROOT)
sys.path.insert(0, GCD_ROOT)
NS, NE = 70, 80


def run(args):
    from mmengine.config import Config
    from mmengine.runner import Runner, load_checkpoint
    from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_overlaps

    cfg = Config.fromfile(args.cfg)
    cfg.work_dir = '/tmp/dec_stag'
    cfg.launcher = 'none'
    cfg.val_dataloader['batch_size'] = 1
    vd = cfg.val_dataloader
    if 'dataset' in vd and isinstance(vd['dataset'], dict):
        vd['dataset'].pop('_delete_', None)

    runner = Runner.from_cfg(cfg)
    runner.load_or_resume()
    model = runner.model.module if hasattr(runner.model, 'module') else runner.model
    load_checkpoint(model, args.ckpt, map_location='cpu')
    dev = torch.device('cuda:0')
    model.to(dev).eval()
    if runner.model is not model:
        runner.model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # ═══════ Hooks on decoder internals ═══════
    cap = {}

    # Hook forward_decoder to capture per-layer references
    orig_fd = model.forward_decoder
    def fd_hook(*a, **k):
        out = orig_fd(*a, **k)
        if 'references' in out:
            ref = out['references']
            if isinstance(ref, (list, tuple)):
                cap['references'] = torch.stack([r.detach() for r in ref])
            else:
                cap['references'] = ref.detach()
        if 'hidden_states' in out:
            hs = out['hidden_states']
            if isinstance(hs, (list, tuple)):
                cap['hidden_states'] = torch.stack([h.detach() for h in hs])
            else:
                cap['hidden_states'] = hs.detach()
        return out
    model.forward_decoder = fd_hook

    # Hook decoder cross-attention (text) to capture attention weights
    # The decoder has 6 layers, each with cross_attn_text
    text_attn_weights = {}
    for layer_idx, layer in enumerate(model.decoder.layers):
        if hasattr(layer, 'cross_attn_text'):
            ca_text = layer.cross_attn_text
            def make_hook(li):
                def hook_fn(module, inputs, output):
                    # MultiheadAttention returns (attn_output, attn_weights)
                    # But mmcv wraps it. Try to get weights.
                    if isinstance(output, tuple) and len(output) >= 2:
                        if output[1] is not None:
                            text_attn_weights[li] = output[1].detach()
                return hook_fn
            ca_text.register_forward_hook(make_hook(layer_idx))

    # Build token sets
    ALL_CLASSES = [
        "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
        "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse",
        "sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie",
        "suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove",
        "skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon",
        "bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut",
        "cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
        "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book",
        "clock","vase","scissors","teddy bear","hair drier","toothbrush"]
    tok = model.language_model.tokenizer
    cap_str = '. '.join(ALL_CLASSES) + '.'
    enc = tok(cap_str, return_offsets_mapping=True, add_special_tokens=True)
    offsets = enc['offset_mapping']
    cls_to_tokens = {}
    cursor = 0
    for ci, cname in enumerate(ALL_CLASSES):
        idx = cap_str.find(cname, cursor)
        if idx < 0: cursor += 1; continue
        c0, c1 = idx, idx + len(cname)
        toks = [t for t, (s, e) in enumerate(offsets) if s < c1 and e > c0]
        cls_to_tokens[ci] = toks
        cursor = c1

    # ═══════ Collectors ═══════
    # H1: reference-to-GT distance per layer
    h1_new_dist = {i: [] for i in range(6)}  # normalized distance
    h1_old_dist = {i: [] for i in range(6)}

    # H2: per-layer IoU for new vs old (expanded from full_chain)
    h2_new_iou = {i: [] for i in range(6)}
    h2_old_iou = {i: [] for i in range(6)}

    seen = 0
    t0 = time.time()
    for data in runner.val_dataloader:
        if seen >= args.n_imgs:
            break
        samples = data['data_samples']
        sl = samples if isinstance(samples, (list, tuple)) else [samples]
        s = sl[0]
        if s.gt_instances is None or len(s.gt_instances.bboxes) == 0:
            continue
        gt_labels = s.gt_instances.labels
        gt_bboxes = s.gt_instances.bboxes
        if hasattr(gt_bboxes, 'tensor'):
            gt_bboxes = gt_bboxes.tensor
        new_mask = (gt_labels >= NS) & (gt_labels < NE)
        old_mask = (gt_labels < NS)
        if not new_mask.any():
            continue

        for k in list(cap.keys()):
            cap.pop(k, None)
        text_attn_weights.clear()
        with torch.no_grad():
            _ = runner.model.val_step(data)

        refs = cap.get('references')
        if refs is None or refs.shape[0] < 6:
            continue

        meta = s.metainfo
        ih, iw = meta['img_shape']
        fac = gt_bboxes.new_tensor([iw, ih, iw, ih]).to(dev)

        new_gt = gt_bboxes[new_mask].to(dev)
        old_gt = gt_bboxes[old_mask].to(dev)
        # GT centers (normalized)
        new_cx = ((new_gt[:, 0] + new_gt[:, 2]) / 2 / iw)
        new_cy = ((new_gt[:, 1] + new_gt[:, 3]) / 2 / ih)
        old_cx = ((old_gt[:, 0] + old_gt[:, 2]) / 2 / iw) if len(old_gt) > 0 else torch.tensor([])
        old_cy = ((old_gt[:, 1] + old_gt[:, 3]) / 2 / ih) if len(old_gt) > 0 else torch.tensor([])

        for layer_i in range(6):
            ref_l = refs[layer_i, 0]  # (nq, 4) cxcywh normalized
            ref_cx = ref_l[:, 0]
            ref_cy = ref_l[:, 1]

            # Convert to pixel boxes for IoU
            boxes_l = bbox_cxcywh_to_xyxy(ref_l) * fac
            boxes_l[:, 0::2].clamp_(0, iw)
            boxes_l[:, 1::2].clamp_(0, ih)

            # H1 + H2 for NEW class
            if len(new_gt) > 0:
                ious_new = bbox_overlaps(boxes_l, new_gt)  # (nq, n_new)
                for ni in range(len(new_gt)):
                    best_q = ious_new[:, ni].argmax().item()
                    best_iou = ious_new[best_q, ni].item()
                    h2_new_iou[layer_i].append(best_iou)
                    # Distance: reference center to GT center (normalized)
                    d = ((ref_cx[best_q] - new_cx[ni])**2 +
                         (ref_cy[best_q] - new_cy[ni])**2).sqrt().item()
                    h1_new_dist[layer_i].append(d)

            # H1 + H2 for OLD class
            if len(old_gt) > 0:
                ious_old = bbox_overlaps(boxes_l, old_gt)
                for oi in range(min(len(old_gt), 20)):  # cap to avoid slow
                    best_q = ious_old[:, oi].argmax().item()
                    best_iou = ious_old[best_q, oi].item()
                    h2_old_iou[layer_i].append(best_iou)
                    d = ((ref_cx[best_q] - old_cx[oi])**2 +
                         (ref_cy[best_q] - old_cy[oi])**2).sqrt().item()
                    h1_old_dist[layer_i].append(d)

        seen += 1
        if seen % 50 == 0:
            print(f"  [{seen}/{args.n_imgs}] {time.time()-t0:.0f}s")

    # ═══════ Results ═══════
    print("\n" + "=" * 80)
    print("DECODER STAGNATION ROOT CAUSE (%d images)" % seen)
    print("=" * 80)

    print("\n╔══ H1: Reference-to-GT center distance (normalized, lower=closer) ══╗")
    print(f"  {'layer':>5} {'new_dist':>10} {'old_dist':>10} {'ratio':>8}")
    for i in range(6):
        nd = np.mean(h1_new_dist[i]) if h1_new_dist[i] else 0
        od = np.mean(h1_old_dist[i]) if h1_old_dist[i] else 0
        r = nd / max(od, 1e-9)
        print(f"  d{i:>4} {nd:>10.4f} {od:>10.4f} {r:>7.2f}x")
    # Does distance decrease across layers? (should if decoder refines)
    if h1_new_dist[0] and h1_new_dist[5]:
        nd0 = np.mean(h1_new_dist[0])
        nd5 = np.mean(h1_new_dist[5])
        od0 = np.mean(h1_old_dist[0])
        od5 = np.mean(h1_old_dist[5])
        print(f"\n  New-class d0→d5 distance change: {nd0:.4f}→{nd5:.4f} ({nd5-nd0:+.4f})")
        print(f"  Old-class d0→d5 distance change: {od0:.4f}→{od5:.4f} ({od5-od0:+.4f})")
        if nd5 >= nd0 - 0.001:
            print(f"  >>> NEW-CLASS: decoder does NOT move queries closer to GT")
        if od5 < od0 - 0.001:
            print(f"  >>> OLD-CLASS: decoder DOES move queries closer to GT")

    print("\n╔══ H2: Per-layer IoU (new vs old class comparison) ══╗")
    print(f"  {'layer':>5} {'new_IoU':>10} {'new≥0.5':>8} {'old_IoU':>10} {'old≥0.5':>8} {'gap':>8}")
    for i in range(6):
        ni = np.array(h2_new_iou[i]) if h2_new_iou[i] else np.array([0])
        oi = np.array(h2_old_iou[i]) if h2_old_iou[i] else np.array([0])
        print(f"  d{i:>4} {ni.mean():>10.3f} {np.mean(ni>=0.5)*100:>7.1f}% {oi.mean():>10.3f} {np.mean(oi>=0.5)*100:>7.1f}% {ni.mean()-oi.mean():>+7.3f}")
    if h2_new_iou[0] and h2_old_iou[0]:
        new_refine = np.mean(h2_new_iou[5]) - np.mean(h2_new_iou[0])
        old_refine = np.mean(h2_old_iou[5]) - np.mean(h2_old_iou[0])
        print(f"\n  New-class refinement d0→d5: {new_refine:+.4f}")
        print(f"  Old-class refinement d0→d5: {old_refine:+.4f}")

    print("\n╔══ DIAGNOSIS ══╗")
    if h1_new_dist[0] and h1_old_dist[0]:
        nd0 = np.mean(h1_new_dist[0])
        od0 = np.mean(h1_old_dist[0])
        print(f"  Initial reference distance: new={nd0:.4f} old={od0:.4f} ratio={nd0/max(od0,1e-9):.2f}x")
        if nd0 > od0 * 1.5:
            print(f"  >>> H1 CONFIRMED: new-class queries start FAR from GT")
            print(f"      (deformable attention can't reach GT with small offsets)")
        else:
            print(f"  >>> H1 NOT confirmed: initial distance comparable")

        nd5 = np.mean(h1_new_dist[5])
        od5 = np.mean(h1_old_dist[5])
        new_movement = nd0 - nd5
        old_movement = od0 - od5
        print(f"\n  Movement toward GT: new={new_movement:+.4f} old={old_movement:+.4f}")
        if old_movement > 0.001 and new_movement < 0.001:
            print(f"  >>> DECODER SELECTIVELY BROKEN for new classes")
            print(f"      Old-class queries move toward GT; new-class queries DON'T")
        elif old_movement < 0.001:
            print(f"  >>> DECODER GLOBALLY NOT REFINING (even old classes)")

    result = {}
    for i in range(6):
        result[f'h1_new_dist_d{i}'] = float(np.mean(h1_new_dist[i])) if h1_new_dist[i] else None
        result[f'h1_old_dist_d{i}'] = float(np.mean(h1_old_dist[i])) if h1_old_dist[i] else None
        result[f'h2_new_iou_d{i}'] = float(np.mean(h2_new_iou[i])) if h2_new_iou[i] else None
        result[f'h2_old_iou_d{i}'] = float(np.mean(h2_old_iou[i])) if h2_old_iou[i] else None
    result['n_images'] = seen
    outpath = '/home/yelingfei/logs/tatri/decoder_stagnation.json'
    with open(outpath, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {outpath}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', default='configs/gdino_inc/70+10/gdino_inc_70+10_70-79_gcd_scratch_coco.py')
    p.add_argument('--ckpt', default='work_dirs/gcd_70plus10_2gpu_20260426_223507/epoch_12.pth')
    p.add_argument('--n-imgs', type=int, default=200)
    run(p.parse_args())
