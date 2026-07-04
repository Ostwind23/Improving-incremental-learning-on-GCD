#!/usr/bin/env python3
"""
Classification Oracle v2 — FIXED.

v1 bug: blindly changed ANY prediction overlapping new-GT, including correct
old-class predictions that happened to overlap a nearby new-GT → destroyed
old-class AP and injected noise into new-class.

v2 fix: only modify predictions where:
  1. The prediction's BEST IoU match is with a new-class GT (not old-class GT)
  2. AND the model's predicted class is WRONG for that GT

Two oracles:
  Oracle-A: For predictions best-matched to new-GT with IoU>0.5,
            replace predicted class with correct new class. Score unchanged.
  Oracle-B: Same as A but also boost score to 0.9 (tests if score is a factor).
"""
import argparse, os, sys, json, copy
import numpy as np
import torch

GCD_ROOT = '/home/yelingfei/projects/GCD'
os.chdir(GCD_ROOT); sys.path.insert(0, GCD_ROOT)
NS, NE = 70, 80


def run(args):
    from mmengine.config import Config
    from mmengine.runner import Runner, load_checkpoint
    from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_overlaps
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    cfg = Config.fromfile(args.cfg)
    cfg.work_dir = '/tmp/cls_oracle2'
    cfg.launcher = 'none'
    cfg.val_dataloader['batch_size'] = 1
    vd = cfg.val_dataloader
    if 'dataset' in vd and isinstance(vd['dataset'], dict):
        vd['dataset'].pop('_delete_', None)
    runner = Runner.from_cfg(cfg); runner.load_or_resume()
    model = runner.model.module if hasattr(runner.model, 'module') else runner.model
    load_checkpoint(model, args.ckpt, map_location='cpu')
    dev = torch.device('cuda:0')
    model.to(dev).eval()
    if runner.model is not model: runner.model.to(dev).eval()
    for p in model.parameters(): p.requires_grad_(False)

    ann_file = './data/coco/annotations/instances_val2017.json'
    coco_gt = COCO(ann_file)
    cat_ids = sorted(coco_gt.getCatIds())
    cat_id_to_label = {cat_id: i for i, cat_id in enumerate(cat_ids)}
    label_to_cat_id = {i: cat_id for cat_id, i in cat_id_to_label.items()}

    results_baseline = []
    results_oracle_a = []  # fix class only
    results_oracle_b = []  # fix class + boost score
    n_fixed = 0
    n_total_new_gt_preds = 0

    seen = 0
    for data in runner.val_dataloader:
        if seen >= args.n_imgs: break
        sl = data['data_samples']
        s = sl[0] if isinstance(sl, (list, tuple)) else sl
        if s.gt_instances is None or len(s.gt_instances.bboxes) == 0:
            seen += 1; continue

        with torch.no_grad():
            results = runner.model.val_step(data)
        result = results[0] if isinstance(results, (list, tuple)) else results
        pred_inst = result.pred_instances
        img_id = int(result.metainfo.get('img_id', seen))

        gt_labels = s.gt_instances.labels
        gt_bboxes = s.gt_instances.bboxes
        if hasattr(gt_bboxes, 'tensor'): gt_bboxes = gt_bboxes.tensor
        all_gt_bboxes = gt_bboxes.to(dev)
        all_gt_labels = gt_labels

        pred_bboxes = pred_inst.bboxes.to(dev)
        pred_scores = pred_inst.scores
        pred_labels = pred_inst.labels

        # Compute IoU of each prediction with ALL GT (both old and new)
        if len(all_gt_bboxes) > 0 and len(pred_bboxes) > 0:
            ious_all = bbox_overlaps(pred_bboxes, all_gt_bboxes)  # (N_pred, N_gt)
        else:
            ious_all = torch.zeros(len(pred_bboxes), len(all_gt_bboxes), device=dev)

        for i in range(len(pred_bboxes)):
            x1, y1, x2, y2 = pred_bboxes[i].cpu().tolist()
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0: continue
            lab = int(pred_labels[i])
            cat_id = label_to_cat_id.get(lab, -1)
            if cat_id < 0: continue

            entry = {
                'image_id': img_id,
                'category_id': cat_id,
                'bbox': [x1, y1, w, h],
                'score': float(pred_scores[i]),
            }
            results_baseline.append(entry)
            entry_a = copy.deepcopy(entry)
            entry_b = copy.deepcopy(entry)

            # Find this prediction's BEST matching GT
            if len(all_gt_bboxes) > 0:
                best_iou, best_gi = ious_all[i].max(dim=0)
                best_iou = best_iou.item()
                best_gi = best_gi.item()
                best_gt_label = int(all_gt_labels[best_gi])

                # Only intervene if:
                # 1. Best GT match is a NEW class (70-79)
                # 2. IoU > 0.5 (genuinely matched, not accidental overlap)
                # 3. Prediction class is WRONG
                if (NS <= best_gt_label < NE and
                    best_iou > 0.5 and
                    lab != best_gt_label):
                    correct_cat_id = label_to_cat_id.get(best_gt_label, -1)
                    if correct_cat_id >= 0:
                        entry_a['category_id'] = correct_cat_id
                        entry_b['category_id'] = correct_cat_id
                        entry_b['score'] = max(float(pred_scores[i]), 0.9)
                        n_fixed += 1
                    n_total_new_gt_preds += 1
                elif NS <= best_gt_label < NE and best_iou > 0.5:
                    n_total_new_gt_preds += 1

            results_oracle_a.append(entry_a)
            results_oracle_b.append(entry_b)

        seen += 1
        if seen % 500 == 0:
            print("  [%d/%d] fixed=%d" % (seen, args.n_imgs, n_fixed), flush=True)

    print("\n" + "=" * 70)
    print("CLASSIFICATION ORACLE v2 (%d images)" % seen)
    print("  Predictions matched to new-GT (IoU>0.5): %d" % n_total_new_gt_preds)
    print("  Of those, class was WRONG (fixed): %d (%.1f%%)" %
          (n_fixed, n_fixed / max(n_total_new_gt_preds, 1) * 100))
    print("=" * 70)

    def eval_coco(results, label):
        if not results: return {}
        import tempfile
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(results, tmp); tmp.close()
        coco_dt = coco_gt.loadRes(tmp.name)

        ev = COCOeval(coco_gt, coco_dt, 'bbox')
        ev.evaluate(); ev.accumulate(); ev.summarize()

        new_cat_ids = [label_to_cat_id.get(c, -1) for c in range(NS, NE)]
        new_cat_ids = [c for c in new_cat_ids if c >= 0]
        old_cat_ids = [label_to_cat_id.get(c, -1) for c in range(NS)]
        old_cat_ids = [c for c in old_cat_ids if c >= 0]

        ev_new = COCOeval(coco_gt, coco_dt, 'bbox')
        ev_new.params.catIds = new_cat_ids
        ev_new.evaluate(); ev_new.accumulate(); ev_new.summarize()

        ev_old = COCOeval(coco_gt, coco_dt, 'bbox')
        ev_old.params.catIds = old_cat_ids
        ev_old.evaluate(); ev_old.accumulate(); ev_old.summarize()

        os.unlink(tmp.name)
        r = {'mAP': ev.stats[0], 'mAP_50': ev.stats[1],
             'new_ap': ev_new.stats[0], 'new_ap50': ev_new.stats[1],
             'ori_ap': ev_old.stats[0], 'ori_ap50': ev_old.stats[1]}
        print("\n  %s: mAP=%.4f new_ap=%.4f ori_ap=%.4f" % (label, r['mAP'], r['new_ap'], r['ori_ap']))
        return r

    r_b = eval_coco(results_baseline, "BASELINE")
    r_a = eval_coco(results_oracle_a, "ORACLE-A (fix class only)")
    r_bb = eval_coco(results_oracle_b, "ORACLE-B (fix class + boost score)")

    print("\n" + "=" * 70)
    print("ORACLE GAINS (v2, safe)")
    print("=" * 70)
    if r_b and r_a:
        print("  Oracle-A (fix class, keep score):")
        print("    new_ap: %.4f -> %.4f (%+.4f)" % (r_b['new_ap'], r_a['new_ap'], r_a['new_ap']-r_b['new_ap']))
        print("    ori_ap: %.4f -> %.4f (%+.4f)" % (r_b['ori_ap'], r_a['ori_ap'], r_a['ori_ap']-r_b['ori_ap']))
        print("    mAP:    %.4f -> %.4f (%+.4f)" % (r_b['mAP'], r_a['mAP'], r_a['mAP']-r_b['mAP']))
    if r_b and r_bb:
        print("  Oracle-B (fix class + boost score):")
        print("    new_ap: %.4f -> %.4f (%+.4f)" % (r_b['new_ap'], r_bb['new_ap'], r_bb['new_ap']-r_b['new_ap']))

    json.dump({'baseline': r_b, 'oracle_a': r_a, 'oracle_b': r_bb,
               'n_fixed': n_fixed, 'n_new_gt_preds': n_total_new_gt_preds, 'n_imgs': seen},
              open('/home/yelingfei/logs/tatri/cls_oracle_v2.json', 'w'), indent=2)
    print("\nSaved: cls_oracle_v2.json")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', default='configs/gdino_inc/70+10/gdino_inc_70+10_70-79_gcd_grmi_12e_coco.py')
    p.add_argument('--ckpt', default='work_dirs/_preserved/grmi_first_best_ep12.pth')
    p.add_argument('--n-imgs', type=int, default=5000)
    run(p.parse_args())
