#!/usr/bin/env python3
"""
Classification Oracle: measure new_ap upper bound if classification were perfect.

Uses GCD's own test pipeline but intercepts predictions:
  - For any prediction box that overlaps a new-class GT with IoU>0.3,
    replace its class score with the correct GT label score = 1.0
  - Re-evaluate AP with corrected predictions

This gives the EXACT AP ceiling from fixing classification alone,
without changing any boxes.

Also measures a softer oracle: only fix predictions where the model
predicted an OLD class but the GT is NEW (the 30% new→old confusion).
"""
import argparse, os, sys, json, copy
import numpy as np
import torch

GCD_ROOT = '/home/yelingfei/projects/GCD'
os.chdir(GCD_ROOT); sys.path.insert(0, GCD_ROOT)

def run(args):
    from mmengine.config import Config
    from mmengine.runner import Runner, load_checkpoint
    from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_overlaps
    from mmdet.structures import DetDataSample
    from mmengine.structures import InstanceData

    NS, NE = 70, 80
    cfg = Config.fromfile(args.cfg)
    cfg.work_dir = '/tmp/cls_oracle'
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

    # We need to run the full test pipeline and capture predictions
    # Then modify predictions and re-compute metrics

    # Approach: use runner's test loop but intercept results
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    ann_file = cfg.val_evaluator.get('ann_file',
               './data/coco/annotations/instances_val2017.json')
    coco_gt = COCO(ann_file)

    # Collect all predictions via val_step
    all_results_baseline = []
    all_results_oracle_full = []
    all_results_oracle_newold = []

    # Map COCO category ids
    cat_ids = sorted(coco_gt.getCatIds())
    cat_id_to_label = {cat_id: i for i, cat_id in enumerate(cat_ids)}
    label_to_cat_id = {i: cat_id for cat_id, i in cat_id_to_label.items()}

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

        img_id = result.metainfo.get('img_id', seen)
        gt_labels = s.gt_instances.labels
        gt_bboxes = s.gt_instances.bboxes
        if hasattr(gt_bboxes, 'tensor'): gt_bboxes = gt_bboxes.tensor

        new_mask = (gt_labels >= NS) & (gt_labels < NE)
        new_gt_bboxes = gt_bboxes[new_mask].to(dev) if new_mask.any() else None
        new_gt_labels = gt_labels[new_mask] if new_mask.any() else None

        pred_bboxes = pred_inst.bboxes  # (N, 4) xyxy
        pred_scores = pred_inst.scores  # (N,)
        pred_labels = pred_inst.labels  # (N,)

        # Baseline predictions → COCO format
        for i in range(len(pred_bboxes)):
            x1, y1, x2, y2 = pred_bboxes[i].cpu().tolist()
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0: continue
            lab = int(pred_labels[i])
            cat_id = label_to_cat_id.get(lab, -1)
            if cat_id < 0: continue
            entry = {
                'image_id': int(img_id),
                'category_id': cat_id,
                'bbox': [x1, y1, w, h],
                'score': float(pred_scores[i]),
            }
            all_results_baseline.append(entry)

            # Oracle: if this prediction overlaps new-class GT, fix its label
            oracle_entry = copy.deepcopy(entry)
            oracle_newold_entry = copy.deepcopy(entry)

            if new_gt_bboxes is not None and len(new_gt_bboxes) > 0:
                ious = bbox_overlaps(
                    pred_bboxes[i:i+1].to(dev), new_gt_bboxes)
                best_iou, best_gi = ious[0].max(dim=0)
                best_iou = best_iou.item()
                best_gi = best_gi.item()

                if best_iou > 0.3:
                    correct_label = int(new_gt_labels[best_gi])
                    correct_cat_id = label_to_cat_id.get(correct_label, -1)

                    # Full oracle: always correct
                    if correct_cat_id >= 0:
                        oracle_entry['category_id'] = correct_cat_id
                        oracle_entry['score'] = max(float(pred_scores[i]), 0.9)

                    # New→Old oracle: only fix if model predicted old but GT is new
                    if lab < NS and correct_cat_id >= 0:
                        oracle_newold_entry['category_id'] = correct_cat_id
                        oracle_newold_entry['score'] = max(float(pred_scores[i]), 0.9)

            all_results_oracle_full.append(oracle_entry)
            all_results_oracle_newold.append(oracle_newold_entry)

        seen += 1
        if seen % 100 == 0:
            print("  [%d/%d]" % (seen, args.n_imgs))

    # Evaluate all three
    print("\n" + "=" * 70)
    print("CLASSIFICATION ORACLE RESULTS (%d images)" % seen)
    print("=" * 70)

    def eval_coco(results, label):
        if not results:
            print("  %s: no results" % label)
            return {}
        import tempfile
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(results, tmp)
        tmp.close()
        coco_dt = coco_gt.loadRes(tmp.name)
        ev = COCOeval(coco_gt, coco_dt, 'bbox')
        ev.evaluate(); ev.accumulate(); ev.summarize()
        os.unlink(tmp.name)

        # Also compute per-class AP for new classes
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

        r = {
            'mAP': ev.stats[0], 'mAP_50': ev.stats[1],
            'new_ap': ev_new.stats[0], 'new_ap50': ev_new.stats[1],
            'ori_ap': ev_old.stats[0], 'ori_ap50': ev_old.stats[1],
        }
        print("\n  %s:" % label)
        print("    mAP=%.4f  mAP_50=%.4f" % (r['mAP'], r['mAP_50']))
        print("    new_ap=%.4f  new_ap50=%.4f" % (r['new_ap'], r['new_ap50']))
        print("    ori_ap=%.4f  ori_ap50=%.4f" % (r['ori_ap'], r['ori_ap50']))
        return r

    r_base = eval_coco(all_results_baseline, "BASELINE")
    r_full = eval_coco(all_results_oracle_full, "ORACLE-FULL (all cls corrected)")
    r_newold = eval_coco(all_results_oracle_newold, "ORACLE-NEW2OLD (only new→old fixed)")

    print("\n" + "=" * 70)
    print("ORACLE GAINS")
    print("=" * 70)
    if r_base and r_full:
        print("  Full oracle (perfect cls):")
        print("    new_ap: %.4f → %.4f (%+.4f)" % (r_base['new_ap'], r_full['new_ap'], r_full['new_ap']-r_base['new_ap']))
        print("    mAP:    %.4f → %.4f (%+.4f)" % (r_base['mAP'], r_full['mAP'], r_full['mAP']-r_base['mAP']))
        print("    ori_ap: %.4f → %.4f (%+.4f)" % (r_base['ori_ap'], r_full['ori_ap'], r_full['ori_ap']-r_base['ori_ap']))
    if r_base and r_newold:
        print("  New→Old oracle (fix confusion to old classes only):")
        print("    new_ap: %.4f → %.4f (%+.4f)" % (r_base['new_ap'], r_newold['new_ap'], r_newold['new_ap']-r_base['new_ap']))

    json.dump({'baseline': r_base, 'oracle_full': r_full, 'oracle_newold': r_newold, 'n_imgs': seen},
              open('/home/yelingfei/logs/tatri/cls_oracle.json', 'w'), indent=2)
    print("\nSaved: cls_oracle.json")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', default='configs/gdino_inc/70+10/gdino_inc_70+10_70-79_gcd_grmi_12e_coco.py')
    p.add_argument('--ckpt', default='work_dirs/_preserved/grmi_first_best_ep12.pth')
    p.add_argument('--n-imgs', type=int, default=5000)
    run(p.parse_args())
