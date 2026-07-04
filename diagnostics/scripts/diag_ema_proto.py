"""EMA visual prototype gate diagnostic — single pass."""
import mmdet.apis, mmdet.engine.hooks  # noqa
import torch, numpy as np, json, time
from mmengine.config import Config
from mmengine.runner import Runner, load_checkpoint

ALL_CLASSES = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse",
    "sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie",
    "suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove",
    "skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon",
    "bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut",
    "cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book",
    "clock","vase","scissors","teddy bear","hair drier","toothbrush"]
NEW_SET = {"toothbrush","hair drier","scissors","teddy bear","toaster",
           "book","clock","vase","sink","refrigerator"}

def auc(pos, neg):
    if len(pos)==0 or len(neg)==0: return None
    scores = np.concatenate([pos, neg])
    labs = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(-scores); ls = labs[order]
    tp = np.cumsum(ls); fp = np.cumsum(1-ls)
    tpr = tp/max(labs.sum(),1); fpr = fp/max((1-labs).sum(),1)
    return float(np.trapz(tpr, fpr))

@torch.no_grad()
def main():
    cfg = Config.fromfile("configs/gdino_inc/70+10/gdino_inc_70+10_70-79_gcd_scratch_coco.py")
    cfg.work_dir = "/tmp/ema_wd"; cfg.launcher = "none"
    cfg.val_dataloader["batch_size"] = 1
    vd = cfg.val_dataloader
    if "dataset" in vd and isinstance(vd["dataset"], dict):
        vd["dataset"].pop("_delete_", None)
    runner = Runner.from_cfg(cfg); runner.load_or_resume()
    model = runner.model.module if hasattr(runner.model, "module") else runner.model
    load_checkpoint(model, "work_dirs/gcd_70plus10_2gpu_20260426_223507/epoch_12.pth", map_location="cpu")
    model.cuda().eval()
    if runner.model is not model: runner.model.cuda(); runner.model.eval()
    for p in model.parameters(): p.requires_grad_(False)

    cap = {}
    orig_fe = model.forward_encoder
    def fe_hook(*a, **k):
        out = orig_fe(*a, **k)
        for key in ["memory", "memory_mask", "spatial_shapes"]:
            v = out.get(key)
            if v is not None: cap[key] = v.detach()
        return out
    model.forward_encoder = fe_hook

    dec_cap = {}
    orig_fd = model.forward_decoder
    def fd_hook(*a, **k):
        out = orig_fd(*a, **k)
        if "hidden_states" in out:
            v = out["hidden_states"]
            dec_cap["hs"] = v.detach() if torch.is_tensor(v) else [x.detach() for x in v]
        if "references" in out:
            v = out["references"]
            dec_cap["refs"] = v.detach() if torch.is_tensor(v) else [x.detach() for x in v]
        return out
    model.forward_decoder = fd_hook

    # Pass 1: collect prototypes (first 100 imgs with new-class)
    proto_accum = {c: [] for c in range(70, 80)}
    n1 = 0; t0 = time.time()
    for data in runner.val_dataloader:
        if n1 >= 100: break
        sl = data["data_samples"]
        s = sl[0] if isinstance(sl, (list, tuple)) else sl
        if s.gt_instances is None or len(s.gt_instances.bboxes) == 0: continue
        has_new = any(ALL_CLASSES[int(l.item())] in NEW_SET for l in s.gt_instances.labels)
        if not has_new: continue
        for k in list(cap.keys()): cap.pop(k, None)
        for k in list(dec_cap.keys()): dec_cap.pop(k, None)
        _ = runner.model.val_step(data)
        hs = dec_cap.get("hs"); refs = dec_cap.get("refs")
        if hs is None: continue
        final_hs = hs[-1] if isinstance(hs, (list, tuple)) else hs[-1]
        final_refs = refs[-1] if isinstance(refs, (list, tuple)) else refs[-1]
        ref_pts = final_refs[0, :, :2].cpu()
        feats = final_hs[0].cpu()
        boxes_t = s.gt_instances.bboxes.tensor.cpu()
        gt_labs = s.gt_instances.labels.cpu()
        H_img, W_img = s.img_shape[:2]
        for gi in range(len(boxes_t)):
            gl = int(gt_labs[gi])
            if gl < 70: continue
            bx = boxes_t[gi]
            x0, y0, x1, y1 = bx[0]/W_img, bx[1]/H_img, bx[2]/W_img, bx[3]/H_img
            inside = (ref_pts[:,0]>=x0)&(ref_pts[:,0]<=x1)&(ref_pts[:,1]>=y0)&(ref_pts[:,1]<=y1)
            if inside.sum() > 0:
                proto_accum[gl].append(feats[inside].mean(0))
        n1 += 1
    print(f"[EMA] Pass1 done: {n1} imgs, {time.time()-t0:.0f}s")
    for c in range(70, 80):
        print(f"  {ALL_CLASSES[c]}: {len(proto_accum[c])} features")

    # Build prototypes
    protos = []
    for c in range(70, 80):
        if proto_accum[c]:
            protos.append(torch.stack(proto_accum[c]).mean(0))
    P = torch.stack(protos)
    P = P / (P.norm(dim=-1, keepdim=True) + 1e-6)
    print(f"[EMA] {len(protos)} class prototypes built")

    # Pass 2: compute per-position similarity (next 200 imgs)
    ema_new, ema_old, ema_bg = [], [], []
    n2 = 0
    for data in runner.val_dataloader:
        if n2 >= 200: break
        sl = data["data_samples"]
        s = sl[0] if isinstance(sl, (list, tuple)) else sl
        if s.gt_instances is None or len(s.gt_instances.bboxes) == 0: continue
        has_new = any(ALL_CLASSES[int(l.item())] in NEW_SET for l in s.gt_instances.labels)
        if not has_new: continue
        for k in list(cap.keys()): cap.pop(k, None)
        _ = runner.model.val_step(data)
        memory = cap.get("memory"); ss = cap.get("spatial_shapes")
        if memory is None: continue
        ssl = ss.cpu().long().tolist()
        N = memory.shape[1]
        mem = memory[0].cpu()
        mem_n = mem / (mem.norm(dim=-1, keepdim=True) + 1e-6)
        sim = (mem_n @ P.T).max(-1)[0].numpy()

        # Label positions
        labels = np.zeros(N, dtype=np.int8)
        boxes_t = s.gt_instances.bboxes.tensor.cpu()
        gt_labs = s.gt_instances.labels.cpu()
        H_img, W_img = s.img_shape[:2]
        for k in range(len(ssl)):
            h, w = ssl[k]
            o = int(sum(ssl[j][0]*ssl[j][1] for j in range(k)))
            lab_grid = np.zeros((h, w), dtype=np.int8)
            for box, gl in zip(boxes_t, gt_labs):
                x0, y0, x1, y1 = box.tolist()
                x0 /= W_img; x1 /= W_img; y0 /= H_img; y1 /= H_img
                c0 = max(0, int(x0*w)); c1 = min(w, int(x1*w)+1)
                r0 = max(0, int(y0*h)); r1 = min(h, int(y1*h)+1)
                if r1 <= r0 or c1 <= c0: continue
                if int(gl) < 70: lab_grid[r0:r1, c0:c1] = np.maximum(lab_grid[r0:r1, c0:c1], 1)
                else: lab_grid[r0:r1, c0:c1] = 2
            labels[o:o+h*w] = lab_grid.flatten()

        if (labels==2).sum()>0: ema_new.append(sim[labels==2])
        if (labels==1).sum()>0: ema_old.append(sim[labels==1])
        if (labels==0).sum()>0: ema_bg.append(sim[labels==0])
        n2 += 1
        if n2 % 50 == 0: print(f"  [EMA pass2] {n2}/200, {time.time()-t0:.0f}s")

    model.forward_encoder = orig_fe
    model.forward_decoder = orig_fd

    en = np.concatenate(ema_new) if ema_new else np.array([])
    eo = np.concatenate(ema_old) if ema_old else np.array([])
    eb = np.concatenate(ema_bg) if ema_bg else np.array([])

    print(f"\n===== EMA VISUAL PROTOTYPE GATE DIAGNOSIS =====")
    print(f"Pass2: {n2} imgs, positions: new={len(en)} old={len(eo)} bg={len(eb)}")
    if len(en) and len(eo) and len(eb):
        print(f"Mean sim: new={np.mean(en):.4f} old={np.mean(eo):.4f} bg={np.mean(eb):.4f}")
        print(f"AUC new vs old:    {auc(en, eo)}")
        print(f"AUC new vs bg:     {auc(en, eb)}")
        print(f"AUC new vs notnew: {auc(en, np.concatenate([eo,eb]))}")
        print(f"Separation: new-old={np.mean(en)-np.mean(eo):.4f}  new-bg={np.mean(en)-np.mean(eb):.4f}")
    else:
        print("No data collected in pass2")

    res = {
        "n_pass1": n1, "n_pass2": n2,
        "counts": {"new": int(len(en)), "old": int(len(eo)), "bg": int(len(eb))},
        "mean_sim": {"new": float(np.mean(en)) if len(en) else None,
                     "old": float(np.mean(eo)) if len(eo) else None,
                     "bg": float(np.mean(eb)) if len(eb) else None},
        "auc_new_vs_old": auc(en, eo),
        "auc_new_vs_bg": auc(en, eb),
        "auc_new_vs_notnew": auc(en, np.concatenate([eo,eb])) if len(eo)+len(eb)>0 else None,
    }
    with open("/home/yelingfei/logs/tatri/ema_proto_diag.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"Saved -> /home/yelingfei/logs/tatri/ema_proto_diag.json")

if __name__ == "__main__":
    main()
