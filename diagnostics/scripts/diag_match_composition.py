"""Verify: what fraction of detection matched queries come from
new-class GT vs old-class pseudo-labels?
And: is the detection gradient on R(M) different between new-only vs all?"""
import mmdet.apis, mmdet.engine.hooks  # noqa
import torch, numpy as np
from mmengine.config import Config
from mmengine.runner import Runner

def main():
    cfg = Config.fromfile("configs/gdino_inc/70+10/grmi_t1_decouple_6e_coco.py")
    cfg.train_cfg.max_epochs = 1; cfg.train_dataloader.batch_size = 2
    cfg.work_dir = "/tmp/matchcomp"; cfg.launcher = "none"
    cfg.default_hooks.checkpoint = dict(type="CheckpointHook", interval=999)
    runner = Runner.from_cfg(cfg); runner.load_or_resume()
    model = runner.model.module if hasattr(runner.model, "module") else runner.model
    model.train()

    head = model.bbox_head
    # New class token boundary (from Part 1: tokens >= 169 are new)
    NEW_TOKEN_START = 169

    # Hook get_targets to capture labels
    orig_gt = head.get_targets
    call_info = {"count": 0, "labels": []}
    def hook_gt(*a, **kw):
        r = orig_gt(*a, **kw)
        call_info["count"] += 1
        call_info["labels"] = r[0]  # labels_list for this call
        return r
    head.get_targets = hook_gt

    dl = iter(runner.train_dataloader)
    stats = {"n_new_matched": [], "n_old_matched": [], "n_bg": []}

    for step in range(15):
        data = next(dl)
        data = model.data_preprocessor(data, True)
        call_info["count"] = 0
        model.zero_grad()
        losses = model(**data, mode="loss")

        # get_targets is called once per decoder layer (6x) + 1x for encoder
        # The last call's labels correspond to the last decoder layer (layer 5)
        labels_list = call_info.get("labels", [])
        if not labels_list:
            continue

        n_new = 0; n_old = 0; n_bg = 0
        for lab in labels_list:
            # lab: (900, 256) multi-hot
            matched = lab.sum(-1) > 0  # (900,)
            for qi in range(lab.shape[0]):
                if not matched[qi]:
                    n_bg += 1
                    continue
                hot_tokens = lab[qi].nonzero(as_tuple=True)[0]
                if (hot_tokens >= NEW_TOKEN_START).any():
                    n_new += 1
                else:
                    n_old += 1
        stats["n_new_matched"].append(n_new)
        stats["n_old_matched"].append(n_old)
        stats["n_bg"].append(n_bg)
        if step < 5:
            print("iter%d: new_matched=%d old_matched=%d bg=%d get_targets_calls=%d" % (
                step, n_new, n_old, n_bg, call_info["count"]))

    head.get_targets = orig_gt

    total_new = sum(stats["n_new_matched"])
    total_old = sum(stats["n_old_matched"])
    total = total_new + total_old
    print("\n===== DETECTION LOSS MATCHING COMPOSITION =====")
    print("Over %d iters:" % len(stats["n_new_matched"]))
    print("  New-class matched (from GT): %d (%.1f%%)" % (total_new, total_new/max(total,1)*100))
    print("  Old-class matched (from pseudo-labels): %d (%.1f%%)" % (total_old, total_old/max(total,1)*100))
    print("  Total matched: %d" % total)
    print()
    if total_old > 0:
        print("CRITICAL: Old-class pseudo-label queries ARE in detection loss!")
        print("  => R(M) receives gradient from BOTH new-class GT and old-class pseudo-labels")
        print("  => Previous claim 'R(M) only gets new-class gradient' was WRONG")
        print("  => The ratio of old-class gradient on R(M) is %.1f%%" % (total_old/total*100))
    else:
        print("Old-class matched = 0, R(M) truly gets only new-class gradient")


if __name__ == "__main__":
    main()
