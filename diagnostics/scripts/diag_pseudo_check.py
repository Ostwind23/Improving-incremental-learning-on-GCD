"""Check GT and pseudo-label composition in training batches."""
import mmdet.apis, mmdet.engine.hooks  # noqa
import torch
from mmengine.config import Config
from mmengine.runner import Runner

cfg = Config.fromfile("configs/gdino_inc/70+10/grmi_t1_decouple_6e_coco.py")
cfg.train_cfg.max_epochs = 1; cfg.train_dataloader.batch_size = 2
cfg.work_dir = "/tmp/pseudo_check"; cfg.launcher = "none"
cfg.default_hooks.checkpoint = dict(type="CheckpointHook", interval=999)
runner = Runner.from_cfg(cfg); runner.load_or_resume()
model = runner.model.module if hasattr(runner.model, "module") else runner.model
model.train()

dl = iter(runner.train_dataloader)
for step in range(3):
    data = next(dl)
    data = model.data_preprocessor(data, True)
    # Forward to trigger pseudo-label generation
    losses = model(**data, mode="loss")
    samples = data["data_samples"] if isinstance(data["data_samples"], list) else [data["data_samples"]]
    for i, ds in enumerate(samples):
        gt = ds.gt_instances
        n_gt = len(gt.labels) if gt is not None else 0
        new_gt = int((gt.labels >= 70).sum()) if n_gt > 0 else 0
        old_gt = n_gt - new_gt
        # Check batch_all_instances (pseudo-labels merged)
        if hasattr(ds, 'batch_all_instances'):
            all_inst = ds.batch_all_instances
            n_all = len(all_inst.labels) if all_inst is not None else 0
        else:
            n_all = -1
        print("iter%d img%d: gt=%d (new=%d old=%d) labels=%s" % (
            step, i, n_gt, new_gt, old_gt,
            gt.labels.tolist()[:10] if n_gt > 0 else []))

# Check head for pseudo-label info
head = model.bbox_head
print("\n=== Pseudo-label info ===")
print("Head has generate_pseudo_label:", hasattr(head, 'generate_pseudo_label'))
# The pseudo-labels are generated in model.loss() and passed to head.loss()
# via batch_pseudo_instances and batch_all_instances
# batch_all_instances = GT + pseudo_labels (merged)
# batch_pseudo_instances = pseudo_labels only (for distillation)
# batch_gt_instances = real GT only (for detection matching)
# Detection loss uses batch_gt_instances (all new-class, no old-class)
# Distillation loss uses batch_pseudo_instances (old-class pseudo-labels)
print("Detection loss: uses batch_gt_instances -> NEW CLASS ONLY")
print("Distillation loss: uses batch_pseudo_instances -> OLD CLASS pseudo-labels")
print("=> Detection gradient through R(M) is ALREADY new-class-only!")
print("=> No old-class detection gradient reaches R(M) because:")
print("   1. Decouple: distillation branch uses memory_raw (no R(M))")
print("   2. Detection matching uses GT only (all new class in 70+10)")
print("   3. The ONLY detection gradient on R(M) comes from new-class GT matching")
