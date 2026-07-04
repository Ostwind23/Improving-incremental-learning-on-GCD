# GT-duplication with IoU filter for new-class instances (COCO 70+10).
# Duplicates new-class real GT 2x before Hungarian matching so more queries
# receive new-class gradients; extra matches with IoU < 0.1 are then filtered.
# Monitor logs loss + GT-dup stats every 250 iters to a JSONL file.
#
# Run on PolyU 2xRTX4090. Compare against GCD baseline 12e
# (mAP 0.464 / new_ap 0.391) and GRMI 12e (mAP 0.459 / new_ap 0.398).

_base_ = './gdino_inc_70+10_70-79_gcd_scratch_coco.py'

max_epochs = 3
train_cfg = dict(max_epochs=max_epochs, val_interval=1)

train_dataloader = dict(
    batch_size=4,
    num_workers=4)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[],
        gamma=0.1)]

optim_wrapper = dict(accumulative_counts=2)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=20),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='coco/bbox_mAP',
        rule='greater'))
log_processor = dict(type='LogProcessor', window_size=20, by_epoch=True)

model = dict(
    bbox_head=dict(
        gt_dup_cfg=dict(
            enable=True,
            dup_factor=2,
            iou_thr=0.1,
            ns=70,
            ne=80,
            monitor=dict(
                enable=True,
                path='/home/yelingfei/logs/tatri/gt_dup_3e_monitor.jsonl',
                interval=250,
            ),
        ),
    ),
)
