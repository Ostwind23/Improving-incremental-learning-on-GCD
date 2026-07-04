# D1: Distillation Truncation at New-Class GT regions (COCO 70+10).
#
# Zeroes per-query KL/L1/GIoU distillation weights for student queries whose
# predicted box overlaps any new-class GT box with IoU >= iou_thr.
# 3-epoch smoke config on single GPU.

_base_ = './gdino_inc_70+10_70-79_gcd_scratch_coco.py'

max_epochs = 3
train_cfg = dict(max_epochs=max_epochs, val_interval=1)

train_dataloader = dict(batch_size=4, num_workers=4)

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
        distill_trunc_cfg=dict(
            enable=True,
            iou_thr=0.1,
            weight=0.0,
            ns=70, ne=80,
            monitor=dict(
                path='/home/yelingfei/logs/tatri/distill_trunc_3e_monitor.jsonl',
                interval=250,
            ),
        ),
    ),
)
