# D2: GT-Guided Selective GRMI (COCO 70+10).
#
# Applies GRMI residual R(M) with higher weight at new-class GT positions
# (alpha=1.0) and lower at old-class/background (beta=0.1), with gamma_boost=3.0.
# 3-epoch smoke config on single GPU.

_base_ = './gdino_inc_70+10_70-79_gcd_grmi_12e_coco.py'

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
    gt_selective_grmi_cfg=dict(
        enable=True,
        alpha=1.0,
        beta=0.1,
        gamma_boost=3.0,
        ns=70, ne=80,
        monitor_path='/home/yelingfei/logs/tatri/sel_grmi_3e_monitor.jsonl',
        monitor_interval=250,
    ),
)
