# GRMI + freeze_gamma (FIXED: register_buffer) 6e.
# γ=0.01 truly frozen (no gradient, no weight decay).
# R(M) MLP still trainable. 6 epochs for clearer trajectory.

_base_ = './gdino_inc_70+10_70-79_gcd_grmi_12e_coco.py'

max_epochs = 6
train_cfg = dict(max_epochs=max_epochs, val_interval=1)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[],
        gamma=0.1)]

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=20),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='coco/bbox_mAP',
        rule='greater'))
log_processor = dict(type='LogProcessor', window_size=20, by_epoch=True)

model = dict(
    residual_inject_cfg=dict(
        enable=True,
        gamma_init=0.01,
        freeze_gamma=True,
        monitor=dict(
            path='/home/yelingfei/logs/tatri/grmi_freeze_gamma_6e_monitor.jsonl',
            interval=250,
        ),
    ),
)
