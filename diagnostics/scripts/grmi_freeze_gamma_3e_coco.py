# GRMI + freeze_gamma 3e experiment (COCO 70+10).
# γ fixed at 0.01 (not trained), R(M) MLP still trainable.
# Tests whether fixing gamma prevents distillation from compressing the
# residual signal (baseline GRMI: γ decays from 0.01 to 0.000366 in 12e).

_base_ = './gdino_inc_70+10_70-79_gcd_grmi_12e_coco.py'

max_epochs = 3
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
            path='/home/yelingfei/logs/tatri/grmi_freeze_gamma_3e_monitor.jsonl',
            interval=250,
        ),
    ),
)
