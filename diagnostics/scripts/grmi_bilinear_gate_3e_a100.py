# GRMI Bilinear + Old-Class Gate, 3 epochs (self-contained for A100).
# Chains directly to GCD scratch config since GRMI/aux configs not uploaded.

_base_ = './gdino_inc_70+10_70-79_gcd_scratch_coco.py'

max_epochs = 3
train_cfg = dict(max_epochs=max_epochs, val_interval=1)

param_scheduler = [
    dict(type='MultiStepLR', begin=0, end=max_epochs, by_epoch=True,
         milestones=[], gamma=0.1)]

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=20),
    checkpoint=dict(type='CheckpointHook', interval=1,
                    save_best='coco/bbox_mAP', rule='greater'))
log_processor = dict(type='LogProcessor', window_size=20, by_epoch=True)

model = dict(
    residual_inject_cfg=dict(
        enable=True,
        mode='bilinear',
        bilinear_bottleneck=64,
        gamma_init=0.5,
        freeze_gamma=False,
        use_old_gate=True,
        monitor=dict(
            path='/root/autodl-tmp/grmi_bilinear_gate_3e_monitor.jsonl',
            interval=250,
        ),
    ),
)
