# GRMI T1 + freeze_gamma + decoupled R(M) from distillation, 6 epochs.
#
# Key change: distillation branch (topology + KL) uses memory_raw (no R(M)),
# detection branch uses memory_enhanced = M + gamma*R(M).
# This prevents R(M) from destabilizing inter_query_loss (topology distillation).
#
# Hypothesis: inter_query_loss should be stable (~0.60 like baseline),
# new_ap should still benefit from R(M), and no ep5-6 decline.

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
        hidden_dim=256,
        gamma_init=0.01,
        freeze_gamma=True,
        monitor=dict(
            path='/home/yelingfei/logs/tatri/grmi_t1_decouple_6e_monitor.jsonl',
            interval=250,
        ),
    ),
)
