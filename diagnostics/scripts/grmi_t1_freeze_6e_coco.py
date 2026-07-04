# GRMI T1 (256→256→256) + freeze_gamma ablation, 6 epochs.
# Purpose: observe R(M) capacity improvement from T0 (128 hidden) to T1 (256 hidden)
# and monitor rm_norm/perturb_pct/delta_perturb trajectory for auto-freeze design.
#
# Changes vs original GRMI:
#   hidden_dim: 128 -> 256 (T1 tier, 131k params vs 66k)
#   freeze_gamma: True (gamma=0.01 as register_buffer, immune to weight decay)
#
# Monitored metrics (JSONL, every 250 iters):
#   rm_norm, mem_norm, rm_ratio, perturb_pct, delta_perturb, hidden_dim, n_params

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
            path='/home/yelingfei/logs/tatri/grmi_t1_freeze_6e_monitor.jsonl',
            interval=250,
        ),
    ),
)
