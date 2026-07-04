# GRMI + decouple + filter_old_pseudo_grad, 6 epochs.
#
# filter_old_pseudo_grad=True: removes old-class pseudo-label detection gradient
# from R(M) by zeroing label_weights/bbox_weights for queries matched to
# old-class tokens (position < 169). cos(E[g_new], E[g_old]) = -0.99 per
# accumulated gradient diagnostic, so this gives R(M) a pure new-class signal.
#
# Reference baseline: grmi_t1_decouple_6e_coco.py (filter_old_pseudo_grad=False)

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
    bbox_head=dict(
        filter_old_pseudo_grad=True,
    ),
    residual_inject_cfg=dict(
        enable=True,
        hidden_dim=256,
        gamma_init=0.01,
        freeze_gamma=True,
        monitor=dict(
            path='/home/yelingfei/logs/tatri/grmi_filter_old_6e_monitor.jsonl',
            interval=250,
        ),
    ),
)
