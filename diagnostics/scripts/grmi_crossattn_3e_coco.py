# GRMI Cross-Attention + filter_old_pseudo_grad, 3 epochs (2 GPU).
# Cross-Attention: R = LN(softmax(Q@K^T/τ) @ Q), Q=LN(W_q(T_new)), K=W_k(M)
_base_ = './gdino_inc_70+10_70-79_gcd_grmi_12e_coco.py'

max_epochs = 3
train_cfg = dict(max_epochs=max_epochs, val_interval=1)

param_scheduler = [
    dict(type='MultiStepLR', begin=0, end=max_epochs, by_epoch=True, milestones=[], gamma=0.1)]

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=20),
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='coco/bbox_mAP', rule='greater'))
log_processor = dict(type='LogProcessor', window_size=20, by_epoch=True)

model = dict(
    bbox_head=dict(filter_old_pseudo_grad=True),
    residual_inject_cfg=dict(
        enable=True,
        mode='crossattn',
        gamma_init=0.5,
        freeze_gamma=False,
        monitor=dict(
            path='/home/yelingfei/logs/tatri/grmi_crossattn_3e_monitor.jsonl',
            interval=250,
        ),
    ),
)
