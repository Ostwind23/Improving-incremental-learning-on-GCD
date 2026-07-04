_base_ = './gdino_inc_70+10_70-79_gcd_vlm_aux_only_12e_bs4_coco.py'
max_epochs = 3
train_cfg = dict(max_epochs=max_epochs, val_interval=1)
model = dict(
    residual_inject_cfg=dict(
        enable=True, in_dim=256, hidden_dim=128,
        gamma_init=0.01, dropout=0.0, act_inference=True),
    tatri_cfg=dict(
        enable=True, in_dim=256, hidden_dim=64,
        mode='per_token', gate_hidden=32))
