_base_ = './gdino_inc_70+10_70-79_gcd_grmi_12e_coco.py'
model = dict(residual_inject_cfg=dict(enable=True, mode='bilinear', bilinear_bottleneck=64, gamma_init=0.5, freeze_gamma=False, monitor=dict(path='/home/yelingfei/logs/tatri/grmi_bilinear_12e_clean_monitor.jsonl', interval=250)))
