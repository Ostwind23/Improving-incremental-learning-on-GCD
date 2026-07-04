from mmengine.config import Config
cfg = Config.fromfile('configs/gdino_inc/70+10/grmi_bilinear_12e_clean.py')
print('residual_inject_cfg:', 'residual_inject_cfg' in cfg.model)
if 'residual_inject_cfg' in cfg.model:
    ric = cfg.model.residual_inject_cfg
    print('  enable:', ric.get('enable'))
    print('  mode:', ric.get('mode'))
    print('  gamma:', ric.get('gamma_init'))
else:
    print('  NOT FOUND')
