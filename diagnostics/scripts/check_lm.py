from mmengine.registry import DefaultScope
DefaultScope.get_instance('mmdet', scope_name='mmdet')
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmdet.registry import MODELS
cfg = Config.fromfile('configs/gdino_inc/70+10/grmi_bilinear_12e_clean.py')
cfg.default_scope = 'mmdet'
m = MODELS.build(cfg.model)
load_checkpoint(m, 'work_dirs/grmi_bilinear_12e_clean/epoch_12.pth', map_location='cpu')
lm = m.language_model
print(f'type={type(lm).__name__}')
for attr in ['body','bert','model','language_backbone','encoder']:
    if hasattr(lm, attr):
        a = getattr(lm, attr)
        print(f'  {attr}: {type(a).__name__}', end='')
        if hasattr(a, 'forward'):
            import inspect
            sig = str(inspect.signature(a.forward)).split('(')[-1][:80]
            print(f'  forward{sig}')
        else:
            print()
