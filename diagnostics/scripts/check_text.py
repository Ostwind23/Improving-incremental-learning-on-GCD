from mmengine.registry import DefaultScope
DefaultScope.get_instance('mmdet', scope_name='mmdet')
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmdet.registry import MODELS

cfg = Config.fromfile('configs/gdino_inc/70+10/grmi_bilinear_12e_clean.py')
cfg.default_scope = 'mmdet'
m = MODELS.build(cfg.model)
load_checkpoint(m, 'work_dirs/grmi_bilinear_12e_clean/epoch_12.pth', map_location='cpu')
print("Model loaded")

bh = m.bbox_head
for attr in dir(bh):
    if 'text' in attr.lower() and not attr.startswith('_'):
        val = getattr(bh, attr, None)
        if val is not None and not callable(val):
            s = f'  bh.{attr}: type={type(val).__name__}'
            if hasattr(val, 'shape'):
                s += f' shape={val.shape}'
            print(s)
