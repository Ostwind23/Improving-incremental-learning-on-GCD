import torch
from transformers import BertModel, BertTokenizer
from pycocotools.coco import COCO
from mmengine.registry import DefaultScope
DefaultScope.get_instance('mmdet', scope_name='mmdet')
from mmengine.config import Config
from mmdet.registry import MODELS

cfg = Config.fromfile('/home/yelingfei/projects/GCD/configs/gdino_inc/70+10/grmi_rms_t1_12e.py')
os.chdir('/home/yelingfei/projects/GCD')
cfg.default_scope = 'mmdet'
m = MODELS.build(cfg.model)
tfm = m.text_feat_map

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained('bert-base-uncased')
coco = COCO('/home/yelingfei/projects/GCD/data/coco/annotations/instances_val2017.json')
names_old = [coco.loadCats([c])[0]['name'] for c in sorted(coco.getCatIds())[0:70]]
prompt = ' . '.join(names_old) + ' .'
tok = tokenizer(prompt, return_tensors='pt', padding='max_length', max_length=256, truncation=True)
with torch.no_grad():
    out = bert(input_ids=tok['input_ids'], attention_mask=tok['attention_mask'])
    emb_768 = out.last_hidden_state.squeeze(0)
    emb_256 = tfm(emb_768)
    T_old_256 = emb_256.mean(dim=0)
    T_old_256 = T_old_256 / T_old_256.norm()
torch.save(T_old_256, '/home/yelingfei/logs/tatri/T_old_256.pt')
print(f'T_old_256 saved: norm={T_old_256.norm():.4f}')
