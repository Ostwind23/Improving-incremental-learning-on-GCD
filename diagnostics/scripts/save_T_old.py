import torch
from transformers import BertModel, BertTokenizer
from pycocotools.coco import COCO
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained('bert-base-uncased')
coco = COCO('data/coco/annotations/instances_val2017.json')
names_old = [coco.loadCats([c])[0]['name'] for c in sorted(coco.getCatIds())[0:70]]
prompt = ' . '.join(names_old) + ' .'
tok = tokenizer(prompt, return_tensors='pt', padding='max_length', max_length=256, truncation=True)
out = bert(input_ids=tok['input_ids'], attention_mask=tok['attention_mask'])
emb = out.last_hidden_state.squeeze(0)
T_old = emb.mean(dim=0)
torch.save(T_old, '/home/yelingfei/logs/tatri/T_old_bert.pt')
print(f'T_old saved: shape={T_old.shape} norm={T_old.norm():.2f}')
