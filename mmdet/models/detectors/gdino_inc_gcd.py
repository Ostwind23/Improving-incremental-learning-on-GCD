# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import os
import json
import copy
from typing import Dict, List, Optional, Tuple, Union
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from mmengine.runner import load_checkpoint, load_state_dict
from mmengine.model import is_model_wrapper
from mmengine import Config
from mmengine.logging import MessageHub
from mmengine.structures import InstanceData
from mmdet.utils import ConfigType, OptConfigType, InstanceList
from mmdet.structures.bbox import bbox2roi
from mmdet.models.utils import multi_apply
from mmdet.registry import MODELS, TASK_UTILS
from .class_token_offset import ClassTokenOffset
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh, bbox_overlaps
from mmdet.models.dense_heads.atss_vlfusion_head import convert_grounding_to_cls_scores
from mmdet.structures import OptSampleList, SampleList
from ..layers import SinePositionalEncoding, CdnQueryGenerator
from ..layers import inverse_sigmoid
from .gdino_inc_distn import GroundingDINO_inc_distn
from mmdet.models.losses.distn_loss import generate_distn_points
from ..utils.vlm_channel1 import NewClassAuxHead, load_vlm_cache, \
    build_encoder_aux_targets, build_new_query_mask
from ..utils.residual_inject import ResidualInject
from ..utils.text_residual_inject import TextResidualInject


def _iou_xyxy(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = ((a[2] - a[0]) * (a[3] - a[1]) +
          (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / ua if ua > 0 else 0.0

@MODELS.register_module()
class GroundingDINO_inc_gcd(GroundingDINO_inc_distn):
    """Implementation of `Grounding DINO: Marrying DINO with Grounded Pre-
    Training for Open-Set Object Detection.

    <https://arxiv.org/abs/2303.05499>`_

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/GroundingDINO>`_.
    """

    COCO_CLASSES = (
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush')

    def __init__(self, *args, prototype_cfg: OptConfigType = None,
                 e1_cfg: OptConfigType = None,
                 vlm_aux_cfg: OptConfigType = None,
                 residual_inject_cfg: OptConfigType = None,
                 dec_aux_cfg: OptConfigType = None,
                 tatri_cfg: OptConfigType = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # --- GRMI: gated residual injection on encoder memory (LGQS-1) ---
        self.residual_inject_cfg = Config._dict_to_config_dict_lazy(
            residual_inject_cfg or dict(enable=False))
        self.residual_inject = None
        if bool(self.residual_inject_cfg.get('enable', False)):
            _mode = str(self.residual_inject_cfg.get('mode', 'mlp'))
            _bb = int(self.residual_inject_cfg.get('bilinear_bottleneck', 64))
            self.residual_inject = ResidualInject(
                in_dim=int(self.residual_inject_cfg.get('in_dim', 256)),
                hidden_dim=int(self.residual_inject_cfg.get('hidden_dim', 128)),
                gamma_init=float(self.residual_inject_cfg.get('gamma_init', 1e-2)),
                dropout=float(self.residual_inject_cfg.get('dropout', 0.0)),
                act_inference=bool(self.residual_inject_cfg.get('act_inference', True)),
                freeze_gamma=bool(self.residual_inject_cfg.get('freeze_gamma', False)),
                mode=_mode,
                bilinear_bottleneck=_bb, use_old_gate=bool(self.residual_inject_cfg.get("use_old_gate",False)), norm_mode=str(self.residual_inject_cfg.get("norm_mode","ln")),rms_target=float(self.residual_inject_cfg.get("rms_target",2.0)), init_std=float(self.residual_inject_cfg.get("init_std",0.01)), ortho_mode=str(self.residual_inject_cfg.get("ortho_mode","none")))
            _ri = self.residual_inject
            _params = sum(p.numel() for p in _ri.parameters())
            print(f"[GRMI] {_mode} enabled: gamma_init={float(_ri.gamma):.4f} params={_params}")
            _grmi_mon = self.residual_inject_cfg.get('monitor', {}) or {}
            self._grmi_mon_path = str(_grmi_mon.get('path', ''))
            self._grmi_mon_interval = int(_grmi_mon.get('interval', 250))
            self._grmi_mon_step = 0
            self._grmi_prev_perturb = 0.0
            
        # --- Build orthogonal projection for old-class subspace ---
        _ortho = str(self.residual_inject_cfg.get('ortho_mode', 'none'))
        if _ortho != 'none' and self.residual_inject is not None:
            # Get old-class text embeddings from text_feat_map
            # Need to build text_feat_map first
            from pycocotools.coco import COCO
            import os as _os2
            ann_file = _os2.path.join('data', 'coco', 'annotations', 'instances_val2017.json')
            _coco = COCO(ann_file)
            _old_names = [_coco.loadCats([c])[0]['name'] for c in sorted(_coco.getCatIds())[0:70]]
            _prompt = ' . '.join(_old_names) + ' .'
            _tokenized = self.language_model.tokenizer([_prompt], padding='max_length', max_length=256, truncation=True, return_tensors='pt')
            _ids = _tokenized['input_ids'].cuda()
            _mask = _tokenized['attention_mask'].cuda()
            with torch.no_grad():
                _bert_out = self.language_model(input_ids=_ids, attention_mask=_mask)
                _text_emb = _bert_out.last_hidden_state
                _text_256 = self.text_feat_map(_text_emb)  # (1, 256, 256)
                T_old = _text_256[:, :70, :].squeeze(0)  # (70, 256)
            self.residual_inject.build_ortho(T_old)
            print(f'[Ortho] Built {_ortho} projection with T_old shape {T_old.shape}')

        self._gs_supp_weight = float(self.residual_inject_cfg.get('gs_supp_weight', 0.001))
        self._gs_ratio_for_log = 0.0
        self._gs_grmi_enabled = bool(self.residual_inject_cfg.get('gs_grmi', False))



        # --- TATRI: text-side gated residual injection ---
        self.tatri_cfg_obj = Config._dict_to_config_dict_lazy(
            tatri_cfg or dict(enable=False))
        self.text_residual_inject = None
        self._new_text_token_mask = None
        if bool(self.tatri_cfg_obj.get('enable', False)):
            self.text_residual_inject = TextResidualInject(
                in_dim=int(self.tatri_cfg_obj.get('in_dim', 256)),
                hidden_dim=int(self.tatri_cfg_obj.get('hidden_dim', 64)),
                gamma_init=float(self.tatri_cfg_obj.get('gamma_init', 0.05)),
                mode=str(self.tatri_cfg_obj.get('mode', 'fixed_gamma')),
                gate_hidden=int(self.tatri_cfg_obj.get('gate_hidden', 32)))
        # --- Decoder multi-layer aux classification heads ---
        self.dec_aux_cfg = Config._dict_to_config_dict_lazy(
            dec_aux_cfg or dict(enable=False))
        self.dec_aux_enable = bool(self.dec_aux_cfg.get('enable', False))
        self.dec_aux_heads = nn.ModuleDict()
        self.dec_aux_layers = []
        if self.dec_aux_enable:
            self.dec_aux_layers = list(self.dec_aux_cfg.get('layers', [3, 4, 5]))
            _da_hidden = int(self.dec_aux_cfg.get('hidden_dim', 128))
            _da_ncls = int(self.dec_aux_cfg.get('num_classes', 10))
            _da_drop = float(self.dec_aux_cfg.get('dropout', 0.0))
            for lid in self.dec_aux_layers:
                self.dec_aux_heads[str(lid)] = NewClassAuxHead(
                    in_dim=256, hidden_dim=_da_hidden,
                    num_classes=_da_ncls, dropout=_da_drop)
            self.dec_aux_weight = float(self.dec_aux_cfg.get('loss_weight', 0.01))
            self.dec_aux_iou_weight = bool(self.dec_aux_cfg.get('iou_weight', True))
            self.dec_aux_iou_floor = float(self.dec_aux_cfg.get('iou_floor', 0.1))
            print(f"[DecAux] layers={self.dec_aux_layers} w={self.dec_aux_weight}")
        self._monitor_cfg = self.dec_aux_cfg.get('monitor', {}) if self.dec_aux_enable else {}
        self._monitor_path = self._monitor_cfg.get('path', '')
        self._monitor_interval = int(self._monitor_cfg.get('interval', 250))
        self._monitor_step = 0

        self.prototype_cfg = Config._dict_to_config_dict_lazy(
            prototype_cfg or dict(enable=False))
        self._raw_e1_cfg = e1_cfg
        self.prototype_enable = bool(self.prototype_cfg.get('enable', False))
        self.prototype_text_embeds = None
        self.prototype_class_ids = None
        self.prototype_prompt_version = self.prototype_cfg.get(
            'prompt_version', 'P0')

        # --- Visual prototype bank for old-class consistency (Scheme A+B) ---
        self.visual_proto_enable = bool(
            self.prototype_cfg.get('visual_proto_enable', False))
        if self.visual_proto_enable:
            num_old = int(self.prototype_cfg.get('start', 70))
            hidden_dim = 256
            self.register_buffer(
                'visual_proto_bank', torch.zeros(num_old, hidden_dim))
            self.register_buffer(
                'visual_proto_counts', torch.zeros(num_old, dtype=torch.long))
            self.register_buffer(
                'visual_proto_frozen', torch.tensor(False))
            self.register_buffer(
                'topo_stored', torch.zeros(num_old, num_old))
            # Scheme D: diagonal variance per class
            self.register_buffer(
                'visual_proto_var', torch.zeros(num_old, hidden_dim))
            self.register_buffer(
                'visual_proto_sq_bank', torch.zeros(num_old, hidden_dim))
            # Scheme E: BERT text similarity matrix (built at freeze time)
            self.register_buffer(
                'text_topo_matrix', torch.zeros(0))
            # Scheme E': VLM similarity matrix (loaded from file)
            vlm_path = self.prototype_cfg.get('vlm_topo_matrix_path', None)
            if vlm_path and os.path.exists(str(vlm_path)):
                self.register_buffer(
                    'vlm_topo_matrix',
                    torch.load(str(vlm_path), map_location='cpu').float())
            else:
                self.register_buffer(
                    'vlm_topo_matrix', torch.zeros(0))
            # New-class topology prior (external model, e.g. SigLIP-2)
            new_topo_path = self.prototype_cfg.get(
                'new_class_topo_matrix_path', None)
            if new_topo_path and os.path.exists(str(new_topo_path)):
                self.register_buffer(
                    'new_class_topo_matrix',
                    torch.load(
                        str(new_topo_path), map_location='cpu').float())
            else:
                self.register_buffer(
                    'new_class_topo_matrix', torch.zeros(0))


        # --- Stage 4 Plan A: CLIP-guided text contrastive loss ---
        self.text_contrast_enable = bool(
            self.prototype_cfg.get('text_contrast_enable', False))
        if self.text_contrast_enable:
            _pairs_raw = self.prototype_cfg.get('text_contrast_pairs', None)
            if _pairs_raw:
                self.text_contrast_pairs = _pairs_raw
            else:
                self.text_contrast_pairs = [
                    (41, 45, 0.904),   # cup vs bowl
                    (48, 53, 0.884),   # sandwich vs pizza
                    (41, 40, 0.874),   # cup vs wine glass
                    (15, 16, 0.927),   # cat vs dog
                    (7,  2,  0.928),   # truck vs car
                    (52, 16, 0.897),   # hot dog vs dog
                    (17, 18, 0.869),   # horse vs sheep
                    (19, 17, 0.877),   # cow vs horse
                ]
            self.text_contrast_weight = float(
                self.prototype_cfg.get('text_contrast_weight', 0.1))
            self.text_contrast_margin = float(
                self.prototype_cfg.get('text_contrast_margin', 0.05))


        # --- Stage 4: query-level confusion-pair margin loss ---
        self.confusion_pair_margin_enable = bool(
            self.prototype_cfg.get('confusion_pair_margin_enable', False))
        self.confusion_pair_margin_weight = float(
            self.prototype_cfg.get('confusion_pair_margin_weight', 0.05))
        self.confusion_pair_margin = float(
            self.prototype_cfg.get('confusion_pair_margin', 0.10))
        self.confusion_pair_iou_thr = float(
            self.prototype_cfg.get('confusion_pair_iou_thr', 0.50))
        _pairs = self.prototype_cfg.get('confusion_pair_margin_pairs', None)
        if _pairs:
            self.confusion_pair_margin_pairs = [(int(a), int(b)) for a, b in _pairs]
        else:
            self.confusion_pair_margin_pairs = [
                (41, 45), (48, 53), (41, 40), (15, 16),
                (7, 2), (52, 16), (17, 18), (19, 17),
                (41, 39), (39, 40), (54, 55), (48, 52),
            ]
        self.confusion_pair_alt_map = {}
        for _a, _b in self.confusion_pair_margin_pairs:
            self.confusion_pair_alt_map.setdefault(_a, []).append(_b)
            self.confusion_pair_alt_map.setdefault(_b, []).append(_a)

        # --- E1: Learnable class token offset for new classes ---
        self.e1_cfg = Config._dict_to_config_dict_lazy(
            getattr(self, '_raw_e1_cfg', None) or dict(enabled=False))
        self.e1_enabled = bool(self.e1_cfg.get('enabled', False))
        self.e1_strong = bool(self.e1_cfg.get('strong', False))
        self.class_token_offset = None
        if self.e1_enabled:
            start_cls = int(self.e1_cfg.get('start', self.start))
            end_cls = int(self.e1_cfg.get('end', self.end))
            new_class_positions = self._get_new_class_token_positions(
                start_cls, end_cls)
            init_scale = float(self.e1_cfg.get('init_scale', 0.0))
            self.class_token_offset = ClassTokenOffset(
                new_class_positions, embed_dim=256, init_scale=init_scale)
        if self.e1_strong:
            for p in self.text_feat_map.parameters():
                p.requires_grad = True
            print(f"[E1-strong] text_feat_map unfrozen, "
                  f"lr_mult={self.e1_cfg.get('text_feat_map_lr_mult', 0.1)}")

        # --- Channel-① VLM auxiliary supervision (external new-class oracle) ---
        self.vlm_aux_cfg = Config._dict_to_config_dict_lazy(
            vlm_aux_cfg or dict(enable=False))
        self.vlm_aux_enable = bool(self.vlm_aux_cfg.get('enable', False))
        self.vlm_aux_head = None
        self._vlm_cache = None
        if self.vlm_aux_enable:
            self.vlm_aux_path = self.vlm_aux_cfg.get(
                'cache_path', '/home/yelingfei/logs/vlm_label_train2017/'
                              'vlm_train_labels.json')
            self.vlm_aux_start = int(self.vlm_aux_cfg.get('start', self.start))
            self.vlm_aux_end = int(self.vlm_aux_cfg.get('end', self.end))
            self.vlm_aux_iou_thr = float(self.vlm_aux_cfg.get('iou_thr', 0.5))
            self.vlm_aux_weight = float(self.vlm_aux_cfg.get('loss_weight', 0.01))
            self.vlm_aux_warmup_iters = int(
                self.vlm_aux_cfg.get('warmup_iters', 0))
            self.vlm_aux_ramp_iters = int(
                self.vlm_aux_cfg.get('ramp_iters', 0))
            # optional distillation masking
            self.vlm_distill_mask_enable = bool(
                self.vlm_aux_cfg.get('distill_mask_enable', False))
            self.vlm_distill_mask_max = int(
                self.vlm_aux_cfg.get('distill_mask_max', 50))
            self.vlm_distill_mask_weight = float(
                self.vlm_aux_cfg.get('distill_mask_weight', 1.0))
            # Direction A: background-suppression branch.
            # When bg_enable=True the aux head gets an extra "background" logit
            # and every encoder position that is NOT inside a (new-class GT,
            # or -- when ignore_old_gt=True -- any old-class GT) is labelled
            # as background. This is the only "free" negative supervision for
            # hallucinated encoder positions; it cannot use real old-class
            # labels (oracle risk), so old-class GT regions are *ignored* by
            # default (ignore_old_gt=True => old-GT positions get label=-1).
            self.vlm_aux_bg_enable = bool(
                self.vlm_aux_cfg.get('bg_enable', False))
            self.vlm_aux_bg_ignore_old_gt = bool(
                self.vlm_aux_cfg.get('bg_ignore_old_gt', True))
            self.vlm_aux_bg_weight = float(
                self.vlm_aux_cfg.get('bg_loss_weight', 1.0))
            # Background warmup (iters): bg CE weight is 0 for the first
            # bg_warmup_iters, letting the new-class positive signal settle.
            self.vlm_aux_bg_warmup_iters = int(
                self.vlm_aux_cfg.get('bg_warmup_iters', 0))
            # Background position subsampling: if bg_max_per_img > 0, randomly
            # keep at most that many background positions per image so the bg
            # CE gradient does not overwhelm the new-class positive signal
            # (there are ~20-30x more bg positions than new-class positions).
            self.vlm_aux_bg_max_per_img = int(
                self.vlm_aux_cfg.get('bg_max_per_img', 0))
            # Neg-mining: positions inside a new-class GT whose supervisor
            # label is vlm_correct==False get the aux head background logit,
            # teaching the head NOT to fire the new class there. Sparse and
            # targeted (unlike generic bg which marks every non-new region).
            self.vlm_aux_neg_enable = bool(
                self.vlm_aux_cfg.get('neg_enable', False))
            self.vlm_aux_neg_weight = float(
                self.vlm_aux_cfg.get('neg_weight', 0.005))
            self.vlm_aux_neg_max_per_img = int(
                self.vlm_aux_cfg.get('neg_max_per_img', 0))
            # Optional cosine schedule for the new-class aux weight, to push
            # past the LGQS phase-transition faster early (high lambda) then
            # taper off (low lambda) so late training settles without
            # disturbing old-class features. When schedule='cosine', lambda
            # goes from aux_weight_max -> aux_weight_min over aux_weight_T
            # epochs. Default 'constant' keeps vlm_aux_weight fixed.
            self.vlm_aux_weight_schedule = str(
                self.vlm_aux_cfg.get('aux_weight_schedule', 'constant'))
            self.vlm_aux_weight_max = float(
                self.vlm_aux_cfg.get('aux_weight_max', self.vlm_aux_weight))
            self.vlm_aux_weight_min = float(
                self.vlm_aux_cfg.get('aux_weight_min', self.vlm_aux_weight))
            self.vlm_aux_weight_T = float(
                self.vlm_aux_cfg.get('aux_weight_T', 12))
            # Experiment C: injection point ablation.
            #   - 'encoder' (default): aux head runs on encoder memory
            #     (B, N_pos, 256), gradient reaches backbone/neck via the
            #     LGQS-prior path. This is Channel-①.
            #   - 'decoder': aux head runs on the LAST decoder hidden state
            #     (B, num_queries, 256) instead of encoder memory. Same VLM
            #     supervision signal, same head, same weight, but injected
            #     AFTER LGQS / inside the SCM/shared-query feature path. If
            #     the LGQS-bottleneck story holds, this should be absorbed by
            #     distillation (≈baseline or worse).
            self.vlm_aux_inject_point = str(
                self.vlm_aux_cfg.get('inject_point', 'encoder'))
            num_new = self.vlm_aux_end - self.vlm_aux_start
            self.vlm_aux_head = NewClassAuxHead(
                in_dim=256, hidden_dim=128, num_classes=num_new, dropout=0.0,
                bg_class=(self.vlm_aux_bg_enable or self.vlm_aux_neg_enable))
            self.register_buffer(
                'vlm_aux_step',
                torch.zeros((), dtype=torch.long),
                persistent=False)
            print(f"[VLM-Aux] enabled: num_new={num_new} "
                  f"weight={self.vlm_aux_weight} "
                  f"warmup={self.vlm_aux_warmup_iters} "
                  f"ramp={self.vlm_aux_ramp_iters} "
                  f"distill_mask={self.vlm_distill_mask_enable} "
                  f"bg_enable={self.vlm_aux_bg_enable} "
                  f"bg_ignore_old_gt={self.vlm_aux_bg_ignore_old_gt} "
                  f"bg_weight={self.vlm_aux_bg_weight} "
                  f"bg_warmup={self.vlm_aux_bg_warmup_iters} "
                  f"bg_max_per_img={self.vlm_aux_bg_max_per_img}")

    @staticmethod
    def _get_new_class_token_positions(start_cls, end_cls):
        """Get BERT token positions for new classes.

        Token positions are deterministic given the fixed GCD prompt
        (80 class names separated by ' . '). Hardcoded from Phase 0a analysis.
        """
        # Phase 0a verified positions for classes 70-79 in GCD 70+10 prompt
        known_positions = {
            70: [169, 170],   # toaster (toast + ##er)
            71: [172],        # sink
            72: [174],        # refrigerator
            73: [176],        # book
            74: [178],        # clock
            75: [180],        # vase
            76: [182],        # scissors
            77: [184, 185],   # teddy bear
            78: [187, 188, 189],  # hair drier (hair + dr + ##ier)
            79: [191, 192],   # toothbrush (tooth + ##brush)
        }
        result = {}
        for cid in range(start_cls, end_cls):
            if cid in known_positions:
                result[cid] = known_positions[cid]
        return result

    def train(self, mode=True):
        return super().train(mode)

    def _prototype_prompt_for_class(self, class_name: str) -> str:
        version = self.prototype_prompt_version
        if version == 'T_photo':
            return f'a photo of a {class_name}'
        if version == 'P1_old':
            old_desc = {
                'book': 'blue spines on shelves',
                'clock': 'wooden clock with black face and white hands',
                'vase': 'blue cylindrical container with logo',
                'teddy bear': 'a person in a brown costume holding a teddy bear',
                'toaster': 'small kitchen appliance for browning bread',
                'sink': 'basin with faucet used for washing',
                'refrigerator': 'large kitchen appliance with doors for cold storage',
                'scissors': 'two metal blades with handles for cutting',
                'hair drier': 'handheld electric device for drying hair',
                'toothbrush': 'small brush with handle for cleaning teeth',
            }
            desc = old_desc.get(class_name, '')
            return f'{class_name}. {desc}' if desc else class_name
        return class_name

    @torch.no_grad()
    def _build_textual_prototypes(self):
        if self.prototype_text_embeds is not None:
            return
        start = int(self.prototype_cfg.get('start', self.start))
        end = int(self.prototype_cfg.get('end', self.end))
        class_ids = list(range(start, end))
        prompts = [
            self._prototype_prompt_for_class(self.COCO_CLASSES[class_id])
            for class_id in class_ids
        ]
        lang_training = self.language_model.training
        map_training = self.text_feat_map.training if self.text_feat_map else False
        self.language_model.eval()
        if self.text_feat_map is not None:
            self.text_feat_map.eval()
        text_dict = self.language_model(prompts)
        if self.text_feat_map is not None:
            text_dict['embedded'] = self.text_feat_map(text_dict['embedded'])
        self.language_model.train(lang_training)
        if self.text_feat_map is not None:
            self.text_feat_map.train(map_training)
        embedded = text_dict['embedded']
        token_mask = text_dict['text_token_mask'].to(embedded.device).float()
        token_mask[:, 0] = 0  # drop [CLS]
        proto = (embedded * token_mask.unsqueeze(-1)).sum(dim=1)
        proto = proto / token_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        proto = torch.nn.functional.normalize(proto, dim=-1)
        self.prototype_text_embeds = proto.detach()
        self.prototype_class_ids = torch.tensor(
            class_ids, device=proto.device, dtype=torch.long)
        print('[B2 prototype] built textual prototypes: '
              f'version={self.prototype_prompt_version}, classes={class_ids}, '
              f'prompts={prompts}')

    # ------------------------------------------------------------------
    # Visual prototype bank: accumulation, freeze, loss (Scheme A+B+E')
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _update_visual_proto_bank(self, new_head_inputs_dict,
                                  batch_data_samples,
                                  batch_pseudo_instances=None):
        """Accumulate old-class query features into visual prototype bank.

        Old classes come from teacher pseudo-labels (batch_pseudo_instances),
        NOT from batch_data_samples.gt_instances (which only has new classes).
        """
        if self.visual_proto_frozen:
            return
        if batch_pseudo_instances is None:
            return

        hidden_states = new_head_inputs_dict['hidden_states']
        references = new_head_inputs_dict['references']
        memory_text = new_head_inputs_dict['memory_text']
        text_token_mask = new_head_inputs_dict['text_token_mask']
        dn_meta = new_head_inputs_dict.get('dn_meta', None)
        last_hidden = hidden_states[-1]

        all_cls_scores, all_bbox_preds = self.bbox_head(
            hidden_states, references, memory_text, text_token_mask)
        (matching_cls_scores, matching_bbox_preds, _, _) = \
            self.bbox_head.split_outputs(
                all_cls_scores, all_bbox_preds, dn_meta)
        last_cls_scores = matching_cls_scores[-1]
        last_bbox_preds = matching_bbox_preds[-1]
        if last_hidden.size(1) != last_cls_scores.size(1):
            last_hidden = last_hidden[:, -last_cls_scores.size(1):, :]

        start = int(self.prototype_cfg.get('start', self.start))
        momentum = float(self.prototype_cfg.get(
            'visual_proto_momentum', 0.999))
        accum_iou_thr = float(self.prototype_cfg.get(
            'visual_proto_accum_iou_thr', 0.5))

        for img_idx, data_sample in enumerate(batch_data_samples):
            if img_idx >= len(batch_pseudo_instances):
                continue
            pseudo_inst = batch_pseudo_instances[img_idx]
            if pseudo_inst is None or len(pseudo_inst.labels) == 0:
                continue
            old_mask_pseudo = pseudo_inst.labels < start
            if old_mask_pseudo.sum() == 0:
                continue
            old_bboxes = pseudo_inst.bboxes[old_mask_pseudo]
            old_labels = pseudo_inst.labels[old_mask_pseudo]

            img_meta = data_sample.metainfo
            img_h, img_w = img_meta['img_shape']
            factor = last_bbox_preds.new_tensor(
                [img_w, img_h, img_w, img_h]).unsqueeze(0)
            abs_bboxes = bbox_cxcywh_to_xyxy(
                last_bbox_preds[img_idx]) * factor

            iou_matrix = bbox_overlaps(abs_bboxes, old_bboxes)
            max_ious, best_query = iou_matrix.max(dim=0)
            valid = max_ious >= accum_iou_thr
            if valid.sum() == 0:
                continue

            pos_inds = best_query[valid]
            matched_labels = old_labels[valid]
            feats = last_hidden[img_idx, pos_inds.long()]

            for cls_id in matched_labels.unique():
                c = cls_id.long().item()
                cls_feats = feats[matched_labels == cls_id]
                mean_feat = cls_feats.mean(dim=0)
                mean_sq = (cls_feats ** 2).mean(dim=0)
                if self.visual_proto_counts[c] == 0:
                    self.visual_proto_bank[c] = mean_feat
                    self.visual_proto_sq_bank[c] = mean_sq
                else:
                    self.visual_proto_bank[c] = (
                        momentum * self.visual_proto_bank[c]
                        + (1 - momentum) * mean_feat)
                    self.visual_proto_sq_bank[c] = (
                        momentum * self.visual_proto_sq_bank[c]
                        + (1 - momentum) * mean_sq)
                self.visual_proto_counts[c] += cls_feats.size(0)

    @torch.no_grad()
    def _freeze_visual_proto_bank(self):
        """Normalize and freeze visual prototype bank, compute topology."""
        initialized = self.visual_proto_counts > 0
        num_init = initialized.sum().item()
        if num_init == 0:
            print('[VisualProto] WARNING: no old classes accumulated, '
                  'skipping freeze')
            self.visual_proto_frozen.fill_(True)
            return

        # Scheme D: compute diagonal variance = E[x^2] - E[x]^2
        self.visual_proto_var[initialized] = (
            self.visual_proto_sq_bank[initialized]
            - self.visual_proto_bank[initialized] ** 2
        ).clamp(min=1e-8)

        self.visual_proto_bank[initialized] = \
            torch.nn.functional.normalize(
                self.visual_proto_bank[initialized], dim=-1)

        self.topo_stored = self.visual_proto_bank @ \
            self.visual_proto_bank.t()

        # Scheme E: build BERT text prototype similarity for ALL old classes
        self._build_text_topo_matrix()

        self.visual_proto_frozen.fill_(True)
        per_class = self.visual_proto_counts[initialized].float()
        var_mean = self.visual_proto_var[initialized].mean().item()
        print(f'[VisualProto] Frozen bank: {num_init}/{self.visual_proto_bank.size(0)} '
              f'classes, total_samples={self.visual_proto_counts.sum().item()}, '
              f'per_class_mean={per_class.mean():.1f}, '
              f'per_class_min={per_class.min():.0f}, '
              f'per_class_max={per_class.max():.0f}, '
              f'var_mean={var_mean:.6f}')

    @torch.no_grad()
    def _build_text_topo_matrix(self):
        """Build BERT text prototype cosine similarity matrix for old classes (Scheme E)."""
        start = int(self.prototype_cfg.get('start', self.start))
        class_ids = list(range(start))
        prompts = [f'a photo of a {self.COCO_CLASSES[c]}' for c in class_ids]

        lang_training = self.language_model.training
        map_training = self.text_feat_map.training if self.text_feat_map else False
        self.language_model.eval()
        if self.text_feat_map is not None:
            self.text_feat_map.eval()

        text_dict = self.language_model(prompts)
        if self.text_feat_map is not None:
            text_dict['embedded'] = self.text_feat_map(text_dict['embedded'])

        self.language_model.train(lang_training)
        if self.text_feat_map is not None:
            self.text_feat_map.train(map_training)

        embedded = text_dict['embedded']
        token_mask = text_dict['text_token_mask'].to(embedded.device).float()
        token_mask[:, 0] = 0
        proto = (embedded * token_mask.unsqueeze(-1)).sum(dim=1)
        proto = proto / token_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        proto = torch.nn.functional.normalize(proto, dim=-1)

        self.text_topo_matrix = (proto @ proto.t()).detach()
        print(f'[VisualProto] Built BERT text topo matrix: '
              f'{self.text_topo_matrix.shape}')


    def _compute_text_contrastive_loss(self, text_dict, batch_data_samples):
        """Plan A: push apart text prototypes for confusion pairs."""
        if not self.text_contrast_enable:
            return {}
        if not hasattr(self, 'token_positive_maps') or self.token_positive_maps is None:
            return {}

        embedded = text_dict['embedded']  # [B, L, 256]

        loss = torch.tensor(0.0, device=embedded.device)
        count = 0

        for (ca, cb, target_cos) in self.text_contrast_pairs:
            key_a = ca + 1
            key_b = cb + 1
            if key_a not in self.token_positive_maps or key_b not in self.token_positive_maps:
                continue

            pos_a = self.token_positive_maps[key_a]
            pos_b = self.token_positive_maps[key_b]

            proto_a = embedded[:, pos_a, :].mean(dim=1).mean(dim=0)  # [256]
            proto_b = embedded[:, pos_b, :].mean(dim=1).mean(dim=0)  # [256]

            cos_sim = F.cosine_similarity(proto_a.unsqueeze(0), proto_b.unsqueeze(0))
            threshold = target_cos - self.text_contrast_margin
            pair_loss = F.relu(cos_sim - threshold)
            loss = loss + pair_loss
            count += 1

        if count == 0:
            return {}

        loss = loss / count * self.text_contrast_weight
        return dict(loss_text_contrast=loss)

    def _compute_visual_proto_losses(self, new_head_inputs_dict,
                                     batch_data_samples,
                                     batch_pseudo_instances=None):
        """Compute visual prototype cosine consistency (A) and topology (B).

        Old-class features come from teacher pseudo-labels.
        """
        if not self.visual_proto_enable or not self.visual_proto_frozen:
            return {}
        if batch_pseudo_instances is None:
            return {}

        epoch = 0.0
        try:
            epoch = float(
                MessageHub.get_current_instance().get_info('epoch'))
        except Exception:
            epoch = 0.0

        vp_warmup = float(self.prototype_cfg.get(
            'visual_proto_warmup_epochs', 1))
        vp_ramp = float(self.prototype_cfg.get(
            'visual_proto_ramp_epochs', 1))
        if epoch < vp_warmup:
            wf = 0.0
        elif vp_ramp > 0 and epoch < vp_warmup + vp_ramp:
            wf = (epoch - vp_warmup + 1.0) / vp_ramp
            wf = max(0.0, min(1.0, wf))
        else:
            wf = 1.0

        zero = new_head_inputs_dict['hidden_states'][-1].sum() * 0.0
        empty_out = {
            'loss_old_cos': zero,
            'loss_old_w2': zero,
            'loss_topo': zero,
            'loss_text_topo': zero,
            'loss_vlm_topo': zero,
            'loss_new_topo': zero,
            'vproto_weight_factor': zero.detach(),
            'vproto_num_classes_seen': zero.detach(),
            'vproto_cos_mean': zero.detach(),
        }
        if wf == 0.0:
            return empty_out

        # --- Collect current-batch per-class features for old classes ---
        hidden_states = new_head_inputs_dict['hidden_states']
        references = new_head_inputs_dict['references']
        memory_text = new_head_inputs_dict['memory_text']
        text_token_mask = new_head_inputs_dict['text_token_mask']
        dn_meta = new_head_inputs_dict.get('dn_meta', None)
        last_hidden = hidden_states[-1]

        all_cls_scores, all_bbox_preds = self.bbox_head(
            hidden_states, references, memory_text, text_token_mask)
        (matching_cls_scores, matching_bbox_preds, _, _) = \
            self.bbox_head.split_outputs(
                all_cls_scores, all_bbox_preds, dn_meta)
        last_cls_scores = matching_cls_scores[-1]
        last_bbox_preds = matching_bbox_preds[-1]
        if last_hidden.size(1) != last_cls_scores.size(1):
            last_hidden = last_hidden[:, -last_cls_scores.size(1):, :]

        start = int(self.prototype_cfg.get('start', self.start))
        iou_low = float(self.prototype_cfg.get('proto_iou_low', 0.3))
        iou_high = float(self.prototype_cfg.get('proto_iou_high', 0.7))
        initialized = self.visual_proto_counts > 0

        current_class_feats = {}  # cls_id -> list of weighted mean tensors

        for img_idx, data_sample in enumerate(batch_data_samples):
            if img_idx >= len(batch_pseudo_instances):
                continue
            pseudo_inst = batch_pseudo_instances[img_idx]
            if pseudo_inst is None or len(pseudo_inst.labels) == 0:
                continue
            old_mask_pseudo = pseudo_inst.labels < start
            if old_mask_pseudo.sum() == 0:
                continue
            old_bboxes = pseudo_inst.bboxes[old_mask_pseudo]
            old_labels_ps = pseudo_inst.labels[old_mask_pseudo]

            img_meta = data_sample.metainfo
            with torch.no_grad():
                img_h, img_w = img_meta['img_shape']
                factor = last_bbox_preds.new_tensor(
                    [img_w, img_h, img_w, img_h]).unsqueeze(0)
                abs_bboxes = bbox_cxcywh_to_xyxy(
                    last_bbox_preds[img_idx]) * factor

                iou_matrix = bbox_overlaps(abs_bboxes, old_bboxes)
                max_ious, best_query = iou_matrix.max(dim=0)
                denom = max(iou_high - iou_low, 1e-6)
                qw = ((max_ious - iou_low) / denom).clamp(0.0, 1.0)
                valid = qw > 0
                if valid.sum() == 0:
                    continue

            pos_inds = best_query[valid]
            feats = last_hidden[img_idx, pos_inds.long()]
            old_labels = old_labels_ps[valid]
            weights = qw[valid]

            for cls_id in old_labels.unique():
                c = cls_id.long().item()
                if not initialized[c]:
                    continue
                mask_c = old_labels == cls_id
                w_c = weights[mask_c]
                f_c = feats[mask_c]
                wmean = (f_c * w_c.unsqueeze(-1)).sum(dim=0) / \
                    w_c.sum().clamp(min=1e-6)
                if c not in current_class_feats:
                    current_class_feats[c] = []
                current_class_feats[c].append(wmean)

        if not current_class_feats:
            return empty_out

        # --- Scheme A: cosine consistency ---
        cos_losses = []
        for c, feat_list in current_class_feats.items():
            current_proto = torch.stack(feat_list).mean(dim=0)
            current_proto = torch.nn.functional.normalize(
                current_proto, dim=-1)
            stored = self.visual_proto_bank[c].to(current_proto.device)
            cos_losses.append(1.0 - (current_proto * stored).sum())

        loss_old_cos = torch.stack(cos_losses).mean()

        # --- Scheme D: simplified W2 distance (diagonal Gaussian) ---
        w2_losses = []
        w2_w = float(self.prototype_cfg.get('loss_old_w2_weight', 0.0))
        if w2_w > 0:
            for c, feat_list in current_class_feats.items():
                all_feats = torch.stack(feat_list)
                cur_mu = all_feats.mean(dim=0)
                cur_var = all_feats.var(dim=0).clamp(min=1e-8) \
                    if all_feats.size(0) > 1 \
                    else torch.ones_like(cur_mu) * 1e-4
                stored_mu = self.visual_proto_bank[c].to(cur_mu.device)
                stored_var = self.visual_proto_var[c].to(cur_mu.device)
                stored_sigma = stored_var.sqrt()
                cur_sigma = cur_var.sqrt()
                w2_sq = ((cur_mu - stored_mu) ** 2).sum() + \
                    ((cur_sigma - stored_sigma) ** 2).sum()
                w2_losses.append(w2_sq)
        loss_old_w2 = torch.stack(w2_losses).mean() \
            if w2_losses else zero

        # --- Scheme B: topology consistency (stored visual topo) ---
        seen = sorted(current_class_feats.keys())
        if len(seen) >= 2:
            cur_protos = []
            for c in seen:
                p = torch.stack(current_class_feats[c]).mean(dim=0)
                cur_protos.append(
                    torch.nn.functional.normalize(p, dim=-1))
            cur_protos = torch.stack(cur_protos)
            topo_cur = cur_protos @ cur_protos.t()
            idx = torch.tensor(
                seen, dtype=torch.long,
                device=self.topo_stored.device)
            topo_target = self.topo_stored[idx][:, idx].to(
                topo_cur.device)
            loss_topo = torch.nn.functional.mse_loss(
                topo_cur, topo_target)
        else:
            loss_topo = zero

        # --- Scheme E: BERT text topo as target ---
        text_topo_w = float(self.prototype_cfg.get(
            'loss_text_topo_weight', 0.0))
        if (text_topo_w > 0 and self.text_topo_matrix.numel() > 0
                and len(seen) >= 2):
            text_sub = self.text_topo_matrix.to(topo_cur.device)
            text_sub = text_sub[idx][:, idx]
            loss_text_topo = torch.nn.functional.mse_loss(
                topo_cur, text_sub)
        else:
            loss_text_topo = zero

        # --- Scheme E': VLM-informed topology (optional) ---
        vlm_w = float(self.prototype_cfg.get('loss_vlm_topo_weight', 0.0))
        vlm_loss_type = str(self.prototype_cfg.get(
            'vlm_topo_loss_type', 'mse'))
        if (vlm_w > 0 and self.vlm_topo_matrix.numel() > 0
                and len(seen) >= 2):
            vlm_sub = self.vlm_topo_matrix.to(topo_cur.device)
            vlm_sub = vlm_sub[idx][:, idx]
            if vlm_loss_type == 'pearson':
                x = topo_cur.flatten()
                y = vlm_sub.flatten()
                x_c = x - x.mean()
                y_c = y - y.mean()
                corr = (x_c * y_c).sum() / (
                    x_c.norm() * y_c.norm() + 1e-6)
                loss_vlm_topo = 1.0 - corr
            elif vlm_loss_type == 'softmax_kl':
                tau = float(self.prototype_cfg.get(
                    'vlm_topo_temperature', 0.1))
                row_norm = bool(self.prototype_cfg.get(
                    'vlm_topo_row_norm', False))
                tc = topo_cur
                vs = vlm_sub
                if row_norm:
                    tc = torch.nn.functional.normalize(tc, p=2, dim=1)
                    vs = torch.nn.functional.normalize(vs, p=2, dim=1)
                p = torch.nn.functional.softmax(vs / tau, dim=1)
                q = torch.nn.functional.log_softmax(tc / tau, dim=1)
                loss_vlm_topo = torch.nn.functional.kl_div(
                    q, p, reduction='batchmean')
            else:
                loss_vlm_topo = torch.nn.functional.mse_loss(
                    topo_cur, vlm_sub)
        else:
            loss_vlm_topo = zero

        # --- New-class topology prior (SigLIP-2 KL on new classes) ---
        new_topo_w = float(
            self.prototype_cfg.get('loss_new_topo_weight', 0.0))
        if (new_topo_w > 0
                and hasattr(self, 'new_class_topo_matrix')
                and self.new_class_topo_matrix.numel() > 0):
            end = int(self.prototype_cfg.get('end', 80))
            new_class_feats = {}
            for img_idx, data_sample in enumerate(batch_data_samples):
                gt_instances = data_sample.gt_instances
                if gt_instances is None or len(gt_instances.labels) == 0:
                    continue
                new_mask = ((gt_instances.labels >= start)
                            & (gt_instances.labels < end))
                if new_mask.sum() == 0:
                    continue
                new_bboxes = gt_instances.bboxes[new_mask]
                new_labels = gt_instances.labels[new_mask]
                img_meta = data_sample.metainfo
                with torch.no_grad():
                    img_h, img_w = img_meta['img_shape']
                    factor = last_bbox_preds.new_tensor(
                        [img_w, img_h, img_w, img_h]).unsqueeze(0)
                    abs_bboxes_new = bbox_cxcywh_to_xyxy(
                        last_bbox_preds[img_idx]) * factor
                    iou_mat = bbox_overlaps(abs_bboxes_new, new_bboxes)
                    max_ious, best_q = iou_mat.max(dim=0)
                    denom = max(iou_high - iou_low, 1e-6)
                    qw_new = ((max_ious - iou_low) / denom).clamp(
                        0.0, 1.0)
                    valid_new = qw_new > 0
                    if valid_new.sum() == 0:
                        continue
                pos_inds_new = best_q[valid_new]
                feats_new = last_hidden[img_idx, pos_inds_new.long()]
                labels_new = new_labels[valid_new]
                weights_new = qw_new[valid_new]
                for cls_id in labels_new.unique():
                    c = cls_id.long().item() - start
                    mask_c = labels_new == cls_id
                    w_c = weights_new[mask_c]
                    f_c = feats_new[mask_c]
                    wmean = (f_c * w_c.unsqueeze(-1)).sum(dim=0) / \
                        w_c.sum().clamp(min=1e-6)
                    if c not in new_class_feats:
                        new_class_feats[c] = []
                    new_class_feats[c].append(wmean)
            new_seen = sorted(new_class_feats.keys())
            if len(new_seen) >= 2:
                new_protos = []
                for c in new_seen:
                    p = torch.stack(new_class_feats[c]).mean(dim=0)
                    new_protos.append(
                        torch.nn.functional.normalize(p, dim=-1))
                new_protos = torch.stack(new_protos)
                new_topo_cur = new_protos @ new_protos.t()
                new_idx = torch.tensor(
                    new_seen, dtype=torch.long,
                    device=self.new_class_topo_matrix.device)
                new_topo_target = self.new_class_topo_matrix[
                    new_idx][:, new_idx].to(new_topo_cur.device)
                ntc = torch.nn.functional.normalize(
                    new_topo_cur, p=2, dim=1)
                nvs = torch.nn.functional.normalize(
                    new_topo_target, p=2, dim=1)
                new_tau = float(self.prototype_cfg.get(
                    'new_class_topo_temperature', 0.1))
                np_ = torch.nn.functional.softmax(nvs / new_tau, dim=1)
                nq_ = torch.nn.functional.log_softmax(
                    ntc / new_tau, dim=1)
                loss_new_topo = torch.nn.functional.kl_div(
                    nq_, np_, reduction='batchmean')
            else:
                loss_new_topo = zero
        else:
            loss_new_topo = zero

        # --- Weights and logging ---
        cos_w = float(self.prototype_cfg.get('loss_old_cos_weight', 0.01))
        topo_w = float(self.prototype_cfg.get('loss_topo_weight', 0.005))

        with torch.no_grad():
            cos_vals = torch.tensor(
                [1.0 - cl.item() for cl in cos_losses])
            cos_mean_val = cos_vals.mean().to(zero.device)
            num_seen = zero.new_tensor(float(len(seen)))
            wf_tensor = zero.new_tensor(float(wf))

        return {
            'loss_old_cos': loss_old_cos * cos_w * wf,
            'loss_old_w2': loss_old_w2 * w2_w * wf,
            'loss_topo': loss_topo * topo_w * wf,
            'loss_text_topo': loss_text_topo * text_topo_w * wf,
            'loss_vlm_topo': loss_vlm_topo * vlm_w * wf,
            'loss_new_topo': loss_new_topo * new_topo_w * wf,
            'vproto_weight_factor': wf_tensor.detach(),
            'vproto_num_classes_seen': num_seen.detach(),
            'vproto_cos_mean': cos_mean_val.detach(),
        }

    def _compute_b2_prototype_loss(self, new_head_inputs_dict, batch_data_samples):
        if not self.prototype_enable:
            return {}
        self._build_textual_prototypes()
        if self.prototype_text_embeds is None:
            return {}

        hidden_states = new_head_inputs_dict['hidden_states']
        references = new_head_inputs_dict['references']
        memory_text = new_head_inputs_dict['memory_text']
        text_token_mask = new_head_inputs_dict['text_token_mask']
        dn_meta = new_head_inputs_dict.get('dn_meta', None)
        last_hidden = hidden_states[-1]

        all_cls_scores, all_bbox_preds = self.bbox_head(
            hidden_states, references, memory_text, text_token_mask)
        (matching_cls_scores, matching_bbox_preds, _, _) = \
            self.bbox_head.split_outputs(all_cls_scores, all_bbox_preds, dn_meta)
        last_cls_scores = matching_cls_scores[-1]
        last_bbox_preds = matching_bbox_preds[-1]
        if last_hidden.size(1) != last_cls_scores.size(1):
            # Drop denoising queries when hidden states still include them.
            last_hidden = last_hidden[:, -last_cls_scores.size(1):, :]

        visual_proto_list = []
        target_list = []
        quality_list = []
        start = int(self.prototype_cfg.get('start', self.start))
        end = int(self.prototype_cfg.get('end', self.end))
        detach_visual = bool(self.prototype_cfg.get('detach_visual', False))
        for img_idx, data_sample in enumerate(batch_data_samples):
            img_meta = data_sample.metainfo
            gt_instances = data_sample.gt_instances
            if len(gt_instances.labels) == 0:
                continue
            with torch.no_grad():
                img_h, img_w = img_meta['img_shape']
                factor = last_bbox_preds.new_tensor(
                    [img_w, img_h, img_w, img_h]).unsqueeze(0)
                abs_bboxes = bbox_cxcywh_to_xyxy(last_bbox_preds[img_idx]) * factor
                pred_instances = InstanceData(
                    scores=last_cls_scores[img_idx], bboxes=abs_bboxes)
                assign_result = self.bbox_head.assigner.assign(
                    pred_instances=pred_instances,
                    gt_instances=gt_instances,
                    img_meta=img_meta)
                pos_inds = torch.nonzero(
                    assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
                if pos_inds.numel() == 0:
                    continue
                pos_gt_inds = assign_result.gt_inds[pos_inds] - 1
                labels = gt_instances.labels[pos_gt_inds.long()]
                matched_pred_boxes = abs_bboxes[pos_inds.long()]
                matched_gt_boxes = gt_instances.bboxes[pos_gt_inds.long()]
                quality = bbox_overlaps(
                    matched_pred_boxes, matched_gt_boxes, is_aligned=True)
            valid = (labels >= start) & (labels < end)
            if valid.sum() == 0:
                continue
            feats = last_hidden[img_idx, pos_inds[valid].long()]
            if detach_visual:
                feats = feats.detach()
            visual_proto_list.append(feats)
            target_list.append((labels[valid] - start).long())
            quality_list.append(quality[valid].to(feats.device))

        zero = last_hidden.sum() * 0.0
        if not visual_proto_list:
            return {
                'loss_proto_ce': zero,
                'loss_proto_align': zero,
                'proto_pos_count': zero.detach(),
                'proto_acc': zero.detach(),
                'proto_weight_factor': zero.detach(),
                'proto_quality_mean': zero.detach(),
                'proto_quality_nonzero_count': zero.detach(),
                'loss_proto_ce_raw': zero.detach(),
            }

        visual_feats = torch.cat(visual_proto_list, dim=0)
        targets = torch.cat(target_list, dim=0)
        quality_scores = torch.cat(quality_list, dim=0).to(
            visual_feats.device, visual_feats.dtype)
        text_proto = self.prototype_text_embeds.to(visual_feats.device, visual_feats.dtype)
        visual_feats = torch.nn.functional.normalize(visual_feats, dim=-1)
        logits = visual_feats @ text_proto.t()
        tau = float(self.prototype_cfg.get('tau', 0.07))
        logits = logits / tau
        base_ce_weight = float(self.prototype_cfg.get('loss_proto_weight', 0.05))
        warmup_epochs = float(self.prototype_cfg.get('warmup_epochs', 0))
        ramp_epochs = float(self.prototype_cfg.get('ramp_epochs', 0))
        epoch = 0.0
        try:
            epoch = float(MessageHub.get_current_instance().get_info('epoch'))
        except Exception:
            epoch = 0.0
        if epoch < warmup_epochs:
            proto_weight_factor = 0.0
        elif ramp_epochs > 0 and epoch < warmup_epochs + ramp_epochs:
            proto_weight_factor = (epoch - warmup_epochs + 1.0) / ramp_epochs
            proto_weight_factor = max(0.0, min(1.0, proto_weight_factor))
        else:
            proto_weight_factor = 1.0
        ce_weight = base_ce_weight * proto_weight_factor
        align_weight = float(self.prototype_cfg.get('loss_align_weight', 0.0))
        per_sample_ce = torch.nn.functional.cross_entropy(
            logits.float(), targets, reduction='none')
        quality_mode = self.prototype_cfg.get('proto_quality_mode', 'none')
        if quality_mode == 'iou':
            quality_weights = quality_scores.clamp(min=0.0, max=1.0)
        elif quality_mode == 'iou_threshold':
            iou_thr = float(self.prototype_cfg.get('proto_iou_thr', 0.5))
            quality_weights = (quality_scores >= iou_thr).to(quality_scores.dtype)
        elif quality_mode == 'iou_linear':
            iou_low = float(self.prototype_cfg.get('proto_iou_low', 0.3))
            iou_high = float(self.prototype_cfg.get('proto_iou_high', 0.7))
            denom = max(iou_high - iou_low, 1e-6)
            quality_weights = ((quality_scores - iou_low) / denom).clamp(0.0, 1.0)
        else:
            quality_weights = torch.ones_like(quality_scores)
        min_proto_pos = int(self.prototype_cfg.get('min_proto_pos', 1))
        nonzero_count = (quality_weights > 0).sum()
        if nonzero_count < min_proto_pos:
            loss_proto_ce = per_sample_ce.sum() * 0.0
        else:
            loss_proto_ce = (per_sample_ce * quality_weights).sum() / \
                quality_weights.sum().clamp(min=1.0)
        matched_text = text_proto[targets]
        loss_proto_align = 1.0 - (visual_feats * matched_text).sum(dim=-1).mean()
        with torch.no_grad():
            proto_acc = (logits.argmax(dim=-1) == targets).float().mean()
            proto_pos_count = targets.new_tensor(float(targets.numel())).to(visual_feats.dtype)
            proto_weight_factor_tensor = targets.new_tensor(
                float(proto_weight_factor)).to(visual_feats.dtype)
            proto_quality_mean = quality_scores.mean()
            proto_quality_nonzero_count = targets.new_tensor(
                float(nonzero_count.item())).to(visual_feats.dtype)
            loss_proto_ce_raw = per_sample_ce.mean()
        return {
            'loss_proto_ce': loss_proto_ce * ce_weight,
            'loss_proto_align': loss_proto_align * align_weight,
            'proto_pos_count': proto_pos_count.detach(),
            'proto_acc': proto_acc.detach(),
            'proto_weight_factor': proto_weight_factor_tensor.detach(),
            'proto_quality_mean': proto_quality_mean.detach(),
            'proto_quality_nonzero_count': proto_quality_nonzero_count.detach(),
            'loss_proto_ce_raw': loss_proto_ce_raw.detach(),
        }


    def _compute_confusion_pair_margin_loss(self, new_head_inputs_dict,
                                            batch_all_instances,
                                            batch_data_samples):
        """Stage 4: single-instance margin loss for known confusion pairs."""
        if not self.confusion_pair_margin_enable:
            return {}
        if batch_all_instances is None or self.token_positive_maps is None:
            return {}

        hidden_states = new_head_inputs_dict['hidden_states']
        references = new_head_inputs_dict['references']
        memory_text = new_head_inputs_dict['memory_text']
        text_token_mask = new_head_inputs_dict['text_token_mask']
        dn_meta = new_head_inputs_dict.get('dn_meta', None)

        all_cls_scores, all_bbox_preds = self.bbox_head(
            hidden_states, references, memory_text, text_token_mask)
        matching_cls_scores, matching_bbox_preds, _, _ = self.bbox_head.split_outputs(
            all_cls_scores, all_bbox_preds, dn_meta)
        last_cls_scores = matching_cls_scores[-1]
        last_bbox_preds = matching_bbox_preds[-1]

        loss_terms = []
        margin_values = []
        for img_idx, inst in enumerate(batch_all_instances):
            if inst is None or len(inst.labels) == 0:
                continue
            labels = inst.labels
            valid_inst = [
                k for k, lab in enumerate(labels.tolist())
                if int(lab) in self.confusion_pair_alt_map
            ]
            if not valid_inst:
                continue

            cls_prob = convert_grounding_to_cls_scores(
                logits=last_cls_scores[img_idx].sigmoid()[None],
                positive_maps=[self.token_positive_maps])[0]

            img_meta = batch_data_samples[img_idx].metainfo
            img_h, img_w = img_meta['img_shape']
            factor = last_bbox_preds.new_tensor([img_w, img_h, img_w, img_h])
            pred_bboxes = bbox_cxcywh_to_xyxy(last_bbox_preds[img_idx]) * factor
            pred_bboxes[:, 0::2].clamp_(min=0, max=img_w)
            pred_bboxes[:, 1::2].clamp_(min=0, max=img_h)

            target_bboxes = inst.bboxes.to(pred_bboxes.device, pred_bboxes.dtype)
            ious = bbox_overlaps(pred_bboxes, target_bboxes)
            best_iou, best_query = ious.max(dim=0)

            for inst_idx in valid_inst:
                if best_iou[inst_idx] < self.confusion_pair_iou_thr:
                    continue
                label = int(labels[inst_idx].item())
                query_idx = int(best_query[inst_idx].item())
                if label >= cls_prob.shape[-1]:
                    continue
                target_score = cls_prob[query_idx, label]
                for alt in self.confusion_pair_alt_map.get(label, []):
                    if alt >= cls_prob.shape[-1]:
                        continue
                    alt_score = cls_prob[query_idx, int(alt)]
                    margin = target_score - alt_score
                    loss_terms.append(F.relu(
                        self.confusion_pair_margin - margin))
                    margin_values.append(margin.detach())

        zero = hidden_states[-1].sum() * 0.0
        if not loss_terms:
            return {
                'loss_confusion_pair_margin': zero,
                'confusion_pair_count': zero.detach(),
            }

        loss = torch.stack(loss_terms).mean() * self.confusion_pair_margin_weight
        with torch.no_grad():
            count = loss.new_tensor(float(len(loss_terms)))
            mean_margin = torch.stack(margin_values).mean()
        return {
            'loss_confusion_pair_margin': loss,
            'confusion_pair_count': count.detach(),
            'confusion_pair_mean_margin': mean_margin.detach(),
        }

    def forward_encoder(self, *args, **kwargs):
        encoder_outputs_dict = super().forward_encoder(*args, **kwargs)
        # GRMI: apply gated residual to memory BEFORE LGQS (student only;
        # teacher self.ori_model uses its own unmodified forward_encoder).
        if self.residual_inject is not None and \
                (self.training or self.residual_inject.act_inference):
            memory = encoder_outputs_dict['memory']
            encoder_outputs_dict['memory_raw'] = memory
            _mt = encoder_outputs_dict.get('memory_text')
            _T_new = _mt[0, 169:189, :] if _mt is not None else None
            encoder_outputs_dict['memory'] = self.residual_inject(memory, _T_new)
        else:
            encoder_outputs_dict['memory_raw'] = encoder_outputs_dict['memory']
        # TATRI: text-side gated residual injection
        if self.text_residual_inject is not None and \
                (self.training or True):
            mt = encoder_outputs_dict['memory_text']
            if self._new_text_token_mask is None and hasattr(self, 'token_positive_maps'):
                tpm = self.token_positive_maps
                T_len = mt.shape[1]
                mask = mt.new_zeros(T_len)
                for k, positions in tpm.items():
                    cls_id = k - 1
                    if 70 <= cls_id < 80:
                        for pos in positions:
                            if pos < T_len:
                                mask[pos] = 1.0
                self._new_text_token_mask = mask
                print(f'[TATRI] new-class mask: {int(mask.sum())}/{T_len} tokens')
            encoder_outputs_dict['memory_text'] = self.text_residual_inject(
                mt, self._new_text_token_mask)
        return encoder_outputs_dict

    def forward_transformer(

        self,
        img_feats: Tuple[Tensor],
        text_dict: Dict,
        batch_data_samples: OptSampleList = None,
        aux_dict: Dict = None
    ) -> Dict:
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)

        encoder_outputs_dict = self.forward_encoder(
            **encoder_inputs_dict, text_dict=text_dict)
        
        if self.training:
            new_decoder_inputs_dict = decoder_inputs_dict.copy()
            old_decoder_inputs_dict = decoder_inputs_dict.copy()
            # forward on newtext
            _memory_raw = encoder_outputs_dict.pop('memory_raw', None)
            new_tmp_dec_in, new_head_inputs_dict = self.pre_decoder_new(
                **encoder_outputs_dict, batch_data_samples=batch_data_samples)
            new_decoder_inputs_dict.update(new_tmp_dec_in)
            # D2-SA v2: capture the actual self-attention transition q_in -> q_sa.
            _sa_cache = {}
            _orig_fwds = {}
            for _li in [4, 5]:
                _sa_cache[_li] = None
                layer = self.decoder.layers[_li]
                _orig_fwds[_li] = layer.forward
                def _make_patch(layer_obj, lid):
                    orig_fn = layer_obj.forward
                    def _sa_patched(*args, **kwargs):
                        q_in = args[0] if args else kwargs.get('query')
                        q_sa_raw = layer_obj.self_attn(
                            query=q_in, key=q_in, value=q_in,
                            query_pos=kwargs.get('query_pos'),
                            key_pos=kwargs.get('query_pos'),
                            attn_mask=kwargs.get('self_attn_mask'))
                        q_sa = layer_obj.norms[0](q_sa_raw)
                        _sa_cache[lid] = {'q_in': q_in, 'q_sa': q_sa}
                        return orig_fn(*args, **kwargs)
                    return _sa_patched
                layer.forward = _make_patch(layer, _li)

            try:
                new_decoder_outputs_dict = self.forward_decoder(**new_decoder_inputs_dict)
            finally:
                for _li, _orig in _orig_fwds.items():
                    self.decoder.layers[_li].forward = _orig
            new_head_inputs_dict['sa_features'] = {
                k: v for k, v in _sa_cache.items() if v is not None}
            new_head_inputs_dict.update(new_decoder_outputs_dict)

            # forward on oldtext -- use memory_raw to decouple R(M) from distillation
            encoder_outputs_for_old = encoder_outputs_dict.copy()
            encoder_outputs_for_old['memory'] = _memory_raw
            encoder_outputs_for_old['text_token_mask'] = self.ori_text_masks

            old_tmp_dec_in, old_head_inputs_dict = self.pre_decoder_old(
                **encoder_outputs_for_old, aux_dict=aux_dict, batch_data_samples=batch_data_samples)
            old_decoder_inputs_dict.update(old_tmp_dec_in)
            old_decoder_outputs_dict = self.forward_decoder(**old_decoder_inputs_dict)
            old_head_inputs_dict.update(old_decoder_outputs_dict)

            return new_head_inputs_dict, old_head_inputs_dict
        
        else:
            encoder_outputs_dict.pop('memory_raw', None)
            tmp_dec_in, head_inputs_dict = self.pre_decoder(
                **encoder_outputs_dict, aux_dict=aux_dict, batch_data_samples=batch_data_samples)
            decoder_inputs_dict.update(tmp_dec_in)      
            decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
            head_inputs_dict.update(decoder_outputs_dict)

            return head_inputs_dict      
        
    def pre_decoder_old(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        memory_text: Tensor,
        text_token_mask: Tensor,
        aux_dict: Dict = None,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        bs, _, c = memory.shape
        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)
        
        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers](output_memory, memory_text,
                                     text_token_mask)
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers].max_text_len
        enc_outputs_coord_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers](output_memory) + output_proposals
        
        # NOTE The DINO selects top-k proposals according to scores of
        # multi-class classification, while DeformDETR, where the input
        # is `enc_outputs_class[..., 0]` selects according to scores of
        # binary classification.
        topk_indices = torch.topk(enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1)[1]
        topk_score = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        topk_coords = topk_coords_unact.sigmoid()

        aux_query, aux_reference, self_attn_mask = generate_distn_points(self.distn_cfg, aux_dict)   
        query = aux_query
        reference_points = aux_reference

        dn_mask, dn_meta = None, None

        # reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask,
            memory_text=memory_text,
            text_attention_mask=~text_token_mask,
        )
        # NOTE DINO calculates encoder losses on scores and coordinates
        # of selected top-k encoder queries, while DeformDETR is of all
        # encoder queries.
        if self.training :
            head_inputs_dict = dict(enc_outputs_class=topk_score, enc_outputs_coord=topk_coords, 
                                    dn_meta=dn_meta) 
        else:
            head_inputs_dict = dict()
        # append text_feats to head_inputs_dict
        head_inputs_dict['memory_text'] = memory_text
        head_inputs_dict['text_token_mask'] = text_token_mask
        return decoder_inputs_dict, head_inputs_dict  
        
    def pre_decoder_new(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        memory_text: Tensor,
        text_token_mask: Tensor,
        aux_dict: Dict = None,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]: 
        
        bs, _, c = memory.shape
        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)
    
        # # forward on new text 
        # text_token_mask = self.new_text_masks

        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers](output_memory, memory_text,
                                     text_token_mask)
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers].max_text_len
        enc_outputs_coord_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers](output_memory) + output_proposals

        # NOTE The DINO selects top-k proposals according to scores of
        # multi-class classification, while DeformDETR, where the input
        # is `enc_outputs_class[..., 0]` selects according to scores of
        # binary classification.
        topk_indices = torch.topk(enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1)[1]
        topk_score = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        
        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()

        query = self.query_embedding.weight[:, None, :]
        query = query.repeat(1, bs, 1).transpose(0, 1)
        
        if self.training :
            if self.dn_cfg is not None:
                dn_label_query, dn_bbox_query, dn_mask, dn_meta = self.dn_query_generator(batch_data_samples)
                query = torch.cat([dn_label_query, query], dim=1)
                reference_points = torch.cat([dn_bbox_query, topk_coords_unact], dim=1)
            else:
                dn_mask, dn_meta = None, None                
                reference_points = topk_coords_unact
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None

        reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask,
            memory_text=memory_text,
            text_attention_mask=~text_token_mask,
        )
        # NOTE DINO calculates encoder losses on scores and coordinates
        # of selected top-k encoder queries, while DeformDETR is of all
        # encoder queries.
        if self.training :
            head_inputs_dict = dict(enc_outputs_class=topk_score, enc_outputs_coord=topk_coords,
                                    dn_meta=dn_meta)
        else:
            head_inputs_dict = dict()
        # append text_feats to head_inputs_dict
        head_inputs_dict['memory_text'] = memory_text
        head_inputs_dict['text_token_mask'] = text_token_mask
        # Channel-①: stash encoder memory + spatial shapes + matched references
        # for the aux head / distill mask, computed in loss().
        if self.vlm_aux_enable:
            head_inputs_dict['vlm_enc_memory'] = memory
            head_inputs_dict['vlm_spatial_shapes'] = spatial_shapes
            head_inputs_dict['vlm_matched_reference'] = topk_coords_unact.sigmoid()
            # Experiment C: also stash decoder query feature positions so the
            # decoder-injection variant can build spatial targets the same way.
            head_inputs_dict['vlm_query_reference'] = topk_coords_unact.sigmoid()
        return decoder_inputs_dict, head_inputs_dict

    def forward_ori_model(
            self,
            img_feats: Tuple[Tensor],
            text_dict: Dict,
            batch_data_samples: OptSampleList = None,
        ):
        encoder_inputs_dict, decoder_inputs_dict = self.ori_model.pre_transformer(
            img_feats, batch_data_samples)

        encoder_outputs_dict = self.ori_model.forward_encoder(
            **encoder_inputs_dict, text_dict=text_dict)

        tmp_dec_in, head_inputs_dict = self.ori_model.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples, query_distn=self.distn_cfg.query_distn)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.ori_model.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)  

        if self.distn_cfg.query_distn.type == 'seperate_queryinit' :
            head_inputs_dict['aux_query'] = tmp_dec_in['query'].clone()
            head_inputs_dict['aux_reference'] = tmp_dec_in['reference_points'].clone()

        head_inputs_dict['ori_text_token_mask'] = head_inputs_dict.pop('text_token_mask')
        head_inputs_dict['ori_memory_text'] = head_inputs_dict.pop('memory_text')
        head_inputs_dict['ori_hidden_states'] = head_inputs_dict.pop('hidden_states')
        head_inputs_dict['ori_references'] = head_inputs_dict.pop('references')
        
        return head_inputs_dict

    def _grmi_monitor_log(self, losses):
        import json as _jg
        if not hasattr(self, "_grmi_mon_path") or not self._grmi_mon_path:
            return
        self._grmi_mon_step += 1
        if self._grmi_mon_step % self._grmi_mon_interval != 0:
            return
        try:
            from mmengine.logging import MessageHub
            hub = MessageHub.get_current_instance()
            epoch = hub.get_info("epoch"); it = hub.get_info("iter")
        except Exception:
            epoch = -1; it = -1
        ri = self.residual_inject
        _gamma = float(ri.gamma.detach().item())
        _rm_n = getattr(ri, "_cached_rm_norm", 0.0)
        _mem_n = getattr(ri, "_cached_mem_norm", 1.0)
        _ratio = _rm_n / max(_mem_n, 1e-8)
        _perturb = _gamma * _ratio
        _prev = getattr(self, "_grmi_prev_perturb", _perturb)
        _delta = abs(_perturb - _prev) / max(abs(_prev), 1e-8) if abs(_prev) > 1e-10 else 0.0
        self._grmi_prev_perturb = _perturb
        rec = {"step": self._grmi_mon_step, "epoch": epoch, "iter": it,
               "gamma": round(_gamma, 6),
               "freeze_gamma": ri.freeze_gamma,
               "rm_norm": round(_rm_n, 4),
               "mem_norm": round(_mem_n, 4),
               "rm_ratio": round(_ratio, 6),
               "perturb_pct": round(_perturb * 100, 6),
               "delta_perturb": round(_delta, 6),
               "hidden_dim": (ri.W_m.out_features if hasattr(ri, "W_m") else ri.net[0].out_features),
               "n_params": sum(p.numel() for p in ri.parameters())}
        for k, v in losses.items():
            if hasattr(v, "item"):
                rec[k] = round(float(v.detach().item()), 6)
        try:
            import os as _osg
            _osg.makedirs(_osg.path.dirname(self._grmi_mon_path), exist_ok=True)
            with open(self._grmi_mon_path, "a") as f:
                f.write(_jg.dumps(rec) + chr(10))
        except Exception:
            pass

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        
        text_prompts = [
            data_samples.text for data_samples in batch_data_samples
        ]
        # text for ori model 
        ori_text_prompts = [
            data_samples.ori_text for data_samples in batch_data_samples
        ]

        gt_labels = [
            data_samples.gt_instances.labels
            for data_samples in batch_data_samples
        ]
        if 'tokens_positive' in batch_data_samples[0]:
            tokens_positive = [
                data_samples.tokens_positive
                for data_samples in batch_data_samples
            ]
            positive_maps = []
            for token_positive, text_prompt, gt_label in zip(
                    tokens_positive, text_prompts, gt_labels):
                tokenized = self.language_model.tokenizer(
                    [text_prompt],
                    padding='max_length'
                    if self.language_model.pad_to_max else 'longest',
                    return_tensors='pt')
                new_tokens_positive = [
                    token_positive[label.item()] for label in gt_label
                ]
                _, positive_map = self.get_positive_map(
                    tokenized, new_tokens_positive)
                positive_maps.append(positive_map)
            new_text_prompts = text_prompts
        else:
            new_text_prompts = []
            positive_maps = []

            if len(set(text_prompts)) == 1:
                tokenized, caption_string, tokens_positive, _ = \
                    self.get_tokens_and_prompts(
                        text_prompts[0], True)

                new_text_prompts = [caption_string] * len(batch_inputs)

                ori_tokenized, ori_caption_string, ori_tokens_positive, _ = \
                    self.get_tokens_and_prompts(
                        ori_text_prompts[0], True)
                ori_text_prompts = [ori_caption_string] * len(batch_inputs)

                # generate global ori_token_positive_maps and ori model prompts
                if self.token_positive_maps is None:
                    token_positive_maps, _ = self.get_positive_map(tokenized, tokens_positive)
                    self.token_positive_maps = token_positive_maps
                    ori_token_positive_maps, _ = self.get_positive_map(ori_tokenized, ori_tokens_positive)
                    self.ori_token_positive_maps = ori_token_positive_maps   

                for gt_label in gt_labels:
                    new_tokens_positive = [
                        tokens_positive[label] for label in gt_label
                    ]
                    # NOTE construct a map such that positive_map[i,j] = True if box i is associated to token j
                    _, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive)
                    positive_maps.append(positive_map)
            else:
                for text_prompt, gt_label in zip(text_prompts, gt_labels):
                    tokenized, caption_string, tokens_positive, _ = \
                        self.get_tokens_and_prompts(
                            text_prompt, True)
                    new_tokens_positive = [
                        tokens_positive[label] for label in gt_label
                    ]
                    _, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive)
                    positive_maps.append(positive_map)
                    new_text_prompts.append(caption_string)

        # new text forward
        text_dict = self.language_model(new_text_prompts)
        # text_dict = self.language_model(ori_text_prompts)
        if self.text_feat_map is not None:
            text_dict['embedded'] = self.text_feat_map(text_dict['embedded'])   # [in_feat = 768, out_feat = 256]

        # E1: apply learnable class token offset
        if self.class_token_offset is not None:
            text_dict['embedded'] = self.class_token_offset(
                text_dict['embedded'])

        for i, data_samples in enumerate(batch_data_samples):
            positive_map = positive_maps[i].to(
                batch_inputs.device).bool().float()
            text_token_mask = text_dict['text_token_mask'][i]
            data_samples.gt_instances.positive_maps = positive_map
            data_samples.gt_instances.text_token_mask = \
                text_token_mask.unsqueeze(0).repeat(
                    len(positive_map), 1)
                        
        # chunked new class text token mask for full class prompt
        if self.new_text_token_mask_chunked is None:
            new_text_token_mask_chunk_pos = self.token_positive_maps[self.start+1][0]
            new_text_token_mask_chunked = copy.deepcopy(text_token_mask)  
            new_text_token_mask_chunked[:new_text_token_mask_chunk_pos] = False
            new_text_token_mask_chunked = new_text_token_mask_chunked.unsqueeze(0).repeat(len(batch_data_samples),1)
            self.new_text_masks = new_text_token_mask_chunked   # text mask for new class  
            self.ori_text_masks = ~new_text_token_mask_chunked

        # ori text forward（given same text as new model but trunc accroding to ori_text_token_mask）
        with torch.no_grad():
            # ori model forward on full text
            if self.distn_cfg.future_class:
                ori_text_dict = self.ori_model.language_model(new_text_prompts)
            else:
                ori_text_dict = self.ori_model.language_model(ori_text_prompts)

            ori_text_dict['embedded'] = self.ori_model.text_feat_map(ori_text_dict['embedded'])
            ori_visual_features = self.ori_model.extract_feat(batch_inputs)
            ori_head_inputs_dict = self.forward_ori_model(ori_visual_features, ori_text_dict, batch_data_samples)
            all_layers_ori_cls_scores, all_layers_ori_bbox_preds = \
                self.ori_model.bbox_head(ori_head_inputs_dict['ori_hidden_states'], 
                                        ori_head_inputs_dict['ori_references'], 
                                        ori_head_inputs_dict['ori_memory_text'], 
                                        ori_head_inputs_dict['ori_text_token_mask'])
            ori_head_inputs_dict['all_layers_ori_cls_scores'] = all_layers_ori_cls_scores
            ori_head_inputs_dict['all_layers_ori_bbox_preds'] = all_layers_ori_bbox_preds
            ori_head_inputs_dict['ori_token_positive_maps'] = self.ori_token_positive_maps
        
            if self.distn_cfg.future_class:
                ori_text_len = self.ori_token_positive_maps[len(data_samples.ori_text)][-1] + 1
                ori_text_token_mask =  ori_head_inputs_dict['ori_text_token_mask'][:,:ori_text_len]
                ori_head_inputs_dict['ori_text_token_mask'] = ori_text_token_mask
            else:
                ori_text_token_mask = ori_head_inputs_dict['ori_text_token_mask']

            if self.distn_cfg.label_distn.type == 'topk_pseudo' or self.distn_cfg.label_distn.type == 'threshold_pseudo':
                topk_query, batch_pseudo_instances, batch_all_instances = \
                    self.bbox_head.generate_pseudo_label(all_layers_ori_cls_scores,
                                                        all_layers_ori_bbox_preds,
                                                        ori_text_token_mask,
                                                        text_token_mask, 
                                                        batch_data_samples,
                                                        self.ori_token_positive_maps)    
                
            ori_head_inputs_dict['batch_pseudo_instances'] = batch_pseudo_instances
            ori_head_inputs_dict['batch_all_instances'] = batch_all_instances
            ori_head_inputs_dict['ori_topk_query'] = topk_query

        visual_features = self.extract_feat(batch_inputs)
        
        aux_dict = None
        if self.distn_cfg.query_distn.type == 'seperate_queryinit':
            num_distn_queries = self.distn_cfg.query_distn.num_aux_query
            assert num_distn_queries <= self.distn_cfg.query_distn.num_matching_query
            aux_query = ori_head_inputs_dict['aux_query'].clone()
            aux_reference = ori_head_inputs_dict.pop('aux_reference')
            aux_query = aux_query[:, :num_distn_queries]
            aux_reference = aux_reference[:, :num_distn_queries]
            aux_enc_coord = ori_head_inputs_dict['enc_outputs_coord'].clone() 
            aux_enc_score = ori_head_inputs_dict['enc_outputs_class'].clone()

            aux_dict = dict(aux_query=aux_query, aux_enc_coord=aux_enc_coord, aux_enc_score=aux_enc_score, 
                            aux_reference=aux_reference, batch_pseudo_instances=batch_pseudo_instances)

        new_head_inputs_dict, old_head_inputs_dict = self.forward_transformer(visual_features,  text_dict, 
                                                                              batch_data_samples, aux_dict)    
        # new_head_inputs_dict['text_token_mask_chunked'] = new_text_token_mask_chunked    
        new_head_inputs_dict['token_positive_maps'] = self.token_positive_maps 

        if 'dn_meta' in ori_head_inputs_dict.keys():
            ori_head_inputs_dict.pop('dn_meta')
        # if 'enc_outputs_class' in ori_head_inputs_dict.keys():
        #     ori_head_inputs_dict.pop('enc_outputs_class')
        #     ori_head_inputs_dict.pop('enc_outputs_coord')
        
        losses = self.bbox_head.loss(new_head_inputs_dict, 
                                     old_head_inputs_dict, 
                                     ori_head_inputs_dict, 
                                    batch_data_samples=batch_data_samples)
        losses.update(self._compute_b2_prototype_loss(
            new_head_inputs_dict, batch_data_samples))
        losses.update(self._compute_confusion_pair_margin_loss(
            new_head_inputs_dict,
            ori_head_inputs_dict.get('batch_all_instances', None),
            batch_data_samples))
        losses.update(self._compute_text_contrastive_loss(
            text_dict, batch_data_samples))

        if self.visual_proto_enable:
            vp_epoch = 0.0
            try:
                vp_epoch = float(
                    MessageHub.get_current_instance().get_info('epoch'))
            except Exception:
                vp_epoch = 0.0
            vp_warmup = float(self.prototype_cfg.get(
                'visual_proto_warmup_epochs', 1))
            pseudo_insts = ori_head_inputs_dict.get(
                'batch_pseudo_instances', None)
            if not self.visual_proto_frozen and vp_epoch < vp_warmup:
                self._update_visual_proto_bank(
                    new_head_inputs_dict, batch_data_samples, pseudo_insts)
            elif not self.visual_proto_frozen:
                self._freeze_visual_proto_bank()
            losses.update(self._compute_visual_proto_losses(
                new_head_inputs_dict, batch_data_samples, pseudo_insts))

        # Channel-①: VLM external auxiliary supervision on encoder memory.
        if self.vlm_aux_enable:
            losses.update(self._compute_vlm_aux_loss(
                new_head_inputs_dict, batch_data_samples))

        # Decoder multi-layer aux classification (d3-d5).
        if self.dec_aux_enable:
            losses.update(self._compute_dec_aux_loss(
                new_head_inputs_dict, batch_data_samples))

        # Training monitor
        if self._monitor_path:
            self._training_monitor_log(losses, new_head_inputs_dict, batch_data_samples)

        # === GS-GRMI: gradient scaling ratio + suppression loss ===
        if self.residual_inject is not None and getattr(self, '_gs_grmi_enabled', False):
            detect_keys = [k for k in losses if any(k.startswith(p) for p in
                           ('loss_cls', 'loss_bbox', 'loss_iou',
                            'enc_loss_cls', 'enc_loss_bbox', 'enc_loss_iou'))
                           or (len(k) > 2 and k[0] == 'd' and k[1].isdigit() and '.loss_' in k)]
            L_detect = sum(losses[k] for k in detect_keys if k in losses
                           and isinstance(losses[k], torch.Tensor) and losses[k].requires_grad)
            L_total = sum(v for v in losses.values()
                          if isinstance(v, torch.Tensor) and v.requires_grad)
            with torch.no_grad():
                if isinstance(L_total, torch.Tensor) and L_total.abs() > 1e-8:
                    ratio = (L_detect / L_total).clamp(0.0, 1.0).item()
                else:
                    ratio = 0.0
                self.residual_inject._gs_ratio = ratio
                self._gs_ratio_for_log = ratio
            # R(M) norm suppression
            if hasattr(self.residual_inject, '_cached_residual'):
                rm = self.residual_inject._cached_residual
                losses['loss_rm_supp'] = self._gs_supp_weight * (rm ** 2).mean()

        if hasattr(self, "_grmi_mon_path") and self._grmi_mon_path:
            self._grmi_monitor_log(losses)
        return losses

    # ────────────────── Channel-① VLM aux ──────────────────
    def _get_vlm_cache(self):
        if self._vlm_cache is None:
            self._vlm_cache = load_vlm_cache(self.vlm_aux_path)
        return self._vlm_cache

    def _effective_aux_weight(self):
        """Effective new-class aux weight for the current epoch.

        - 'constant': returns the fixed vlm_aux_weight (default, backward-compatible).
        - 'cosine':   cosine-anneals aux_weight_max -> aux_weight_min over
                      aux_weight_T epochs, to push past the LGQS phase-transition
                      fast early then settle without disturbing old classes.
        """
        if self.vlm_aux_weight_schedule != 'cosine':
            return self.vlm_aux_weight
        import math
        try:
            epoch = float(MessageHub.get_current_instance().get_info('epoch'))
        except Exception:
            epoch = 0.0
        T = max(float(self.vlm_aux_weight_T), 1.0)
        progress = min(max(epoch / T, 0.0), 1.0)
        w = self.vlm_aux_weight_min + 0.5 * (
            self.vlm_aux_weight_max - self.vlm_aux_weight_min) * (1 + math.cos(math.pi * progress))
        return w

    def _vlm_ramp_scale(self, ref_tensor):
        """Warmup (0 for warmup_iters) then linear ramp to 1.0 over ramp_iters."""
        self.vlm_aux_step += 1
        step = self.vlm_aux_step.to(device=ref_tensor.device, dtype=ref_tensor.dtype)
        warmup = float(self.vlm_aux_warmup_iters)
        ramp = float(self.vlm_aux_ramp_iters)
        after_warmup = (step > warmup).to(dtype=ref_tensor.dtype) if warmup > 0 \
            else ref_tensor.new_tensor(1.0)
        if ramp > 0:
            scale = (step - warmup) / ramp
            return scale.clamp(min=0.0, max=1.0) * after_warmup
        return after_warmup

    def _build_vlm_aux_targets_robust(self, batch_data_samples, spatial_shapes,
                                      cache):
        """Augmentation-aware target builder.

        The VLM cache stores boxes in ORIGINAL image coords. After the train
        pipeline (Resize + RandomCrop + RandomFlip) the GT boxes live in
        augmented coords and the two can no longer be matched by naive
        normalization. We instead use each image's recorded GT box (in
        augmented coords) as the spatial anchor and look up the VLM label
        by matching the GT box against the cache in the ORIGINAL coordinate
        space (using scale_factor + flip to undo the augmentation).

        This guarantees that every new-class GT present in the batch gets a
        VLM label whenever that GT exists in the cache, independent of the
        random crop/flip.
        """
        device = spatial_shapes.device
        start = self.vlm_aux_start
        end = self.vlm_aux_end
        num_new = end - start
        iou_thr = self.vlm_aux_iou_thr
        sh = spatial_shapes.cpu().long().tolist()
        level_offsets = []
        total = 0
        for (H, W) in sh:
            level_offsets.append(total)
            total += H * W
        # precompute per-level centers (normalized by level H/W)
        level_centers = []
        for (H, W) in sh:
            gy, gx = torch.meshgrid(
                torch.arange(H, dtype=torch.float32),
                torch.arange(W, dtype=torch.float32), indexing='ij')
            cx = (gx + 0.5) / W
            cy = (gy + 0.5) / H
            level_centers.append(torch.stack([cx.reshape(-1), cy.reshape(-1)], dim=-1))

        labels_list, conf_list, mask_list = [], [], []
        bg_enable = bool(self.vlm_aux_bg_enable)
        ignore_old_gt = bool(self.vlm_aux_bg_ignore_old_gt)
        bg_label = self.vlm_aux_head.bg_label  # = num_new when bg on, else -1
        for sample in batch_data_samples:
            labels = torch.full((total,), -1, dtype=torch.long, device=device)
            conf = torch.zeros(total, dtype=torch.float32, device=device)
            mask = torch.zeros(total, dtype=torch.float32, device=device)
            # Direction A: every position defaults to "background" when bg is on.
            # New-class confirmed positions get overwritten below with 0..num_new-1.
            # Old-class GT positions get overwritten with -1 (IGNORE) so the
            # background branch never suppresses a region the teacher might be
            # detecting an old object in (no oracle on old classes).
            if bg_enable:
                labels.fill_(bg_label)
                mask.fill_(1.0)
                conf.fill_(1.0)
            gt = sample.gt_instances
            meta = sample.metainfo
            if gt is None or len(gt.bboxes) == 0:
                labels_list.append(labels); conf_list.append(conf); mask_list.append(mask)
                continue
            img_id = meta.get('img_id', None)
            vlm_entry = cache.get(str(img_id), None) if img_id is not None else None
            if not isinstance(vlm_entry, dict) or 'new_class_gts' not in vlm_entry:
                vlm_entry = None
            # When bg is OFF we keep the original behaviour: no VLM cache =>
            # no targets. When bg is ON we still want background targets even
            # without a VLM cache (new-class supervision is simply absent).
            if not bg_enable and vlm_entry is None:
                labels_list.append(labels); conf_list.append(conf); mask_list.append(mask)
                continue

            # GT boxes in augmented img_shape coords
            gt_bboxes = gt.bboxes.detach()
            if hasattr(gt_bboxes, 'tensor'):
                gt_bboxes = gt_bboxes.tensor
            gt_bboxes = gt_bboxes.cpu().float()
            gt_labels = gt.labels.detach().cpu().long()
            img_shape = meta.get('img_shape', None)
            if img_shape is None:
                labels_list.append(labels); conf_list.append(conf); mask_list.append(mask)
                continue
            H_img, W_img = float(img_shape[0]), float(img_shape[1])

            # Direction A: IGNORE old-class GT regions (no oracle). When bg is on
            # and ignore_old_gt=True, every encoder position inside an old-class
            # GT box is set to -1 (ignore) so it receives no supervision.
            if bg_enable and ignore_old_gt:
                for g_idx in range(gt_bboxes.shape[0]):
                    lab = int(gt_labels[g_idx].item())
                    if start <= lab < end:
                        continue  # new class, handled by VLM branch below
                    gx1, gy1, gx2, gy2 = gt_bboxes[g_idx].tolist()
                    g_norm = [gx1 / W_img, gy1 / H_img,
                              gx2 / W_img, gy2 / H_img]
                    for lvl in range(len(sh)):
                        centers = level_centers[lvl]
                        inside = ((centers[:, 0] >= g_norm[0]) &
                                  (centers[:, 0] <= g_norm[2]) &
                                  (centers[:, 1] >= g_norm[1]) &
                                  (centers[:, 1] <= g_norm[3]))
                        if not inside.any():
                            continue
                        off = level_offsets[lvl]
                        idx = off + torch.nonzero(
                            inside, as_tuple=False).squeeze(-1)
                        idx_dev = idx.to(device)
                        labels[idx_dev] = -1   # IGNORE
                        mask[idx_dev] = 0.0
                        conf[idx_dev] = 0.0

            # Map each batch GT box back to ORIGINAL coords, then IoU-match
            # against VLM cache entries (which are in original coords).
            sf = meta.get('scale_factor', None)
            sf = self._parse_scale_factor(sf)
            flip = bool(meta.get('flip', False))
            flip_dir = meta.get('flip_direction', 'horizontal')
            oH = float(meta.get('ori_shape', [H_img, W_img])[0])
            oW = float(meta.get('ori_shape', [H_img, W_img])[1])

            # New-class positive supervision requires a VLM cache entry.
            # Skip the VLM loop entirely if none (bg targets already set above).
            vlm_gts = vlm_entry['new_class_gts'] if vlm_entry is not None else None
            if vlm_gts:
                for g_idx in range(gt_bboxes.shape[0]):
                    lab = int(gt_labels[g_idx].item())
                    if not (start <= lab < end):
                        continue
                    gx1, gy1, gx2, gy2 = gt_bboxes[g_idx].tolist()
                    # undo augmentation: augmented -> original image coords.
                    # GT coords are in img_shape space; scale_factor maps
                    # ori->img, so undo by dividing (guarded for RandomCrop: crop
                    # offset is not recorded, so matching degrades but still works
                    # when the GT survives the crop intact).
                    if sf is not None:
                        ox1 = gx1 / sf[0]; ox2 = gx2 / sf[0]
                        oy1 = gy1 / sf[1]; oy2 = gy2 / sf[1]
                    else:
                        ox1, oy1, ox2, oy2 = gx1, gy1, gx2, gy2
                    if flip and flip_dir == 'horizontal':
                        ox1, ox2 = (oW - ox2), (oW - ox1)
                    elif flip and flip_dir == 'vertical':
                        oy1, oy2 = (oH - oy2), (oH - oy1)
                    g_orig = [ox1, oy1, ox2, oy2]

                    best_iou, best_entry = 0.0, None
                    for ve in vlm_gts:
                        vx, vy, vw, vh = ve['bbox']
                        iou = _iou_xyxy(g_orig, [vx, vy, vx + vw, vy + vh])
                        if iou > best_iou:
                            best_iou, best_entry = iou, ve
                    if best_entry is None or best_iou < iou_thr:
                        continue
                    is_neg = not best_entry.get('vlm_correct', False)
                    if is_neg and not (bg_enable or self.vlm_aux_neg_enable):
                        continue
                    local_label = int(best_entry.get('local_label', lab - start))
                    local_label = max(0, min(num_new - 1, local_label))
                    vlm_conf = float(best_entry.get('vlm_confidence', 1.0))

                    if is_neg and self.vlm_aux_neg_enable and bg_label >= 0:
                        assign_label = bg_label
                        assign_conf = float(best_entry.get('vlm_confidence', 1.0))
                    else:
                        if is_neg:
                            continue
                        assign_label = local_label
                        assign_conf = vlm_conf

                    # assign label to encoder positions whose center falls in the
                    # GT box (augmented coords, normalized by img_shape)
                    g_norm = [gx1 / W_img, gy1 / H_img, gx2 / W_img, gy2 / H_img]
                    for lvl in range(len(sh)):
                        centers = level_centers[lvl]
                        inside = ((centers[:, 0] >= g_norm[0]) &
                                  (centers[:, 0] <= g_norm[2]) &
                                  (centers[:, 1] >= g_norm[1]) &
                                  (centers[:, 1] <= g_norm[3]))
                        if not inside.any():
                            continue
                        off = level_offsets[lvl]
                        idx = off + torch.nonzero(inside, as_tuple=False).squeeze(-1)
                        idx_dev = idx.to(device)
                        labels[idx_dev] = assign_label
                        conf[idx_dev] = assign_conf
                        mask[idx_dev] = 1.0
            labels_list.append(labels); conf_list.append(conf); mask_list.append(mask)
        return labels_list, conf_list, mask_list

    @staticmethod
    def _parse_scale_factor(sf):
        """Normalize scale_factor to (sx, sy) floats or None."""
        if sf is None:
            return None
        try:
            import numpy as np
            if hasattr(sf, 'cpu'):
                sf = sf.cpu()
            arr = np.array(sf, dtype=float).reshape(-1)
        except Exception:
            return None
        if arr.size == 0:
            return None
        if arr.size == 1:
            return (float(arr[0]), float(arr[0]))
        if arr.size >= 2:
            return (float(arr[0]), float(arr[1]))
        return None

    def _build_vlm_query_targets(self, batch_data_samples, reference_points,
                                 cache):
        """Experiment C: build PER-QUERY new-class aux targets for the
        decoder-injection variant.

        Mirrors ``_build_vlm_aux_targets_robust`` but assigns labels to the
        ``num_queries`` (900) decoder queries instead of the N_pos encoder
        memory positions. A query is labelled with the local new-class index
        iff its (sigmoided) reference-point center falls inside a
        VLM-confirmed new-class GT box (matched back to original coords the
        same augmentation-aware way as the encoder path). Unmatched queries
        get label=-1 / mask=0 (ignored). The returned tensors therefore have
        length ``num_queries`` so they align 1:1 with
        ``hidden_states[-1]`` (B, num_queries, 256).
        """
        device = reference_points.device
        start = self.vlm_aux_start
        end = self.vlm_aux_end
        num_new = end - start
        iou_thr = self.vlm_aux_iou_thr
        bs, num_queries = reference_points.shape[:2]
        labels_list, conf_list, mask_list = [], [], []
        for b_idx, sample in enumerate(batch_data_samples):
            labels = torch.full((num_queries,), -1, dtype=torch.long,
                                device=device)
            conf = torch.zeros(num_queries, dtype=torch.float32,
                               device=device)
            mask = torch.zeros(num_queries, dtype=torch.float32,
                               device=device)
            gt = sample.gt_instances
            meta = sample.metainfo
            if gt is None or len(gt.bboxes) == 0:
                labels_list.append(labels); conf_list.append(conf)
                mask_list.append(mask)
                continue
            img_id = meta.get('img_id', None)
            vlm_entry = cache.get(str(img_id), None) if img_id is not None \
                else None
            if not isinstance(vlm_entry, dict) or 'new_class_gts' not in vlm_entry:
                labels_list.append(labels); conf_list.append(conf)
                mask_list.append(mask)
                continue
            gt_bboxes = gt.bboxes.detach()
            if hasattr(gt_bboxes, 'tensor'):
                gt_bboxes = gt_bboxes.tensor
            gt_bboxes = gt_bboxes.cpu().float()
            gt_labels = gt.labels.detach().cpu().long()
            img_shape = meta.get('img_shape', None)
            if img_shape is None:
                labels_list.append(labels); conf_list.append(conf)
                mask_list.append(mask)
                continue
            H_img, W_img = float(img_shape[0]), float(img_shape[1])
            sf = meta.get('scale_factor', None)
            sf = self._parse_scale_factor(sf)
            flip = bool(meta.get('flip', False))
            flip_dir = meta.get('flip_direction', 'horizontal')
            oH = float(meta.get('ori_shape', [H_img, W_img])[0])
            oW = float(meta.get('ori_shape', [H_img, W_img])[1])
            vlm_gts = vlm_entry['new_class_gts']
            ref_cx = reference_points[b_idx, :, 0].cpu()
            ref_cy = reference_points[b_idx, :, 1].cpu()
            for g_idx in range(gt_bboxes.shape[0]):
                lab = int(gt_labels[g_idx].item())
                if not (start <= lab < end):
                    continue
                gx1, gy1, gx2, gy2 = gt_bboxes[g_idx].tolist()
                if sf is not None:
                    ox1 = gx1 / sf[0]; ox2 = gx2 / sf[0]
                    oy1 = gy1 / sf[1]; oy2 = gy2 / sf[1]
                else:
                    ox1, oy1, ox2, oy2 = gx1, gy1, gx2, gy2
                if flip and flip_dir == 'horizontal':
                    ox1, ox2 = (oW - ox2), (oW - ox1)
                elif flip and flip_dir == 'vertical':
                    oy1, oy2 = (oH - oy2), (oH - oy1)
                g_orig = [ox1, oy1, ox2, oy2]
                best_iou, best_entry = 0.0, None
                for ve in vlm_gts:
                    vx, vy, vw, vh = ve['bbox']
                    iou = _iou_xyxy(g_orig, [vx, vy, vx + vw, vy + vh])
                    if iou > best_iou:
                        best_iou, best_entry = iou, ve
                if best_entry is None or best_iou < iou_thr:
                    continue
                if not best_entry.get('vlm_correct', False):
                    continue
                local_label = int(best_entry.get('local_label', lab - start))
                local_label = max(0, min(num_new - 1, local_label))
                vlm_conf = float(best_entry.get('vlm_confidence', 1.0))
                g_norm = [gx1 / W_img, gy1 / H_img, gx2 / W_img, gy2 / H_img]
                inside = ((ref_cx >= g_norm[0]) & (ref_cx <= g_norm[2]) &
                          (ref_cy >= g_norm[1]) & (ref_cy <= g_norm[3]))
                idx = torch.nonzero(inside, as_tuple=False).squeeze(-1)
                if idx.numel() == 0:
                    continue
                labels[idx] = local_label
                conf[idx] = vlm_conf
                mask[idx] = 1.0
            labels_list.append(labels); conf_list.append(conf)
            mask_list.append(mask)
        return labels_list, conf_list, mask_list

    def _compute_dec_aux_loss(self, new_head_inputs_dict, batch_data_samples):
        from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_overlaps
        from mmdet.utils import reduce_mean
        zero = next(iter(self.dec_aux_heads.values())).net[0].weight.new_tensor(0.0)
        out = {}
        hidden_states = new_head_inputs_dict.get('hidden_states', None)
        references = new_head_inputs_dict.get('references', None)
        dn_meta = new_head_inputs_dict.get('dn_meta', None)
        if hidden_states is None:
            for lid in self.dec_aux_layers:
                out[f'loss_dec_aux_d{lid}'] = zero.detach()
            out['dec_aux_n_pos'] = zero.detach()
            return out
        cache = self._get_vlm_cache()
        bs = hidden_states.shape[1]
        num_dn = dn_meta.get('num_denoising_queries', 0) if dn_meta else 0
        total_pos = 0
        per_layer_loss = {lid: zero.clone() for lid in self.dec_aux_layers}
        for b in range(bs):
            sample = batch_data_samples[b]
            gt = sample.gt_instances
            gt_labels = gt.labels
            gt_bboxes = gt.bboxes
            if hasattr(gt_bboxes, 'tensor'):
                gt_bboxes = gt_bboxes.tensor
            new_mask = (gt_labels >= self.vlm_aux_start) & (gt_labels < self.vlm_aux_end)
            if not new_mask.any():
                continue
            new_gt_labels = gt_labels[new_mask]
            new_gt_bboxes = gt_bboxes[new_mask]
            new_local_labels = new_gt_labels - self.vlm_aux_start
            img_meta = sample.metainfo
            img_h, img_w = img_meta['img_shape']
            factor = new_gt_bboxes.new_tensor([img_w, img_h, img_w, img_h])
            for lid in self.dec_aux_layers:
                hs_layer = hidden_states[lid][b] if isinstance(hidden_states, (list, tuple)) else hidden_states[lid, b]
                ref_layer = references[lid][b] if isinstance(references, (list, tuple)) else references[lid, b]
                if num_dn > 0:
                    hs_layer = hs_layer[num_dn:]
                    ref_layer = ref_layer[num_dn:]
                pred_bboxes = bbox_cxcywh_to_xyxy(ref_layer) * factor
                pred_bboxes[:, 0::2].clamp_(0, img_w)
                pred_bboxes[:, 1::2].clamp_(0, img_h)
                ious = bbox_overlaps(pred_bboxes, new_gt_bboxes)
                best_iou, best_qi = ious.max(dim=0)
                logits = self.dec_aux_heads[str(lid)](hs_layer.unsqueeze(0).float())
                for g_idx in range(len(new_local_labels)):
                    iou_val = best_iou[g_idx]
                    qi = best_qi[g_idx]
                    local_lab = new_local_labels[g_idx]
                    if iou_val < 0.01:
                        continue
                    w = torch.clamp(iou_val, min=self.dec_aux_iou_floor) if self.dec_aux_iou_weight else 1.0
                    logit = logits[0, qi].unsqueeze(0)
                    target = local_lab.unsqueeze(0).long()
                    ce = F.cross_entropy(logit, target) * w
                    per_layer_loss[lid] = per_layer_loss[lid] + ce
                    total_pos += 1
        n_pos = zero.detach().new_tensor(float(total_pos))
        n_pos_avg = torch.clamp(reduce_mean(n_pos), min=1.0)
        for lid in self.dec_aux_layers:
            out[f'loss_dec_aux_d{lid}'] = self.dec_aux_weight * per_layer_loss[lid] / n_pos_avg
        out['dec_aux_n_pos'] = n_pos.detach()
        return out

    def _training_monitor_log(self, losses, new_head_inputs_dict, batch_data_samples):
        import json as _json
        import torch.distributed as dist
        if not self._monitor_path or self._monitor_step % self._monitor_interval != 0:
            self._monitor_step += 1
            return
        self._monitor_step += 1
        # Only write on rank 0 to avoid DDP file contention
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        try:
            from mmengine.logging import MessageHub
            hub = MessageHub.get_current_instance()
            epoch = hub.get_info('epoch')
            iter_val = hub.get_info('iter')
        except Exception:
            epoch = -1
            iter_val = -1
        record = {'step': self._monitor_step - 1, 'epoch': epoch, 'iter': iter_val}
        for k, v in losses.items():
            if isinstance(v, torch.Tensor):
                record[k] = round(float(v.detach().item()), 6)
        if self.residual_inject is not None:
            record['grmi_gamma'] = round(float(self.residual_inject.gamma.detach().item()), 6)

            # TATRI metrics
            if self.text_residual_inject is not None:
                tri = self.text_residual_inject
                if hasattr(tri, 'gamma'):
                    record['tatri_gamma'] = round(float(tri.gamma.detach().item()), 6)
                if hasattr(tri, 'gate') and self._new_text_token_mask is not None:
                    try:
                        with torch.no_grad():
                            mt_snap = new_head_inputs_dict.get('memory_text', None)
                            if mt_snap is not None:
                                g = tri.gate(mt_snap)
                                msk = self._new_text_token_mask
                                new_g = g[0, msk > 0.5, 0]
                                if len(new_g) > 0:
                                    record['tatri_gate_mean'] = round(float(new_g.mean()), 4)
                                    record['tatri_gate_min'] = round(float(new_g.min()), 4)
                                    record['tatri_gate_max'] = round(float(new_g.max()), 4)
                    except Exception:
                        pass
        if 'hidden_states' in new_head_inputs_dict:
            hs = new_head_inputs_dict['hidden_states']
            for lid in self.dec_aux_layers:
                if lid < hs.shape[0]:
                    record[f'd{lid}_query_norm'] = round(float(hs[lid].detach().norm(dim=-1).mean().item()), 4)
            record['d5_query_norm'] = round(float(hs[-1].detach().norm(dim=-1).mean().item()), 4)
        try:
            with open(self._monitor_path, 'a') as f:
                f.write(_json.dumps(record) + '\n')
        except Exception:
            pass


    def _compute_vlm_aux_loss(self, new_head_inputs_dict, batch_data_samples):
        """Channel-①: auxiliary new-class classification on encoder memory,
        plus optional distillation masking for queries in VLM-confirmed
        new-class regions. Direction A adds an independent background-
        suppression CE over every non-ignored encoder position.

        Experiment C: when ``inject_point == 'decoder'`` the aux head runs on
        the last decoder hidden state (B, num_queries, 256) and targets are
        per-query (VLM-confirmed new-class GT boxes whose reference-point
        center falls inside them), NOT per encoder position. The background
        branch and distillation-mask logic only apply to the encoder path.
        """
        zero = next(self.vlm_aux_head.parameters()).new_tensor(0.0)
        out = dict(
            loss_vlm_aux_cls=zero.detach(),
            loss_vlm_aux_bg=zero.detach(),
            loss_vlm_aux_neg=zero.detach(),
            vlm_aux_pos=zero.detach().new_tensor(0.0),
            vlm_aux_n_confirmed=zero.detach().new_tensor(0.0),
            vlm_aux_n_bg=zero.detach().new_tensor(0.0),
            vlm_aux_n_neg=zero.detach().new_tensor(0.0),
            vlm_aux_n_query_masked=zero.detach().new_tensor(0.0))

        cache = self._get_vlm_cache()
        bg_enable = bool(self.vlm_aux_bg_enable)
        bg_label = self.vlm_aux_head.bg_label
        num_new = self.vlm_aux_end - self.vlm_aux_start

        # NOTE: ``_vlm_ramp_scale`` advances ``vlm_aux_step`` each call, so it
        # must be invoked exactly once per iteration. We therefore call it
        # inside the chosen branch, not in this preamble.

        # ---------------- Experiment C: decoder-injection branch ----------------
        if self.vlm_aux_inject_point == 'decoder':
            ramp = self._vlm_ramp_scale(zero)
            aux_w = self._effective_aux_weight()
            hidden_states = new_head_inputs_dict.get('hidden_states', None)
            matched_ref = new_head_inputs_dict.get(
                'vlm_query_reference', None)
            if hidden_states is None or matched_ref is None:
                return out
            last_hidden = hidden_states[-1]  # (bs, num_queries, 256)
            if last_hidden.dim() == 3 and \
                    last_hidden.shape[1] != matched_ref.shape[1]:
                # Drop any DN prefix if present so queries and refs align.
                last_hidden = last_hidden[:, -matched_ref.shape[1]:, :]
            aux_logits = self.vlm_aux_head(last_hidden.float())
            labels_list, conf_list, mask_list = \
                self._build_vlm_query_targets(
                    batch_data_samples, matched_ref, cache)
            cls_loss_sum = aux_logits.sum() * 0.0
            n_confirmed = 0
            total_pos = 0
            for b_idx, (labels, conf, m) in enumerate(
                    zip(labels_list, conf_list, mask_list)):
                sel = m > 0
                total_pos += int(labels.shape[0])
                if not sel.any():
                    continue
                logits_b = aux_logits[b_idx][sel]
                lab_sel = labels[sel]
                conf_sel = conf[sel]
                new_sel = (lab_sel >= 0) & (lab_sel < num_new)
                if new_sel.any():
                    nl = lab_sel[new_sel]
                    nlg = logits_b[new_sel]
                    ce = F.cross_entropy(nlg, nl, reduction='none')
                    cls_loss_sum = cls_loss_sum + (ce * conf_sel[new_sel]).sum()
                    n_confirmed += int(new_sel.sum().item())
            from mmdet.utils import reduce_mean
            n_conf_t = zero.detach().new_tensor(float(n_confirmed))
            n_conf_red = torch.clamp(reduce_mean(n_conf_t), min=1.0)
            out['loss_vlm_aux_cls'] = ramp * aux_w * \
                (cls_loss_sum / n_conf_red)
            out['vlm_aux_pos'] = zero.detach().new_tensor(float(total_pos))
            out['vlm_aux_n_confirmed'] = zero.detach().new_tensor(
                float(n_confirmed))
            out['vlm_aux_n_bg'] = zero.detach().new_tensor(0.0)
            # distillation mask only meaningful on encoder path; no-op here
            self._vlm_new_query_mask = None
            return out

        # ---------------- Encoder-injection path (Channel-① default) ----------
        ramp = self._vlm_ramp_scale(zero)
        aux_w = self._effective_aux_weight()
        memory = new_head_inputs_dict.get('vlm_enc_memory', None)
        spatial_shapes = new_head_inputs_dict.get('vlm_spatial_shapes', None)
        if memory is None or spatial_shapes is None:
            return out

        labels_list, conf_list, mask_list = self._build_vlm_aux_targets_robust(
            batch_data_samples, spatial_shapes, cache)

        # forward aux head on full encoder memory (gradient -> backbone/neck)
        aux_logits = self.vlm_aux_head(memory.float())  # (bs, N_pos, K[+bg])

        device = memory.device
        total_pos = 0
        n_confirmed = 0          # new-class positions (aux-only path, unchanged)
        n_bg = 0                 # background positions (Direction A path)
        n_neg = 0                # supervisor-rejected new-class positions
        # Initialise the loss sums as a differentiable zero tied to aux_logits so
        # that under DDP every aux-head parameter always receives a gradient,
        # even on iterations with zero matched positions (otherwise DDP raises
        # "Expected to have finished reduction ... parameters not used in loss").
        cls_loss_sum = aux_logits.sum() * 0.0
        bg_loss_sum = aux_logits.sum() * 0.0
        neg_loss_sum = aux_logits.sum() * 0.0
        for b_idx, (labels, conf, m) in enumerate(
                zip(labels_list, conf_list, mask_list)):
            sel = m > 0
            total_pos += int(labels.shape[0])
            if not sel.any():
                continue
            logits_b = aux_logits[b_idx][sel]
            lab_sel = labels[sel]
            conf_sel = conf[sel]
            # new-class positions: label in [0, num_new)
            new_sel = (lab_sel >= 0) & (lab_sel < num_new)
            if new_sel.any():
                nl = lab_sel[new_sel]
                nlg = logits_b[new_sel]
                ce = F.cross_entropy(nlg, nl, reduction='none')
                cls_loss_sum = cls_loss_sum + (ce * conf_sel[new_sel]).sum()
                n_confirmed += int(new_sel.sum().item())
            # neg-mining: bg_label-marked positions from supervisor-reject path.
            # Requires bg OFF (so all bg_label positions are neg, separable).
            if self.vlm_aux_neg_enable and bg_label >= 0:
                neg_sel = (lab_sel == bg_label)
                if neg_sel.any():
                    nlg = logits_b[neg_sel]
                    nconf = conf_sel[neg_sel]
                    nlbl = lab_sel[neg_sel]
                    cap = int(self.vlm_aux_neg_max_per_img)
                    if cap > 0 and int(neg_sel.sum().item()) > cap:
                        nidx = torch.nonzero(neg_sel, as_tuple=False).squeeze(-1)
                        perm = torch.randperm(nidx.shape[0], device=nidx.device)[:cap]
                        keep = nidx[perm]
                        neg_sel = torch.zeros_like(neg_sel)
                        neg_sel[keep] = True
                        nlbl = lab_sel[neg_sel]
                        nlg = logits_b[neg_sel]
                        nconf = conf_sel[neg_sel]
                    nce = F.cross_entropy(nlg, nlbl, reduction='none')
                    neg_loss_sum = neg_loss_sum + (nce * nconf).sum()
                    n_neg += int(neg_sel.sum().item())
            # background positions: label == bg_label (only when bg on)
            if bg_enable and bg_label >= 0:
                bg_sel = (lab_sel == bg_label)
                if bg_sel.any():
                    bl = lab_sel[bg_sel]
                    blg = logits_b[bg_sel]
                    bg_conf = conf_sel[bg_sel]
                    # subsample bg positions so their gradient does not
                    # overwhelm the new-class positive signal.
                    cap = int(self.vlm_aux_bg_max_per_img)
                    if cap > 0 and int(bg_sel.sum().item()) > cap:
                        bg_idx = torch.nonzero(bg_sel, as_tuple=False).squeeze(-1)
                        perm = torch.randperm(
                            bg_idx.shape[0], device=bg_idx.device)[:cap]
                        keep = bg_idx[perm]
                        # rebuild a mask of length len(lab_sel) keeping only cap bg
                        bg_sel = torch.zeros_like(bg_sel)
                        bg_sel[keep] = True
                        bl = lab_sel[bg_sel]
                        blg = logits_b[bg_sel]
                        bg_conf = conf_sel[bg_sel]
                    bgce = F.cross_entropy(blg, bl, reduction='none')
                    bg_loss_sum = bg_loss_sum + (bgce * bg_conf).sum()
                    n_bg += int(bg_sel.sum().item())

        from mmdet.utils import reduce_mean

# ---- Rich-text LGQS augmentation ----
_RICH_TEXT_PATH = '/tmp/rich_text_emb.pt'
import os as _os_rt; import torch as _torch_rt
if _os_rt.path.exists(_RICH_TEXT_PATH):
    _RICH_TEXT_NEW = _torch_rt.load(_RICH_TEXT_PATH, map_location='cpu')
else:
    _RICH_TEXT_NEW = None
# -------------------------------------

        # new-class CE: normalize by #confirmed (aux-only behaviour preserved)
        n_conf_t = zero.detach().new_tensor(float(n_confirmed))
        n_conf_red = torch.clamp(reduce_mean(n_conf_t), min=1.0)
        # effective aux weight (optional cosine schedule over epochs)
        aux_w = self._effective_aux_weight()
        out['loss_vlm_aux_cls'] = ramp * aux_w * \
            (cls_loss_sum / n_conf_red)
        # background CE: separate weight; optional warmup (0 for first
        # bg_warmup_iters), then full weight. Same ramp as cls.
        bg_warmup = float(self.vlm_aux_bg_warmup_iters)
        cur_step = float(self.vlm_aux_step.item())
        if bg_warmup > 0 and cur_step <= bg_warmup:
            bg_ramp = zero.detach().new_tensor(0.0)
        else:
            bg_ramp = ramp
        n_bg_t = zero.detach().new_tensor(float(n_bg))
        n_bg_red = torch.clamp(reduce_mean(n_bg_t), min=1.0)
        out['loss_vlm_aux_bg'] = bg_ramp * self.vlm_aux_bg_weight * \
            (bg_loss_sum / n_bg_red)
        n_neg_t = zero.detach().new_tensor(float(n_neg))
        n_neg_red = torch.clamp(reduce_mean(n_neg_t), min=1.0)
        if self.vlm_aux_neg_enable:
            out['loss_vlm_aux_neg'] = ramp * self.vlm_aux_neg_weight * \
                (neg_loss_sum / n_neg_red)
        else:
            out['loss_vlm_aux_neg'] = zero.detach()
        out['vlm_aux_pos'] = zero.detach().new_tensor(float(total_pos))
        out['vlm_aux_n_confirmed'] = zero.detach().new_tensor(float(n_confirmed))
        out['vlm_aux_n_bg'] = zero.detach().new_tensor(float(n_bg))
        out['vlm_aux_n_neg'] = zero.detach().new_tensor(float(n_neg))

        # Distillation masking: reduce weight on queries inside VLM-confirmed
        # new-class GT. Implemented as a *negative bonus* that cancels part of
        # the existing label/box distillation weight on those queries. Because
        # distillation losses are normalized over queries, we scale down the
        # matching distillation entries via the head's stored mask. We stash
        # the mask on the head so loss_by_feat_old can apply it; if that path
        # is not wired, this remains a no-op stat.
        if self.vlm_distill_mask_enable:
            matched_ref = new_head_inputs_dict.get('vlm_matched_reference', None)
            if matched_ref is not None:
                qmask = build_new_query_mask(
                    batch_data_samples, matched_ref, cache,
                    self.vlm_aux_start, self.vlm_aux_end,
                    iou_thr=self.vlm_aux_iou_thr,
                    max_mask=self.vlm_distill_mask_max)
                self._vlm_new_query_mask = qmask
                out['vlm_aux_n_query_masked'] = qmask.sum().detach()
            else:
                self._vlm_new_query_mask = None
        else:
            self._vlm_new_query_mask = None

        return out


 
