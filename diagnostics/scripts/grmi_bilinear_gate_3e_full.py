# GRMI Bilinear + Old-Class Gate, 3 epochs, FULLY SELF-CONTAINED for A100.
# No _base_ dependencies — all settings inline.

# ── Model ──
model = dict(
    type='GroundingDINO_inc_gcd',
    num_queries=900,
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type='DetDataPreprocessor', mean=[123.675,116.28,103.53],
        std=[58.395,57.12,57.375], bgr_to_rgb=True, pad_mask=False),
    backbone=dict(
        type='SwinTransformer', embed_dims=96, depths=[2,2,6,2],
        num_heads=[3,6,12,24], window_size=7, mlp_ratio=4,
        qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0,
        drop_path_rate=0.2, patch_norm=True, out_indices=(1,2,3),
        with_cp=True, convert_weights=False, frozen_stages=-1),
    neck=dict(
        type='ChannelMapper', in_channels=[192,384,768], out_channels=256,
        num_outs=4, kernel_size=1, norm_cfg=dict(type='GN',num_groups=32),
        act_cfg=None, bias=True),
    encoder=dict(
        num_layers=6, num_cp=6,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4, dropout=0.0),
            ffn_cfg=dict(embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)),
        text_layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=4, dropout=0.0),
            ffn_cfg=dict(embed_dims=256, feedforward_channels=1024, ffn_drop=0.0)),
        fusion_layer_cfg=dict(v_dim=256, l_dim=256, embed_dim=1024, num_heads=4, init_values=1e-4)),
    decoder=dict(
        num_layers=6, return_intermediate=True, post_norm_cfg=None,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            cross_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            cross_attn_text_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            ffn_cfg=dict(embed_dims=256, feedforward_channels=2048, ffn_drop=0.0))),
    positional_encoding=dict(num_feats=128, normalize=True, offset=0.0, temperature=20),
    language_model=dict(
        type='BertModel', name='bert-base-uncased', max_tokens=256,
        pad_to_max=False, use_sub_sentence_represent=True,
        special_tokens_list=['[CLS]','[SEP]','.','?'], add_pooling_layer=False),
    bbox_head=dict(
        type='GroundingDINOHead_inc_gcd',
        sync_cls_avg_factor=True, setting='full_text', trunc_class=[70,80],
        contrastive_cfg=dict(max_text_len=256, bias=None, log_scale=None),
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0),
        distn_cfg=dict(
            label_distn=dict(type='threshold_pseudo', label_iou_th=0.7, sigma=0.4,
                             loss_ld=dict(type='KnowledgeDistillationKLDivLoss', T=100, loss_weight=1.0),
                             mode='hardlabel'),
            feat_distn=dict(type='inter-class', subtype='opt1',
                            img_loss=dict(type='L2Loss', reduction='mean', loss_weight=3.0),
                            text_loss=dict(type='L2Loss', reduction='mean', loss_weight=5.0)),
            query_distn=dict(type='seperate_queryinit', num_aux_query=900, num_matching_query=900),
            ori_config_file='configs/gdino_inc/70+10/gdino_inc_70+10_0-69_scratch_coco.py',
            future_class=False)),
    train_cfg=dict(assigner=dict(
        type='HungarianAssigner',
        match_costs=[dict(type='BinaryFocalLossCost',weight=2.0),
                     dict(type='BBoxL1Cost',weight=5.0,box_format='xywh'),
                     dict(type='IoUCost',iouloss='giou',weight=2.0)])),
    test_cfg=dict(max_per_img=300),
    frozen_cfg=dict(backbone_frozen=False, encoder_frozen=False, decoder_frozen=False,
                    head_frozen=False, language_model_frozen=True, neck_frozen=False),
    dn_cfg=None,
    residual_inject_cfg=dict(
        enable=True, mode='bilinear', bilinear_bottleneck=64,
        gamma_init=0.5, freeze_gamma=False, use_old_gate=True,
        monitor=dict(path='/root/autodl-tmp/grmi_bilinear_gate_3e_monitor.jsonl', interval=250)),
)

# ── Data ──
dataset_type = 'CocoIncDataset'
data_root = '/root/autodl-tmp/gcd_work/data/coco/'
backend_args = None
lang_model_name = 'bert-base-uncased'

train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(prob=0.5, type='RandomFlip'),
    dict(transforms=[[
        dict(keep_ratio=True, scales=[(480,1333),(512,1333),(544,1333),(576,1333),(608,1333),(640,1333),(672,1333),(704,1333),(736,1333),(768,1333),(800,1333)], type='RandomChoiceResize')],
        [dict(keep_ratio=True, scales=[(400,4200),(500,4200),(600,4200)], type='RandomChoiceResize'),
         dict(allow_negative_crop=True, crop_size=(384,600), crop_type='absolute_range', type='RandomCrop'),
         dict(keep_ratio=True, scales=[(480,1333),(512,1333),(544,1333),(576,1333),(608,1333),(640,1333),(672,1333),(704,1333),(736,1333),(768,1333),(800,1333)], type='RandomChoiceResize')]], type='RandomChoice'),
    dict(meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor','flip','flip_direction','text','ori_text','custom_entities'), type='PackDetInputs')]

test_pipeline = [
    dict(backend_args=None, imdecode_backend='pillow', type='LoadImageFromFile'),
    dict(backend='pillow', keep_ratio=True, scale=(800,1333), type='FixScaleResize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor','text','ori_text','custom_entities'), type='PackDetInputs')]

train_dataloader = dict(
    batch_size=4, num_workers=4, persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(type=dataset_type, data_root=data_root, return_classes=True,
        setting='full_text', start=70, end=80,
        ann_file='annotations/70+10/instances_train2017_70-79.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=4, num_workers=2, persistent_workers=True,
    drop_last=False, sampler=dict(shuffle=False, type='DefaultSampler'),
    dataset=dict(type=dataset_type, data_root=data_root, return_classes=True,
        setting='full_text', start=70, end=80, test_mode=True,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'), pipeline=test_pipeline))

val_evaluator = dict(type='IncCocoMetric', ann_file=data_root+'annotations/instances_val2017.json', metric='bbox')

# ── Training config ──
max_epochs = 3
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

load_from = '/root/autodl-tmp/gcd_work/work_dirs/gdino_inc_70+10_0-69_scratch_coco/epoch_12.pth'
start = 70; end = 80
resume = False
launcher = 'none'
work_dir = '/root/autodl-tmp/gcd_work/work_dirs/grmi_bilinear_gate_3e'
log_level = 'INFO'

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=20),
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='coco/bbox_mAP', rule='greater'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'))
custom_hooks = [dict(type='Increment_distn_hook')]
log_processor = dict(type='LogProcessor', window_size=20, by_epoch=True)
env_cfg = dict(cudnn_benchmark=False, mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0), dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='DetLocalVisualizer', vis_backends=[dict(type='LocalVisBackend')], name='visualizer')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=5e-5, weight_decay=0.0001),
    paramwise_cfg=dict(custom_keys=dict(backbone=dict(lr_mult=0.1), language_model=dict(lr_mult=0.1))),
    clip_grad=dict(max_norm=0.1, norm_type=2))
param_scheduler = [dict(type='MultiStepLR', begin=0, end=max_epochs, by_epoch=True, milestones=[], gamma=0.1)]
auto_scale_lr = dict(base_batch_size=32, enable=False)
