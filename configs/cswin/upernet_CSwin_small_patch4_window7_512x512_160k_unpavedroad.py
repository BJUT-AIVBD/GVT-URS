_base_ = [
    '../_base_/models/upernet_CSwin.py', '../_base_/datasets/UnpavedRoad.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
model = dict(
    backbone=dict(
        patch_size=4,
        embed_dim=64,
        depth=[2, 4, 32, 2],
        split_size=[1, 2, 8, 8],
        num_heads=[2, 4, 8, 16],
        mlp_ratio=4.,
        img_size=512,
        in_chans=3,
        num_classes=3,
    ),

    decode_head=dict(
        loss_decode=[dict(type='DiceLoss', loss_weight=0.25, name='dl'),
                     dict(type='CrossEntropyLoss', loss_weight=0.75, name='ce')],
        in_channels=[96, 192, 384, 768],
        num_classes=3
    ),
    auxiliary_head=dict(
        loss_decode=[dict(type='DiceLoss', loss_weight=0.1, name='dl'),
                     dict(type='CrossEntropyLoss', loss_weight=0.3, name='ce')],
        in_channels=384,
        num_classes=3
    ))

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2)
