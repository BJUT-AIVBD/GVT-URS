_base_ = [
    '../_base_/models/upernet_cswin_real.py', '../_base_/datasets/UnpavedRoad_binary.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
model = dict(
    backbone=dict(
        type='DCSWin',
        img_size=[800, 1200],
        embed_dim=64,
        depth=[2, 4, 32, 2],
        num_heads=[2, 4, 8, 16],
        split_size=[1, 2, 7, 7],
        dilate=4,
        drop_path_rate=0.4,
        use_chk=False,
    ),
    decode_head=dict(
        in_channels=[64, 128, 256, 512],
        num_classes=2
    ),
    auxiliary_head=dict(
        in_channels=256,
        num_classes=2
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

data = dict(samples_per_gpu=2)
