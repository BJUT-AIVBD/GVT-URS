# dataset settings
dataset_type = 'UnpavedRoadBinary'
data_root = '/media/lws/Store/Dataset/unpaved_road_dataset/ALL_to_1200_800/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'), # reduce_zero_label=True
    dict(type='Resize', img_scale=(1216, 768), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='EqHist'),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),

]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1216, 768),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            # dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            # dict(type='EqHist'),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),

        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        # img_dir='train/image_multi_1200_800src_otheraug_V2',
        # ann_dir='train/mask_multi_1200_800src_otheraug_V2_adetype',
        # img_dir='train/image_V5',
        # ann_dir='train/mask_V5_gray',
        # img_dir='train/image_V6',
        # ann_dir='train/mask_V6_pseudo',
        # img_dir='train/image_V7',
        # ann_dir='train/mask_V7_pseudo',
        # img_dir='train/image_V8',
        # ann_dir='train/mask_V8_pseudo',
        # img_dir='train/image_V9',
        # ann_dir='train/mask_V9_pseudo',
        # img_dir='train/image_V10',
        # ann_dir='train/mask_V11_pseudo',
        img_dir='train/image_V12',
        ann_dir='train/mask_V12_pseudo',
        pipeline=train_pipeline),
#split='splits/train.txt',
    val=dict(
        type=dataset_type,
        data_root=data_root,
        # img_dir='test/image_V5',
        # ann_dir='test/mask_V5_gray',
        # img_dir='test/image_multi_src',
        # ann_dir='test/mask_multi_src_V2',
        # img_dir='test/image_V9',
        # ann_dir='test/mask_V9_pseudo',
        # img_dir='test/image_V10/src',
        # ann_dir='test/mask_V11_pseudo/src',
        img_dir='test/image_V12/src',
        ann_dir='test/mask_V12_pseudo/src',
        pipeline=test_pipeline),
#split='splits/val.txt',
    test=dict(
        type=dataset_type,
        data_root=data_root,
        # img_dir='test/image_multi_withoutaug_V2',
        # ann_dir='test/mask_multi_withoutaug_V2',
        # img_dir='test/image_V5',
        # ann_dir='test/mask_V5_gray',
        # img_dir='test/image_eqHist',
        # ann_dir='test/mask_V5_gray',
        # img_dir='test/image_2400_1600',
        # ann_dir='test/mask_V5_gray',
        # img_dir='test/image_multi_src',
        # ann_dir='test/mask_multi_src_V2',
        # img_dir='test/image_src_split_3000_2000',
        # ann_dir='test/mask_src_split_3000_2000',
        # img_dir='test/image_V8',
        # ann_dir='test/mask_V8',
        # img_dir='test/image_V9',
        # ann_dir='test/mask_V9_pseudo',
        # img_dir='test/image_V10/src',
        # ann_dir='test/mask_V10_pseudo/src',
        # img_dir='test/image_V10/aug',
        # ann_dir='test/mask_V11_pseudo/aug',
        img_dir='test/image_V12/aug',
        ann_dir='test/mask_V12_pseudo/aug',
        pipeline=test_pipeline))

#split='splits/test.txt',
