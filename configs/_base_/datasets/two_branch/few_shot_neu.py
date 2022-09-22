# dataset settings
img_norm_cfg = dict(
    mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)
multi_scales = (32, 64, 128, 256, 512, 800)
train_multi_pipelines = dict(
    main=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(
            type='Resize',
            img_scale=[(1333, 800)],
            keep_ratio=True,
            multiscale_mode='value'),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ],
    auxiliary=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='CropInstance', context_ratio=1 / 7.),
        dict(
            type='ResizeToMultiScale',
            multi_scales=[(s * 8 / 7., s * 8 / 7.) for s in multi_scales]),
        dict(
            type='MultiImageRandomCrop',
            multi_crop_sizes=[(s, s) for s in multi_scales]),
        dict(type='MultiImageNormalize', **img_norm_cfg),
        dict(type='MultiImageRandomFlip', flip_ratio=0.5),
        dict(type='MultiImagePad', size_divisor=32),
        dict(type='MultiImageFormatBundle'),
        dict(type='MultiImageCollect', keys=['img', 'gt_labels'])
    ])
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
# classes splits are predefined in FewShotCocoDataset
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    auxiliary_samples_per_gpu=2,
    auxiliary_workers_per_gpu=2,
    train=dict(
        type='TwoBranchDataset',
        save_dataset=True,
        reweight_dataset=False,
        dataset=dict(
            type='FewShotNEUDataset',
            save_dataset=True,
            data_root="./dataset/",
            ann_cfg=[
                dict(
                    type='ann_file',
                    ann_file='data_split/trainval.txt')    
            ],
            img_prefix="images",
            multi_pipelines=train_multi_pipelines,
            classes='ALL_CLASSES',
            instance_wise=True,
            dataset_name='main_dataset'),
        auxiliary_dataset=dict(
            copy_from_main_dataset=True,
            instance_wise=True,
            dataset_name='auxiliary_dataset')),
    val=dict(
        type='FewShotNEUDataset',
        data_root="./dataset/",
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file='data_split/test.txt')    
        ],
        img_prefix="images",
        pipeline=test_pipeline,
        classes='ALL_CLASSES'),
    test=dict(
        type='FewShotNEUDataset',
        data_root="./dataset/",
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file='data_split/test.txt')    
        ],
        img_prefix="images",
        pipeline=test_pipeline,
        test_mode=True,
        classes='ALL_CLASSES'))

