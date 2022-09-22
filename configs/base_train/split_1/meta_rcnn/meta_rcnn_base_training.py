_base_ = [
    '../../../_base_/datasets/nway_kshot/base_neu.py',
    '../../../_base_/models/meta-rcnn_r101_c4.py', 
    '../../../_base_/train_settings.py',
]

# dataset setting
num_support_ways = 3
num_support_shots = 10

data = dict(
    train=dict(
        num_support_ways=num_support_ways,
        num_support_shots=num_support_shots,
        dataset=dict(classes='BASE_CLASSES_SPLIT1')),
    val=dict(classes='BASE_CLASSES_SPLIT1'),
    test=dict(classes='BASE_CLASSES_SPLIT1'),
    model_init =dict(classes='BASE_CLASSES_SPLIT1'))

# model settings
model = dict(
    roi_head=dict(bbox_head=dict(num_classes=3, num_meta_classes=3)))

evaluation = dict(interval=500)
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[6000, 8000])
runner = dict(_delete_=True,type='IterBasedRunner', max_iters=10000)
checkpoint_config = dict(interval=100000)
log_config = dict(interval=50)