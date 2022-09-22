_base_ = [
    '../../../_base_/datasets/nway_kshot/few_shot_neu.py',
    '../../../_base_/models/fsdetview_r101_c4.py',
    '../../../_base_/fine_tune_settings.py',
]
# classes splits are predefined in FewShotCocoDataset
# FewShotCocoDefaultDataset predefine ann_cfg for model reproducibility

num_support_ways = 6
num_support_shots = 5
sample_seed=1
split=2
data = dict(
    train=dict(
        save_dataset=True,
        num_support_ways=num_support_ways,
        num_support_shots=num_support_shots,
        num_used_support_shots=num_support_shots,
        dataset=dict(
            type='FewShotNEUDefaultDataset',
            ann_cfg=[dict(method='FSDetView', setting=f'SPLIT{split}_SEED{sample_seed}_{num_support_shots}SHOT')],
            num_novel_shots=num_support_shots,
            num_base_shots=num_support_shots,
            classes=f'ALL_CLASSES_SPLIT{split}',
        )),
    val=dict(classes=f'ALL_CLASSES_SPLIT{split}'),
    test=dict(classes=f'ALL_CLASSES_SPLIT{split}'),
    model_init=dict(classes=f'ALL_CLASSES_SPLIT{split}',num_novel_shots=num_support_shots, num_base_shots=num_support_shots))
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)

# model settings
model = dict(frozen_parameters=[
    'backbone'],
    roi_head = dict(bbox_head = dict(num_classes=num_support_ways, num_meta_classes=num_support_ways))
             )
evaluation = dict(class_splits=['BASE_CLASSES_SPLIT2', 'NOVEL_CLASSES_SPLIT2'])
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[7800])