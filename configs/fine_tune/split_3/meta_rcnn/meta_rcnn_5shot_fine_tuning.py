_base_ = [
    '../../_base_/datasets/nway_kshot/few_shot_fsneu.py',
    '../../_base_/models/meta-rcnn_r50_c4.py',
    '../../_base_/fine_tune_settings.py',
]
# classes splits are predefined in FewShotCocoDataset
# FewShotCocoDefaultDataset predefine ann_cfg for model reproducibility

num_support_ways = 6
num_support_shots = 5
seed=1
split=3
data = dict(
    train=dict(
        save_dataset=True,
        num_support_ways=num_support_ways,
        num_support_shots=num_support_shots,
        num_used_support_shots=num_support_shots,
        dataset=dict(
            type='FewShotNEUDefaultDataset',
            ann_cfg=[dict(method='MetaRCNN', setting=f'SPLIT{split}_SEED{seed}_{num_support_shots}SHOT')],
            num_novel_shots=num_support_shots,
            num_base_shots=num_support_shots,
            classes=f'ALL_CLASSES_SPLIT{split}',
        )),
    val=dict(classes=f'ALL_CLASSES_SPLIT{split}'),
    test=dict(classes=f'ALL_CLASSES_SPLIT{split}'),
    model_init=dict(classes=f'ALL_CLASSES_SPLIT{split}',num_novel_shots=num_support_shots, num_base_shots=num_support_shots))

# model settings
model = dict(frozen_parameters=[
    'backbone', 'shared_head', 'rpn_head', 'aggregation_layer'],
    roi_head = dict(bbox_head = dict(num_classes=num_support_ways, num_meta_classes=num_support_ways))
             )
