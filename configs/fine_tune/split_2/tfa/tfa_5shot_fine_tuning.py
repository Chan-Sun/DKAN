_base_ = [
    "../../../_base_/datasets/fine_tune_based/few_shot_neu.py",
    './tfa_r101_fpn.py',
    '../../../_base_/fine_tune_settings.py',
]
# classes splits are predefined in FewShotCocoDataset
# FewShotCocoDefaultDataset predefine ann_cfg for model reproducibility
num_support_ways = 6
num_support_shots = 5
seed=1
split=2
data = dict(
    train=dict(
        type='FewShotNEUDefaultDataset',
        ann_cfg=[dict(method='TFA', setting=f'SPLIT{split}_SEED{seed}_{num_support_shots}SHOT')],
        num_novel_shots=num_support_shots,
        num_base_shots=num_support_shots,
        classes=f'ALL_CLASSES_SPLIT{split}'),
    val=dict(classes=f'ALL_CLASSES_SPLIT{split}'),
    test=dict(classes=f'ALL_CLASSES_SPLIT{split}'))

model = dict(
    roi_head=dict(bbox_head=dict(num_classes=6)),
    )
# base model needs to be initialized with following script:
#   tools/detection/misc/initialize_bbox_head.py
# please refer to configs/detection/tfa/README.md for more details.
load_from = ("./Weights/base_model_random_init_bbox_head.pth")
evaluation = dict(class_splits=['BASE_CLASSES_SPLIT2', 'NOVEL_CLASSES_SPLIT2'])
