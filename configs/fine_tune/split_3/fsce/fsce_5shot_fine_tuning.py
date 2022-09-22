_base_ = [
    "../../../_base_/datasets/fine_tune_based/few_shot_neu.py",
    './fsce_r101_fpn.py',
    '../../../_base_/fine_tune_settings.py',
]
# classes splits are predefined in FewShotCocoDataset
# FewShotCocoDefaultDataset predefine ann_cfg for model reproducibility
num_support_ways = 6
num_support_shots = 5
sample_seed=1
split=3
data = dict(
    train=dict(
        type='FewShotNEUDefaultDataset',
        ann_cfg=[dict(method='FSCE', setting=f'SPLIT{split}_SEED{sample_seed}_{num_support_shots}SHOT')],
        num_novel_shots=num_support_shots,
        num_base_shots=num_support_shots,
        classes=f'ALL_CLASSES_SPLIT{split}',
        instance_wise=True),
    val=dict(classes=f'ALL_CLASSES_SPLIT{split}'),
    test=dict(classes=f'ALL_CLASSES_SPLIT{split}'))

model = dict(
    roi_head=dict(bbox_head=dict(num_classes=num_support_ways)),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(pos_iou_thr=0.5, neg_iou_thr=0.5, min_pos_iou=0.5))))

# base model needs to be initialized with following script:
#   tools/detection/misc/initialize_bbox_head.py
# please refer to configs/detection/fsce/README.md for more details.
# load_from = 'path of base training model'

load_from = ('work_dirs/fsce_r101_fpn_coco_base-training/'
             'base_model_random_init_bbox_head.pth')
evaluation = dict(class_splits=['BASE_CLASSES_SPLIT3', 'NOVEL_CLASSES_SPLIT3'])
