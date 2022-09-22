_base_ = [
    '../../../_base_/datasets/two_branch/few_shot_neu.py',
    '../../../_base_/models/mpsr_r101_fpn.py',
    '../../../_base_/fine_tune_settings.py',
]
# classes splits are predefined in FewShotCocoDataset
# FewShotCocoDefaultDataset predefine ann_cfg for model reproducibility
num_support_ways = 6
num_support_shots = 5
seed=1
split=1
data = dict(
    train=dict(
        dataset=dict(
            type='FewShotNEUDefaultDataset',
            ann_cfg=[dict(method='MPSR', setting=f'SPLIT{split}_SEED{seed}_{num_support_shots}SHOT')],
            num_novel_shots=num_support_shots,
            num_base_shots=num_support_shots,
            classes=f'ALL_CLASSES_SPLIT{split}')),
    val=dict(classes=f'ALL_CLASSES_SPLIT{split}'),
    test=dict(classes=f'ALL_CLASSES_SPLIT{split}'))
model = dict(
    roi_head=dict(
        bbox_roi_extractor=dict(roi_layer=dict(aligned=False)),
        bbox_head=dict(
            num_classes=num_support_ways,
            init_cfg=[
                dict(
                    type='Normal',
                    override=dict(type='Normal', name='fc_cls', std=0.001))
            ])))
