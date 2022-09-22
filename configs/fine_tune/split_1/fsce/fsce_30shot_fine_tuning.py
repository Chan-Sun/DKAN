_base_ = "./fsce_5shot_fine_tuning.py"
# classes splits are predefined in FewShotCocoDataset
# FewShotCocoDefaultDataset predefine ann_cfg for model reproducibility
num_support_shots = 10
sample_seed=1
split=1
data = dict(
    train=dict(
        ann_cfg=[dict(method='FSCE', setting=f'SPLIT{split}_SEED{sample_seed}_{num_support_shots}SHOT')],
        num_novel_shots=num_support_shots,
        num_base_shots=num_support_shots))