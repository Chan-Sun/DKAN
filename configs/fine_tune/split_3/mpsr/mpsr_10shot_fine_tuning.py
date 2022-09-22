_base_ = ["./mpsr_5shot_fine_tuning.py"]
# classes splits are predefined in FewShotCocoDataset
# FewShotCocoDefaultDataset predefine ann_cfg for model reproducibility
num_support_shots = 10
seed=1
split=3
data = dict(
    train=dict(
        dataset=dict(
            ann_cfg=[dict(method='MPSR', setting=f'SPLIT{split}_SEED{seed}_{num_support_shots}SHOT')],
            num_novel_shots=num_support_shots,
            num_base_shots=num_support_shots)))