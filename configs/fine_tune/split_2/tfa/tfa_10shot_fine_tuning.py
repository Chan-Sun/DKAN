_base_ = ["./tfa_5shot_fine_tuning.py"]
# classes splits are predefined in FewShotCocoDataset
# FewShotCocoDefaultDataset predefine ann_cfg for model reproducibility
shots=10
data = dict(
    train=dict(
            ann_cfg=[dict(method='TFA', setting=f'SPLIT2_SEED1_{shots}SHOT')],
            num_novel_shots=shots,
            num_base_shots=shots))
