_base_ = ["./fsdetview_5shot_fine_tuning.py"]

# classes splits are predefined in FewShotCocoDataset
# FewShotCocoDefaultDataset predefine ann_cfg for model reproducibility

num_support_shots = 30
sample_seed=1
split=2
data = dict(
    train=dict(
        save_dataset=True,
        num_support_shots=num_support_shots,
        num_used_support_shots=num_support_shots,
        dataset=dict(
            ann_cfg=[dict(method='FSDetView', setting=f'SPLIT{split}_SEED{sample_seed}_{num_support_shots}SHOT')],
            num_novel_shots=num_support_shots,
            num_base_shots=num_support_shots,
        )),
    model_init=dict(num_novel_shots=num_support_shots, num_base_shots=num_support_shots))