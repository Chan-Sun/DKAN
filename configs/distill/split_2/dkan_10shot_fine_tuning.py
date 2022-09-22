_base_ = ["./dkan_5shot_fine_tuning.py"]

shots=10
split=2
seed=1
data = dict(
    train=dict(
        ann_cfg=[dict(method='TFA', setting=f'SPLIT{split}_SEED{seed}_{shots}SHOT')],
        num_novel_shots=shots,
        num_base_shots=shots))
