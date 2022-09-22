_base_ = ["./meta_rcnn_5shot_fine_tuning.py"]
# classes splits are predefined in FewShotCocoDataset
# FewShotCocoDefaultDataset predefine ann_cfg for model reproducibility

num_support_shots = 10
seed=1
split=3
data = dict(
    train=dict(
        save_dataset=True,
        num_support_shots=num_support_shots,
        num_used_support_shots=num_support_shots,
        dataset=dict(
            ann_cfg=[dict(method='MetaRCNN', setting=f'SPLIT{split}_SEED{seed}_{num_support_shots}SHOT')],
            num_novel_shots=num_support_shots,
            num_base_shots=num_support_shots,
        )),
    model_init=dict(num_novel_shots=num_support_shots, num_base_shots=num_support_shots))
