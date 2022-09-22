_base_ = [
    '../../../_base_/datasets/fine_tune_based/few_shot_neu.py',
    '../../../_base_/fine_tune_settings.py',
    "../../tfa_st.py"
]
student_class = 6
teacher_class = 6
shots=5
student = dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint="/home/user/sun_chen/Projects/KDTFA/weights/split_3/tfa_random_init_bbox_head.pth"
        ),
        backbone=dict(frozen_stages=4),
        roi_head=dict(
            bbox_head=dict(
            type='CosineSimBBoxHead',
            num_classes=student_class,
            num_shared_fcs=2,
            scale=20
            )))

teacher = dict(        
        init_cfg=dict(
            type='Pretrained',
            checkpoint="/home/user/sun_chen/Projects/KDTFA/weights/split_3/tfa_random_init_bbox_head.pth"
        ),
        roi_head=dict(
            bbox_head=dict(
                num_classes=teacher_class,
                )))
split=3
seed=1
data = dict(
    train=dict(
        type='FewShotNEUDefaultDataset',
        ann_cfg=[dict(method='TFA', setting=f'SPLIT{split}_SEED{seed}_{shots}SHOT')],
        num_novel_shots=shots,
        num_base_shots=shots,
        classes=f'ALL_CLASSES_SPLIT{split}',
        instance_wise=True),
    val=dict(classes=f'ALL_CLASSES_SPLIT{split}'),
    test=dict(classes=f'ALL_CLASSES_SPLIT{split}'))
evaluation = dict(interval=2000, metric='mAP',save_best='mAP',class_splits=['BASE_CLASSES_SPLIT3', 'NOVEL_CLASSES_SPLIT3'])
runner = dict(type='IterBasedRunner', max_iters=8000)
weight=5
tau=1.0
distill_cfg = [
            dict(
                student_module='roi_head.bbox_head.fc_cls',
                teacher_module='roi_head.bbox_head.fc_cls',
                losses=[
                    dict(
                        type='ICKLDivergence',
                        name='loss_logits',
                        tau=5,
                        loss_weight=0.01,
                        base_class=3)]),
            dict(
            student_module = 'neck.fpn_convs.3.conv',
            teacher_module = 'neck.fpn_convs.3.conv',
            losses=[
                dict(
                    type='ChannelWiseDivergence',
                    name='loss_cw_fpn_3',
                    tau = tau,
                    loss_weight =weight)]),
            dict(
            student_module = 'neck.fpn_convs.2.conv',
            teacher_module = 'neck.fpn_convs.2.conv',
            losses=[
                dict(
                    type='ChannelWiseDivergence',
                    name='loss_cw_fpn_2',
                    tau = tau,
                    loss_weight =weight)]),
            dict(
            student_module = 'neck.fpn_convs.1.conv',
            teacher_module = 'neck.fpn_convs.1.conv',
            losses=[
                dict(
                    type='ChannelWiseDivergence',
                    name='loss_cw_fpn_1',
                    tau = tau,
                    loss_weight =weight)]),
            dict(
            student_module = 'neck.fpn_convs.0.conv',
            teacher_module = 'neck.fpn_convs.0.conv',
            losses=[
                dict(
                    type='ChannelWiseDivergence',
                    name='loss_cw_fpn_0',
                    tau = tau,
                    loss_weight=weight)])
            ]  

# algorithm setting
algorithm = dict(
    architecture=dict(model=student),
    distiller=dict(
            teacher=teacher,
            components=distill_cfg))

find_unused_parameters = True