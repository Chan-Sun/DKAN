_base_ = [
    '../../../_base_/datasets/fine_tune_based/base_neu.py',
    '../../../_base_/models/faster_rcnn_r50_caffe_fpn.py',
    '../../../_base_/train_settings.py',
]

data = dict(
    train=dict(classes='BASE_CLASSES_SPLIT3'),
    val=dict(classes='BASE_CLASSES_SPLIT3'),
    test=dict(classes='BASE_CLASSES_SPLIT3'))


# model settings
model = dict(
    roi_head=dict(bbox_head=dict(num_classes=3)))
