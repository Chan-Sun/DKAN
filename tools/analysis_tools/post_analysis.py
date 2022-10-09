import mmcv
import os
import torch
from torch.distributed import launch
from mmcv import Config
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel,MMDistributedDataParallel
from mmcv.runner import (get_dist_info, load_checkpoint,
                         wrap_fp16_model)
from mmdet.datasets import replace_ImageToTensor
from mmdet.models import build_detector
from mmdet.utils import setup_multi_processes
import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
os.chdir("../../"+current_path)
sys.path.append("../../"+current_path)
from dkan.dataset.data_builder import build_dataloader, build_dataset,get_copy_dataset_type
from analyze_results import ResultVisualizer,bbox_map_eval
from mmdet.datasets import get_loading_pipeline

def model_output(cfg,ckpt_path,gpu_id=[0]):

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # build the model and load checkpoint
    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.pop('frozen_parameters', None)

    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, ckpt_path, map_location='cpu')
    model = fuse_conv_bn(model)
    # in case the test dataset is concatenated
    cfg.data.test.test_mode = True
    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
    if samples_per_gpu > 1:
        # Replace 'ImageToTensor' to 'DefaultFormatBundle'
        cfg.data.test.pipeline = replace_ImageToTensor(
            cfg.data.test.pipeline)
    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)
    
    if cfg.data.get('model_init', None) is not None:
        cfg.data.model_init.pop('copy_from_train_dataset')
        
        model_init_samples_per_gpu = cfg.data.model_init.pop(
            'samples_per_gpu', 1)
        model_init_workers_per_gpu = cfg.data.model_init.pop(
            'workers_per_gpu', 1)
        
        if cfg.data.model_init.get('ann_cfg', None) is None:
            train_data = build_dataset(cfg.data.train)
            cfg.data.model_init.type = \
                get_copy_dataset_type(cfg.data.model_init.type)
            cfg.data.model_init.ann_cfg = [
                dict(data_infos=train_data.get_support_data_infos())
            ]
        model_init_dataset = build_dataset(cfg.data.model_init)
        # disable dist to make all rank get same data
        model_init_dataloader = build_dataloader(
            model_init_dataset,
            samples_per_gpu=model_init_samples_per_gpu,
            workers_per_gpu=model_init_workers_per_gpu,
            dist=False,
            shuffle=False)


    cfg.gpu_ids = gpu_id
    rank, _ = get_dist_info()

    model.CLASSES = dataset.CLASSES
    
    model = MMDataParallel(model, device_ids=cfg.gpu_ids)
    if cfg.data.get('model_init', None) is not None:
        from mmfewshot.detection.apis import (single_gpu_model_init,
                                                single_gpu_test)
        single_gpu_model_init(model, model_init_dataloader)
        results = single_gpu_test(model, data_loader)
    else:
        model.eval()
        results = []
        prog_bar = mmcv.ProgressBar(len(dataset))
        for i, data in enumerate(data_loader):
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data)
            batch_size = len(result)
            results.extend(result)
            for _ in range(batch_size):
                prog_bar.update()


    eval_kwargs = cfg.get('evaluation', {}).copy()
    # hard-code way to remove EvalHook args
    for key in [
            'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
            'rule', 'dynamic_intervals'
    ]:
        eval_kwargs.pop(key, None)
    eval_kwargs.update(dict(metric="mAP"))
    metric = dataset.evaluate(results, **eval_kwargs)
    print(metric)

    return results

def bbox_visualize(cfg,outputs,show_dir):
            
    cfg.data.test.test_mode = True
    os.makedirs(show_dir,exist_ok=True)
    cfg.data.test.pop('samples_per_gpu', 0)
    if cfg.data.train.type == "NWayKShotDataset":
        cfg.data.test.pipeline = get_loading_pipeline(cfg.data.train.dataset.multi_pipelines.query)
    elif cfg.data.train.type == "TwoBranchDataset":
        cfg.data.test.pipeline = get_loading_pipeline(cfg.data.train.dataset.multi_pipelines.main)      
    else:
        cfg.data.test.pipeline = get_loading_pipeline(cfg.data.train.pipeline) 

    datasets = build_dataset(cfg.data.test)

    result_visualizer = ResultVisualizer(score_thr=0.3)
    

    prog_bar = mmcv.ProgressBar(len(outputs))
    _mAPs = {}
    for i, (result, ) in enumerate(zip(outputs)):
        # self.dataset[i] should not call directly
        # because there is a risk of mismatch
        data_info = datasets.prepare_train_img(i)
        mAP = bbox_map_eval(result, data_info['ann_info'])
        _mAPs[i] = mAP
        # _mAPs[i] = 1
        prog_bar.update()

    # descending select topk image
    _mAPs = list(sorted(_mAPs.items(), key=lambda kv: kv[1]))
    result_visualizer._save_image_gts_results(datasets, outputs, _mAPs, show_dir)
    # result_visualizer._save_img_gt(datasets,show_dir)

def save2json(cfg,results,save_path=None):
    if "data" in cfg.keys():
        cfg.data.test.test_mode = True
        dataset = build_dataset(cfg.data.test)
    else:
        cfg.test_mode = True
        dataset = build_dataset(cfg)
    json_file = dataset._det2json(results)
    if save_path!=None:
        mmcv.dump(json_file,save_path)
    return json_file


if __name__=="__main__":
    # path = "/home/sunchen/Projects/KDTFA/results/ckpt/split_1/5_shot"
    # path = "/home/sunchen/Projects/KDTFA/results/ckpt/split_2/5_shot"
    path = "/home/sunchen/Projects/KDTFA/results/ckpt/split_3/5_shot"
    # method_list = ["mpsr","fsce","tfa"]
    method_list = ["mpsr","fsce","tfa","fsdetview"]
    # method_list = ["fsdetview"]
    
    # show_dir = "/home/sunchen/Projects/KDTFA/results/visualization/SPLIT 2"
    show_dir = "/home/sunchen/Projects/KDTFA/results/visualization/SPLIT 3"

    for method in method_list:
        file_list = [os.path.join(path,method,file_name) for file_name in os.listdir(os.path.join(path,method))]
        
        for file_name in file_list:
            if os.path.splitext(file_name)[-1] == ".py":
                cfg_path=os.path.join(path,method,file_name)
                
        for file_name in file_list:   
            if "best" in file_name:
                ckpt_path=os.path.join(path,method,file_name)
                break
            elif "iter_8000" in file_name:
                ckpt_path=os.path.join(path,method,file_name)
        result_file = show_dir+f"/{method}.pkl"

        cfg = Config.fromfile(cfg_path)
        cfg.data.test = cfg.data.val
        # cfg.merge_from_dict("")

        # outputs = model_output(cfg,ckpt_path,gpu_id=[0])
        # mmcv.dump(outputs, result_file)
        
        outputs = mmcv.load(result_file)
        # method="all_3"
        bbox_visualize(cfg,outputs,show_dir+f"/{method}")
        # from confusion_matrix import calculate_confusion_matrix,plot_confusion_matrix,main
        # import argparse
        # args = argparse.Namespace(
        #     config = cfg_path,
        #     cfg_options = None,
        #     prediction_path = result_file,
        #     nms_iou_thr = None,
        #     show =False,
        #     score_thr = 0,
        #     tp_iou_thr = 0.5,
        #     save_dir = f"/home/user/sun_chen/Projects/KDTFA/results/confusion_matrix/"
        #     )    
        # main(args)

        # if isinstance(cfg.data.test, dict):
        #     cfg.data.test.test_mode = True
        # elif isinstance(cfg.data.test, list):
        #     for ds_cfg in cfg.data.test:
        #         ds_cfg.test_mode = True
        # dataset = build_dataset(cfg.data.test)

        # confusion_matrix = calculate_confusion_matrix(dataset, outputs,
        #                                             0,
        #                                             None,
        #                                             0.5)
        # plot_confusion_matrix(
        #     confusion_matrix,
        #     dataset.CLASSES + ('background', ),
        #     save_dir=args.save_dir,
        #     show=args.show)