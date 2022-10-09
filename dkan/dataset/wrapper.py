import copy
import json
import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union,Iterator
from torch.utils.data import DataLoader,ConcatDataset
import numpy as np
from mmdet.datasets.builder import DATASETS

from mmfewshot.detection.datasets.base import BaseFewShotDataset

@DATASETS.register_module()
class MultiStageBaseDataset:
    """A dataset wrapper of TwoBranchDataset.

    Wrapping novel_dataset and base_dataset to a single dataset and thus
    building TwoBranchDataset requires two dataset. The behavior of
    TwoBranchDataset is determined by `mode`. Dataset will return images
    and annotations according to `mode`, e.g. fetching data from
    novel_dataset if `mode` is 'novel'. The default `mode` is 'novel' and
    by using convert function `convert_novel_to_base` the `mode`
    will be converted into 'base'.

    Args:
        novel_dataset (:obj:`BaseFewShotDataset`):
            novel dataset to be wrapped.
        base_dataset (:obj:`BaseFewShotDataset` | None):
            base dataset to be wrapped. If base dataset is None,
            base dataset will copy from novel dataset.
    """

    def __init__(self,
                 novel_dataset: BaseFewShotDataset = None,
                 base_datasets: Optional[BaseFewShotDataset] = None
                 ) -> None:
        self.novel_dataset = novel_dataset
        self.base_dataset = base_datasets
        self.CLASSES = self.novel_dataset.CLASSES
        self.base_classes = self.base_dataset.CLASSES
        # self.dataset = ConcatDataset([self.base_dataset,self.novel_dataset])
        self.novel_idx_map = list(range(len(self.novel_dataset)))
        self.base_idx_map = list(range(len(self.base_dataset)))
        self._novel_len = len(self.novel_idx_map)
        self._base_len = len(self.base_idx_map)
        self.dataset_len = self._base_len+self._novel_len
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def __getitem__(self, idx: int) -> Dict:
        class_mode = idx//self._novel_len
        
        if class_mode ==0:
            idx %= self._novel_len
            idx = self.novel_idx_map[idx]
            return self.novel_dataset.prepare_train_img(idx)
        else:
            idx %= self._base_len
            idx = self.base_idx_map[idx]
            return self.base_dataset.prepare_train_img(idx)

    def __len__(self) -> int: 
        """Length of dataset."""
        return self.dataset_len

    def save_data_infos(self, output_path: str) -> None:
        """Save data infos of novel and base data."""
        self.novel_dataset.save_data_infos(output_path)
        paths = output_path.split('.')
        self.base_dataset.save_data_infos(
            '.'.join(paths[:-1] + ['base', paths[-1]]))

    def _set_group_flag(self) -> None:
        # disable the group sampler, because in few shot setting,
        # one group may only has two or three images.
        self.flag = np.zeros(len(self), dtype=np.uint8)


class MultiStageBaseDataloader:
    """A dataloader wrapper.

    It Create a iterator to iterate two different dataloader simultaneously.
    Note that `TwoBranchDataloader` dose not support `EpochBasedRunner`
    and the length of dataloader is decided by main dataset.

    Args:
        main_data_loader (DataLoader): DataLoader of main dataset.
        auxiliary_data_loader (DataLoader): DataLoader of auxiliary dataset.
    """

    def __init__(self, main_data_loader: DataLoader,
                 auxiliary_data_loader: DataLoader) -> None:
        self.dataset = main_data_loader.dataset
        self.main_data_loader = main_data_loader
        self.auxiliary_data_loader = auxiliary_data_loader

    def __iter__(self) -> Iterator:
        # if infinite sampler is used, this part of code only run once
        self.main_iter = iter(self.main_data_loader)
        self.auxiliary_iter = iter(self.auxiliary_data_loader)
        return self

    def __next__(self) -> Dict:
        # The iterator actually has infinite length. Note that it can NOT
        # be used in `EpochBasedRunner`, because the `EpochBasedRunner` will
        # enumerate the dataloader forever.
        try:
            main_data = next(self.main_iter)
        except StopIteration:
            self.main_iter = iter(self.main_data_loader)
            main_data = next(self.main_iter)
        try:
            auxiliary_data = next(self.auxiliary_iter)
        except StopIteration:
            self.auxiliary_iter = iter(self.auxiliary_data_loader)
            auxiliary_data = next(self.auxiliary_iter)
        return {'main_data': main_data, 'auxiliary_data': auxiliary_data}

    def __len__(self) -> int:
        return len(self.main_data_loader)
