
"""The data layer used during training to train a Fast R-CNN network.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import torch

from lib.model.utils.config import cfg
from lib.roi_data_layer.minibatch import get_minibatch

import numpy as np
import random
import time
import pdb


class roibatchLoader(data.Dataset):
    def __init__(self, roidb, normalize=None, phase='train'):
        self._roidb = roidb
        self.max_num_box = cfg.MAX_NUM_GT_TWINS     # 20
        self.normalize = normalize
        self.phase = phase

    def __getitem__(self, index):   # index:(0,13711)
        # get the anchor index for current sample index
        item = self._roidb[index]   # item形如{'frames': array([[  0,   0, 768,   1]]), 'fg_name': '/home/tx/Dataset/tx/THUMOS14/val/video_validation_0000934', 'flipped': False, 'durations': array([30.]), 'bg_name': '/home/tx/Dataset/tx/THUMOS14/val/video_validation_0000934', 'max_classes': array([18.]), 'gt_classes': array([18.]), 'wins': array([[235., 265.]]), 'max_overlaps': array([1.])}
        blobs = get_minibatch([item], self.phase)

        data = torch.from_numpy(blobs['data'])  # blobs['data']的形状:[batch_size, 3, 512, 112, 112]
        length, height, width = data.shape[-3:]
        data = data.contiguous().view(3, length, height, width)

        gt_windows = torch.from_numpy(blobs['gt_windows'])#blobs['gt_windows']的形状为(1,3),前两位是时序片段在视频中的“起止帧时刻”(大概是这个意思),第三位是该时序片段的类别
        gt_windows_padding = gt_windows.new(self.max_num_box, gt_windows.size(1)).zero_()   # gt_windows_padding为20行3列的tensor包裹的0矩阵
        ####################################################################################################
        num_gt = min(gt_windows.size(0), self.max_num_box)  # num_gt = 1(此处貌似有误,实际上是(1~20)间的数)
        gt_windows_padding[:num_gt, :] = gt_windows[:num_gt]    # 把该时序片段的起始帧时刻赋值给gt_windows_padding的第一行所有列(3)(此处暂时把gt_windows_padding看成形状为(20,3)的列表)
        
        if self.phase == 'test':
            video_info = ''
            for key, value in item.items():
                video_info = video_info + " {}: {}\n".format(key, value)
            # drop the last "\n"
            video_info = video_info[:-1]
            return data, gt_windows_padding, num_gt, video_info
        else:
            return data, gt_windows_padding, num_gt  # 先看成如下形状:data(3,768,112,112),gt_windows_padding(20,3),num_gt(1)   其中num_gt那个数取值范围为(1~20)

    def __len__(self):
        return len(self._roidb)     # 13712
