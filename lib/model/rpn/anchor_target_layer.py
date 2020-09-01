from __future__ import absolute_import
# --------------------------------------------------------
# R-C3D
# Copyright (c) 2017 Boston University
# Licensed under The MIT License [see LICENSE for details]
# Written by Huijuan Xu
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Shiguang Wang
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import numpy.random as npr

from model.utils.config import cfg
from .generate_anchors import generate_anchors
from .twin_transform import clip_twins, twins_overlaps_batch, twin_transform_batch

import pdb

DEBUG = False

try:
    long        # Python 2
except NameError:
    long = int  # Python 3


class _AnchorTargetLayer(nn.Module):
    """
        Assign anchors to ground-truth targets. Produces anchor classification
        labels and bounding-box regression targets.
    """
    def __init__(self, feat_stride, scales):
        super(_AnchorTargetLayer, self).__init__()

        self._feat_stride = feat_stride  # 8
        self._anchors = torch.from_numpy(generate_anchors(base_size=feat_stride, scales=np.array(scales))).float()#generate_anchors(8, [2 4 5 6 8 9 10 12 14 16])
        """_anchors = tensor([[ -4  11]
                             [-12  19]
                             [-16  23]
                             [-20  27]
                             [-28  35]
                             [-32  39]
                             [-36  43]
                             [-44  51]
                             [-52  59]
                             [-60  67]])"""
        self._num_anchors = self._anchors.size(0)   # 10

        # allow boxes to sit over the edge by a small amount
        self._allowed_border = 0  # default is 0

    def forward(self, input):   # input=(rpn_cls_score, gt_twins) [(1,20,96,1,1), (1, 20, 3)] gt_twins前两列代表起止帧,第三列代表标签
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate 9 anchor boxes centered on cell i
        #   apply predicted twin deltas at cell i to each of the 9 anchors
        # filter out-of-video anchors
        # measure GT overlap

        rpn_cls_score = input[0]    # rpn_cls_score (1,20,96,1,1)    二分类
        # GT boxes (batch_size, n, 3), each row of gt box contains (x1, x2, label)
        gt_twins = input[1]  # gt_twins (1, 20, 3)
        # im_info = input[2]
        # num_boxes = input[2]

        batch_size = gt_twins.size(0)   # 1

        # map of shape (..., L, H, W)
        length, height, width = rpn_cls_score.shape[-3:]    # (96,1,1)
        # Enumerate all shifts
        shifts = np.arange(0, length) * self._feat_stride   # shifts = np.arange(0, 96) * 8
        shifts = torch.from_numpy(shifts.astype(float))
        shifts = shifts.contiguous().type_as(rpn_cls_score)    # (96) shifts = np.arange(0, 96) * 8
        # Enumerate all shifted anchors:
        #
        # add A anchors (1, A, 2) to
        # cell K shifts (K, 1, 1) to get
        # shift anchors (K, A, 2)
        # reshape to (K*A, 2) shifted anchors
        A = self._num_anchors  # 10
        K = shifts.shape[0]  # 96

        self._anchors = self._anchors.type_as(rpn_cls_score)  # move to specific context
        all_anchors = self._anchors.view((1, A, 2)) + shifts.view(K, 1, 1)    # (96,10,2)
        all_anchors = all_anchors.view(K * A, 2)    # (960, 2)
        total_anchors = int(K * A)  # 960

        keep = ((all_anchors[:, 0] >= -self._allowed_border) &  # 把all_anchors每个锚框限制在(0,96*8)内
                (all_anchors[:, 1] < long(length * self._feat_stride) + self._allowed_border))

        inds_inside = torch.nonzero(keep).view(-1)  # (876) 帧索引 经过(0,960)这个边界抑制后得到876个锚框,别与768搞混了
        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]   # (876,2)原始锚框,两列分别为起止帧
        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = gt_twins.new(batch_size, inds_inside.size(0)).fill_(-1)    # (1,876)所有元素全部填充-1

        twin_inside_weights = gt_twins.new(batch_size, inds_inside.size(0)).zero_()     # (1,876)所有元素全部填充0
        twin_outside_weights = gt_twins.new(batch_size, inds_inside.size(0)).zero_()    # (1,876)所有元素全部填充0
        # print("anchors {}".format(anchors.shape)) #(876, 2)
        # print("gt_twins {}".format(gt_twins.shape)) #(1, 20, 3)
        # assume anchors(batch_size, N, 2) and gt_wins(batch_size, K, 2), respectively, overlaps will be (batch_size, N, K)
        overlaps = twins_overlaps_batch(anchors, gt_twins)  # 可能是(1,876,20) 计算876个锚(anchor)与20个真实框(gt_twins)的重叠度(IOU)
        # find max_overlaps for each dt: (batch_size, N)
        max_overlaps, argmax_overlaps = torch.max(overlaps, 2)  # (1,876), (1,876)找到第二个维度的最大值,即每个锚(876)与真实框之间的最大重叠度 max_overlaps为每行最大值,argmax_overlaps为对应的索引
        # find max_overlaps for each gt: (batch_size, K)
        gt_max_overlaps, _ = torch.max(overlaps, 1)  # (1,20)找到第一个维度的最大值,即每个真实框(20)与锚之间的最大重叠度 gt_max_overlaps为每列最大值
        if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:  # True
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0#(1,876)此时labels部分为0部分为-1,0表示背景(当876个锚与20个真实框的IOU小于0.3时，就认为是背景)
        gt_max_overlaps[gt_max_overlaps == 0] = 1e-5    # (1,20)
        keep = torch.sum(overlaps.eq(gt_max_overlaps.view(batch_size, 1, -1).expand_as(overlaps)), 2)#(1,876)这个列表里的元素要么是0要么大于0
        if torch.sum(keep) > 0:
            labels[keep > 0] = 1    # (1,876)此时labels部分为0部分为-1部分为1,1表示前景(正样本),-1忽略
        # fg label: above threshold IOU
        labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1  # (1,876)max_overlaps>0.7设置为正样本

        if cfg.TRAIN.RPN_CLOBBER_POSITIVES:  # False
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)   # num_fg=128

        sum_fg = torch.sum((labels == 1).int(), 1)  # 正样本数
        sum_bg = torch.sum((labels == 0).int(), 1)  # 负样本数
        for i in range(batch_size):
            # subsample positive labels if we have too many
            if sum_fg[i] > num_fg:  # 如果正样本数>128
                fg_inds = torch.nonzero(labels[i] == 1).view(-1)    # (1)正样本在labels(1,876)里的索引，个数为>128 <876

                # torch.randperm seems has a bug on multi-gpu setting that cause the segfault.
                # See https://github.com/pytorch/pytorch/issues/1868 for more details.
                # use numpy instead.
                # rand_num = torch.randperm(fg_inds.size(0)).type_as(gt_twins).long()
                rand_num = torch.from_numpy(np.random.permutation(fg_inds.size(0))).type_as(gt_twins).long()
                disable_inds = fg_inds[rand_num[:fg_inds.size(0)-num_fg]]
                labels[i][disable_inds] = -1

#           num_bg = cfg.TRAIN.RPN_BATCHSIZE - sum_fg[i]
            num_bg = cfg.TRAIN.RPN_BATCHSIZE - torch.sum((labels == 1).int(), 1)[i]  # num_bg=(256-正样本数)

            # subsample negative labels if we have too many
            if sum_bg[i] > num_bg:  # 如果负样本数>(256-正样本数) 主要走这条路
                bg_inds = torch.nonzero(labels[i] == 0).view(-1)    # (1)负样本在labels(1,876)里的索引，个数为>(256-正样本数) <876
                # rand_num = torch.randperm(bg_inds.size(0)).type_as(gt_twins).long()
                rand_num = torch.from_numpy(np.random.permutation(bg_inds.size(0))).type_as(gt_twins).long()  # (1)维度和bg_inds一样
                disable_inds = bg_inds[rand_num[:bg_inds.size(0)-num_bg]]   # (1)部分的负样本
                labels[i][disable_inds] = -1    # (1,876) 部分的负样本部分的负样本变成无关样本,负样本减少了

        offset = torch.arange(0, batch_size)*gt_twins.size(1)   # (1)只有一个元素0
        argmax_overlaps = argmax_overlaps + offset.view(batch_size, 1).type_as(argmax_overlaps)#(1,876)每行重叠度最大值索引加上偏移，实际上偏移为0,还是没变化
        twin_targets = _compute_targets_batch(anchors, gt_twins.view(-1,3)[argmax_overlaps.view(-1), :].view(batch_size,-1, 3))   # (tx暂时不懂,将来是分类任务)(1,876,2)
        # use a single value instead of 2 values for easy index.
        twin_inside_weights[labels==1] = cfg.TRAIN.RPN_TWIN_INSIDE_WEIGHTS[0]   # (1,876)权重列表 窗口内正样本的权重为1  (2分类)

        if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:   # True
            num_examples = torch.sum(labels[i] >= 0)    # (1) 常数256(why) 正负样本的和
            positive_weights = 1.0 / num_examples.float()   # 1/265
            negative_weights = 1.0 / num_examples.float()   # 1/265
        else:
            assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                    (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
            positive_weights = cfg.TRAIN.RPN_POSITIVE_WEIGHT
            negative_weights = 1 - positive_weights                    

        twin_outside_weights[labels == 1] = positive_weights    # (1,876)权重列表 窗口外正负样本权重相等,均为1/256    整个列表接近于0
        twin_outside_weights[labels == 0] = negative_weights    # (1,876)权重列表 窗口外正负样本权重相等,均为1/256    整个列表接近于0
        #               (1,876), 960,          (876),      1,           -1
        labels = _unmap(labels, total_anchors, inds_inside, batch_size, fill=-1)#(1,960) 前876个数为处理后的labels(包含1,0,-1),后面几个数填充-1
        #                     (1,876,2),   960,             (876),       1,         0
        twin_targets = _unmap(twin_targets, total_anchors, inds_inside, batch_size, fill=0)#(1,960,2) 前876个数为处理后的twin_targets(1,876,2),后面两列几个数填充0
        #                            (1,876),             960,           (876),       1,           0
        twin_inside_weights = _unmap(twin_inside_weights, total_anchors, inds_inside, batch_size, fill=0)#(1,960) 前876个数为处理后的twin_inside_weights(1,876),后面几个数填充0       整个列表有几个1
        #                            (1,876),               960,           (876),       1,           0
        twin_outside_weights = _unmap(twin_outside_weights, total_anchors, inds_inside, batch_size, fill=0)#(1,960) 前876个数为处理后的twin_outside_weights(1,876),后面几个数填充0    整个列表接近于0

        outputs = []
        #                     1,         96,     1,      1,     10
        labels = labels.view(batch_size, length, height, width, A).permute(0,4,1,2,3).contiguous()  # (1, 10, 96, 1, 1)
        labels = labels.view(batch_size, 1, A * length, height, width)  # (1, 1, 960, 1, 1)  2分类
        outputs.append(labels)  # 0:(1, 1, 960, 1, 1)
        #                                1,           96,    1,      1,     20
        twin_targets = twin_targets.view(batch_size, length, height, width, A*2).permute(0,4,1,2,3).contiguous() # (1,20,96,1,1)  多分类(好像是窗口回归)
        outputs.append(twin_targets)    # 0:(1, 1, 960, 1, 1)  1:(1, 20, 96, 1, 1)

        anchors_count = twin_inside_weights.size(1)     # 960
        twin_inside_weights = twin_inside_weights.view(batch_size,anchors_count,1).expand(batch_size, anchors_count, 2)   # (1,960,2)  窗口回归(内窗口)
        twin_inside_weights = twin_inside_weights.contiguous().view(batch_size, length, height, width, 2*A)\
                            .permute(0,4,1,2,3).contiguous()        # (1,20,96,1,1)  窗口回归(内窗口)
        outputs.append(twin_inside_weights)  # 0:(1, 1, 960, 1, 1)  1:(1, 20, 96, 1, 1)  2:(1,20,96,1,1)

        twin_outside_weights = twin_outside_weights.view(batch_size,anchors_count,1).expand(batch_size, anchors_count, 2)   # (1,960,2)  窗口回归(外窗口)
        twin_outside_weights = twin_outside_weights.contiguous().view(batch_size, length, height, width, 2*A)\
                            .permute(0,4,1,2,3).contiguous()        # (1,20,96,1,1)  窗口回归(外窗口)
        outputs.append(twin_outside_weights)  # 0:(1, 1, 960, 1, 1) 1:(1, 20, 96, 1, 1) 2:(1,20,96,1,1) 3:(1,20,96,1,1)

        return outputs  # 0:(1, 1, 960, 1, 1) 1:(1, 20, 96, 1, 1) 2:(1,20,96,1,1) 3:(1,20,96,1,1)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

#         (1,876), 960, (876), 1,           -1
#         (1,876,2),960,(876), 1,           0
#         (1,876),960,  (876), 1,           0
#         (1,876),960,  (876), 1,           0
def _unmap(data, count, inds, batch_size, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    # for labels, twin_inside_weights and twin_outside_weights
    if data.dim() == 2:
        ret = data.new(batch_size, count).fill_(fill)
        ret[:, inds] = data
    # for twin_targets
    else:
        ret = data.new(batch_size, count, data.size(2)).fill_(fill)
        ret[:, inds,:] = data
    return ret


def _compute_targets_batch(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an video."""

    return twin_transform_batch(ex_rois, gt_rois[:, :, :2])
