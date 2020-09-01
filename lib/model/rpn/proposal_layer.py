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
import math
import yaml
from lib.model.utils.config import cfg
from .generate_anchors import generate_anchors
from .twin_transform import twin_transform_inv, clip_twins
from lib.model.nms.nms_wrapper import nms

import pdb

DEBUG = False


class _ProposalLayer(nn.Module):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular twins (called "anchors").
    """
    def __init__(self, feat_stride, scales, out_scores=False):
        super(_ProposalLayer, self).__init__()

        self._feat_stride = feat_stride     # 8
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
        self._out_scores = out_scores
        # TODO: add scale_ratio for video_len ??
        # rois blob: holds R regions of interest, each is a 3-tuple
        # (n, x1, x2) specifying an video batch index n and a
        # rectangle (x1, x2)
        # top[0].reshape(1, 3)
        #
        # # scores blob: holds scores for R regions of interest
        # if len(top) > 1:
        #     top[1].reshape(1, 1, 1, 1)

    def forward(self, input):   # input=(rpn_cls_prob, rpn_twin_pred, cfg_key) [(1,20,96,1,1), (1,20,96,1,1), 'TRAIN']

        # Algorithm:
        #
        # for each (H, W) location i
        #   generate A anchor twins centered on cell i
        #   apply predicted twin deltas at cell i to each of the A anchors
        # clip predicted twins to video
        # remove predicted twins with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)

        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs
        scores = input[0][:, self._num_anchors:, :, :, :]   # rpn_cls_prob (1,10,96,1,1) 貌似只取了前景
        twin_deltas = input[1]  # rpn_twin_pred (1,20,96,1,1)
        cfg_key = input[2]  # 'TRAIN'
        pre_nms_topN  = cfg[cfg_key].RPN_PRE_NMS_TOP_N  # 12000
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N  # 2000
        nms_thresh    = cfg[cfg_key].RPN_NMS_THRESH  # 0.8
        min_size      = cfg[cfg_key].RPN_MIN_SIZE   # 0

        # 1. Generate proposals from twin deltas and shifted anchors
        length, height, width = scores.shape[-3:]   # (96,1,1)

        if DEBUG:
            print('score map size: {}'.format(scores.shape))

        batch_size = twin_deltas.size(0)    # 1

        # Enumerate all shifts
        shifts = np.arange(0, length) * self._feat_stride   # shifts = np.arange(0, 96) * 8
        shifts = torch.from_numpy(shifts.astype(float))
        shifts = shifts.contiguous().type_as(scores)    # shifts = np.arange(0, 96) * 8
        # print(shifts.shape)     # torch.Size([96])
        # Enumerate all shifted anchors:
        #
        # add A anchors (1, A, 2) to
        # cell K shifts (K, 1, 1) to get
        # shift anchors (K, A, 2)
        # reshape to (1, K*A, 2) shifted anchors
        # expand to (batch_size, K*A, 2)
        A = self._num_anchors  # 10
        K = shifts.shape[0]  # 96
        self._anchors = self._anchors.type_as(scores)
        anchors = self._anchors.view(1, A, 2) + shifts.view(K, 1, 1)    # (96,10,2)
        anchors = anchors.view(1, K * A, 2).expand(batch_size, K * A, 2)    # (1, 960, 2)
        # Transpose and reshape predicted twin transformations to get them
        # into the same order as the anchors:
        #
        # twin deltas will be (batch_size, 2 * A, L, H, W) format
        # transpose to (batch_size, L, H, W, 2 * A)
        # reshape to (batch_size, L * H * W * A, 2) where rows are ordered by (l, h, w, a)
        # in slowest to fastest order
        twin_deltas = twin_deltas.permute(0, 2, 3, 4, 1).contiguous()   # rpn_twin_pred (1,96,1,1,20)
        twin_deltas = twin_deltas.view(batch_size, -1, 2)   # rpn_twin_pred (1,96*1*1*10,2)
        # Same story for the scores:
        #
        # scores are (batch_size, A, L, H, W) format
        # transpose to (batch_size, L, H, W, A)
        # reshape to (batch_size, L * H * W * A) where rows are ordered by (l, h, w, a)
        scores = scores.permute(0, 2, 3, 4, 1).contiguous()  # rpn_cls_prob (1,96,1,1,10)
        scores = scores.view(batch_size, -1)   # rpn_cls_prob (1,96*1*1*10)

        # Convert anchors into proposals via twin transformations
        #                              (1,960,2),(1,960,2),1
        proposals = twin_transform_inv(anchors, twin_deltas, batch_size)#(960个原始锚框,偏移,batch_size)(原始锚框第一列表示起始帧,第二列表示结束帧)(偏移第一列表示中心偏移,第二列表示长度偏移)
        # 预测的新锚框(1,960,2)第一列表示预测起始帧,第二列表示预测结束帧              #     rpn网络里的回归

        # 2. clip predicted wins to video
        #                      (1,960,2), 96*8,                       1
        proposals = clip_twins(proposals, length * self._feat_stride, batch_size)   # 把proposals值范围抑制在(0,96*8)之间,其实没起作用
        # 3. remove predicted twins with either length < threshold
        # assign the score to 0 if it's non keep.
        no_keep = self._filter_twins_reverse(proposals, min_size)   # 去除小于min_size的窗口,但实际min_size=0,所以此句无用
        scores[no_keep] = 0  # scores是前景(1, 960)   每个值对应每帧图片是前景的概率

        scores_keep = scores    # 二分类(1,960)前景的概率
        proposals_keep = proposals  # 回归(1,960,2)预测起始帧 预测结束帧
        # sorted in descending order
        _, order = torch.sort(scores_keep, 1, True)  # (1,960)order是(0~959(scores里的下标))构成的列表,表示scores里的概率按从大到小排列
 
        # print ("scores_keep {}".format(scores_keep.shape))
        # print ("proposals_keep {}".format(proposals_keep.shape))
        # print ("order {}".format(order.shape))

        output = scores.new(batch_size, post_nms_topN, 3).zero_()   # (1,2000,3)全0的tensor类型列表

        if self._out_scores:    # False
            output_score = scores.new(batch_size, post_nms_topN, 2).zero_()

        for i in range(batch_size):

            proposals_single = proposals_keep[i]    # (960,2) 预测起始帧 预测结束帧
            scores_single = scores_keep[i]  # (960)前景的概率

            # 4. sort all (proposal, score) pairs by score from highest to lowest
            # 5. take top pre_nms_topN (e.g. 6000)
            order_single = order[i]  # (960)scores里的下标,scores里的概率按从大到小排列

            if pre_nms_topN > 0 and pre_nms_topN < scores_keep.numel():  # False
                order_single = order_single[:pre_nms_topN]

            proposals_single = proposals_single[order_single, :]    #(960,2)把proposals里的960个特征按其是前景概率的大小从大到小排列,后面两列仍然是预测起始和结束帧
            scores_single = scores_single[order_single].view(-1, 1)#(960,1)把proposals里的960个特征按其是前景概率的大小从大到小排列,后面一列是对应的从大到小的概率

            # 6. apply nms (e.g. threshold = 0.7)
            # 7. take after_nms_topN (e.g. 300)
            # 8. return the top proposals (-> RoIs top)

            keep_idx_i = nms(torch.cat((proposals_single, scores_single), 1), nms_thresh, force_cpu=not cfg.USE_GPU_NMS)#scores_single并在proposals_single的右侧形成(960,3),然后经过nms函数
            # keep_idx_i(<960, 1),取出scores_single>0.8的行作为前景,那列为>0.8时960个特征对应的索引
            keep_idx_i = keep_idx_i.long().view(-1)
            # keep_idx_i(<960)
            if post_nms_topN > 0:
                keep_idx_i = keep_idx_i[:post_nms_topN]  # 没啥变化，post_nms_topN=2000,而keep_idx_i只有<960个数
            # keep_idx_i(<960)
            proposals_single = proposals_single[keep_idx_i, :]  # (<960,2)取出经过nms抑制后的proposals_single,后面两列是可能的前景的起止帧
            scores_single = scores_single[keep_idx_i, :]    # (<960,1)取出经过nms抑制后的scores_single,后面一列是可能的前景概率

            # padding 0 at the end.
            num_proposal = proposals_single.size(0)  # <960个,经过nms抑制后proposal的个数
            # print ("num_proposal: ", num_proposal)
            output[i,:,0] = i   # (1,2000,3)仍然全0
            output[i,:num_proposal,1:] = proposals_single#(1,2000,3)[其中(1,<960,3)<960的部分是前景,第一列全0存放未来的21类标签,后两列是可能的前景的起止帧;(960,2000)的部分全0,可能代表背景]
            if self._out_scores:    # False
                output_score[i, :, 0] = i
                output_score[i, :num_proposal, 1] = scores_single

        if self._out_scores:    # False
            return output, output_score
        else:
            return output#(1,2000,3)[其中(1,<960,3)<960的部分是前景,第一列全0存放未来的21类标签,后两列是可能的前景的起止帧;(960,2000)的部分全0,可能代表背景]

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def _filter_twins_reverse(self, twins, min_size):
        """get the keep index of all twins with length smaller than min_size. 
        twins will be (batch_size, C, 2), keep will be (batch_size, C)"""
        ls = twins[:, :, 1] - twins[:, :, 0] + 1
        no_keep = (ls < min_size)
        return no_keep
