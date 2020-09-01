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
import numpy as np
import pdb

def twin_transform(ex_rois, gt_rois):
    # ex_rois will be (C, 2)
    # gt_rois will be (C, 2)
    # targets will be (C, 2)
    ex_lengths = ex_rois[:, 1] - ex_rois[:, 0] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_lengths

    gt_lengths = gt_rois[:, 1] - gt_rois[:, 0] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_lengths

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_lengths
    targets_dl = torch.log(gt_lengths / ex_lengths)

    targets = np.stack(
        (targets_dx, targets_dl), 1)
    return targets


def twin_transform_inv(wins, deltas, batch_size):  # (960个锚框,偏移,batch_size)
    # wins will be (batch_size, C, 2)
    # deltas will be (batch_size, C, 2)
    # pred_wins will be (batch_size, C, 2)

    lengths = wins[:, :, 1] - wins[:, :, 0] + 1.0   # 锚框窗口长度(1,960,1)
    ctr_x = wins[:, :, 0] + 0.5 * lengths   # 锚框窗口中心(1,960,1)
    # after the slice operation, dx will be (batch_size, C, 1), Not (batch_size, C)
    dx = deltas[:, :, 0::2]  # 0::2代表从0开始跳2步取出对应元素,此时dx为960个偏移中的第一列，即中心偏移(1,960,1)
    dl = deltas[:, :, 1::2]  # dl为960个偏移中的第二列，即长度偏移(1,960,1)
    # 公式
    pred_ctr_x = dx * lengths.unsqueeze(2) + ctr_x.unsqueeze(2)  # 预测的新锚框中心(1,960,1,1)
    pred_l = torch.exp(dl) * lengths.unsqueeze(2)  # 预测的新锚框长度(1,960,1,1)

    pred_wins = deltas.clone()  # 960个锚框(1,960,2)
    # x1
    pred_wins[:, :, 0::2] = pred_ctr_x - 0.5 * pred_l   # 预测的新锚框起始帧
    # x2
    pred_wins[:, :, 1::2] = pred_ctr_x + 0.5 * pred_l   # 预测的新锚框结束帧

    return pred_wins    # 预测的新锚框(1,960,2)


def clip_twins(wins, video_length, batch_size):
    """
    Clip wins to video boundaries.
    """
    wins.clamp_(0, video_length-1)
    return wins

def twin_transform_batch(ex_rois, gt_rois):
    # ex_rois will be (C, 2) -- anchors are the same for all the batchs.
    #              or (batch_size, C, 2)
    # gt_rois will be (batch_size, C, 2) or (batch_size, C, 3)
    # targes will be (batch_size, C, 2)
    # Note that the outer dimension equals to 1 will be discarded
    if ex_rois.dim() == 2:
        ex_lengths = ex_rois[:, 1] - ex_rois[:, 0] + 1.0
        ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_lengths

        gt_lengths = gt_rois[:, :, 1] - gt_rois[:, :, 0] + 1.0
        gt_ctr_x = gt_rois[:, :, 0] + 0.5 * gt_lengths

        targets_dx = (gt_ctr_x - ex_ctr_x.view(1,-1).expand_as(gt_ctr_x)) / ex_lengths
        targets_dl = torch.log(gt_lengths / ex_lengths.view(1,-1).expand_as(gt_lengths))

    elif ex_rois.dim() == 3:
        ex_lengths = ex_rois[:, :, 1] - ex_rois[:, :, 0] + 1.0
        ex_ctr_x = ex_rois[:, :, 0] + 0.5 * ex_lengths

        gt_lengths = gt_rois[:, :, 1] - gt_rois[:, :, 0] + 1.0
        gt_ctr_x = gt_rois[:, :, 0] + 0.5 * gt_lengths

        targets_dx = (gt_ctr_x - ex_ctr_x) / ex_lengths
        targets_dl = torch.log(gt_lengths / ex_lengths)
    else:
        raise ValueError('ex_roi input dimension is not correct.')

    targets = torch.stack(
        (targets_dx, targets_dl),2)

    return targets

def twins_overlaps(anchors, gt_twins):
    """
    anchors: (N, 2) ndarray of float
    gt_twins: (K, 2) ndarray of float
    overlaps: (N, K) ndarray of overlap between twins and query_twins
    """
    N = anchors.size(0)
    K = gt_twins.size(0)

    gt_twins_len = (gt_twins[:,1] - gt_twins[:,0] + 1).view(1, K)
    gt_len_zero = (gt_twin_len == 1)
    anchors_len = (anchors[:,1] - anchors[:,0] + 1).view(N, 1)
    anchors_len_zero = (anchors_len_zero==1)

    twins = anchors.view(N, 1, 2).expand(N, K, 2)
    query_twins = gt_twins.view(1, K, 2).expand(N, K, 2)

    ilen = torch.min(twins[:,:,1], query_twins[:,:,1]) - torch.max(twins[:,:,0], query_twins[:,:,0]) + 1
    ilen[ilen < 0] = 0

    ua = anchors_len + gt_twins_len - ilen
    overlaps = ilen / ua

    # mask the overlap
    overlaps.mask_fill_(gt_len_zero.view(1, K).expand(N, K), 0)
    overlaps.mask_fill_(anchors_len_zero.view(N, 1).expand(N, K), -1)

    return overlaps


# 可以计算两种来源的重叠度1: 2:
def twins_overlaps_batch(anchors, gt_twins):    # [(876,2), (1, 20, 3)] [(876个锚,两列分别为其起止帧), (gt_twins前两列代表真实起止帧,第三列代表真实标签)]
    """
    anchors: 
        For RPN: (N, 2) ndarray of float or (batch_size, N, 2) ndarray of float
        For TDCNN: (batch_size, N, 3) ndarray of float
    gt_twins: (batch_size, K, 3) ndarray of float, (x1, x2, class_id)
    overlaps: (batch_size, N, K) ndarray of overlap between twins and query_twins
    """
    batch_size = gt_twins.size(0)   # 1

    if anchors.dim() == 2:  # True

        N = anchors.size(0)  # 876
        K = gt_twins.size(1)    # 20
        anchors = anchors.view(1, N, 2).expand(batch_size, N, 2).contiguous()   # (1,876,2)锚,两列分别为其起止帧
        gt_twins = gt_twins[:,:,:2].contiguous()    # (1,20,2)真实窗口,两列分别为其起止帧

        gt_twins_x = (gt_twins[:,:,1] - gt_twins[:,:,0] + 1)    # (1,20)真实窗口长度
        gt_twins_len = gt_twins_x.view(batch_size, 1, K)    # (1,1,20)真实窗口长度(行向量)

        anchors_twins_x = (anchors[:,:,1] - anchors[:,:,0] + 1)     # (1,876)锚窗口长度
        anchors_len = anchors_twins_x.view(batch_size, N, 1)    # (1,876,1)锚窗口长度(行向量)

        gt_len_zero = (gt_twins_x == 1)  # (1,20)元素全为0
        anchors_len_zero = (anchors_twins_x == 1)   # (1,876)元素全为0

        twins = anchors.view(batch_size, N, 1, 2).expand(batch_size, N, K, 2)   # (1, 876, 20, 2)
        query_twins = gt_twins.view(batch_size, 1, K, 2).expand(batch_size, N, K, 2)    # (1, 876, 20, 2)

        ilen = (torch.min(twins[:,:,:,1], query_twins[:,:,:,1]) -   # (1,876,20)公共的窗口长度(IOU的分子)
            torch.max(twins[:,:,:,0], query_twins[:,:,:,0]) + 1)
        ilen[ilen < 0] = 0

        ua = anchors_len + gt_twins_len - ilen  # (1,876,20)IOU的分母
        overlaps = ilen / ua    # (1,876,20)重叠度,相当于fasterrcnn里求交并比(IOU)

        # mask the overlap here.    掩盖重叠
        overlaps.masked_fill_(gt_len_zero.view(batch_size, 1, K).expand(batch_size, N, K), 0)
        overlaps.masked_fill_(anchors_len_zero.view(batch_size, N, 1).expand(batch_size, N, K), -1)

    elif anchors.dim() == 3:
        N = anchors.size(1)
        K = gt_twins.size(1)

        if anchors.size(2) == 2:
            anchors = anchors[:,:,:2].contiguous()
        # (video_idx, x1, x2)
        else:
            anchors = anchors[:,:,1:3].contiguous()

        gt_twins = gt_twins[:,:,:2].contiguous()

        gt_twins_x = (gt_twins[:,:,1] - gt_twins[:,:,0] + 1)
        gt_twins_len = gt_twins_x.view(batch_size, 1, K)

        anchors_twins_x = (anchors[:,:,1] - anchors[:,:,0] + 1)
        anchors_len = anchors_twins_x.view(batch_size, N, 1)

        gt_len_zero = (gt_twins_x == 1)
        anchors_len_zero = (anchors_twins_x == 1)
        twins = anchors.view(batch_size, N, 1, 2).expand(batch_size, N, K, 2)
        query_twins = gt_twins.view(batch_size, 1, K, 2).expand(batch_size, N, K, 2)

        ilen = (torch.min(twins[:,:,:,1], query_twins[:,:,:,1]) -
            torch.max(twins[:,:,:,0], query_twins[:,:,:,0]) + 1)
        ilen[ilen < 0] = 0

        ua = anchors_len + gt_twins_len - ilen
        overlaps = ilen / ua

        # mask the overlap here.
        overlaps.masked_fill_(gt_len_zero.view(batch_size, 1, K).expand(batch_size, N, K), 0)
        overlaps.masked_fill_(anchors_len_zero.view(batch_size, N, 1).expand(batch_size, N, K), -1)
    else:
        raise ValueError('anchors input dimension is not correct.')

    return overlaps  # 重叠度,可能是(1,876,20)
