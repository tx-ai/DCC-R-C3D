from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from lib.model.utils.config import cfg
from .proposal_layer import _ProposalLayer
from .anchor_target_layer import _AnchorTargetLayer
from lib.model.utils.net_utils import _smooth_l1_loss, mask_rpn_losses

import numpy as np
import math
import pdb
import time

DEBUG = False


class _RPN(nn.Module):
    """ region proposal network """
    def __init__(self, din, out_scores=False):
        super(_RPN, self).__init__()
        
        self.din = din  # get depth of input feature map, e.g., 512
        self.anchor_scales = cfg.ANCHOR_SCALES  # 程序中设置成了[2,4,5,6,8,9,10,12,14,16]
        self.feat_stride = cfg.FEAT_STRIDE[0]   # 8
        self.out_scores = out_scores    # False
        self.mask_upsample_rate = 1

        # define the convrelu layers processing input feature map
        self.RPN_Conv1 = nn.Conv3d(self.din, 512, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=True) # 若输入是(2, 512, 96, 7, 7) tx(1)
        self.RPN_Conv2 = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=True)#则输出是(2, 512, 96, 2, 2)    tx(1)
        # self.RPN_Conv2 = nn.Conv3d(self.din, 512, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 0, 0), bias=True)  # 则输出是(2, 512, 96, 2, 2) tx(1)
        self.RPN_output_pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))    # 则输出是(2, 512, 96, 1, 1)符合论文要求

        self.Conv_up = nn.ConvTranspose3d(512, 512, kernel_size=(4, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))   # tx(3)

        # define bg/fg classifcation score layer
        self.nc_score_out = len(self.anchor_scales) * 2  # 2(bg/fg) * 10 (anchors)  两个2含义不同
        self.RPN_cls_score = nn.Conv3d(512, self.nc_score_out, 1, 1, 0)     # nn.Conv3d(512, 20, 1, 1, 0)

        # define anchor twin offset prediction layer
        self.nc_twin_out = len(self.anchor_scales) * 2  # 2(coords) * 10 (anchors)  两个2含义不同
        self.RPN_twin_pred = nn.Conv3d(512, self.nc_twin_out, 1, 1, 0)  # nn.Conv3d(512, 20, 1, 1, 0)

        # define proposal layer
        self.RPN_proposal = _ProposalLayer(self.feat_stride, self.anchor_scales, self.out_scores)#(tx)(1,2000,3)[其中(1,<960,3)<960的部分是前景,第一列全0存放未来的21类标签,后两列是可能的前景的起止帧;(960,2000)的部分全0,可能代表背景]

        # define anchor target layer
        self.RPN_anchor_target = _AnchorTargetLayer(self.feat_stride, self.anchor_scales)#(tx)0:(1, 1, 960, 1, 1) 1:(1, 20, 96, 1, 1) 2:(1,20,96,1,1) 3:(1,20,96,1,1) 2分类 多分类好像是窗口回归 窗口回归(内窗口) 窗口回归(外窗口)

        self.rpn_loss_cls = 0
        self.rpn_loss_twin = 0
        self.rpn_loss_mask = 0

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3],
            input_shape[4]
        )
        return x

    def forward(self, base_feat, gt_twins):  # [(1, 512, 96, 7, 7), (1, 20, 3)] gt_twins前两列代表起止帧,第三列代表标签

        batch_size = base_feat.size(0)

        # return feature map after convrelu layer
        rpn_conv1 = F.relu(self.RPN_Conv1(base_feat), inplace=True)   # tx(1)
        rpn_conv2 = F.relu(self.RPN_Conv2(rpn_conv1), inplace=True)    # tx(1)
        rpn_output_pool = self.RPN_output_pool(rpn_conv2)  # (1,512,96,1,1) 此处经历了第二个网络:Temporal Proposal Subnet(类似RPN)

        rpn_output_pool = F.relu(self.Conv_up(rpn_output_pool), inplace=True)  # tx(3)  (1,512,128,1,1)

        # get rpn classification score
        rpn_cls_score = self.RPN_cls_score(rpn_output_pool)  # (1,512,96,1,1)->(1,20,96,1,1) 二分类
        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)  # (1,20,96,1,1)->(1,2,960,1,1)
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, dim=1)  # 按行SoftMax,行和为1(可以理解成把像素归一化，且每行为一个batch) (1,2,960,1,1)
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)#(1,2,960,1,1)->(1,20,96,1,1)

        # get rpn offsets to the anchor twins
        rpn_twin_pred = self.RPN_twin_pred(rpn_output_pool)  # (1,512,96,1,1)->(1,20,96,1,1) 回归(中心点、长度)

        # proposal layer
        cfg_key = 'TRAIN' if self.training else 'TEST'

        # rois = self.RPN_proposal((rpn_cls_prob.data, rpn_twin_pred.data, cfg_key))
        if self.out_scores:  # False
            rois, rois_score = self.RPN_proposal((rpn_cls_prob.data, rpn_twin_pred.data, cfg_key))
        else:   # (回归)
            rois = self.RPN_proposal((rpn_cls_prob.data, rpn_twin_pred.data, cfg_key))#(1,2000,3)[其中(1,<960,3)<960的部分是前景,第一列全0存放未来的21类标签,后两列是可能的前景的起止帧;(960,2000)的部分全0,可能代表背景]
        self.rpn_loss_cls = 0
        self.rpn_loss_twin = 0
        self.rpn_loss_mask = 0
        self.rpn_label = None

        # generating training labels and build the rpn loss
        if self.training:
            assert gt_twins is not None
            # rpn_data = [label_targets, twin_targets, twin_inside_weights, twin_outside_weights]
            # label_targets: (batch_size, 1, A * length, height, width)
            # twin_targets: (batch_size, A*2, length, height, width), the same as twin_inside_weights and twin_outside_weights
            # (二分类)
            #                                 (1,20,96,1,1)二分类,     (1, 20, 3) gt_twins前两列代表真实起止帧,第三列代表标签
            rpn_data = self.RPN_anchor_target((rpn_cls_score.data, gt_twins))#rpn_data 0:(1, 1, 960, 1, 1) 1:(1, 20, 96, 1, 1) 2:(1,20,96,1,1) 3:(1,20,96,1,1) 2分类 多分类(好像是窗口回归) 窗口回归(内窗口) 窗口回归(外窗口)

            # compute classification loss
            rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 4, 1).contiguous().view(batch_size, -1, 2)   # (1,960,2) 二分类,后两列可能是前景和背景概率分数

            self.rpn_label = rpn_data[0].view(batch_size, -1)# (1, 960) 前876个数为处理后的labels(包含1,0,-1),后面几个数填充-1
            rpn_keep = Variable(self.rpn_label.view(-1).ne(-1).nonzero().view(-1))#(256)非0元素所在所有列(索引),即所有1和-1的索引(好像是1和0)(与前景有关)
            rpn_cls_score = torch.index_select(rpn_cls_score.view(-1,2), 0, rpn_keep)#(256,2)256个1和-1(好像是1和0)(前景和无关项(可能的前景))对应的前景和背景概率(以rpn_keep(256)的值为索引,取出256个rpn_cls_score的第0维(行))
            self.rpn_label = torch.index_select(self.rpn_label.view(-1), 0, rpn_keep.data)#(256)256个1和-1(好像是1和0)(前景和无关项(可能的前景))对应的labels(相当于去除了背景0)
            self.rpn_label = Variable(self.rpn_label.long())#(256)256个1和0对应的labels(相当于去除了-1)
            #          (256,2)对应帧的1和0对应的前背景概率(预测值),(256)对应帧的1和0(真实值)
            self.rpn_loss_cls = F.cross_entropy(rpn_cls_score, self.rpn_label)#(二分类损失)交叉熵函数,因为函数公式中有个-log，当预测越接近1(预测正确),函数值越小,所以可看成损失,即损失(函数值)越小,预测得越正确
            fg_cnt = torch.sum(self.rpn_label.data.ne(0))
            # 真实窗口起止帧,   列表中前景部分为1(少量),    列表基本接近于0(背景)
            rpn_twin_targets, rpn_twin_inside_weights, rpn_twin_outside_weights = rpn_data[1:]   # (1, 20, 96, 1, 1)窗口回归

            # compute twin regression loss
            rpn_twin_inside_weights = Variable(rpn_twin_inside_weights)
            rpn_twin_outside_weights = Variable(rpn_twin_outside_weights)
            rpn_twin_targets = Variable(rpn_twin_targets)
            #                                    预测窗口起止帧,  真实窗口起止帧,     真实窗口 前景填充为1(少量), 真实窗口 全是背景(0)
            self.rpn_loss_twin = _smooth_l1_loss(rpn_twin_pred, rpn_twin_targets, rpn_twin_inside_weights, rpn_twin_outside_weights, sigma=3, dim=[1,2,3,4])#(窗口回归损失)有对应公式,同样损失值越小代表预测越准

        if self.out_scores:  # False
            return rois, rois_score, rpn_cls_prob, rpn_twin_pred, self.rpn_loss_cls, self.rpn_loss_twin, self.rpn_label, self.rpn_loss_mask
        else:
            # (1,2000,3),(1,20,96,1,1),(1,20,96,1,1), (1),               (1),                (256),          (1) =0
            return rois, rpn_cls_prob, rpn_twin_pred, self.rpn_loss_cls, self.rpn_loss_twin, self.rpn_label, self.rpn_loss_mask
            #                                         (二分类损失)         (回归损失)

    def init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RPN_Conv1, 0, 0.01, cfg.TRAIN.TRUNCATED)  # tx(1)
        normal_init(self.RPN_Conv2, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RPN_twin_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self.init_weights()
        
    def generate_mask_label(self, gt_twins, feat_len):
        """ 
        gt_twins will be (batch_size, n, 3), where each gt will be (x1, x2, class_id)
        # feat_len is the length of mask-task features, self.feat_stride * feat_len = video_len
        # according: self.feat_stride, and upsample_rate
        # mask will be (batch_size, feat_len), -1 -- ignore, 1 -- fg, 0 -- bg
        """
        batch_size = gt_twins.size(0)
        mask_label = torch.zeros(batch_size, feat_len).type_as(gt_twins)
        for b in range(batch_size):
           single_gt_twins = gt_twins[b]
           single_gt_twins[:, :2] = (single_gt_twins[:, :2] / self.feat_stride).int()
           twins_start = single_gt_twins[:, 0]
           _, indices = torch.sort(twins_start)
           single_gt_twins = torch.index_select(single_gt_twins, 0, indices).long().cpu().numpy()

           starts = np.minimum(np.maximum(0, single_gt_twins[:,0]), feat_len-1)
           ends = np.minimum(np.maximum(0, single_gt_twins[:,1]), feat_len)
           for x in zip(starts, ends):
              mask_label[b, x[0]:x[1]+1] = 1

        return mask_label
