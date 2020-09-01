import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from lib.model.utils.config import cfg
from lib.model.rpn.rpn import _RPN
from lib.model.roi_temporal_pooling.modules.roi_temporal_pool import _RoITemporalPooling
from lib.model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from lib.model.utils.net_utils import _smooth_l1_loss
from lib.model.utils.non_local_dot_product import NONLocalBlock3D

DEBUG = False


class _TDCNN(nn.Module):
    """ faster RCNN """
    def __init__(self):
        super(_TDCNN, self).__init__()
        # self.classes = classes
        self.n_classes = cfg.NUM_CLASSES    # 21
        # loss
        self.RCNN_loss_cls = 0  # 多分类损失
        self.RCNN_loss_twin = 0  # 窗口回归损失

        # define rpn
        #                   self.dout_base_model=512     # (1,2000,3),(1,20,96,1,1),(1,20,96,1,1),  (1),               (1),                (256),          (1) =0
        self.RCNN_rpn = _RPN(self.dout_base_model)  #     return rois, rpn_cls_prob, rpn_twin_pred, self.rpn_loss_cls, self.rpn_loss_twin, self.rpn_label, self.rpn_loss_mask

        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_temporal_pool = _RoITemporalPooling(cfg.POOLING_LENGTH, cfg.POOLING_HEIGHT, cfg.POOLING_WIDTH, cfg.DEDUP_TWINS)


        if cfg.USE_ATTENTION:   # 不走这条路
            self.RCNN_attention = NONLocalBlock3D(self.dout_base_model, inter_channels=self.dout_base_model)
        
    def prepare_data(self, video_data):
        return video_data

    def forward(self, video_data, gt_twins):
        batch_size = video_data.size(0)
        # print(batch_size)

        gt_twins = gt_twins.data
        # prepare data
        video_data = self.prepare_data(video_data)  # 这个video_data有变化？  (1, 3, 768, 112, 112)
        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(video_data)  # 经过c3d的前五层得到特征图(得到512 x L/8 x H/16 x W/16大小的特征图) (1,512,96,7,7)

        # feed base feature map tp RPN to obtain rois
        # (1,2000,3),(1),           (1)                      (1,512,96,7,7), (1,20,3)gt_twins前两列代表起止帧,第三列代表标签
        rois, _, _, rpn_loss_cls, rpn_loss_twin, _, _ = self.RCNN_rpn(base_feat, gt_twins)  # 经过rpn网络得到rois
        # rois[其中(1,<960,3)<960的部分是前景,第1列全0存21类标签,后两列是前景的起止帧;(960,2000)的部分全0,可能代表背景]
        # if it is training phase, then use ground truth twins for refining
        if self.training:   # 走这条(暂时理解成对rois的一些限制)
            roi_data = self.RCNN_proposal_target(rois, gt_twins)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data
            # (1,128,3),(1,128),(1,128,2), (1,128,2),      (1,128,2)
            rois_label = Variable(rois_label.view(-1).long())   # (128)
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))   # (128,2)
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))  # (128,2)
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))   # (128,2)
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_twin = 0

        rois = Variable(rois)
        # do roi pooling based on predicted rois
        if cfg.POOLING_MODE == 'pool':  # True
            pooled_feat = self.RCNN_roi_temporal_pool(base_feat, rois.view(-1, 3))  # (128, 512, 4, 2, 2)
        if cfg.USE_ATTENTION:   # False
            pooled_feat = self.RCNN_attention(pooled_feat)
        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)   # 分类网络  (128,4096)

        # compute twin offset, twin_pred will be (128, 402)
        twin_pred = self.RCNN_twin_pred(pooled_feat)    # nn.Linear(4096, 2 * 21)  实际是(128,42)

        if self.training:   # 走这条
            # select the corresponding columns according to roi labels, twin_pred will be (128, 2)
            twin_pred_view = twin_pred.view(twin_pred.size(0), int(twin_pred.size(1) / 2), 2)
            twin_pred_select = torch.gather(twin_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 2))
            twin_pred = twin_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)    # nn.Linear(4096, 21)
        cls_prob = F.softmax(cls_score, dim=1)  # 多分类

        if DEBUG:   # False
            print("tdcnn.py--base_feat.shape {}".format(base_feat.shape))
            print("tdcnn.py--rois.shape {}".format(rois.shape))
            print("tdcnn.py--tdcnn_tail.shape {}".format(pooled_feat.shape))
            print("tdcnn.py--cls_score.shape {}".format(cls_score.shape))
            print("tdcnn.py--twin_pred.shape {}".format(twin_pred.shape))

        RCNN_loss_cls = 0
        RCNN_loss_twin = 0

        if self.training:   # 走这条
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)  # (多分类损失)

            # bounding box regression L1 loss
            RCNN_loss_twin = _smooth_l1_loss(twin_pred, rois_target, rois_inside_ws, rois_outside_ws)   # (回归损失)

            # RuntimeError caused by mGPUs and higher pytorch version: https://github.com/jwyang/faster-rcnn.pytorch/issues/226
            rpn_loss_cls = torch.unsqueeze(rpn_loss_cls, 0)
            rpn_loss_twin = torch.unsqueeze(rpn_loss_twin, 0)
            RCNN_loss_cls = torch.unsqueeze(RCNN_loss_cls, 0)
            RCNN_loss_twin = torch.unsqueeze(RCNN_loss_twin, 0)

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)  # 分类预测？
        twin_pred = twin_pred.view(batch_size, rois.size(1), -1)   # 回归预测？

        if self.training:   # 走这条
            return rois, cls_prob, twin_pred, rpn_loss_cls, rpn_loss_twin, RCNN_loss_cls, RCNN_loss_twin, rois_label
        else:
            return rois, cls_prob, twin_pred

    def _init_weights(self):
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
        self.RCNN_rpn.init_weights()
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)  # self.RCNN_cls_score = nn.Linear(4096, 21)   cfg.TRAIN.TRUNCATED = False
        normal_init(self.RCNN_twin_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)  # self.RCNN_twin_pred = nn.Linear(4096, 42)

    def create_architecture(self):
        self._init_modules()    # 初始化特征提取网络
        self._init_weights()
