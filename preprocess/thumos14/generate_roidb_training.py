# coding=utf-8
# --------------------------------------------------------
# R-C3D
# Copyright (c) 2017 Boston University
# Licensed under The MIT License [see LICENSE for details]
# Written by Huijuan Xu
# --------------------------------------------------------

import os
import copy
import json
import pickle
import subprocess
import numpy as np
import cv2
from util import *
import glob

FPS = 25
ext = '.mp4'
LENGTH = 512
min_length = 3
overlap_thresh = 0.7
STEP = LENGTH / 4  # 192
WINS = [LENGTH * 1]
FRAME_DIR = '/home/tx/Dataset/tx/THUMOS14'
META_DIR = os.path.join(FRAME_DIR, 'annotation_')

print('Generate Training Segments')
train_segment = dataset_label_parser(META_DIR+'val', 'val', use_ambiguous=False)  # use_ambiguous=False:去掉背景类


def generate_roi(rois, video, start, end, stride, split):
  tmp = {}
  tmp['wins'] = (rois[:, :2] - start) / stride  # (tx暂时不懂:保存每个rois的“窗口”到tmp字典？)
  tmp['durations'] = tmp['wins'][:, 1] - tmp['wins'][:, 0]  # 保存每个rois的长度到tmp字典
  tmp['gt_classes'] = rois[:, 2]  # 保存每个rois的类别到tmp字典
  tmp['max_classes'] = rois[:, 2]   # (tx不懂)???和上一个类别的区别是啥
  tmp['max_overlaps'] = np.ones(len(rois))  # (tx不懂)生成一个len(rois)长度的单位矩阵，表示最大重合度，有何作用？
  tmp['flipped'] = False
  tmp['frames'] = np.array([[0, start, end, stride]])   # (tx不懂)每个rois的帧？
  tmp['bg_name'] = os.path.join(FRAME_DIR, split, video)  # 背景,每个视频帧文件夹的绝对路径
  tmp['fg_name'] = os.path.join(FRAME_DIR, split, video)  # (tx不懂)???和上一个的区别
  if not os.path.isfile(os.path.join(FRAME_DIR, split, video, 'image_' + str(end-1).zfill(5) + '.jpg')):
    print(os.path.join(FRAME_DIR, split, video, 'image_' + str(end-1).zfill(5) + '.jpg'))
    raise
  return tmp


def generate_roidb(split, segment):
  VIDEO_PATH = os.path.join(FRAME_DIR, split)
  video_list = set(os.listdir(VIDEO_PATH))#生成所有帧文件夹的文件名的set形式({'video_validation_0000415', 'video_validation_0000363', ...)
  duration = []
  roidb = []
  for vid in segment:  # name of every video folder:'video_validation_0000415', 'video_validation_0000363', ...
    if vid in video_list:  # 程序肯定走这条路
      length = len(os.listdir(os.path.join(VIDEO_PATH, vid)))  # length：某个帧文件夹里的帧数
      db = np.array(segment[vid])   # 某个视频内所有的[[start1,end1,class1],[start2,end2,class2],...]
      if len(db) == 0:  # 滤掉没有动作的视频
        continue
      db[:, :2] = db[:, :2] * FPS  # 将帧时刻转化成第几帧(针对每个动作做同样的操作,列表每行就代表一个动作的[起始帧时刻,结束帧时刻,帧间动作类别])

      for win in WINS:  # 生成窗口?只会执行一次？
        # inner of windows
        stride = int(win / LENGTH)  # stride=1
        # Outer of windows
        step = int(stride * STEP)  # step=192
        # Forward Direction
        for start in range(0, max(1, length - win + 1), step):  # 设定一个起始帧
          end = min(start + win, length)  # 设定那个起始帧对应的结束帧
          assert end <= length  # 确保结束帧小于等于该视频的总帧数
          rois = db[np.logical_not(np.logical_or(db[:, 0] >= end, db[:, 1] <= start))]  # zhiyao shipin pianduan he huadongchuangkou de mou chuangkou zhanbian de pianduan doubaoliu daoyiqi

          # Remove duration less than min_length
          if len(rois) > 0: # ye youkeneng gai chuangkou meiyou yige dongzuo pianduan
            duration = rois[:, 1] - rois[:, 0]  # 计算某个动作(标签)的帧持续时间
            rois = rois[duration >= min_length]   # 留下长度大于等于3的rois

          # Remove overlap less than overlap_thresh
          if len(rois) > 0:
            time_in_wins = (np.minimum(end, rois[:, 1]) - np.maximum(start, rois[:, 0]))*1.0  # (tx暂时不懂)可能是一个视频边缘部分动作的最大持续时间
            overlap = time_in_wins / (rois[:, 1] - rois[:, 0])    # 计算重叠度
            assert min(overlap) >= 0  # 确保重叠度的范围是(0,1)
            assert max(overlap) <= 1
            rois = rois[overlap >= overlap_thresh]  # 留下重叠度大于0.7的rois

          # Append data
          if len(rois) > 0:
            rois[:, 0] = np.maximum(start, rois[:, 0])  # qu rois yu chuangkou chonghe de bufen zuowei xinde rois
            rois[:, 1] = np.minimum(end, rois[:, 1])
            tmp = generate_roi(rois, vid, start, end, stride, split)# 封装一个tmp字典,包含{'durations','gt_classes','max_overlaps','frames','bg_name'等信息}
            roidb.append(tmp)   # 把每个rois的字典信息放入roidb列表
            if USE_FLIPPED:  # (tx暂时不懂)
               flipped_tmp = copy.deepcopy(tmp)
               flipped_tmp['flipped'] = True
               roidb.append(flipped_tmp)

        # Backward Direction
        for end in range(length, win-1, - step):  # (tx暂时不懂)貌似和上一个for循环做同样的事情
          start = end - win
          assert start >= 0
          rois = db[np.logical_not(np.logical_or(db[:, 0] >= end, db[:, 1] <= start))]

          # Remove duration less than min_length
          if len(rois) > 0:
            duration = rois[:, 1] - rois[:, 0]
            rois = rois[duration > min_length]

          # Remove overlap less than overlap_thresh
          if len(rois) > 0:
            time_in_wins = (np.minimum(end, rois[:, 1]) - np.maximum(start, rois[:, 0]))*1.0
            overlap = time_in_wins / (rois[:, 1] - rois[:, 0])
            assert min(overlap) >= 0
            assert max(overlap) <= 1
            rois = rois[overlap > overlap_thresh]

          # Append data
          if len(rois) > 0:
            rois[:, 0] = np.maximum(start, rois[:, 0])
            rois[:, 1] = np.minimum(end, rois[:, 1])
            tmp = generate_roi(rois, vid, start, end, stride, split)  # 封装一个tmp字典,包含{'durations','gt_classes','max_overlaps','frames','bg_name'等信息}
            roidb.append(tmp)
            if USE_FLIPPED:
               flipped_tmp = copy.deepcopy(tmp)
               flipped_tmp['flipped'] = True
               roidb.append(flipped_tmp)

  return roidb


if __name__ == '__main__':

    USE_FLIPPED = True      
    train_roidb = generate_roidb('val', train_segment)  # 把每个rois的字典信息放入roidb列表
    print(len(train_roidb))
    print("Save dictionary")
    pickle.dump(train_roidb, open('train_data_25fps_flipped.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)  # 把roidb列表保存成pickle形式

