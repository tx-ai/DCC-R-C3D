# --------------------------------------------------------
# R-C3D
# Copyright (c) 2017 Boston University
# Licensed under The MIT License [see LICENSE for details]
# Written by Huijuan Xu
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Shiguang Wang
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
import cv2
import os
from lib.model.utils.config import cfg
from lib.model.utils.blob import prep_im_for_blob, video_list_to_blob
from lib.model.utils.transforms import GroupMultiScaleCrop
import pdb
from multiprocessing import Pool, cpu_count
import threading

DEBUG = False


def get_minibatch(roidb, phase='train'):
    """Given a roidb, construct a minibatch sampled from it."""
    num_videos = len(roidb)     # 1
    # print('11111111111111111111111111111111111',num_videos)
    assert num_videos == 1, "Single batch only"
    # Sample random scales to use for each video in this batch
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.LENGTH),
                                    size=num_videos)    # npr.randint(0,1,1)    random_scale_inds=[0]

    # Get the input video blob, formatted for caffe
    video_blob = _get_video_blob(roidb, random_scale_inds, phase=phase)  #video_blob的shape：[batch_size, 3, 512, 112, 112]
    blobs = {'data': video_blob}
    
    if phase != 'train':    # 不走这条路
        blobs['gt_windows'] = np.zeros((1, 3), dtype=np.float32)
        return blobs
        
    # gt windows: (x1, x2, cls)
    gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]  # gt_inds = [0]
    gt_windows = np.empty((len(gt_inds), 3), dtype=np.float32)  # 产生大小为1行3列的数值接近于0的矩阵
    gt_windows[:, 0:2] = roidb[0]['wins'][gt_inds, :]   # gt_windows的前两列(即前两位)是时序片段的“起止帧时刻”(大概是这个意思)
    gt_windows[:, -1] = roidb[0]['gt_classes'][gt_inds]     # gt_windows的最后一列(即第三位)是该时序片段的类别
    blobs['gt_windows'] = gt_windows    # blobs['gt_windows']的形状为(1,3),前两位是时序片段在视频中的“起止帧时刻”(大概是这个意思),第三位是该时序片段的类别

    return blobs    # blobs是二维字典,包含键:'data'和'gt_windows'


class MyThread(threading.Thread):
    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)
        
    def get_result(self):
        try:
            return self.result
        except Exception:
            return None


def prepare_im_func(prefix, random_idx, frame_idx, flipped):        
    frame_path = os.path.join(prefix, 'image_'+str(frame_idx).zfill(5)+'.jpg')
    frame = cv2.imread(frame_path)
    # process the boundary frame
    if frame is None:          
        frames = sorted(os.listdir(prefix))
        frame_path = frame_path = os.path.join(prefix, frames[-1])
        frame = cv2.imread(frame_path)         
    
    frame = prep_im_for_blob(frame, cfg.PIXEL_MEANS, tuple(cfg.TRAIN.FRAME_SIZE[::-1]), cfg.TRAIN.CROP_SIZE, random_idx)
       
    if flipped:
        frame = frame[:, ::-1, :]

    if DEBUG:
        cv2.imshow('frame', frame/255.0)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    return frame        


def _get_video_blob(roidb, scale_inds, phase='train'):  # ([{'gt_classes': array([18.]), 'bg_name':... }], [0], 'train')
    """Builds an input blob from the videos in the roidb at the specified
    scales.
    """
    processed_videos = []
    
    for i, item in enumerate(roidb):  # i= 0,item的shape： {'gt_classes': array([18.]), 'bg_name':... }
        # just one scale implementated
        video_length = cfg.TRAIN.LENGTH[scale_inds[0]]  # video_length = 512
        video = np.zeros((video_length, cfg.TRAIN.CROP_SIZE,    # (512, 112, 112, 3)
                        cfg.TRAIN.CROP_SIZE, 3))
        j = 0

        if phase == 'train':
            random_idx = [np.random.randint(cfg.TRAIN.FRAME_SIZE[1]-cfg.TRAIN.CROP_SIZE),   # [np.random.randint(59), np.random.randint(16)]
                            np.random.randint(cfg.TRAIN.FRAME_SIZE[0]-cfg.TRAIN.CROP_SIZE)]
            # TODO: data argumentation
            #image_w, image_h, crop_w, crop_h = cfg.TRAIN.FRAME_SIZE[1], cfg.TRAIN.FRAME_SIZE[0], cfg.TRAIN.CROP_SIZE, cfg.TRAIN.CROP_SIZE
            #offsets = GroupMultiScaleCrop.fill_fix_offset(False, image_w, image_h, crop_w, crop_h) 
            #random_idx = offsets[ npr.choice(len(offsets)) ]
        else:
            random_idx = [int((cfg.TRAIN.FRAME_SIZE[1]-cfg.TRAIN.CROP_SIZE) / 2), 
                      int((cfg.TRAIN.FRAME_SIZE[0]-cfg.TRAIN.CROP_SIZE) / 2)]
                                      
        if DEBUG:
            print("offsets: {}, random_idx: {}".format(offsets, random_idx))
            
        video_info = item['frames'][0]  # item['frames'][0]的shape:[0,1317,2085,1]
        step = video_info[3] if cfg.INPUT == 'video' else 1  # step = 1
        prefix = item['fg_name'] if video_info[0] else item['bg_name']  # 视频帧文件夹的绝对路径
        
        if cfg.TEMP_SPARSE_SAMPLING:       
            if phase == 'train':
                segment_offsets = npr.randint(step, size=len(range(video_info[1], video_info[2], step)))
            else:
                segment_offsets = np.zeros(len(range(video_info[1], video_info[2], step))) + step // 2
        else:  # 走这条路
            segment_offsets = np.zeros(len(range(video_info[1], video_info[2], step)))  # 时序片段的帧长度

        for i, idx in enumerate(range(video_info[1], video_info[2], step)):  # idx为该时序片段内的所有帧的下标
            frame_idx = int(segment_offsets[i]+idx+1)   # frame_idx为该时序片段内的所有帧的下标+1
            frame_path = os.path.join(prefix, 'image_'+str(frame_idx).zfill(5)+'.jpg')  # frame_path为该时序片段内的所有帧的绝对路径
            frame = cv2.imread(frame_path)  # 读取该路径下帧的图片
            # process the boundary frame
            if frame is None:   # 若该时序片段下的帧为空，则读取该视频帧文件夹的最后一帧作为frame
                frames = sorted(os.listdir(prefix))     # frames为帧文件夹下所有帧图片名字构成的列表,且列表内的图片名字的数字由小到大排列
                frame_path = frame_path = os.path.join(prefix, frames[-1])
                frame = cv2.imread(frame_path)
            frame = prep_im_for_blob(frame, cfg.PIXEL_MEANS, tuple(cfg.TRAIN.FRAME_SIZE[::-1]), cfg.TRAIN.CROP_SIZE, random_idx)
            if item['flipped']:     # 不走这条路(flipped=False)
                frame = frame[:, ::-1, :]

            if DEBUG:   # 不走这条路(DEBUG=False)
                cv2.imshow('frame', frame/255.0)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            video[j] = frame    # 把每一个时序片段的所有帧重新装入video这个列表里,video形状:[video_info[2]-video_info[1], 112, 112, 3]
            j = j + 1
            
        video[j:video_length] = video[j-1]  # 若video长度不足512,则把video的最后一帧重复填充直至第512帧,最终video形状:[512, 112, 112, 3]
        
    processed_videos.append(video)  # 把每个video合在一起(可能是形成一个batch)(但实际上这级的for循环只会执行一次?所以batch_size在这里=1?) processed_videos的shape：[batch_size, 512, 112, 112, 3]
    # Create a blob to hold the input images, dimension trans CLHW
    blob = video_list_to_blob(processed_videos)     # blob的shape：[batch_size, 3, 512, 112, 112]

    return blob     # blob的shape：[batch_size, 3, 512, 112, 112]
