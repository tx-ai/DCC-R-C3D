#coding=utf-8
# --------------------------------------------------------
# R-C3D
# Copyright (c) 2017 Boston University
# Licensed under The MIT License [see LICENSE for details]
# Written by Huijuan Xu
# --------------------------------------------------------

import os
from util import *
import json
import glob

fps = 25
ext = '.mp4'
VIDEO_DIR = '/media/tx/LaCie/田翔/数据集/tx/THUMOS 2014'
FRAME_DIR = '/home/tx/Dataset/tx/THUMOS14'

META_DIR = os.path.join(FRAME_DIR, 'annotation_')


def generate_frame(split):
  SUB_FRAME_DIR = os.path.join(FRAME_DIR, split)
  mkdir(SUB_FRAME_DIR)
  segment = dataset_label_parser(META_DIR+split, split, use_ambiguous=True)#把“某个视频”作为键(200个视频)，该视频某段标注的动作的起始、结束时刻和动作类别作为值，放入segment列表
  video_list = segment.keys()  # video_list为200个视频的名称
  for vid in video_list:  # (tx标注：好像只用到了帧，没有用到视频里的起始和结束的时刻帧)
    filename = os.path.join(VIDEO_DIR, split, vid+ext)  # filename为200个视频之一的绝对路径
    outpath = os.path.join(FRAME_DIR, split, vid)   # outpath为200个视频之一的输出成帧文件夹的绝对路径
    outfile = os.path.join(outpath, "image_%5d.jpg")  # outfile为200个视频之一的输出成帧文件夹里面的所有视频帧的绝对路径
    mkdir(outpath)
    ffmpeg(filename, outfile, fps)  # 以25fps的帧率将原始视频的视频帧输出到指定的帧文件夹
    for framename in os.listdir(outpath):  # 只是对帧的一个resize
      resize(os.path.join(outpath, framename))
    frame_size = len(os.listdir(outpath))  # frame_size为一个帧文件夹的总帧数
    print(filename, fps, frame_size)


# generate_frame('val')
generate_frame('test')
# generate_frame('testing')
