# --------------------------------------------------------
# R-C3D
# Copyright (c) 2017 Boston University
# Licensed under The MIT License [see LICENSE for details]
# Written by Huijuan Xu
# --------------------------------------------------------

import subprocess
import shutil
import os, errno
import cv2
#import scipy.io
import glob
from collections import defaultdict
import shutil
import math
#from sklearn.cluster import KMeans
#import matplotlib.pyplot as plt
import numpy as np


# meta_dir = '/home/tx/Dataset/tx/THUMOS14/annotation_val'
# split = 'val'
def dataset_label_parser(meta_dir, split, use_ambiguous=False):
  class_id = defaultdict(int)
  with open(os.path.join(meta_dir, 'detclasslist.txt'), 'r') as f:
    lines = f.readlines()
    for l in lines:
      cname = l.strip().split()[-1]   # 被标记的那20类的英文类名
      cid = int(l.strip().split()[0])   # 被标记的那20类的标签(7 9 12 21 22 23 24 26 31 33 36 40 45 51 68 79 85 92 93 97)
      class_id[cname] = cid
      if use_ambiguous:
        class_id['Ambiguous'] = 21  # (tx不懂：为啥要修改这个类标签？难道这个标签被设置成了背景？)
    segment = {}
    # video_instance = set()
  for cname in class_id.keys():
    tmp = '{}_{}.txt'.format(cname, split)  # ”这个动作“的全部涉及到的视频
    with open(os.path.join(meta_dir, tmp)) as f:
      lines = f.readlines()
      for l in lines:
        vid_name = l.strip().split()[0]   # 对应训练集中“某个视频”的前缀(除去.mp4)
        start_t = float(l.strip().split()[1])   # 对应该视频“这个动作”的起始时刻
        end_t = float(l.strip().split()[2])   # 对应该视频“这个动作”的结束时刻
        # video_instance.add(vid_name)
        # initionalize at the first time
        if not vid_name in segment.keys():
          segment[vid_name] = [[start_t, end_t, class_id[cname]]]  # 程序一般都走这条路：把“某个视频”作为键，该视频某段标注的动作的起始、结束时刻和动作类别作为值，放入segment列表
        else:
          segment[vid_name].append([start_t, end_t, class_id[cname]])
  # print(list(segment.keys())[0])
  # print(len(list(segment.values())[0]))
  # print(list(segment.values())[0])

  # sort segments by start_time
  for vid in segment:
    segment[vid].sort(key=lambda x: x[0])   # (tx不懂:按照视频名称末尾上升的顺序，对segment列表进行排序？)
  if True:#(tx暂时不懂:可能改变了'segment.txt'文件内容?方便查看200个视频各个动作以及其起始时刻？(一知半解:原来的segment.txt文件标签是1~20,修改后是真实标签:7 9 12 21 22 23 24 26 31 33 36 40 45 51 68 79 85 92 93 97))
    keys = list(segment.keys())  # 修改后的'segment.txt'文件记录了200个视频帧文件夹中某个动作的起始和结束时刻信息
    keys.sort()
    with open('segment.txt', 'w') as f:   # split='val'时用这个(仅仅是为了区分不同txt文件而已)
      for k in keys:
        f.write("{}\n{}\n\n".format(k, segment[k]))
  return segment


# def get_segment_len(segment):
#   segment_len = []
#   for vid_seg in segment.values():
#     for seg in vid_seg:
#       l = seg[1] - seg[0]
#       assert l > 0
#       segment_len.append(l)
#   return segment_len
#
#
# def mkdir(path):
#   try:
#     os.makedirs(path)
#   except OSError as e:
#     if e.errno != errno.EEXIST:
#       raise
#
#
# def rm(path):
#   try:
#     shutil.rmtree(path)
#   except OSError as e:
#     if e.errno != errno.ENOENT:
#       raise
#
#
# def ffmpeg(filename, outfile, fps):
#   command = ["ffmpeg", "-i", filename, "-q:v", "1", "-r", str(fps), outfile]
#   pipe = subprocess.Popen(command, stdout=subprocess.PIPE, stderr = subprocess.STDOUT)
#   pipe.communicate()
#
#
# def resize(filename, size = (171, 128)):
#   img = cv2.imread(filename, 100)
#   img2 = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
#   cv2.imwrite(filename, img2, [100])
#
#
# # get segs_len from segments by: segs_len = [ s[1]-s[0] for v in segments.values() for s in v ]
# def kmeans(segs_len, K=5, vis=False):
#   X = np.array(segs_len).reshape(-1, 1)
#   cls = KMeans(K).fit(X)
#   print( "the cluster centers are: ")
#   print( cls.cluster_centers_)
#   if vis:
#     markers = ['^','x','o','*','+']
#     for i in range(K):
#       members = cls.labels_ == i
#       plt.scatter(X[members,0],X[members,0],s=60,marker=markers[min(i,K-1)],c='b',alpha=0.5)
#       plt.title(' ')
#       plt.show()


# dataset_label_parser('/home/lyp/disk2/tx/THUMOS14/annotation_val', 'val', use_ambiguous=False)
