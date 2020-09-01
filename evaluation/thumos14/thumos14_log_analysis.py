# --------------------------------------------------------
# R-C3D
# Copyright (c) 2017 Boston University
# Licensed under The MIT License [see LICENSE for details]
# Written by Huijuan Xu
# --------------------------------------------------------

import sys, os, errno
import numpy as np
import csv
import json
import copy
import argparse
import subprocess


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
FRAME_DIR = '/home/tx/Dataset/tx/THUMOS14'
META_DIR = os.path.join(FRAME_DIR, 'annotation_')


def nms(dets, thresh=0.4):  # dets:每个真实标签对应的预测起止帧时刻和概率标签
    """Pure Python NMS baseline."""
    if len(dets) == 0: return []
    x1 = dets[:, 0]  # 预测起始帧时刻
    x2 = dets[:, 1]  # 预测结束帧时刻
    scores = dets[:, 2]  # 概率标签
    lengths = x2 - x1   # 行为窗口时长
    order = scores.argsort()[::-1]  # 按概率标签的索引从大到小排列
    keep = []
    while order.size > 0:
        i = order[0]    # 取出概率标签最大值的索引
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])  # 预测窗口的起始帧时刻尽量向右移
        xx2 = np.minimum(x2[i], x2[order[1:]])  # 预测窗口的结束帧时刻尽量向左移
        inter = np.maximum(0.0, xx2 - xx1)  # 预测窗口缩小后的时长
        ovr = inter / (lengths[i] + lengths[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]   # (索引)重叠的窗口只有重叠度足够小(<0.4),才认为找到正确的数据(概率标签、预测起始帧时刻),并且取出这些符合要求中概率值最大的作为最终数据
        order = order[inds + 1]
    return keep     # 1W个时序片段中里面20个类别已经分好了,取出每个真实标签对应的索引,这些索引将来可以遍历其预测起止帧时刻和概率标签数据


def generate_classes(meta_dir, split, use_ambiguous=False):
    class_id = {0: 'Background'}
    with open(os.path.join(meta_dir, 'detclasslist.txt'), 'r') as f:
        lines = f.readlines()
        for l in lines:
            cname = l.strip().split()[-1]
            cid = int(l.strip().split()[0])
            class_id[cid] = cname
        if use_ambiguous:
            class_id[21] = 'Ambiguous'

    return class_id  # 21个类别的标签+类别名
'''
def get_segments(data, thresh, framerate):
    segments = []
    vid = 'Background'
    find_next = False
    tmp = {'label' : 0, 'score': 0, 'segment': [0, 0]}
    for l in data:
      # video name and sliding window length
      if "fg_name :" in l:
         vid = l.split('/')[-1]

      # frame index, time, confident score
      elif "frames :" in l:
         start_frame=int(l.split()[4])
         end_frame=int(l.split()[5])
         stride = int(l.split()[6].split(']')[0])

      elif "activity:" in l:
         label = int(l.split()[1])
         tmp['label'] = label
         find_next = True

      elif "im_detect" in l:
         return vid, segments

      elif find_next:
         try: 
           left_frame = float(l.split()[0].split('[')[-1])*stride + start_frame
           right_frame = float(l.split()[1])*stride + start_frame
         except:
           left_frame = float(l.split()[1])*stride + start_frame
           right_frame = float(l.split()[2])*stride + start_frame
         if (left_frame < end_frame) and (right_frame <= end_frame):
           left  = left_frame / 25.0
           right = right_frame / 25.0
           try: 
             score = float(l.split()[-1].split(']')[0])
           except:
             score = float(l.split()[-2])
           if score > thresh:
             tmp1 = copy.deepcopy(tmp)
             tmp1['score'] = score
             tmp1['segment'] = [left, right]
             segments.append(tmp1)
         elif (left_frame < end_frame) and (right_frame > end_frame):
             if (end_frame-left_frame)*1.0/(right_frame-left_frame)>=0:
                 right_frame = end_frame
                 left  = left_frame / 25.0
                 right = right_frame / 25.0
                 try: 
                   score = float(l.split()[-1].split(']')[0])
                 except:
                   score = float(l.split()[-2])
                 if score > thresh:
                     tmp1 = copy.deepcopy(tmp)
                     tmp1['score'] = score
                     tmp1['segment'] = [left, right]
                     segments.append(tmp1)

'''


# data:列表,每个元素为test_log中fg_name到im_detect之间的行
def get_segments(data, thresh, framerate):
    segments = []
    vid = 'Background'
    find_next = False
    tmp = {'label' : 0, 'score': 0, 'segment': [0, 0]}
    for l in data:
        # video name and sliding window length
        if "fg_name:" in l:
            vid = l.split('/')[-1]  # 视频的名称,如:video_test_0000622

        # frame index, time, confident score
        elif "frames:" in l:
            start_frame = int(l.split()[3]) # 起始帧
            end_frame = int(l.split()[4])   # 结束帧
            stride = int(l.split()[5].split(']')[0])    # 步长1

        elif "activity:" in l:
            label = int(l.split()[1])   # (1~20)标签
            tmp['label'] = label
            find_next = True

        elif "im_detect" in l:  # 每一个时序片段输出一次
            return vid, segments    # vid:视频的名称,如:video_test_0000622 segments的任意一行:tmp['label']该类的真实标签 tmp1['score']是该类别的概率标签 tmp1['segment']预测的起止帧时刻
            # segments本身是列表,里面包裹着1W多个字典,每个字典3个键值
        elif find_next:
            try: 
                left_frame = float(l.split()[0].split('[')[-1])*stride + start_frame  # 可理解成窗口起始帧的偏移
                right_frame = float(l.split()[1])*stride + start_frame               # 可理解成窗口结束帧的偏移
            except:  # 一般不会走这条
                left_frame = float(l.split()[1])*stride + start_frame
                right_frame = float(l.split()[2])*stride + start_frame

            try:
                score = float(l.split()[-1].split(']')[0])                  # 是该类别的概率
            except:  # 一般不会走这条
                score = float(l.split()[-2])    
                            
            if (left_frame >= right_frame):  # 一般不会走这条
                print("???", l)
                continue
                
            if right_frame > end_frame:
                # print("right out", right_frame, end_frame)
                right_frame = end_frame
                                
            left  = left_frame / framerate  # 算出起始帧时刻
            right = right_frame / framerate  # 算出结束帧时刻
            if score > thresh:  # 几乎都会走这条
                tmp1 = copy.deepcopy(tmp)   # 该类的真实标签
                tmp1['score'] = score   # 是该类别的概率标签
                tmp1['segment'] = [left, right]     # 预测的起止帧时刻
                segments.append(tmp1)


#            test_log文件,0.005,   25
def analysis_log(logfile, thresh, framerate):
    with open(logfile, 'r') as f:
        lines = f.read().splitlines()
    predict_data = []
    res = {}
    for l in lines:
        if "fg_name:" in l:  # github上此处小问题
            predict_data = []   # 暂存,所以清空
        predict_data.append(l)  # 暂存形如[fg_name: /home/tx/Dataset/tx/THUMOS14/test/video_test_0000622, frames: [[0 2496 3264 1]],...]的数据 (注:predict_data是3维矩阵)
        if "im_detect:" in l:
            vid, segments = get_segments(predict_data, thresh, framerate)
            # segments本身是列表,里面包裹着多个字典,每个字典3个键值
            # vid:视频的名称,如:video_test_0000622 segments的任意一行:tmp['label']该类的真实标签 tmp1['score']是该类别的概率标签 tmp1['segment']预测的起止帧时刻
            if vid not in res:
                res[vid] = []   # 最终形成包含213个元素的字典,每个字典的键为测试视频名称如video_test_0000622
            res[vid] += segments    # 每个字典的值为segments
    return res  # 包含213个元素的字典


# segmentations:包含213个元素的字典
def select_top(segmentations, nms_thresh=0.99999, num_cls=0, topk=0):
  res = {}
  for vid, vinfo in segmentations.items():  # vid视频名称, vinfo三个信息:该类的真实标签 该类别的概率标签 预测的起止帧时刻
    # select most likely classes
    if num_cls > 0:  # Flase
      ave_scores = np.zeros(21)
      for i in xrange(1, 21):
        ave_scores[i] = np.sum([d['score'] for d in vinfo if d['label']==i])
      labels = list(ave_scores.argsort()[::-1][:num_cls])
    else:
      labels = list(set([d['label'] for d in vinfo]))   # 三个信息中该视频的所有真实标签

    # NMS
    res_nms = []
    for lab in labels:  # 执行20次
      nms_in = [d['segment'] + [d['score']] for d in vinfo if d['label'] == lab]    # 每个真实标签对应的预测起止帧时刻和概率标签
      keep = nms(np.array(nms_in), nms_thresh)  # (1)20次,每次都会返回该真实标签下选出的1W个时序片段当中的索引
      for i in keep:
        # tmp = {'label':classes[lab], 'score':nms_in[i][2], 'segment': nms_in[i][0:2]}
        tmp = {'label': lab, 'score':nms_in[i][2], 'segment': nms_in[i][0:2]}
        res_nms.append(tmp)  # 列表,列表中的元素是(<1W,3)字典,字典键:真实标签、概率标签、预测起止帧时刻

    # select topk
    scores = [d['score'] for d in res_nms]  # (<1W,1)概率标签
    sortid = np.argsort(scores)[-topk:]  # <1W个中后200个概率标签的索引
    res[vid] = [res_nms[id] for id in sortid]   # 213次,每次形成字典中的一个元素,该元素键:视频名,值:列表(200,3)子字典(子字典键:真实标签、概率标签、预测起止帧时刻)
  return res    # 213元素的字典


parser = argparse.ArgumentParser(description="log analysis.py")
parser.add_argument('log_file', type=str, help="test log file path")
parser.add_argument('--framerate', type=int, help="frame rate of videos extract by ffmpeg")
parser.add_argument('--thresh', type=float, default=0.005, help="filter those dets low than the thresh, default=0.0005")
parser.add_argument('--nms_thresh', type=float, default=0.4, help="nms thresh, default=0.3")
parser.add_argument('--topk', type=int, default=200, help="select topk dets, default=200")
parser.add_argument('--num_cls', type=int, default=0, help="select most likely classes, default=0")  

args = parser.parse_args()
classes = generate_classes(META_DIR+'test', 'test', use_ambiguous=False)    # 21个类别的标签+类别名
segmentations = analysis_log(args.log_file, thresh = args.thresh, framerate=args.framerate)  # 包含213个元素的字典
segmentations = select_top(segmentations, nms_thresh=args.nms_thresh, num_cls=args.num_cls, topk=args.topk)  # 包含213个元素的字典
res = {'version': 'VERSION 1.3',
       'external_data': {'used': True, 'details': 'C3D pre-trained on activity-1.3 training set'},
       'results': {}}
for vid, vinfo in segmentations.items():
  res['results'][vid] = vinfo   # res['results']是包含213元素的字典,元素键:视频名,值:200元素的子字典(子字典键:真实标签label、概率标签score、预测起止帧时刻segment)
# segmentations与res['results']是等价的
#with open('results.json', 'w') as outfile:
#  json.dump(res, outfile)


with open('tmp.txt', 'w') as outfile:
  for vid, vinfo in segmentations.items():
    for seg in vinfo:
      outfile.write("{} {} {} {} {}\n".format(vid, seg['segment'][0], seg['segment'][1], int(seg['label']) ,seg['score']))


def matlab_eval():
    print('Computing results with the official Matlab eval code')
    path = os.path.join(THIS_DIR, 'Evaluation')
    cmd = 'cp tmp.txt {} && '.format(THIS_DIR)
    cmd += 'cd {} && '.format(path)
    cmd += 'matlab -nodisplay -nodesktop '
    cmd += '-r "dbstop if error; '
    cmd += 'eval_thumos14(); quit;"'

    print('Runing: \n {}'.format(cmd))
    status = subprocess.call(cmd, shell=True)

# matlab_eval()
