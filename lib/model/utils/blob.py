# --------------------------------------------------------
# R-C3D
# Copyright (c) 2017 Boston University
# Licensed under The MIT License [see LICENSE for details]
# Written by Huijuan Xu
# --------------------------------------------------------

"""Blob helper functions."""

import numpy as np
# from scipy.misc import imread, imresize
import cv2

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


def video_list_to_blob(videos):     # videos的shape：[batch_size, 512, 112, 112, 3]
    """Convert a list of videos into a network input.

    Assumes videos are already prepared (means subtracted, BGR order, ...).
    """
    shape = videos[0].shape    # [512, 112, 112, 3]
    num_videos = len(videos)    # batch_size
    blob = np.zeros((num_videos, shape[0], shape[1], shape[2], shape[3]),
                    dtype=np.float32)
    for i in xrange(num_videos):
        blob[i] = videos[i]     # 单纯的把video列表复制到blob列表(只不过blob列表是float型,video列表是int型)
    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, length, height, width)
    channel_swap = (0, 4, 1, 2, 3)
    blob = blob.transpose(channel_swap)  # [batch_size, 512, 112, 112, 3]->[batch_size, 3, 512, 112, 112]
    return blob     # blob的shape：[batch_size, 3, 512, 112, 112]


def prep_im_for_blob(im, pixel_means, target_size, crop_size, random_idx):
    """Mean subtract, resize and crop an frame for use in a blob."""
    im = im.astype(np.float32, copy=False)
    im = cv2.resize(im, target_size, interpolation=cv2.INTER_LINEAR)
    im -= pixel_means
    x = random_idx[1]
    y = random_idx[0]
    return im[x:x+crop_size, y:y+crop_size]
