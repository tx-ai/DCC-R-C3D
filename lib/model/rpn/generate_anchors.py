# --------------------------------------------------------
# R-C3D
# Copyright (c) 2017 Boston University
# Licensed under The MIT License [see LICENSE for details]
# Written by Huijuan Xu
# --------------------------------------------------------

import numpy as np
import pdb


def generate_anchors(base_size=8, scales=2**np.arange(3, 6)):#generate_anchors(8, [2 4 5 6 8 9 10 12 14 16])
    """
    Generate anchor (reference) windows by enumerating aspect 
    scales wrt a reference (0, 7) window.
    """
    base_anchor = np.array([1, base_size]) - 1  # [0 7]
    anchors = _scale_enum(base_anchor, scales)
    """anchors = [[ -4  11]
                 [-12  19]
                 [-16  23]
                 [-20  27]
                 [-28  35]
                 [-32  39]
                 [-36  43]
                 [-44  51]
                 [-52  59]
                 [-60  67]]"""
    return anchors


def _whctrs(anchor):    # _whctrs([0 7])
    """
    Return width, height, x center, and y center for an anchor (window).
    """
    l = anchor[1] - anchor[0] + 1   # l = 8
    x_ctr = anchor[0] + 0.5 * (l - 1)   # x_ctr = 3.5
    return l, x_ctr 


def _mkanchors(ls, x_ctr):
    """
    Given a vector of lengths (ls) around a center
    (x_ctr), output a set of anchors (windows).
    """
    ls = ls[:, np.newaxis]  # np.newaxis表示1行10列变成10行1列 ls=np.array([[16],[32],[40],[48],[64],[72],[80],[96],[112],[128]])
    anchors = np.hstack((x_ctr - 0.5 * (ls - 1),    # np.hstack(([[-4][-12][-16][-20][-28][-32][-36][-44][-52][-60]],[[11][19][23][27][35][39][43][51][59][67]]))
                         x_ctr + 0.5 * (ls - 1)))
    """anchors = [[ -4  11]
                 [-12  19]
                 [-16  23]
                 [-20  27]
                 [-28  35]
                 [-32  39]
                 [-36  43]
                 [-44  51]
                 [-52  59]
                 [-60  67]]"""
    return anchors


def _scale_enum(anchor, scales):  # _scale_enum([0 7], [2 4 5 6 8 9 10 12 14 16])
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """
    l, x_ctr = _whctrs(anchor)  # l = 8, x_ctr = 3.5
    ls = l * scales  # ls = 8*[2 4 5 6 8 9 10 12 14 16] = [ 16  32  40  48  64  72  80  96 112 128]
    anchors = _mkanchors(ls, x_ctr)
    """anchors = [[ -4  11]
                 [-12  19]
                 [-16  23]
                 [-20  27]
                 [-28  35]
                 [-32  39]
                 [-36  43]
                 [-44  51]
                 [-52  59]
                 [-60  67]]"""
    return anchors

if __name__ == '__main__':
    import time
    t = time.time()
    a = generate_anchors(scales=np.array([2, 4, 5, 6, 8, 9, 10, 12, 14, 16]))
    print (time.time() - t)
    print (a)
    from IPython import embed; embed()
