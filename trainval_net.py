from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import time
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from lib.roi_data_layer.roibatchLoader import roibatchLoader
from lib.model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from lib.model.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient


from lib.model.tdcnn.c3d import C3D, c3d_tdcnn
from lib.model.tdcnn.i3d import I3D, i3d_tdcnn
from lib.model.tdcnn.resnet import resnet_tdcnn
from lib.model.tdcnn.eco import eco_tdcnn

# from apex import amp

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"    # 指定gpu

dataset = 'thumos14'
net = 'c3d'
start_epoch = 1
max_epochs = 5
disp_interval = 100
save_dir = "./models"
output_dir = "./output"
nw = 12
gpus = [0, 1]
bs = 2
roidb_dir = "./preprocess"

lr = 0.0001
lr_decay_step = 6
lr_decay_gamma = 0.1

session = 1

resume = False
checksession = 1
checkepoch = 6
checkpoint = 6856


class sampler(Sampler):
    def __init__(self, train_size, batch_size):
        self.num_data = train_size
        self.num_per_batch = int(train_size / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0, batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if train_size % batch_size:
            self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
            self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
        self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
            self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

        return iter(self.rand_num_view)

    def __len__(self):
        return self.num_data


def get_roidb(path):
    data = pickle.load(open(path, 'rb'))
    return data


def train_net(tdcnn_demo, dataloader, optimizer, lr, epoch, disp_interval, session):
    # setting to train mode
    tdcnn_demo.train()
    loss_temp = 0
    start = time.time()

    data_start = time.time()
    for step, (video_data, gt_twins, num_gt) in enumerate(dataloader):  # ([2, 3, 768, 112, 112] [2, 20, 3] [2]) num_gt取值范围(1~20)
        video_data = video_data.cuda()
        gt_twins = gt_twins.cuda()
        data_time = time.time()-data_start
        tdcnn_demo.zero_grad()

        rois, cls_prob, twin_pred, \
        rpn_loss_cls, rpn_loss_twin, RCNN_loss_cls, RCNN_loss_twin, \
        rois_label = tdcnn_demo(video_data, gt_twins)

        loss = rpn_loss_cls.mean() + rpn_loss_twin.mean() \
           + RCNN_loss_cls.mean() + RCNN_loss_twin.mean()
        loss_temp += loss.item()

        # backward
        optimizer.zero_grad()
        # with amp.scale_loss(loss, optimizer) as scaled_loss: scaled_loss.backward
        loss.backward()
        # if args.net == "vgg16": clip_gradient(tdcnn_demo, 100.)
        optimizer.step()

        if step % disp_interval == 0:
            end = time.time()
            if step > 0:
                loss_temp /= disp_interval

            loss_rpn_cls = rpn_loss_cls.mean().item()
            loss_rpn_twin = rpn_loss_twin.mean().item()
            loss_rcnn_cls = RCNN_loss_cls.mean().item()
            loss_rcnn_twin = RCNN_loss_twin.mean().item()
            fg_cnt = torch.sum(rois_label.data.ne(0))
            bg_cnt = rois_label.data.numel() - fg_cnt
            gt_cnt = num_gt.sum().item()

            print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                                    % (session, epoch, step+1, len(dataloader), loss_temp, lr))
            print("\t\t\tfg/bg=(%d/%d), gt_twins: %d, time cost: %f" % (fg_cnt, bg_cnt, gt_cnt, end-start))
            print("\t\t\trpn_cls: %.4f, rpn_twin: %.4f, rcnn_cls: %.4f, rcnn_twin %.4f" \
                          % (loss_rpn_cls, loss_rpn_twin, loss_rcnn_cls, loss_rcnn_twin))
            print("one step data time: %.4f" % (data_time))

            loss_temp = 0
            start = time.time()
        data_start = time.time()

    end = time.time()
    print(end - start)


if __name__ == '__main__':
    imdb_name = "train_data_25fps_flipped.pkl"
    num_classes = 21
    set_cfgs = ['ANCHOR_SCALES', '[2,4,5,6,8,9,10,12,14,16]', 'NUM_CLASSES', num_classes]

    cfg_file = "cfgs/{}_{}.yml".format(net, dataset)

    cfg.CUDA = True
    cfg.USE_GPU_NMS = True

    if cfg_file is not None:
        cfg_from_file(cfg_file)
    if set_cfgs is not None:
        cfg_from_list(set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)
    # for reproduce
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(cfg.RNG_SEED)

    cudnn.benchmark = True

    # train set
    roidb_path = roidb_dir + "/" + dataset + "/" + imdb_name
    roidb = get_roidb(roidb_path)   # roidb是(13712,)大小的列表,每个元素形如{'frames': array([[  0,   0, 768,   1]]), 'fg_name': '/home/tx/Dataset/tx/THUMOS14/val/video_validation_0000934', 'flipped': False, 'durations': array([30.]), 'bg_name': '/home/tx/Dataset/tx/THUMOS14/val/video_validation_0000934', 'max_classes': array([18.]), 'gt_classes': array([18.]), 'wins': array([[235., 265.]]), 'max_overlaps': array([1.])}
    print('{:d} roidb entries'.format(len(roidb)))
    model_dir = save_dir + "/" + net + "/" + dataset
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    output_dir = output_dir + "/" + net + "/" + dataset
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # sampler_batch = sampler(train_size, args.batch_size)
    dataset = roibatchLoader(roidb)#return data, gt_windows_padding, num_gt (data形状:[batch_size, 3, 512, 112, 112], gt_windows_padding形状:20行3列, num_gt = 1)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, num_workers=nw, shuffle=True)

    # initilize the network here.
    tdcnn_demo = c3d_tdcnn(pretrained=True)
    tdcnn_demo.create_architecture()    # 初始化特征提取网络

    print(tdcnn_demo)   # 加载了activitynet预训练网络参数的c3d网络?(再次查看:rpn+特征提取网络？)包括_RPN、RCNN_base、RCNN_top、RCNN_cls_score、RCNN_twin_pred等网络

    params = []
    for key, value in dict(tdcnn_demo.named_parameters()).items():  # (tx疑问:具体后面要用这个param列表做啥还不知道)
        if value.requires_grad:  # 除了RCNN_base层的前两层卷积层(6层)的参数为False外,RCNN_base层的其余三层卷积层都为True;然后_RPN、RCNN_top、RCNN_cls_score、RCNN_twin_pred的参数都为True
            print(key)
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                    'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    if resume:   # resume需要人工设置default = True or False
        load_name = os.path.join(model_dir, 'tdcnn_{}_{}_{}.pth'.format(checksession, checkepoch, checkpoint))#这三个值都需要人工设置
        checkpoint = torch.load(load_name)
        session = checkpoint['session']
        start_epoch = checkpoint['epoch'] + 1
        tdcnn_demo.load_state_dict(checkpoint['model'])
        optimizer_tmp = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)
        optimizer_tmp.load_state_dict(checkpoint['optimizer'])
        lr = optimizer_tmp.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
            print("loaded checkpoint %s" % (load_name))

    if torch.cuda.is_available():
        tdcnn_demo = tdcnn_demo.cuda()
        # tdcnn_demo, optimizer = amp.initialize(tdcnn_demo, optimizer, opt_level="O1")
        tdcnn_demo = nn.parallel.DataParallel(tdcnn_demo, device_ids = gpus)

    for epoch in range(start_epoch, max_epochs + 1):
        if epoch % (lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, lr_decay_gamma)
            lr *= lr_decay_gamma

        train_net(tdcnn_demo, dataloader, optimizer, lr, epoch, disp_interval, session)

        if len(gpus) > 1:
            save_name = os.path.join(model_dir, 'tdcnn_{}_{}_{}.pth'.format(session, epoch, len(dataloader)))
            save_checkpoint({
                'session': session,
                'epoch': epoch,
                'model': tdcnn_demo.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'pooling_mode': cfg.POOLING_MODE
            }, save_name)
        else:
            save_name = os.path.join(model_dir, 'tdcnn_{}_{}_{}.pth'.format(session, epoch, len(dataloader)))
            save_checkpoint({
                'session': session,
                'epoch': epoch,
                'model': tdcnn_demo.state_dict(),
                'optimizer': optimizer.state_dict(),
                'pooling_mode': cfg.POOLING_MODE
            }, save_name)
        print('save model: {}'.format(save_name))
