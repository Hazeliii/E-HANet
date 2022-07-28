from __future__ import print_function, division
import sys
sys.path.append('core')
from pathlib import Path
import argparse
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from model import bam_split
from loader.train_loader_dsec import *
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from utils.helper_functions import move_dict_to_cuda
from pathlib import *
os.environ["CUDA_VISIBLE_DEVICES"] = "3, 2"  # 4和5

MAX_FLOW = 400
SUM_FREQ = 50
VAL_FREQ = 5000


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def dataloading_demo(dataloader, batch_size, visualize=True):
    with torch.no_grad():
        for data in dataloader:
            if batch_size == 1 and visualize:
                disp = data['disparity_gt'].numpy.squeeze()


def fetch_optimizer(args, model):
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
    return optimizer, scheduler


def sequence_loss(flow_pred, flow_gt, valid, gamma=0.8, smooth_weight=0.5):
    # flow_gt.shape torch.Size([4, 480, 640, 2]) flow_pred list,里面12个元素 每个：torch.Size([4, 2, 480, 640])
    # 则需要对flow_pred的shape进行转换
    # flow_gt = flow_gt.permute(0, 3, 1, 2)
    n_predictions = len(flow_pred)
    flow_loss = 0.0
    smooth_loss = 0.0

    # 去掉无效像素点和特别大的光流
    mask = (valid > 0)  # [4, 480, 640]
    '''
    print('valid.shape:', valid.shape)
    print('mask.shape:', mask.shape)
    mask.shape: torch.Size([4, 480, 640])
    valid.shape: torch.Size([4, 480, 640])
    '''
    gt_masked = flow_gt[mask, :]

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        flow_pred_i = flow_pred[i].permute(0, 2, 3, 1)
        i_loss = (flow_pred_i - flow_gt).abs()
        # i_loss.shape: torch.Size([4, 480, 640, 2]) mask[:, :, :, None].shape: torch.Size([4, 480, 640, 1])
        flow_loss += i_weight * (mask[:, :, :, None] * i_loss).mean()
        # smooth_loss += compute_smooth_loss(flow_pred_i)
    # smooth_loss *= smooth_weight / float(n_predictions)

    # total_loss = flow_loss + smooth_loss

    flow_pred_last = flow_pred[-1].permute(0, 2, 3, 1)
    flow_last_masked = flow_pred_last[mask, :]
    EPE = torch.sum((gt_masked - flow_last_masked)**2, dim=-1).sqrt()
    AEE = EPE.mean().item()
    # print('EE.shape:', EPE.mean().shape) torch.Size([])
    n_points = flow_last_masked.shape[0]

    thresh = 3
    percent_AEE = float((EPE < thresh).sum()) / float(n_points + 1e-5)
    outlier = 1. - percent_AEE

    metrics = {
        'supervised_loss': flow_loss,
        # 'smooth_loss': smooth_loss,
        # 'total_loss': total_loss,
        'AEE': AEE,
        'n_points': n_points,
        'outlier': outlier
    }
    return flow_loss, metrics


class Logger:
    def __init__(self, model, scheduler, path:Path):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 50000
        self.running_loss = {}
        self.writer = None
        self.train_epe_list = []
        self.train_steps_list = []
        self.val_epe_list = []
        self.val_step_list = []
        self.txt_file = path / 'metrics.txt'

    def _print_training_status(self):
        # metrics_data = [torch.mean(self.running_loss[k]) for k in sorted(self.running_loss.keys())]
        metrics_data = [np.sum(self.running_loss[k])/ SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps + 1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:^10.4f}, " * len(metrics_data)).format(*metrics_data)

        # print the training status
        print('setps\t scheduler.get_last_lr')
        print(training_str)
        print('{:^10}{:^10}{:^10}{:^10}'.
              format('AEE', 'n_points', 'outlier',  'loss'))
        print(metrics_str)

        with self.txt_file.open(mode='a+') as file:
            file.write(metrics_str)
            file.write('\n')

        # logging running loss to total loss
        self.train_epe_list.append(np.mean(self.running_loss['AEE']))
        self.train_steps_list.append(self.total_steps)

        for k in self.running_loss:
            self.running_loss[k] = []

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = []

            self.running_loss[key].append(metrics[key])

        if self.total_steps % SUM_FREQ == SUM_FREQ - 1:  # 每50次打印一次结果
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def plot_train(logger, args, total_steps):
    # plot training plot
    plt.figure()
    plt.plot(logger.train_steps_list, logger.train_epe_list)
    plt.xlabel('x_steps')
    plt.ylabel('EPE')
    plt.title('Running training error (EPE)')
    plt.savefig(args.output+'/train_epe_%d.png'%total_steps, bbox_inches='tight')
    plt.close()


def train(args):
    model = nn.DataParallel(bam_split.ERAFT(args, n_first_channels=args.num_voxel_bins).cuda(), device_ids=args.gpus)
    # model = ERAFT(args, n_first_channels=args.num_voxel_bins)
    print('Parameters Count:%d' % count_parameters(model))

    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)

    # model.cuda()
    model.train()   # 作用是启用batch normalization和drop out

    dsec_dir = Path(args.dataset)
    assert dsec_dir.is_dir()
    dataset_provider = DatasetProvider(dsec_dir)
    train_dataset = dataset_provider.get_train_dataset()

    batch_size = 4
    num_workers = 0
    epoch = 35
    total_steps = 50000

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              drop_last=True)
    optimizer, scheduler = fetch_optimizer(args, model)
    print('args.mixed_precision:', args.mixed_precision)
    scaler = GradScaler(enabled=args.mixed_precision)
    output_path = Path(args.output)
    logger = Logger(model, scheduler, output_path)

    should_keep_training = True
    # gpu = torch.device('cuda:'+str(args.gpus))
    print('The number of training samples:', len(train_loader))
    while should_keep_training:
        print("-----------------------epoch:{}-----------------------".format(epoch))
        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            gt_flow, valid_2D, vox_old, vox_new = [x.cuda() for x in data_blob]
            '''
            data_blob = move_dict_to_cuda(data_blob, gpu)
            vox_old = data_blob['event_volume_old']
            vox_new = data_blob['event_volume_new']
            gt_flow = data_blob['gt_flow']
            valid_2D = data_blob['valid_2D']
            '''

            # 每一次迭代都有一次flow_pre,则flow_pre为n*b*w*h*2
            delta_f, flow_prediction = model(vox_old, vox_new)

            loss, metrics = sequence_loss(flow_prediction, gt_flow, valid_2D, smooth_weight=args.smoothness_weight)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
            # 梯度裁剪原理：既然在BP过程中会产生梯度消失（就是偏导无限接近0，导致长时记忆无法更新），那么最简单粗暴的方法，设定阈值，当梯度小于阈值时，更新的梯度为阈值

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            logger.push(metrics)

            if total_steps % VAL_FREQ == VAL_FREQ - 1:
                # 每5000次保存一次模型
                PATH = '%s/%s_%d.pth' % (args.output, args.name, total_steps+1)
                torch.save(model.state_dict(), PATH)
                plot_train(logger, args, total_steps)
                print('The checkpoint is saved to %s .'% PATH)

                results = {}
                # validation

                logger.write_dict(results)
                model.train()

            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break
        epoch += 1

    logger.close()
    PATH = '%s/%s.pth' % (args.output, args.name)
    torch.save(model.state_dict(), PATH)
    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='bam_motion', help="name your experiment")
    parser.add_argument('--dataset', type=str, default='/storage1/dataset/DSEC')
    parser.add_argument('--validation', type=str, nargs='+')  # 表示参数可设置一个或多个
    parser.add_argument('--output', default='checkpoints_dsec')

    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--batch_size', default=4)
    parser.add_argument('--num_voxel_bins', default=15)
    parser.add_argument('--num_steps', default=200000)

    parser.add_argument('--iters', default=12)
    parser.add_argument('--lr', type=float, default=0.0004)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--smoothness_weight', type=float, default=0.5,
                        help='Weight for the smoothness term in the loss function.')

    parser.add_argument('--num_heads', default=4, help='number of heads in attention and aggregation')
    parser.add_argument('--position_only', default=False, action='store_false',
                        help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=True, action='store_true',
                        help='use position and content-wise attention')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    # parser.add_argument('--gpus', default=[0, 3])
    parser.add_argument('--gpus', default=[0, 1])
    parser.add_argument('--type', default='standard', help='warm_start or standard' )
    parser.add_argument('--corr_levels', default=4)
    parser.add_argument('--corr_radius', default=4)
    args = parser.parse_args()
    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    print('batch_size={}, num_heads={}'.format(args.batch_size, args.num_heads))
    # --restore_ckpt checkpoints_dsec/bam_motion_50000.pth
    train(args)






