from __future__ import print_function, division
import sys
import argparse
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from model import bam_split
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from loader.train_loader_mvsec import *
from utils import image_utils
sys.path.append('core')

os.environ["CUDA_VISIBLE_DEVICES"] = "1, 0"  # 6和5

MAX_FLOW = 400
SUM_FREQ = 50
VAL_FREQ = 5000


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps + 100,
                                              pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
    return optimizer, scheduler


def charbonnier_loss(delta, alpha=0.45, epsilon=1e-3):
    '''
    p(x) = (x^2 + e^2)^a
    '''
    loss = torch.mean(torch.pow((delta ** 2 + epsilon ** 2), alpha))
    return loss


def compute_smooth_loss(flow):
    '''
    flow:B*w*h*2
    参考ev-flow中的中的smooth loss
    选取每个像素点四周的8个相邻像素
    '''
    flow_ucrop = flow[..., 1:]
    flow_dcrop = flow[..., :-1]
    flow_lcrop = flow[..., 1:, :]
    flow_rcrop = flow[..., :-1, :]

    flow_ulcrop = flow[..., 1:, 1:]
    flow_drcrop = flow[..., :-1, :-1]
    flow_dlcrop = flow[..., :-1, 1:]
    flow_urcrop = flow[..., 1:, :-1]

    smooth_loss = charbonnier_loss(flow_lcrop - flow_rcrop) \
                  + charbonnier_loss(flow_ucrop - flow_dcrop) \
                  + charbonnier_loss(flow_ulcrop - flow_drcrop) \
                  + charbonnier_loss(flow_dlcrop - flow_urcrop)
    smooth_loss /= 4
    return smooth_loss


def sequence_loss(flow_pred, flow_gt, valid, gamma=0.8):
    """
    supervised loss
    @param flow_pred: list,里面12个元素 每个：torch.Size([B, 2, H, W]) 则需要对flow_pred的shape进行转换
    @param flow_gt: torch.Size([B, 2, H, W]) 则需要对shape进行转换->(B,H,W,2)
    @param valid: gt_mask, torch.Size([B, 256, 256])
    @param gamma:
    @param smooth_weight:
    @return:
    """
    flow_gt = flow_gt.permute(0, 2, 3, 1)
    n_predictions = len(flow_pred)
    flow_loss = 0.0

    # 去掉无效像素点和特别大的光流
    mask = (valid > 0)  # [4, H, W]
    gt_masked = flow_gt[mask, :]
    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        flow_pred_i = flow_pred[i].permute(0, 2, 3, 1)
        i_loss = (flow_pred_i - flow_gt).abs()
        flow_loss += i_weight * (mask[:, :, :, None] * i_loss).mean()
    '''
    print(flow_loss)
    print(type(flow_loss))
    
    tensor(15.3385, device='cuda:0', dtype=torch.float64, grad_fn=<AddBackward0>)
    <class 'torch.Tensor'>
    '''

    flow_pred_last = flow_pred[-1].permute(0, 2, 3, 1)
    flow_last_masked = flow_pred_last[mask, :]
    EPE = torch.sum((gt_masked - flow_last_masked) ** 2, dim=-1).sqrt()
    AEE = EPE.mean().item()
    # print('EE.shape:', EPE.mean().shape) torch.Size([])
    n_points = flow_last_masked.shape[0]

    thresh = 3
    percent_AEE = float((EPE < thresh).sum()) / float(n_points + 1e-5)
    outlier = 1. - percent_AEE

    metrics = {
        'AEE': AEE,
        'n_points': n_points,
        'outlier': outlier,
        'loss': flow_loss,
    }
    # oroder:AEE  loss n_points outlier
    return flow_loss, metrics


class Logger:
    def __init__(self, model, scheduler, path: Path):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None
        self.train_epe_list = []
        self.train_steps_list = []
        self.val_epe_list = []
        self.val_step_list = []
        self.txt_file = path / 'bam_voxel_metrics.txt'

    def _print_training_status(self):
        # metrics_data = [torch.mean(self.running_loss[k]) for k in sorted(self.running_loss.keys())]
        metrics_data = [np.sum(self.running_loss[k]) / SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps + 1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:^10.4f}, " * len(metrics_data)).format(*metrics_data)

        # print the training status
        print('setps\t scheduler.get_last_lr')
        print(training_str)
        '''
        unsupervised
        print('{:^10}{:^10}{:^10}{:^10}{:^15}{:^10}'.
              format('AEE', 'loss', 'n_points', 'outlier', 'smooth_loss', 'warp_loss'))
        '''
        print('{:^10}{:^10}{:^10}{:^10}'.
              format('AEE', 'loss', 'n_points', 'outlier'))
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

        if self.total_steps % SUM_FREQ == SUM_FREQ - 1:  # 每100次打印一次结果
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
    plt.savefig(args.output + '/train_epe_%d.png' % total_steps, bbox_inches='tight')
    plt.close()


def train(args):
    config_path = args.config_path
    filters_pth = args.filters_json
    config = json.load(open(config_path))
    filters = json.load(open(filters_pth))

    num_bins = config['data_loader']['train']['args']['num_voxel_bins']
    print('num_bins is ', num_bins)
    model = nn.DataParallel(bam_split.ERAFT(args, num_bins*2).cuda(),
                            device_ids=args.gpus)

    # model = ERAFT(args, n_first_channels=args.num_voxel_bins)
    print('Parameters Count:%d' % count_parameters(model))

    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)

    # model.cuda()
    model.train()  # 作用是启用batch normalization和drop out

    MVSEC_dir = Path(args.dataset)
    assert MVSEC_dir.is_dir()
    if args.type == 'standard':
        train_set = MvsecFlow(
            args=config["data_loader"]["train"]["args"],
            filters=filters,
            type='train',
            path=args.dataset
        )
    elif args.type == 'warm_start':
        train_set = MvsecFlowRecurrent(
            args=config["data_loader"]["train"]["args"],
            filters=filters,
            type='train',
            path=args.dataset
        )
    else:
        raise NotImplementedError

    batch_size = 4
    num_workers = 0
    train_set_loader = DataLoader(train_set, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers,
                                  drop_last=True)

    optimizer, scheduler = fetch_optimizer(args, model)
    total_steps = 0
    epochs = 0

    print('args.mixed_precision:', args.mixed_precision)
    print('args.type:', args.type)

    scaler = GradScaler(enabled=args.mixed_precision)
    output_path = Path(args.output)
    logger = Logger(model, scheduler, output_path)

    should_keep_training = True
    # gpu = torch.device('cuda:'+str(args.gpus))
    while should_keep_training:
        epochs += 1
        print('-------------------------epoch:{}-------------------------'.format(epochs))
        if args.type == 'standard':
            for i_batch, data_blob in enumerate(train_set_loader):
                optimizer.zero_grad()
                gt_flow, valid_2D, vox_old, vox_new = data_blob['flow'].cuda(), data_blob['gt_valid_mask'].cuda(), \
                                                data_blob['event_volume_new'].cuda(), data_blob['event_volume_old'].cuda()
                delta_f, flow_prediction = model(vox_old, vox_new)
                loss, metrics = sequence_loss(flow_prediction, gt_flow, valid_2D)
                # loss, metrics = motion_compensation_loss(flow_prediction, events_list, gt_flow, valid_2D)
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
                    PATH = '%s/%d_%s.pth' % (args.output, total_steps + 1, args.name)
                    torch.save(model.state_dict(), PATH)
                    plot_train(logger, args, total_steps)
                    print('The checkpoint is saved to %s .' % PATH)
                    results = {}
                    # validation
                    logger.write_dict(results)
                    model.train()

                total_steps += 1
                if total_steps > args.num_steps:
                    should_keep_training = False
                    break
            epochs += 1

        elif args.type == 'warm_start':
            for i_batch, sequences in enumerate(train_set_loader):
                optimizer.zero_grad()
                init_flow = None
                for l in range(len(sequences)):
                    sample = sequences[l]
                    gt_flow, valid_2D, vox_old, vox_new = sample['flow'].cuda(), sample['gt_valid_mask'].cuda(), \
                                                    sample['event_volume_new'].cuda(), sample['event_volume_old'].cuda()
                    delta_f, flow_prediction = model(vox_old, vox_new, flow_init=init_flow)
                    init_flow = image_utils.forward_interpolate_pytorch(delta_f)
                    loss, metrics = sequence_loss(flow_prediction, gt_flow, valid_2D)
                    # loss, metrics = motion_compensation_loss(flow_prediction, events_list, gt_flow, valid_2D)
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
                    PATH = '%s/%d_%s.pth' % (args.output, total_steps + 1, args.name)
                    torch.save(model.state_dict(), PATH)
                    plot_train(logger, args, total_steps)
                    print('The checkpoint is saved to %s .' % PATH)

                    results = {}
                    # validation

                    logger.write_dict(results)
                    model.train()

                total_steps += 1

                if total_steps > args.num_steps:
                    should_keep_training = False
                    break

    logger.close()
    PATH = '%s/%s.pth' % (args.output, args.name)
    torch.save(model.state_dict(), PATH)
    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='se_bam_voxel', help="name your experiment")
    parser.add_argument('--dataset', type=str, default='/storage1/dataset/MVSEC/mvsec_45HZ')
    parser.add_argument('--validation', type=str, nargs='+')  # 表示参数可设置一个或多个
    parser.add_argument('--output', default='checkpoints_mvsec_outdoor2')
    parser.add_argument('--config_path', type=str, default='config/train_mvsec_45_outdoor.json')
    parser.add_argument('--filters_json', type=str, default='config/mvsec_train_outdoor2_filter.json')

    parser.add_argument('--restore_ckpt', help="restore checkpoint", type=str)
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
    parser.add_argument('--position_and_content', default=False, action='store_true',
                        help='use position and content-wise attention')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    # parser.add_argument('--gpus', default=[0, 3])
    parser.add_argument('--gpus', default=[0, 1])
    parser.add_argument('--type', default='standard', help='warm_start or standard')
    parser.add_argument('--corr_levels', default=4)
    parser.add_argument('--corr_radius', default=4)
    args = parser.parse_args()
    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    print('batch_size={}, num_heads={}'.format(args.batch_size, args.num_heads))

    train(args)
