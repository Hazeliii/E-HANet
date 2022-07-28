import os

import numpy as np
import torch as th
from torchvision import utils
from utils.helper_functions import *
import utils.visualization as visualization
import utils.filename_templates as TEMPLATES
import utils.helper_functions as helper
import utils.logger as logger
from utils import image_utils
from pathlib import Path
from torchvision import transforms
MAX_FLOW = 400


def sequence_loss(flow_pred, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    # flow_gt.shape torch.Size([4, 480, 640, 2]) flow_pred list,里面12个元素 每个：torch.Size([4, 2, 480, 640])
    # 则先对flow_gt的shape进行转换
    flow_gt = flow_gt.permute(0, 2, 3, 1)

    # 去掉无效像素点和特别大的光流
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    # mag.shape torch.Size([4, 480, 640])
    mask0 = valid >= 0.5
    mask1 = mag < max_flow  # mask: torch.Size([4, 480, 640])
    valid = mask0 & mask1

    epe = torch.sum((flow_pred - flow_gt)**2, dim=1).sqrt()  # 逐像素误差
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }
    return metrics


def flow_error_dense(flow_gt, flow_pred, valid, dataset):
    # flow_gt.shape torch.Size([4, 480, 640, 2])
    # flow_pred list,里面12个元素 每个：torch.Size([4, 2, 480, 640]) => [4, 480, 640, 2]
    if dataset == 'mvsec_45Hz':
        flow_gt = flow_gt.permute(0, 2, 3, 1)

    flow_pred= flow_pred.permute(0, 2, 3, 1)
    mask = (valid > 0)    #选出ground truth中有光流的数据
    # print('mask.shape:', mask.shape)
    gt_masked = flow_gt[mask, :]
    pre_masked = flow_pred[mask, :]
    ee = np.linalg.norm(gt_masked.cpu() - pre_masked.cpu(), axis=-1)
    n_points = ee.shape[0]
    aee = np.mean(ee)
    thresh = 3.
    precent_AEE = float((ee < thresh).sum()) / float(ee.shape[0] + 1e-5)
    outlier = 1. - precent_AEE
    PE1 = float((ee > 1.).sum()) / float(ee.shape[0] + 1e-5)
    PE2 = float((ee > 2.).sum()) / float(ee.shape[0] + 1e-5)
    metrics = {
        'AEE': aee,
        'outlier': outlier,
        'n_points': n_points,
        'PE1': PE1,
        'PE3': PE2
    }
    return metrics


class Logger:
    def __init__(self, path):
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None
        self.train_epe_list = []
        self.train_steps_list = []
        self.txt_file = Path(path) / 'metrics_mvsec_20HZ.txt'

    def _print_training_status(self):
        metrics_data = [np.mean(self.running_loss[k]) for k in sorted(self.running_loss.keys())]
        metrics_str = ("{:^10.4f}, " * len(metrics_data)).format(*metrics_data)

        # print the training status
        # print('epe|\t 1px|\t 2px|\t 3px|\t 5px')
        print('{:^10}{:^10}{:^10}{:^10}{:^10}'.format('AEE', '1PE', '3PE', 'n_points', 'outlier', ))
        print(metrics_str)

        # logging running loss to total loss

        for k in self.running_loss:
            self.running_loss[k] = []

        with self.txt_file.open(mode='a+') as file:
            file.write(metrics_str)
            file.write('\n')

        self.train_epe_list.append(np.mean(self.running_loss['AEE']))

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = []

            self.running_loss[key].append(metrics[key])

        if self.total_steps % 10 == 0:  # 每10次打印一次结果
            self._print_training_status()
            self.running_loss = {}
        return self.txt_file


class Test(object):
    """
    Test class

    """

    def __init__(self, model, config,
                 data_loader, visualizer, test_logger=None, save_path=None, additional_args=None):
        self.downsample = False # Downsampling for Rebuttal
        self.model = model
        self.config = config
        self.data_loader = data_loader
        self.additional_args = additional_args
        self.name = config['name']
        self.cropper = transforms.CenterCrop((256, 256))
        if config['cuda'] and not torch.cuda.is_available():
            print('Warning: There\'s no CUDA support on this machine, '
                                'training is performed on CPU.')
        else:
            self.gpu = torch.device('cuda:' + str(config['gpu']))
            self.model = self.model.to(self.gpu)
        if save_path is None:
            self.save_path = helper.create_save_path(config['save_dir'].lower(),
                                           config['name'].lower())
        else:
            self.save_path=save_path
        if logger is None:
            self.logger = logger.Logger(self.save_path)
        else:
            self.logger = test_logger
        if isinstance(self.additional_args, dict) and 'name_mapping_test' in self.additional_args.keys():
            visu_add_args = {'name_mapping' : self.additional_args['name_mapping_test']}
        else:
            visu_add_args = None
        self.visualizer = visualizer(data_loader, self.save_path, additional_args=visu_add_args)

    def summary(self):
        self.logger.write_line("====================================== TEST SUMMARY ======================================", True)
        self.logger.write_line("Model:\t\t\t" + self.model.__class__.__name__, True)
        self.logger.write_line("Tester:\t\t" + self.__class__.__name__, True)
        self.logger.write_line("Test Set:\t" + self.data_loader.dataset.__class__.__name__, True)
        self.logger.write_line("\t-Dataset length:\t"+str(len(self.data_loader)), True)
        self.logger.write_line("\t-Batch size:\t\t" + str(self.data_loader.batch_size), True)
        self.logger.write_line("==========================================================================================", True)

    def run_network(self, epoch):
        raise NotImplementedError

    def move_batch_to_cuda(self, batch):
        raise NotImplementedError

    def visualize_sample(self, batch):
        self.visualizer(batch)

    def visualize_sample_dsec(self, batch, batch_idx):
        self.visualizer(batch, batch_idx, None)

    def get_estimation_and_target(self, batch):
        # Returns the estimation and target of the current batch
        raise NotImplementedError

    def mvsec_test(self):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        pre_flow_list = list()
        gt_flow_list = list()
        events_list = list()
        logger = Logger(self.save_path)
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.data_loader):
                # Move Data to GPU
                if next(self.model.parameters()).is_cuda:
                    batch = self.move_batch_to_cuda(batch)
                # Network Forward Pass
                self.run_network(batch)
                print("Sample {}/{}".format(batch_idx + 1, len(self.data_loader)))

                # Visualize
                '''
                if hasattr(batch, 'keys') and 'loader_idx' in batch.keys() \
                        or (isinstance(batch,list) and hasattr(batch[0], 'keys') and 'loader_idx' in batch[0].keys()):
                    self.visualize_sample(batch)
                else:
                    # DSEC Special Snowflake
                    self.visualize_sample_dsec(batch, batch_idx)
                    #print('Not Visualizing')
                '''

                # epe
                flow_gt = batch['flow_gt']
                valid_2D = batch['valid_2D']
                flow_est = batch['flow_est']
                events = batch['events']
                # metrics = sequence_loss(flow_est, flow_gt, valid_2D)
                metrics = flow_error_dense(flow_gt, flow_est, valid_2D, 'mvsec_45Hz')
                pre_flow_list.append(flow_est)
                gt_flow_list.append(flow_gt)
                events_list.append(events)
                metrics_txt_pth = logger.push(metrics)

        return pre_flow_list, gt_flow_list, events_list, metrics_txt_pth

    def dsec_test(self):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        pre_flow_list = list()
        gt_flow_list = list()
        events_list = list()
        logger = Logger(self.save_path)
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.data_loader):
                # Move Data to GPU
                if next(self.model.parameters()).is_cuda:
                    batch = self.move_batch_to_cuda(batch)
                # Network Forward Pass
                self.run_network(batch)
                print("Sample {}/{}".format(batch_idx + 1, len(self.data_loader)))

                # Visualize
                '''
                if hasattr(batch, 'keys') and 'loader_idx' in batch.keys() \
                        or (isinstance(batch,list) and hasattr(batch[0], 'keys') and 'loader_idx' in batch[0].keys()):
                    self.visualize_sample(batch)
                else:
                    # DSEC Special Snowflake
                    self.visualize_sample_dsec(batch, batch_idx)
                    #print('Not Visualizing')
                '''
                # epe
                flow_gt = batch['flow_gt']
                valid_2D = batch['valid_2D']
                flow_est = batch['flow_est']

                # metrics = sequence_loss(flow_est, flow_gt, valid_2D)
                metrics = flow_error_dense(flow_gt, flow_est, valid_2D, 'dsec')
                pre_flow_list.append(flow_est)
                gt_flow_list.append(flow_gt)
                metrics_txt_pth = logger.push(metrics)

        return pre_flow_list, events_list, metrics_txt_pth

    def _test(self):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.data_loader):
                # Move Data to GPU
                if next(self.model.parameters()).is_cuda:
                    batch = self.move_batch_to_cuda(batch)
                # Network Forward Pass
                self.run_network(batch)
                print("Sample {}/{}".format(batch_idx + 1, len(self.data_loader)))

                # Visualize
                if hasattr(batch, 'keys') and 'loader_idx' in batch.keys() \
                        or (isinstance(batch, list) and hasattr(batch[0], 'keys') and 'loader_idx' in batch[0].keys()):
                    self.visualize_sample(batch)
                else:
                    # DSEC Special Snowflake
                    self.visualize_sample_dsec(batch, batch_idx)
                    # print('Not Visualizing')

        # Log Generation
        log = {}
        return log


class TestRaftEvents(Test):
    def move_batch_to_cuda(self, batch):
        return move_dict_to_cuda(batch, self.gpu)    
    
    def get_estimation_and_target(self, batch):
        if not self.downsample:
            if 'gt_valid_mask' in batch.keys():
                return batch['flow_est'].cpu().data, (batch['flow'].cpu().data, batch['gt_valid_mask'].cpu().data)
            return batch['flow_est'].cpu().data, batch['flow'].cpu().data
        else:
            f_est = batch['flow_est'].cpu().data
            f_gt = torch.nn.functional.interpolate(batch['flow'].cpu().data, scale_factor=0.5)
            if 'gt_valid_mask' in batch.keys():
                f_mask = torch.nn.functional.interpolate(batch['gt_valid_mask'].cpu().data, scale_factor=0.5)
                return f_est, (f_gt, f_mask)
            return f_est, f_gt

    def run_network(self, batch):
        # RAFT just expects two images as input. cleanest. code. ever.
        if not self.downsample:
                im1 = batch['event_volume_old']
                im2 = batch['event_volume_new']
        else:
            im1 = torch.nn.functional.interpolate(batch['event_volume_old'], scale_factor=0.5)
            im2 = torch.nn.functional.interpolate(batch['event_volume_new'], scale_factor=0.5)
        _, batch['flow_list'] = self.model(image1=im1,
                                           image2=im2)
        if self.name == 'mvsec_45Hz':
            batch['flow_est'] = self.cropper(batch['flow_list'][-1])
        else:
            batch['flow_est'] = batch['flow_list'][-1]


class TestRaftEventsWarm(Test):
    def __init__(self, model, config,
                 data_loader, visualizer, test_logger=None, save_path=None, additional_args=None):
        super(TestRaftEventsWarm, self).__init__(model, config,
                                                 data_loader, visualizer, test_logger, save_path,
                                                 additional_args=additional_args)
        self.subtype = config['subtype'].lower()
        print('Tester Subtype: {}'.format(self.subtype))
        self.net_init = None # Hidden state of the refinement GRU
        self.flow_init = None
        self.idx_prev = None
        self.init_print=False
        assert self.data_loader.batch_size == 1, 'Batch size for recurrent testing must be 1'

    def move_batch_to_cuda(self, batch):
        return move_list_to_cuda(batch, self.gpu)

    def get_estimation_and_target(self, batch):
        if not self.downsample:
            if 'gt_valid_mask' in batch[-1].keys():
                return batch[-1]['flow_est'].cpu().data, (batch[-1]['flow'].cpu().data, batch[-1]['gt_valid_mask'].cpu().data)
            return batch[-1]['flow_est'].cpu().data, batch[-1]['flow'].cpu().data
        else:
            f_est = batch[-1]['flow_est'].cpu().data
            f_gt = torch.nn.functional.interpolate(batch[-1]['flow'].cpu().data, scale_factor=0.5)
            if 'gt_valid_mask' in batch[-1].keys():
                f_mask = torch.nn.functional.interpolate(batch[-1]['gt_valid_mask'].cpu().data, scale_factor=0.5)
                return f_est, (f_gt, f_mask)
            return f_est, f_gt

    def visualize_sample(self, batch):
        self.visualizer(batch[-1])

    def visualize_sample_dsec(self, batch, batch_idx):
        self.visualizer(batch[-1], batch_idx, None)

    def check_states(self, batch):
        # 0th case: there is a flag in the batch that tells us to reset the state (DSEC)
        if 'new_sequence' in batch[0].keys():
            if batch[0]['new_sequence'].item() == 1:
                self.flow_init = None
                self.net_init = None
                self.logger.write_line("Resetting States!", True)
        else:
            # During Validation, reset state if a new scene starts (index jump)
            if self.idx_prev is not None and batch[0]['idx'].item() - self.idx_prev != 1:
                self.flow_init = None
                self.net_init = None
                self.logger.write_line("Resetting States!", True)
            self.idx_prev = batch[0]['idx'].item()

    def run_network(self, batch):
        self.check_states(batch)
        for l in range(len(batch)):
            # Run Recurrent Network for this sample

            if not self.downsample:
                im1 = batch[l]['event_volume_old']
                im2 = batch[l]['event_volume_new']
            else:
                im1 = torch.nn.functional.interpolate(batch[l]['event_volume_old'], scale_factor=0.5)
                im2 = torch.nn.functional.interpolate(batch[l]['event_volume_new'], scale_factor=0.5)
            flow_low_res, batch[l]['flow_list'] = self.model(image1=im1,
                                                                image2=im2,
                                                                flow_init=self.flow_init)

        batch[l]['flow_est'] = batch[l]['flow_list'][-1]
        self.flow_init = image_utils.forward_interpolate_pytorch(flow_low_res)
        batch[l]['flow_init'] = self.flow_init
