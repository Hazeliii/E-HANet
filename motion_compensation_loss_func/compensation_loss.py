import torch
import torch.nn.functional as F
from motion_compensation_loss_func.warp_and_gen_IWE import *
import numpy as np

'''
基于运动补偿的损失函数，包括
1.Squared average timestamp images objective (Zhu et al, Unsupervised Event-based
        Learning of Optical Flow, Depth, and Egomotion, CVPR19)
2.sobel_based_loss
3.Stoffregen et al, Event Cameras, Contrast Maximization and Reward Functions: an Analysis, CVPR19
'''


class motion_compensation_losses:
    def __init__(self, events, flow_pre, device):
        # print('In class motion_compensation_losses .')
        # self.warp_fun = Warp_MVSEC_events(events, flow_pre, device)
        self.warped_xs, self.warped_ys, self.ts, self.ps = Warp_MVSEC_events(events, flow_pre, device).warp_events()
        self.device = device

    def sobel_xy(self, im):
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype='float32')
        sobel_x = np.reshape(sobel_x, (1, 3, 3))

        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype='float32')
        sobel_y = np.reshape(sobel_y, (1, 3, 3))

        sobel = np.concatenate(
            [np.repeat(sobel_x, 1, axis=0).reshape((1, 1, 3, 3)),
             np.repeat(sobel_y, 1, axis=0).reshape((1, 1, 3, 3))],
            axis=0
        )
        # print('sobel.shape:', sobel.shape) sobel.shape: (2, 1, 3, 3)

        conv = nn.Conv2d(1, 2, kernel_size=3, padding=1, bias=False)
        conv.weight.data = torch.from_numpy(sobel)
        conv.cuda()
        edge = conv(Variable(im))
        # print('edge.shape:', edge.shape) edge.shape: torch.Size([4, 2, 256, 256])
        edge_x = edge[0, 0, :, :].squeeze()
        edge_y = edge[0, 1, :, :].squeeze()
        # print('edge_x.shape, edge_y.shape:', edge_x.shape, edge_y.shape) torch.Size([256, 256]) torch.Size([256, 256])
        return edge_x, edge_y

    def zhu_based_loss(self):
        '''
        Squared average timestamp images objective (Zhu et al, Unsupervised Event-based
        Learning of Optical Flow, Depth, and Egomotion, CVPR19)
        Loss given by g(x)^2*h(x)^2 where g(x) is image of average timestamps of positive events
        and h(x) is image of average timestamps of negative events.
        '''
        gen_func = GenIwe(self.warped_xs, self.warped_ys, self.ts, self.ps)
        img_pos, img_neg = gen_func.events_to_timestamp_image()
        loss = torch.sum(img_pos * img_pos) + torch.sum(img_neg * img_neg)
        return loss

    def sobel_based_loss(self):
        # 基于sobel算子的边缘强度函数作为损失函数，并将edge_x和edge_y分别归一化到[0,1]
        gen_func = GenIwe(self.warped_xs, self.warped_ys, self.ts, self.ps, using_polatity=False)
        events_count_img = gen_func.events_to_count_img()

        # 归一化到[0,255]
        max1 = torch.tensor(255.0, dtype=torch.float).cuda()
        img_max = torch.max(events_count_img)
        img_min = torch.min(events_count_img)
        norm_events_count_img = (max1 * (events_count_img - img_min) / (img_max - img_min))

        events_count_img = torch.unsqueeze(torch.unsqueeze(norm_events_count_img, dim=0), dim=0)  # (1, 1, H, W)
        edge_x, edge_y = self.sobel_xy(events_count_img)
        # edge_x, edge_y归一化到[0,1]
        max2 = torch.tensor(1.0, dtype=torch.float).cuda()
        edge_x_abs, edge_y_abs = torch.abs(edge_x), torch.abs(edge_y)
        x_max, x_min = torch.max(edge_x_abs), torch.min(edge_x_abs)
        y_max, y_min = torch.max(edge_y_abs), torch.min(edge_y_abs)
        norm_edge_x = (max2 * (edge_x_abs - x_min) / (x_max - x_min))
        norm_edge_y = (max2 * (edge_y_abs - y_min) / (y_max - y_min))
        print()
        edge = torch.mean(norm_edge_x + norm_edge_y)  # [0, 2]
        return torch.tensor(2., device=self.device) - edge  # min=0, max=2, the min the better

    def sos_objective(self, img):
        """
        Loss given by g(x)^2 where g(x) is IWE
        """
        sos = torch.mean(img * img)   # [0,1]
        return -sos

    def soe_objective(self, img):
        """
        Sum of exponentials objective (Stoffregen et al, Event Cameras, Contrast
         Maximization and Reward Functions: an Analysis, CVPR19)
         Loss given by e^g(x) where g(x) is IWE
        """
        exp = torch.exp(img)
        soe = torch.mean(exp)
        return -soe

    def moa_objective(self, img):
        """
        Max of accumulations objective
        Loss given by max(g(x)) where g(x) is IWE
        """
        moa = torch.max(img)
        return moa

    def isoa_objective(self, img, thresh=0.5):
        """
        Inverse sum of accumulations objective
        Loss given by sum(1 where g(x)>1 else 0) where g(x) is IWE.
        This formulation has similar properties to original ISoA, but negation makes derivative
        more stable than inversion.
        """
        isoa = torch.sum(torch.where(img > thresh, 1., 0.))
        return isoa

    def sosa_objective(self, img, p=3.):
        """
        Sum of Supressed Accumulations objective
        Loss given by e^(-p*g(x)) where g(x) is IWE. p is arbitrary shifting factor,
        higher values give better noise performance but lower accuracy.
        """
        exp = torch.exp(-p * img)
        sosa = torch.sum(exp)  # [32636.928, 65536] 256*256= 65536
        return -sosa

    def r1_objective(self, img, p=3.):
        """
        R1 objective (Stoffregen et al, Event Cameras, Contrast
        Maximization and Reward Functions: an Analysis, CVPR19)
         Loss given by SOS and SOSA combined
        """
        sos = self.sos_objective(img)
        sosa = self.sosa_objective(img, p)
        return -sos * sosa

    def contrast_max_based_loss(self):
        # print('In contrast_max_baed_loss.')
        gen_func = GenIwe(self.warped_xs, self.warped_ys, self.ts, self.ps)
        events_count_img = gen_func.events_to_count_img()
        # 归一化到[0,1]
        max1 = torch.tensor(1.0, dtype=torch.float).cuda()
        img_max = torch.max(events_count_img)
        img_min = torch.min(events_count_img)
        norm_events_count_img = (max1 * (events_count_img - img_min) / (img_max - img_min))
        return self.r1_objective(norm_events_count_img)  # [-65536,0]









