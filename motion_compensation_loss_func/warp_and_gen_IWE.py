import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class Warp_MVSEC_events:
    def __init__(self, events, flow, device, t_ref=None):
        # print('In class Warp_MVSEC_events .')
        ts = torch.as_tensor(events[:, 0], device=device)
        xs = torch.as_tensor(events[:, 1] - 45., device=device)
        ys = torch.as_tensor(events[:, 2] - 2., device=device)
        ps = torch.as_tensor(events[:, 3], device=device)

        crop_mask_x = ((xs >= 0) & (xs <= 255))
        crop_mask_y = ((ys >= 0) & (ys <= 255))
        self.crop_mask = torch.as_tensor(crop_mask_x * crop_mask_y, device=device)

        self.ts = torch.masked_select(ts, self.crop_mask)
        self.xs = torch.masked_select(xs, self.crop_mask)
        self.ys = torch.masked_select(ys, self.crop_mask)
        self.ps = torch.masked_select(ps, self.crop_mask)
        self.flow = flow

        if len(xs.shape) > 1:
            self.ts, self.xs, self.ys, self.ps = self.ts.squeeze(), self.xs.squeeze(), self.ys.squeeze(), self.ps.squeeze()
        while len(self.flow.size()) < 4:
            self.flow = self.flow.unsqueeze(0)  # 2*h*w->1*2*h*w

        if t_ref is None:
            self.t_ref = ts[0]
        # print('Done Warp_MVSEC_events init .')

    def warp_events(self):
        # print('In warp events .')
        if len(self.xs.size()) == 1:
            events_indices = torch.transpose(torch.stack((self.xs, self.ys), dim=0), 0, 1)  # N*2, 2 means (x, y)
        else:
            events_indices = torch.transpose(torch.stack((self.xs, self.ys), dim=1), 0, 1)
        events_indices = torch.reshape(events_indices, [1, 1, len(self.xs), 2])

        # Event indices need to be between -1 and 1 for F.gridsample
        events_indices[:, :, :, 0] = events_indices[:, :, :, 0] / (self.flow.shape[-1] - 1) * 2.0 - 1.0
        events_indices[:, :, :, 1] = events_indices[:, :, :, 1] / (self.flow.shape[-2] - 1) * 2.0 - 1.0
        # use float64 dtype
        flow_at_event = F.grid_sample((self.flow / 50.).double(), events_indices, align_corners=True)

        # print('flow_at_event.shape:', flow_at_event.shape) torch.Size([1, 2, 1, 9554])

        dt = (self.ts - self.t_ref).squeeze() / 1000.
        warped_xs = self.xs - flow_at_event[:, 0, :, :].squeeze() * dt
        warped_ys = self.ys - flow_at_event[:, 1, :, :].squeeze() * dt

        return warped_xs, warped_ys, self.ts, self.ps


class GenIwe:
    def __init__(self, xs, ys, ts, ps, using_polatity=False, sensor_size=(256, 256), clip_out_of_range=True,
                 interpolation='bilinear', padding=True, timestamp_reverse=False):
        '''
           :param xs:list of event x coordinates
           :param ys:list of event y coordinates
           :param ts:list of event timestamps
           :param ps:list of event polarities
           :param sensor_size:the size of the event sensor/output voxels
           :param clip_out_of_range: if the events go beyond the desired image size,
              clip the events to fit into the image
           :param interpolation: which interpolation to use. Options=None,'bilinear'
           :param padding:if bilinear interpolation, allow padding the image by 1 to allow events to fit
           :param timestamp_reverse:reverse the timestamps of the events, for backward warp
        '''
        self.xs = xs
        self.ys = ys
        self.ts = ts
        self.ps = ps
        self.using_polatity = using_polatity
        self.sensor_size = sensor_size
        self.clip_out_of_range = clip_out_of_range
        self.interpolation = interpolation
        self.padding = padding
        self.device = xs.device
        self.timestamp_reverse= timestamp_reverse

    def events_to_timestamp_image(self):
        '''
        Method to generate the average timestamp images from 'Zhu19, Unsupervised Event-based Learning
        of Optical Flow, Depth, and Egomotion'.
        Returns
        -------
       img_pos: timestamp image of the positive events
       img_neg: timestamp image of the negative events
        '''
        ts, xs, ys, ps = self.ts.squeeze(), self.xs.squeeze(), self.ys.squeeze(), self.ps.squeeze()
        if self.padding:
            img_size = (self.sensor_size[0] + 1, self.sensor_size[1] + 1)
        else:
            img_size = self.sensor_size

        zero_v = torch.tensor([0.], device=self.device)
        ones_v = torch.tensor([1.], device=self.device)
        mask = torch.ones(xs.size(), device=self.device)

        if self.clip_out_of_range:
            # print('clip_out_of_range')
            if self.interpolation is None and self.padding == False:
                # print('If .')
                clipx = img_size[1]
                clipy = img_size[0]
            else:
                clipx = img_size[1] - 1
                clipy = img_size[0] - 1
            mask = torch.where(xs >= clipx, zero_v, ones_v) * torch.where(ys >= clipy, zero_v, ones_v)

        pos_events_mask = torch.where(ps > 0, ones_v, zero_v)  # 1 for positive events else 0
        neg_events_mask = torch.where(ps <= 0, ones_v, zero_v)
        epsilon = 1e-6

        if self.timestamp_reverse:
            normalized_ts = ((-ts + ts[-1]) / (ts[-1] - ts[0] + epsilon)).squeeze()
        else:
            normalized_ts = ((ts - ts[0]) / (ts[-1] - ts[0] + epsilon)).squeeze()

        pxs = xs.floor().float()
        dxs = (xs - pxs).float()
        pys = ys.floor().float()
        dys = (ys - pys).float()
        pxs = (pxs * mask).long()
        pys = (pys * mask).long()

        pos_weighted = (normalized_ts * pos_events_mask).float()  # 选出positive时间对应的normalized timestamp
        neg_weighted = (normalized_ts * neg_events_mask).float()

        img_pos = torch.zeros(img_size).to(self.device)
        img_pos_cnt = torch.ones(img_size).to(self.device)  # 在1的基础上进行加？
        img_neg = torch.zeros(img_size).to(self.device)
        img_neg_cnt = torch.ones(img_size).to(self.device)

        self.interpolate_to_img(pxs, pys, dxs, dys, pos_weighted, img_pos)
        self.interpolate_to_img(pxs, pys, dxs, dys, pos_events_mask, img_pos_cnt)
        self.interpolate_to_img(pxs, pys, dxs, dys, neg_weighted, img_neg)
        self.interpolate_to_img(pxs, pys, dxs, dys, neg_events_mask, img_neg_cnt)

        # Avoid division by 0
        img_pos_cnt[img_pos_cnt == 0] = 1
        img_neg_cnt[img_neg_cnt == 0] = 1
        # print('img_pos_cnt.shape:', img_pos_cnt.shape)   torch.Size([257, 257])

        img_pos = img_pos.div(img_pos_cnt)
        img_neg = img_neg.div(img_neg_cnt)
        return img_pos, img_neg

    def interpolate_to_img(self, pxs, pys, dxs, dys, weights, img):
        """
        Accumulate x and y coords to an image using bilinear interpolation
        （pxs,pys）————-(pxs+1,pys)
          |             |
          |  （xs,ys）   |
          ——————————------
        """
        img.index_put_((pys, pxs), weights * (1.0 - dxs) * (1.0 - dys), accumulate=True)
        img.index_put_((pys, pxs + 1), weights * dxs * (1.0 - dys), accumulate=True)
        img.index_put_((pys + 1, pxs), weights * (1.0 - dxs) * dys, accumulate=True)
        img.index_put_((pys + 1, pxs + 1), weights * dxs * dys, accumulate=True)
        return img

    def events_to_count_img(self):
        '''
        generate count or polarity_based IWE, according to the parameter 'using_polatity'
        return:count_based img/ polarity_based img
        '''
        if self.padding:
            img_size = (self.sensor_size[0] + 1, self.sensor_size[1] + 1)
        else:
            img_size = self.sensor_size
        mask = torch.ones(self.xs.size(), device=self.device)
        if self.clip_out_of_range:
            zero_v = torch.tensor([0.], device=self.device)
            ones_v = torch.tensor([1.], device=self.device)
            if self.interpolation is None and self.padding == False:
                # print('If .')
                clipx = img_size[1]
                clipy = img_size[0]
            else:
                clipx = img_size[1] - 1
                clipy = img_size[0] - 1
            mask = torch.where(self.xs >= clipx, zero_v, ones_v) * torch.where(self.ys >= clipy, zero_v, ones_v)

        pxs = self.xs.floor().float()
        dxs = (self.xs - pxs).float()
        pys = self.ys.floor().float()
        dys = (self.ys - pys).float()
        pxs = (pxs * mask).long()
        pys = (pys * mask).long()
        counts = torch.ones(self.ps.size(), device=self.device)
        if self.using_polatity:
            masked_counts = self.ps * mask  # 使用极性
        else:
            masked_counts = counts * mask  # 不使用极性
        img = torch.zeros(img_size).to(self.device)
        self.interpolate_to_img(pxs, pys, dxs, dys, masked_counts, img)
        return img

