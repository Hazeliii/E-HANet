import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from train_MVSEC import sobel_xy


def warp_events_flow_torch(events, flow, device, t_ref=None):
    """
    warp events accords to the optical flow
    :param events: a set of events, [t, x, y, p]
    :param flow: 2D tensor containing the flow at each (x,y) position 2*h*w
    :param gt_flow: ([2, 256, 256])
    :param t0: the reference time to warp events to.If none, will use the timestamp of the first event
    :return warped_xs, warped_ys
    """

    ts = torch.as_tensor(events[:, 0], device=device)
    xs = torch.as_tensor(events[:, 1] - 45., device=device)
    ys = torch.as_tensor(events[:, 2] - 2., device=device)
    ps = torch.as_tensor(events[:, 3], device=device)

    # 只保留减去之后，x在【0，255】，y在【0，255】之内的事件
    crop_mask_x = ((xs >= 0) & (xs <= 255))
    crop_mask_y = ((ys >= 0) & (ys <= 255))
    crop_mask = torch.as_tensor(crop_mask_x * crop_mask_y, device=device)

    ts = torch.masked_select(ts, crop_mask)
    xs = torch.masked_select(xs, crop_mask)
    ys = torch.masked_select(ys, crop_mask)
    ps = torch.masked_select(ps, crop_mask)

    if len(xs.shape) > 1:
        ts, xs, ys, ps = ts.squeeze(), xs.squeeze(), ys.squeeze(), ps.squeeze()
    while len(flow.size())<4:
        flow = flow.unsqueeze(0)  # 2*h*w->1*2*h*w

    if t_ref is None:
        t_ref = ts[0]
    if len(xs.size()) == 1:
        events_indices = torch.transpose(torch.stack((xs, ys), dim=0), 0, 1)  # N*2, 2 means (x, y)
    else:
        events_indices = torch.transpose(torch.stack((xs, ys), dim=1), 0, 1)
    events_indices = torch.reshape(events_indices, [1, 1, len(xs), 2])

    # Event indices need to be between -1 and 1 for F.gridsample
    events_indices[:, :, :, 0] = events_indices[:, :, :, 0]/(flow.shape[-1]-1)*2.0-1.0
    events_indices[:, :, :, 1] = events_indices[:, :, :, 1]/(flow.shape[-2]-1)*2.0-1.0
    # use float64 dtype
    flow_at_event = F.grid_sample((flow/50.).double(), events_indices, align_corners=True)

    # print('flow_at_event.shape:', flow_at_event.shape) torch.Size([1, 2, 1, 9554])

    dt = (ts-t_ref).squeeze() / 1000.
    warped_xs = xs-flow_at_event[:, 0, :, :].squeeze() *dt
    warped_ys = ys-flow_at_event[:, 1, :, :].squeeze() *dt

    return warped_xs, warped_ys, ts, ps


def events_to_count_img(xs, ys, ts, ps, using_polatity=False, sensor_size=(256, 256), clip_out_of_range=True,
                              interpolation='bilinear', padding=True, timestamp_reverse=False):
    if padding:
        img_size = (sensor_size[0]+1, sensor_size[1]+1)
    else:
        img_size = sensor_size
    device = xs.device
    mask = torch.ones(xs.size(), device=device)
    if clip_out_of_range:
        zero_v = torch.tensor([0.], device=device)
        ones_v = torch.tensor([1.], device=device)
        if interpolation is None and padding==False:
            # print('If .')
            clipx = img_size[1]
            clipy = img_size[0]
        else:
            clipx = img_size[1]-1
            clipy = img_size[0]-1
        mask = torch.where(xs >= clipx, zero_v, ones_v)*torch.where(ys >= clipy, zero_v, ones_v)

    pxs = xs.floor().float()
    dxs = (xs - pxs).float()
    pys = ys.floor().float()
    dys = (ys - pys).float()
    pxs = (pxs * mask).long()
    pys = (pys * mask).long()
    counts = torch.ones(ps.size(), device=device)
    if using_polatity:
        masked_counts = ps*mask  # 使用极性
    else:
        masked_counts = counts * mask  # 不使用极性
    img = torch.zeros(img_size).to(device)
    interpolate_to_img(pxs, pys, dxs, dys, masked_counts, img)
    return img


def sobel_based_loss(events, flow_pre, device):
    # 基于sobel算子的边缘强度函数作为损失函数，并将edge_x和edge_y分别归一化到[0,1]
    warped_xs, warped_ys, ts, ps = warp_events_flow_torch(events, flow_pre, device)
    events_count_img = events_to_count_img(warped_xs, warped_ys, ts, ps)

    # 归一化到[0,255]
    max1 = torch.tensor(255.0, dtype=torch.float).cuda()
    img_max = torch.max(events_count_img)
    img_min = torch.min(events_count_img)
    norm_events_count_img = (max1 * (events_count_img - img_min) / (img_max - img_min))

    events_count_img = torch.unsqueeze(torch.unsqueeze(norm_events_count_img, dim=0), dim=0)  # (1, 1, H, W)
    edge_x, edge_y = sobel_xy(events_count_img)
    # edge_x, edge_y归一化到[0,1]
    max2 = torch.tensor(1.0, dtype=torch.float).cuda()
    edge_x_abs, edge_y_abs = torch.abs(edge_x), torch.abs(edge_y)
    x_max, x_min = torch.max(edge_x_abs), torch.min(edge_x_abs)
    y_max, y_min = torch.max(edge_y_abs), torch.min(edge_y_abs)
    norm_edge_x = (max2 * (edge_x_abs - x_min) / (x_max - x_min))
    norm_edge_y = (max2 * (edge_y_abs - y_min) / (y_max - y_min))
    print()
    edge = torch.mean(norm_edge_x+norm_edge_y)  # [0, 2]
    return torch.tensor(2., device=device) - edge  # min=0, max=2, the min the better
