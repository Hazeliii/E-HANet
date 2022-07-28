import torch
import torch.nn.functional as F
import numpy


def interpolate_to_img(pxs, pys, dxs, dys, weights, img):
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


def events_to_timestamp_image(xs, ys, ts, ps, sensor_size=(256, 256), clip_out_of_range=True,
                              interpolation='bilinear', padding=True, timestamp_reverse=False):
    """
    Method to generate the average timestamp images from 'Zhu19, Unsupervised Event-based Learning
    of Optical Flow, Depth, and Egomotion'.
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
    Returns
    -------
    img_pos: timestamp image of the positive events
    img_neg: timestamp image of the negative events
    """
    device = xs.device
    ts, xs, ys, ps = ts.squeeze(), xs.squeeze(), ys.squeeze(), ps.squeeze()
    if padding:
        img_size = (sensor_size[0]+1, sensor_size[1]+1)
    else:
        img_size = sensor_size

    zero_v = torch.tensor([0.], device=device)
    ones_v = torch.tensor([1.], device=device)
    mask = torch.ones(xs.size(), device=device)

    if clip_out_of_range:
        # print('clip_out_of_range')
        if interpolation is None and padding==False:
            # print('If .')
            clipx = img_size[1]
            clipy = img_size[0]
        else:
            clipx = img_size[1]-1
            clipy = img_size[0]-1
        mask = torch.where(xs >= clipx, zero_v, ones_v) * torch.where(ys >= clipy, zero_v, ones_v)

    pos_events_mask = torch.where(ps>0, ones_v, zero_v)  # 1 for positive events else 0
    neg_events_mask = torch.where(ps<=0, ones_v, zero_v)
    epsilon = 1e-6

    if timestamp_reverse:
        normalized_ts = ((-ts + ts[-1])/(ts[-1] - ts[0] + epsilon)).squeeze()
    else:
        normalized_ts = ((ts - ts[0])/(ts[-1] - ts[0] + epsilon)).squeeze()

    pxs = xs.floor().float()
    dxs = (xs - pxs).float()
    pys = ys.floor().float()
    dys = (ys - pys).float()
    pxs = (pxs*mask).long()
    pys = (pys*mask).long()

    pos_weighted = (normalized_ts*pos_events_mask).float()  # 选出positive时间对应的normalized timestamp
    neg_weighted = (normalized_ts*neg_events_mask).float()

    img_pos = torch.zeros(img_size).to(device)
    img_pos_cnt = torch.ones(img_size).to(device)  # 在1的基础上进行加？
    img_neg = torch.zeros(img_size).to(device)
    img_neg_cnt = torch.ones(img_size).to(device)

    interpolate_to_img(pxs, pys, dxs, dys, pos_weighted, img_pos)
    interpolate_to_img(pxs, pys, dxs, dys, pos_events_mask, img_pos_cnt)
    interpolate_to_img(pxs, pys, dxs, dys, neg_weighted, img_neg)
    interpolate_to_img(pxs, pys, dxs, dys, neg_events_mask, img_neg_cnt)

    # Avoid division by 0
    img_pos_cnt[img_pos_cnt == 0] = 1
    img_neg_cnt[img_neg_cnt == 0] = 1
    # print('img_pos_cnt.shape:', img_pos_cnt.shape)   torch.Size([257, 257])

    img_pos = img_pos.div(img_pos_cnt)
    img_neg = img_neg.div(img_neg_cnt)
    return img_pos, img_neg


def zhu_based_loss(events, flow_pre, device):
    """
    Squared average timestamp images objective (Zhu et al, Unsupervised Event-based
    Learning of Optical Flow, Depth, and Egomotion, CVPR19)
    Loss given by g(x)^2*h(x)^2 where g(x) is image of average timestamps of positive events
    and h(x) is image of average timestamps of negative events.
    @param events:a set of events N*(t,x,y,p)
    @param flow_pre:flow , ([2, H, W]) tensor
    return:the loss
    """
    warped_xs, warped_ys, ts, ps = warp_events_flow_torch(events, flow_pre, device)
    img_pos, img_neg = events_to_timestamp_image(warped_xs, warped_ys, ts, ps)
    loss = torch.sum(img_pos * img_pos) + torch.sum(img_neg * img_neg)
    return loss










