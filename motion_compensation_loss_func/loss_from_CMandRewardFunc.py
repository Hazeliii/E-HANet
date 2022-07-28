import torch
from img_edge_based_loss import events_to_count_img
'''
论文Event Cameras, Contrast Maximization and Reward Functions: an Analysis
中提到的损失函数，默认作为神经网络损失函数，越小越好
https://github.com/TimoStoff/events_contrast_maximization/issues
'''


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


def sos_objective(img):
    """
    Loss given by g(x)^2 where g(x) is IWE
    """
    sos = torch.mean(img*img)
    return -sos


def soe_objective(img):
    """
    Sum of exponentials objective (Stoffregen et al, Event Cameras, Contrast
     Maximization and Reward Functions: an Analysis, CVPR19)
     Loss given by e^g(x) where g(x) is IWE
    """
    exp = torch.exp(img)
    soe = torch.mean(exp)
    return -soe


def moa_objective(img):
    """
    Max of accumulations objective
    Loss given by max(g(x)) where g(x) is IWE
    """
    moa = torch.max(img)
    return moa


def isoa_objective(img, thresh=0.5):
    """
    Inverse sum of accumulations objective
    Loss given by sum(1 where g(x)>1 else 0) where g(x) is IWE.
    This formulation has similar properties to original ISoA, but negation makes derivative
    more stable than inversion.
    """
    isoa = torch.sum(torch.where(img>thresh, 1., 0.))
    return isoa


def sosa_objective(img, p=3.):
    """
    Sum of Supressed Accumulations objective
    Loss given by e^(-p*g(x)) where g(x) is IWE. p is arbitrary shifting factor,
    higher values give better noise performance but lower accuracy.
    """
    exp = torch.exp(-p*img)
    sosa = torch.sum(exp)
    return -sosa


def r1_objective(img, p=3.):
    """
    R1 objective (Stoffregen et al, Event Cameras, Contrast
    Maximization and Reward Functions: an Analysis, CVPR19)
     Loss given by SOS and SOSA combined
    """
    sos = sos_objective(img)
    sosa = sosa_objective(img, p)
    return -sos*sosa


def contrast_max_baed_loss(events, flow_pre, device):
    warped_xs, warped_ys, ts, ps = warp_events_flow_torch(events, flow_pre, device)
    events_count_img = events_to_count_img(warped_xs, warped_ys, ts, ps)
    # 归一化到[0,1]
    max1 = torch.tensor(1.0, dtype=torch.float).cuda()
    img_max = torch.max(events_count_img)
    img_min = torch.min(events_count_img)
    norm_events_count_img = (max1 * (events_count_img - img_min) / (img_max - img_min))
    return r1_objective(norm_events_count_img)  # [0, 1]*sum([e^-3, 1])



