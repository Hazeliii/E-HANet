import numpy as np
import torch
import torch as th


def dictionary_of_numpy_arrays_to_tensors(sample):
    """Transforms dictionary of numpy arrays to dictionary of tensors."""
    if isinstance(sample, dict):
        return {
            key: dictionary_of_numpy_arrays_to_tensors(value)
            for key, value in sample.items()
        }
    if isinstance(sample, np.ndarray):
        if len(sample.shape) == 2:
            return th.from_numpy(sample).float().unsqueeze(0)
        else:
            return th.from_numpy(sample).float()
    return sample


class EventSequenceToVoxelGrid_Pytorch(object):
    # Source: https://github.com/uzh-rpg/rpg_e2vid/blob/master/utils/inference_utils.py#L480
    def __init__(self, num_bins, gpu=False, gpu_nr=0, normalize=True, forkserver=True):
        if forkserver:
            try:
                th.multiprocessing.set_start_method('forkserver')
            except RuntimeError:
                pass
        self.num_bins = num_bins
        self.normalize = normalize
        if gpu:
            if not th.cuda.is_available():
                print('Warning: There\'s no CUDA support on this machine!')
            else:
                self.device = th.device('cuda:' + str(gpu_nr))
        else:
            self.device = th.device('cpu')

    def __call__(self, event_sequence):
        """
        Build a voxel grid with bilinear interpolation in the time domain from a set of events.
        :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
        :param num_bins: number of bins in the temporal axis of the voxel grid
        :param width, height: dimensions of the voxel grid
        :param device: device to use to perform computations
        :return voxel_grid: PyTorch event tensor (on the device specified)
        """

        events = event_sequence.features.astype('float')

        width = event_sequence.image_width
        height = event_sequence.image_height

        assert (events.shape[1] == 4)
        assert (self.num_bins > 0)
        assert (width > 0)
        assert (height > 0)

        with th.no_grad():

            events_torch = th.from_numpy(events)
            # with DeviceTimer('Events -> Device (voxel grid)'):
            events_torch = events_torch.to(self.device)

            # with DeviceTimer('Voxel grid voting'):
            voxel_grid = th.zeros(self.num_bins, height, width, dtype=th.float32, device=self.device).flatten()

            # normalize the event timestamps so that they lie between 0 and num_bins
            last_stamp = events_torch[-1, 0]
            first_stamp = events_torch[0, 0]

            assert last_stamp.dtype == th.float64, 'Timestamps must be float64!'
            # assert last_stamp.item()%1 == 0, 'Timestamps should not have decimals'

            deltaT = last_stamp - first_stamp

            if deltaT == 0:
                deltaT = 1.0

            events_torch[:, 0] = (self.num_bins - 1) * (events_torch[:, 0] - first_stamp) / deltaT
            ts = events_torch[:, 0]
            xs = events_torch[:, 1].long()
            ys = events_torch[:, 2].long()
            pols = events_torch[:, 3].float()
            pols[pols == 0] = -1  # polarity should be +1 / -1
            # 选取一组事件，按照时间对事件进行加权，越新的事件权重越大
            # added by WQM
            '''
            pols[pols == 0] = 1  # equals to counts instead of polarity
            t_weight = (events_torch[:, 0] - first_stamp) / deltaT
            pos_weighted = (pols * t_weight).float()  # torch.float32,the same with voxel_grid
            '''
            tis = th.floor(ts)
            tis_long = tis.long()
            dts = ts - tis

            # original:
            vals_left = pols * (1.0 - dts.float())
            vals_right = pols * dts.float()

            '''
            WQM
            vals_left = pos_weighted * (1.0 - dts.float())
            vals_right = pos_weighted * dts.float()
            '''
            valid_indices = tis < self.num_bins
            valid_indices &= tis >= 0  # [0, B-1]

            if events_torch.is_cuda:
                datatype = th.cuda.LongTensor
            else:
                datatype = th.LongTensor

            voxel_grid.index_add_(dim=0,
                                  index=(xs[valid_indices] + ys[valid_indices]
                                         * width + tis_long[valid_indices] * width * height).type(
                                      datatype),
                                  source=vals_left[valid_indices])

            valid_indices = (tis + 1) < self.num_bins
            valid_indices &= tis >= 0

            voxel_grid.index_add_(dim=0,
                                  index=(xs[valid_indices] + ys[valid_indices] * width
                                         + (tis_long[valid_indices] + 1) * width * height).type(datatype),
                                  source=vals_right[valid_indices])

            voxel_grid = voxel_grid.view(self.num_bins, height, width)

        if self.normalize:
            mask = th.nonzero(voxel_grid, as_tuple=True)
            if mask[0].size()[0] > 0:
                mean = voxel_grid[mask].mean()
                std = voxel_grid[mask].std()
                if std > 0:
                    voxel_grid[mask] = (voxel_grid[mask] - mean) / std
                else:
                    voxel_grid[mask] = voxel_grid[mask] - mean

        return voxel_grid


# Added by WQM
# 先将事件按照正负极分类，然后将正负极事件分别生成C*H*W的体素网格，最后再一片正极一片负极进行拼接，生成（2*C）*H*W的体素网格
class EventSequenceToPositiveNegativeTensor(object):
    def __init__(self, num_bins, gpu=False, gpu_nr=0, normalize=True, forkserver=True):
        # print('In EventSequenceToPositiveNegativeTensor.init by WQM. The num_bins is ', num_bins)
        if forkserver:
            try:
                th.multiprocessing.set_start_method('forkserver')
            except RuntimeError:
                pass

        self.num_bins = num_bins
        self.normalize = normalize
        if gpu:
            if not th.cuda.is_available():
                print('Warning: There is no CUDA support on this machine!')
            else:
                self.device = th.device('cuda:'+str(gpu_nr))
        else:
            self.device = th.device('cpu')

    def __call__(self, event_sequence):
        '''
        Build a voxel grid with bilinear interpolation in the time domain from a set of events.
        :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
        :param num_bins: number of bins in the temporal axis of the voxel grid
        :param width, height: dimensions of the voxel grid
        :param device: device to use to perform computations
        :return voxel_grid: PyTorch event tensor (on the device specified)
        '''
        events = event_sequence.features.astype('float')
        width = event_sequence.image_width
        height = event_sequence.image_height

        assert (events.shape[1] == 4)
        assert (self.num_bins > 0)
        assert (width > 0)
        assert (height > 0)

        with th.no_grad():
            events_torch = th.from_numpy(events)
            events_torch = events_torch.to(self.device)

            ts = events_torch[:, 0]
            xs = events_torch[:, 1].long()
            ys = events_torch[:, 2].long()
            pols = events_torch[:, 3].float()
            pols[pols == 0] = -1

            mask = (pols == 1)
            neg_mask = (pols == -1)

            pos_ts = torch.masked_select(ts, mask)
            pos_xs = torch.masked_select(xs, mask)
            pos_ys = torch.masked_select(ys, mask)
            pos_p = torch.masked_select(pols, mask)

            assert pos_ts.size() == pos_xs.size() == pos_ys.size() == pos_p.size()

            neg_ts = torch.masked_select(ts, neg_mask)
            neg_xs = torch.masked_select(xs, neg_mask)
            neg_ys = torch.masked_select(ys, neg_mask)
            neg_p = torch.masked_select(pols, neg_mask)
            assert neg_ts.size() == neg_xs.size() == neg_ys.size() == neg_p.size()

            if events_torch.is_cuda:
                self.datatype = th.cuda.LongTensor
            else:
                self.datatype = th.LongTensor

            if pos_ts.size() == torch.Size([0]) or neg_ts.size() == torch.Size([0]):
                voxel_grid = self.generate_img(ts, xs, ys, pols, height, width, num_bins=self.num_bins*2)
            else:
                positive_voxel_grid = self.generate_img(pos_ts, pos_xs, pos_ys, pos_p, height, width)
                negative_voxel_grid = self.generate_img(neg_ts, neg_xs, neg_ys, neg_p, height, width)
                # stack
                # positive_voxel_grid.size(): torch.Size([3, 260, 346]
                # voxel_grid.size(): torch.Size([6, 260, 346])
                voxel_grid = th.zeros(self.num_bins * 2, height, width, dtype=th.float32, device=self.device)
                for i in range(self.num_bins):
                    # 一片正极（H*W）一片负极（H*W）拼接
                    positive_slicer = positive_voxel_grid[i]
                    negative_slicer = negative_voxel_grid[i]
                    voxel_grid[i * 2] = positive_slicer
                    voxel_grid[i * 2 + 1] = negative_slicer

            return voxel_grid

    def generate_img(self, ts, xs, ys, pols, height, width, num_bins=3):
        voxel_grid = th.zeros(num_bins, height, width, dtype=th.float32, device=self.device).flatten()
        last_stamp = ts[-1]
        first_stamp = ts[0]
        assert last_stamp.dtype == th.float64, 'Timestamps must be float64!'

        delta_T = last_stamp - first_stamp
        if delta_T == 0:
            delta_T = 1.0

        ts = (num_bins - 1) * (ts - first_stamp) / delta_T
        tis = th.floor(ts)
        tis_long = tis.long()
        dts = ts - tis
        vals_left = pols * (1.0 - dts.float())
        vals_right = pols * dts.float()

        valid_indices = tis < num_bins
        valid_indices &= tis >= 0

        voxel_grid.index_add_(dim=0,
                              index=(xs[valid_indices] + ys[valid_indices] * width +
                                     tis_long[valid_indices] * width * height).type(self.datatype),
                              source=vals_left[valid_indices])

        valid_indices = (tis + 1) < num_bins
        valid_indices &= tis >= 0
        voxel_grid.index_add_(dim=0,
                              index=(xs[valid_indices] + ys[valid_indices] * width +
                                     (tis_long[valid_indices]+1) * width * height).type(self.datatype),
                              source=vals_right[valid_indices])

        voxel_grid = voxel_grid.view(num_bins, height, width)

        if self.normalize:
            mask = th.nonzero(voxel_grid, as_tuple=True)
            if mask[0].size()[0] > 0:
                mean = voxel_grid[mask].mean()
                std = voxel_grid[mask].std()
                if std > 0:
                    voxel_grid[mask] = (voxel_grid[mask] - mean) / std
                else:
                    voxel_grid[mask] = voxel_grid[mask] - mean
        return voxel_grid







