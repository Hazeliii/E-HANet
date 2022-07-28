import math
import random
import weakref
from pathlib import Path
from typing import Dict, Tuple
import h5py
import imageio
import torch
from torch.utils.data import Dataset
from utils.dsec_utils import VoxelGrid, flow_16bit_to_float
import numpy as np
from numba import jit
from torchvision import transforms
from torchvision.transforms.functional import rotate, hflip


class EventSlicer:
    def __init__(self, h5f: h5py.File):
        # 从hdf5文件中加载events
        self.h5f = h5f
        self.events = dict()
        for dest_str in ['p', 'x', 'y', 't']:
            self.events[dest_str] = self.h5f['events/{}'.format(dest_str)]

        self.ms_to_idx = np.asarray(self.h5f['ms_to_idx'], dtype='int64')
        self.t_offset = int(h5f['t_offset'][()])
        self.t_final = int(self.events['t'][-1]) + self.t_offset

    def get_events(self, t_start_us: int, t_end_us: int) -> Dict[str, np.ndarray]:
        """Get events (p, x, y, t) within the specified time window
        Parameters
        ----------
        t_start_us: start time in microseconds
        t_end_us: end time in microseconds
        Returns
        -------
        events: dictionary of (p, x, y, t) or None if the time window cannot be retrieved
        """
        assert t_start_us < t_end_us

        # We assume that the times are top-off-day, hence subtract offset:
        t_start_us -= self.t_offset
        t_end_us -= self.t_offset

        t_start_ms, t_end_ms = self.get_conservative_windows_ms(t_start_us, t_end_us)
        t_start_ms_idx = self.ms2idx(t_start_ms)
        t_end_ms_idx = self.ms2idx(t_end_ms)

        if t_start_ms_idx is None or t_end_ms_idx is None:
            # Cannot guarantee window size anymore
            return None

        events = dict()
        time_array_conservative = np.asarray(self.events['t'][t_start_ms_idx:t_end_ms_idx])
        idx_start_offset, idx_end_offset = self.get_time_indices_offsets(time_array_conservative, t_start_us, t_end_us)
        t_start_us_idx = t_start_ms_idx + idx_start_offset
        t_end_us_idx = t_start_ms_idx + idx_end_offset
        # Again add t_offset to get gps time
        events['t'] = time_array_conservative[idx_start_offset:idx_end_offset] + self.t_offset
        for dset_str in ['p', 'x', 'y']:
            events[dset_str] = np.asarray(self.events[dset_str][t_start_us_idx:t_end_us_idx])
            assert events[dset_str].size == events['t'].size
        return events


    @staticmethod
    def get_conservative_windows_ms(ts_start_us: int, ts_end_us) -> Tuple[int, int]:
        """Compute a conservative time window of time with millisecond resolution.
        We have a time to index mapping for each millisecond. Hence, we need
        to compute the lower and upper millisecond to retrieve events.
        Parameters
        ----------
        ts_start_us:    start time in microseconds
        ts_end_us:      end time in microseconds
        Returns
        -------
        window_start_ms:    conservative start time in milliseconds
        window_end_ms:      conservative end time in milliseconds
        """
        assert ts_end_us > ts_start_us
        window_start_ms = math.floor(ts_start_us / 1000)
        window_end_ms = math.ceil(ts_end_us / 1000)
        return window_start_ms, window_end_ms

    @staticmethod
    @jit(nopython=True)
    def get_time_indices_offsets(time_array: np.ndarray, time_start_us: int, time_end_us: int) -> Tuple[list, list]:
        """Compute index offset of start and end timestamps in microseconds
        Parameters
        ----------
        time_array:     timestamps (in us) of the events
        time_start_us:  start timestamp (in us)
        time_end_us:    end timestamp (in us)
        Returns
        -------
        idx_start:  Index within this array corresponding to time_start_us
        idx_end:    Index within this array corresponding to time_end_us
        such that (in non-edge cases)
        time_array[idx_start] >= time_start_us
        time_array[idx_end] >= time_end_us
        time_array[idx_start - 1] < time_start_us
        time_array[idx_end - 1] < time_end_us
        this means that
        time_start_us <= time_array[idx_start:idx_end] < time_end_us
        """
        assert time_array.ndim == 1
        idx_start = -1
        if time_array[-1] < time_start_us:
            return time_array.size, time_array.size
        else:
            for idx_from_start in range(0, time_array.size, 1):
                if time_array[idx_from_start] >= time_start_us:
                    idx_start = idx_from_start
                    break
        assert idx_start >= 0

        idx_end = time_array.size
        for idx_from_end in range(time_array.size - 1, -1, -1):
            if time_array[idx_from_end] >= time_end_us:
                idx_end = idx_from_end
            else:
                break

        assert time_array[idx_start] >= time_start_us
        if idx_end < time_array.size:
            assert time_array[idx_end] >= time_end_us
        if idx_start > 0:
            assert time_array[idx_start - 1] < time_start_us
        if idx_end > 0:
            assert time_array[idx_end - 1] < time_end_us
        return idx_start, idx_end

    def ms2idx(self, time_ms: int) -> int:
        assert time_ms >= 0
        if time_ms >= self.ms_to_idx.size:
            return None
        return self.ms_to_idx[time_ms]


class Sequences(Dataset):
    # seq_name(e.g.zurich_city_11_a)
    # ├── flow(forward)
    # │   ├── event
    # │   │   ├── 000000.png
    # │   │   └── ...
    # │   └── timestamps.txt
    # └── events
    #     ├── left
    #     │   ├── events.h5
    #     │   └── rectify_map.h5
    #     └── right
    #         ├── events.h5
    #         └── rectify_map.h5
    def __init__(self, seq_path: Path, delta_t_ms: int = 100, num_bins: int = 15, name_idx=0, visualize=False):
        assert num_bins >= 1
        assert seq_path.is_dir()

        self.height = 480
        self.width = 640
        self.num_bins = num_bins

        self.voxel_grid = VoxelGrid((self.num_bins, self.height, self.width), normalize=True)
        # self.locations = ['left']

        self.delta_t_us = delta_t_ms * 1000
        self.name_idx = name_idx
        self.visualize_samples = visualize

        # load flow timestamps
        flow_dir = seq_path / 'flow'
        assert flow_dir.is_dir()
        timestamp_file = flow_dir / 'timestamps.txt'
        assert timestamp_file.is_file()
        file = np.genfromtxt(
            timestamp_file,
            delimiter=','
        )
        self.start_timestamps = file[:, 0]
        self.end_timestamps = file[:, 1]

        # load flow paths
        ev_flow_dir = flow_dir / 'forward'
        assert ev_flow_dir.is_dir()
        flow_gt_pathstring = list()
        for entry in ev_flow_dir.iterdir():
            assert str(entry.name).endswith('.png')
            flow_gt_pathstring.append(str(entry))
        flow_gt_pathstring.sort()
        self.flow_gt_pathstring = flow_gt_pathstring
        #  print('flow_gt_pathstring:', self.flow_gt_pathstring)

        assert len(self.flow_gt_pathstring) == self.start_timestamps.size
        # 去掉第一个flow和timestamp 因为没有此前的events
        # assert int(Path(self.flow_gt_pathstring[0]).stem) == 0
        # self.flow_gt_pathstring.pop(0)
        # self.timestamps = self.timestamps[1:]

        ev_dir_location = seq_path / 'events_left'
        ev_data_file = ev_dir_location / 'events.h5'
        ev_rect_file = ev_dir_location / 'rectify_map.h5'
        h5f_location = h5py.File(str(ev_data_file), 'r')
        self.h5f = h5f_location
        self.event_slicers = EventSlicer(h5f_location)
        with h5py.File(str(ev_rect_file), 'r') as h5_rect:
            self.rectify_ev_maps = h5_rect['rectify_map'][()]

        # self._finalizer = weakref.finalize(self, self.close_callback, self.h5f)  # 弱引用
        self.flip = transforms.Compose([
            transforms.RandomHorizontalFlip()
        ])
        self.angle = random.uniform(-30., 30.)

    @staticmethod
    def close_callback(h5f_dict):
        for k, h5f in h5f_dict.items():
            h5f.close()

    @staticmethod
    def get_flow(flowfile: Path):
        assert flowfile.exists()
        assert flowfile.suffix == '.png'
        flow_16bit = imageio.imread(str(flowfile), format='PNG-FI')
        flow, valid2D = flow_16bit_to_float(flow_16bit)
        return flow, valid2D

    def __len__(self):
        return len(self.flow_gt_pathstring)

    def rectify_events(self, x: np.ndarray, y: np.ndarray):
        # assert location in self.locations
        rectify_map = self.rectify_ev_maps
        assert rectify_map.shape == (self.height, self.width, 2), rectify_map.shape
        assert x.max() < self.width
        assert y.max() < self.height
        return rectify_map[y, x]

    def events_to_grid(self, p, t, x, y, device: str='cup'):
        t = (t - t[0]).astype('float32')
        t = (t/t[-1])
        x = x.astype('float32')
        y = y.astype('float32')
        pol = p.astype('float32')
        event_data_torch = {
            'p':torch.from_numpy(pol),
            't':torch.from_numpy(t),
            'x':torch.from_numpy(x),
            'y':torch.from_numpy(y)
        }
        return self.voxel_grid.convert(event_data_torch)

    def get_events(self, index):
        '''
        ts_start = [self.timestamps[index] - self.delta_t_us, self.timestamps[index]]
        ts_end = [self.timestamps[index], self.timestamps[index] + self.delta_t_us]
        '''
        ts_start = [self.start_timestamps[index] - self.delta_t_us, self.start_timestamps[index]]
        ts_end = [self.end_timestamps[index] - self.delta_t_us, self.end_timestamps[index]]
        event_data = self.event_slicers.get_events(ts_start[1], ts_end[1])
        p = event_data['p']
        t = event_data['t']
        x = event_data['x']
        y = event_data['y']

        xy_rect = self.rectify_events(x, y)
        x_rect = xy_rect[:, 0]
        y_rect = xy_rect[:, 1]

        return t, x_rect

    def get_data_sample(self, index, crop_window=None, flip=None):
        names = ['event_volume_old', 'event_volume_new']
        # First entry corresponds to all events BEFORE the flow map
        # Second entry corresponds to all events AFTER the flow map (corresponding to the actual fwd flow)
        ts_start = [self.start_timestamps[index] - self.delta_t_us, self.start_timestamps[index]]
        ts_end = [self.end_timestamps[index] - self.delta_t_us, self.end_timestamps[index]]
        # 对应的flow的下标
        file_index = index
        flow_gt_path = Path(self.flow_gt_pathstring[index])
        flow, valid_2D = self.get_flow(flow_gt_path)

        output = {
            # 'file_index': file_index,
            # 'timestamp': self.start_timestamps[index],
            'gt_flow': flow,
            'valid_2D': valid_2D
        }
        for i in range(len(names)):
            event_data = self.event_slicers.get_events(ts_start[i], ts_end[i])
            p = event_data['p']
            t = event_data['t']
            x = event_data['x']
            y = event_data['y']

            xy_rect = self.rectify_events(x, y)
            x_rect = xy_rect[:, 0]
            y_rect = xy_rect[:, 1]

            if crop_window is not None:
                # Cropping (+- 2 for safety reasons)
                x_mask = (x_rect >= crop_window['start_x'] - 2) & (
                        x_rect < crop_window['start_x'] + crop_window['crop_width'] + 2)
                y_mask = (y_rect >= crop_window['start_y'] - 2) & (
                        y_rect < crop_window['start_y'] + crop_window['crop_height'] + 2)
                mask_combined = x_mask & y_mask
                p = p[mask_combined]
                t = t[mask_combined]
                x_rect = x_rect[mask_combined]
                y_rect = y_rect[mask_combined]

            if self.voxel_grid is None:
                raise NotImplementedError
            else:
                event_representation = self.events_to_grid(p, t, x_rect, y_rect)
                output[names[i]] = event_representation
            output['name_map'] = self.name_idx
        '''
        flow = self.flip(torch.from_numpy(flow))  # (480, 640, 2)
        valid_2D = self.flip(torch.from_numpy(valid_2D))  # (480, 640)
        event_voxel_old = self.flip(output[names[0]])
        event_voxel_new = self.flip(output[names[1]])

        flow = rotate(flow, self.angle)
        valid_2D = rotate(torch.unsqueeze(valid_2D, 2), self.angle)
        event_voxel_old = rotate(event_voxel_old, self.angle)
        event_voxel_new = rotate(event_voxel_new, self.angle)
        '''

        return flow,valid_2D, output[names[0]], output[names[1]]

    def __getitem__(self, index):
        sample = self.get_data_sample(index)
        return sample


class DatasetProvider:
    def __init__(self, dataset_path: Path, delta_t_ms: int = 100, num_bins=15):
        train_path = dataset_path / 'test'
        assert dataset_path.is_dir(), str(dataset_path)
        assert train_path.is_dir(), str(train_path)

        train_sequences = list()
        for child in train_path.iterdir():
            train_sequences.append(Sequences(child, delta_t_ms, num_bins))

        self.train_dataset = torch.utils.data.ConcatDataset(train_sequences)

    def get_train_dataset(self):
        return self.train_dataset
