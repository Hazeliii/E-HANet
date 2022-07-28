import numpy as np
import os
from decimal import *
import h5py
import pandas as pd


def read_events(events_pth):
    print('read_events Begin')
    events_h5 = h5py.File(events_pth)
    # image_ts = np.loadtxt(image_time_pth)
    # show(events_h5)
    events_left = events_h5['davis']['left']['events']
    xs = list()
    ys = list()
    ts = list()
    ps = list()
    i = 0
    for events in events_left:
        x = events[0]
        y = events[1]
        time = events[2]
        p = events[3]
        xs.append(x)
        ys.append(y)
        ts.append(time)
        ps.append(p)
        i += 1
        if i % 20000 == 0:
            print('Reading {} events...'.format(i))
    print('read_events Done')
    return ts, xs, ys, ps


def gen_event_h5(ts, xs, ys, ps, image_time_pth, saved_pth):
    image_times = np.loadtxt(image_time_pth)
    idx_pre = 0
    for i in range(len(image_times)):
        time_i = image_times[i]
        idx_i = np.searchsorted(ts, time_i, side='right')
        events_i = list()
        if idx_i > idx_pre:
            for ev_idx in range(idx_pre, idx_i):
                x = xs[ev_idx]
                y = ys[ev_idx]
                t = ts[ev_idx]
                p = ps[ev_idx]
                event = np.array([t, x, y, p])
                events_i.append(event)
            idx_pre = idx_i
        if i % 100 == 0:
            print('Processing {} h5 file...'.format(i))
        if len(events_i) == 0:
            continue
        h5_file_name = '{:06d}.h5'.format(i)
        df = pd.DataFrame(events_i, columns=['ts', 'x', 'y', 'p'])
        h5_file = pd.HDFStore(os.path.join(saved_pth, h5_file_name))
        h5_file['myDataset'] = df
        h5_file.close()

    print('total evnet.h5 files:', i)
        # print('%s id Done'%h5_file_name)


if __name__ == '__main__':

    # deal_with_flow(x_flow_pth, y_flow_pth, saved_pth)
    # deal_with_flow_time(time_pth, time_saved_pth)

    sequence_name = ['indoor_flying_2', 'indoor_flying_3', 'indoor_flying_4', 'outdoor_day_1', 'outdoor_day_2']
    root = '/storage1/dataset/MVSEC/events_hdf5'
    saved_root = '/storage1/dataset/MVSEC/mvsec_45HZ'

    for name in sequence_name:
        print('Processing %s ...' % name)
        event_pth = os.path.join(root, '{}_data.hdf5'.format(name))
        saved_event_root = os.path.join(saved_root, name, 'davis/left/events')
        image_time_pth = os.path.join(saved_root, name, 'timestamps_images.txt')
        if not os.path.exists(saved_event_root):
            os.makedirs(saved_event_root)

        ts, xs, ys, ps = read_events(event_pth)
        gen_event_h5(ts, xs, ys, ps, image_time_pth, saved_event_root)