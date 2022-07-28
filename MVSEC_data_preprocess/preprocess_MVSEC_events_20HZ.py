import numpy as np
import os
from decimal import *
import h5py
import pandas as pd


def read_events(events_pth, event_txt_pth):
    print('read_events Begin')
    events_h5 = h5py.File(events_pth)
    events_left = events_h5['davis']['left']['events']
    i = 0
    with open(event_txt_pth, 'w') as file:
        for events in events_left:
            x = events[0]
            y = events[1]
            time = events[2]
            p = events[3]
            time_str = '{} {} {} {}'.format(time, x, y, p)
            file.write(time_str)
            file.write('\n')
            i += 1
            if i % 2000 == 0:
                print('Now processing {} events'.format(i))
        file.close()
    print('read_events Done')


def gen_h5_file_partly(event_times_pth, events_pth, flow_time_pth, saved_pth):
    flow_times = np.loadtxt(flow_time_pth)
    print('Load flow_time Done.')
    ts = np.loadtxt(event_times_pth)
    print('Load events_time Done.')
    events_h5 = np.loadtxt(events_pth)
    print('Load events Done.')

    start_i = 0
    time_start = flow_times[start_i]
    idx_pre = np.searchsorted(ts, time_start, side='right')
    print('len(flow_times):', len(flow_times))

    for i in range(start_i, len(flow_times)-1):  # 总共的.h5文件，与image_times相同
        time_next = flow_times[i+1]
        idx_i = np.searchsorted(ts, time_next, side='right')
        events_i = list()

        if idx_i > idx_pre:
            for ev_idx in range(idx_pre, idx_i):
                event = events_h5[ev_idx]
                time = event[0]
                x = event[1]
                y = event[2]
                p = event[3]
                event_array = np.array([time, x, y, p])
                events_i.append(event_array)
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
    print('total evnet.h5 files:', i+1)


def gen_event_h5(ts, xs, ys, ps, image_time_pth, saved_pth):
    image_times = np.loadtxt(image_time_pth)
    start_i = 8702
    time_start = image_times[start_i]
    idx_pre = np.searchsorted(ts, time_start, side='right')
    for i in range(start_i+1, len(image_times)):
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

    sequence_name = ['outdoor_day_2']
    root = '/storage1/dataset/MVSEC/events_hdf5'
    saved_root = '/storage1/dataset/MVSEC/mvsec_20HZ'
    time_txt_root = '/storage1/dataset/MVSEC/mvsec_45HZ'
    for name in sequence_name:
        print('Processing %s ...' % name)
        event_pth = os.path.join(root, '{}_data.hdf5'.format(name))
        saved_event_root = os.path.join(saved_root, name, 'davis/left/events')
        flow_time_pth = os.path.join(saved_root, name, 'timestamps_flow.txt')
        # MVSEC_45HZ有准备好的
        events_time_pth = os.path.join(time_txt_root, name, 'event_times.txt')
        events_txt_pth = os.path.join(time_txt_root, name, 'events.txt')
        if not os.path.exists(saved_event_root):
            os.makedirs(saved_event_root)
        # read_events(event_pth, events_txt_pth)
        gen_h5_file_partly(events_time_pth, events_txt_pth, flow_time_pth, saved_event_root)


