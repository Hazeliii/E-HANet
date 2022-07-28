import numpy as np
import os
from decimal import *
import h5py
import pandas as pd


def deal_with_flow(gt_files, saved_pth):
    '''
    将flow_groundtruth分开，x_flow_dist中的每一行与y_flow_dist中的每一行组合，生成xy_flow(2,260,346)，每一组光流存为一个.npy文件
    parameters:x_flow_pth, y_flow_pth分别存储x,y光流，为.npy形式
    '''
    x_flow = gt_files['x_flow_dist']
    y_flow = gt_files['y_flow_dist']

    assert x_flow.shape == y_flow.shape
    for i in range(len(x_flow)):
        xy_flow_name = '{:06d}.npy'.format(i)
        xy_flow_pth = os.path.join(saved_pth, xy_flow_name)
        x_flow_i = x_flow[i]
        y_flow_i = y_flow[i]
        xy_flow_i = np.stack((x_flow_i, y_flow_i), axis=0)
        np.save(xy_flow_pth, xy_flow_i)
    print('Total %d flows are processed!'% i)


def deal_with_flow_time(gt_files, saved_pth):
    time_npy_file = gt_files['timestamps']
    time_txt_pth = os.path.join(saved_pth, 'timestamps_flow.txt')

    with open(time_txt_pth, 'w+') as file:
        for i in range(len(time_npy_file)):
            time_i = time_npy_file[i]
            time_long = Decimal(time_i).quantize(Decimal('0.000000000'))
            str = '{:e}'.format(time_long)
            file.write(str)
            file.write('\n')
    file.close()
    print('The timestamps_flow.txt is done ! ')


def deal_with_image_time(time_list, saved_pth):
    time_txt_pth = os.path.join(saved_pth, 'timestamps_images.txt')

    with open(time_txt_pth, 'w+') as file:
        for i in range(len(time_list)):
            time_i = time_list[i]
            time_long = Decimal(time_i).quantize(Decimal('0.000000000'))
            str = '{:e}'.format(time_long)
            file.write(str)
            file.write('\n')
    file.close()
    print('The timestamps_flow.txt is done ! ')


if __name__ == '__main__':
    sequence_name = ['outdoor_day_2']
    root = '/storage1/dataset/MVSEC/outdoor_day_2_gt_flow_dist.npz'
    saved_root = '/storage1/dataset/MVSEC/mvsec_45HZ'

    gt_files = np.load(root)
    print('gt_files.files:', gt_files.files)

    time_saved_root = os.path.join(saved_root, 'outdoor_day_2')
    flow_saved_root = os.path.join(time_saved_root, 'optical_flow')
    if not os.path.exists(flow_saved_root):
        os.makedirs(flow_saved_root)

    deal_with_flow_time(gt_files, time_saved_root)
    deal_with_flow(gt_files, flow_saved_root)

    '''

    for name in sequence_name:
        print('Processing %s ...'%name)
        time_saved_root = os.path.join(saved_root, name)

        flow_saved_root = os.path.join(time_saved_root, 'optical_flow')

        if not os.path.exists(flow_saved_root):
            os.makedirs(flow_saved_root)
        
        h5_root = os.path.join(root,  '{}_data.hdf5'.format(name))
        h5_flie = h5py.File(h5_root)
        image_time_date = h5_flie['davis']['left']['image_raw_ts']
        ts_list = list()
        for time in image_time_date:
            ts_list.append(time)
        deal_with_image_time(ts_list, time_saved_root)
        
        dataset_name = os.path.join(root, '{}_gt_flow_dist'.format(name))
        x_flow_pth = os.path.join(dataset_name, 'x_flow_dist.npy')
        y_flow_pth = os.path.join(dataset_name, 'y_flow_dist.npy')
        flow_time_pth = os.path.join(dataset_name, 'timestamps.npy')

        deal_with_flow_time(flow_time_pth, time_saved_root)
        deal_with_flow(x_flow_pth, y_flow_pth, flow_saved_root)
        '''
