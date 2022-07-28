import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "1"
from loader.loader_mvsec import *
from loader.loader_dsec import *
from utils.logger import Logger
import utils.helper_functions as helper
import json
from torch.utils.data.dataloader import DataLoader
from utils import visualization as visu
import argparse
from test import *
import torch.nn
from model import bam_split
from utils.dsec_utils import RepresentationType
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # 6


def initialize_tester(config):
    # Warm Start
    if config['subtype'].lower() == 'warm_start':
        return TestRaftEventsWarm
    # Classic
    else:
        return TestRaftEvents


def get_visualizer(args):
    # DSEC dataset
    if args.dataset.lower() == 'dsec':
        return visualization.DsecFlowVisualizer
    # MVSEC dataset
    else:
        return visualization.FlowVisualizerEvents


def test(args):
    # Choose correct config file
    if args.dataset.lower() == 'dsec':
        if args.type.lower() == 'warm_start':
            config_path = 'config/dsec_warm_start.json'
        elif args.type.lower() == 'standard':
            config_path = 'config/dsec_standard.json'
        else:
            raise Exception('Please provide a valid argument for --type. [warm_start/standard]')
    elif args.dataset.lower() == 'mvsec':
        if args.frequency == 20:
            config_path = 'config/mvsec_20.json'
        elif args.frequency == 45:
            config_path = 'config/mvsec_45_test_outdoor.json'
        else:
            raise Exception('Please provide a valid argument for --frequency. [20/45]')
    else:
        raise Exception('Please provide a valid argument for --dataset. [dsec/mvsec]')

    filter_pth = 'config/mvsec_test_outdoor1_filter.json'

    # Load config file
    config = json.load(open(config_path))
    filters = json.load(open(filter_pth))

    # Logger
    # my_logger = Logger(save_path)
    # my_logger.initialize_file("test")

    # Instantiate Dataset
    # Case: DSEC Dataset
    additional_loader_returns = None
    if args.dataset.lower() == 'dsec':
        # Dsec Dataloading
        loader = DatasetProvider(
            dataset_path=Path(args.path),
            representation_type=RepresentationType.VOXEL,
            delta_t_ms=100,
            type=config['subtype'].lower(),
            visualize=args.visualize)
        # loader.summary(my_logger)
        test_set = loader.get_test_dataset()
        additional_loader_returns = {'name_mapping_test': loader.get_name_mapping_test()}

    # Case: MVSEC Dataset
    else:
        if config['subtype'].lower() == 'standard':
            test_set = MvsecFlow(
                args=config["data_loader"]["test"]["args"],
                filters=filters,
                type='test',
                path=args.path
            )
        elif config['subtype'].lower() == 'warm_start':
            test_set = MvsecFlowRecurrent(
                args=config["data_loader"]["test"]["args"],
                type='test',
                path=args.path
            )
        else:
            raise NotImplementedError
            # test_set.summary(my_logger)

    # Instantiate Dataloader
    test_set_loader = DataLoader(test_set,
                                 batch_size=config['data_loader']['test']['args']['batch_size'],
                                 shuffle=config['data_loader']['test']['args']['shuffle'],
                                 num_workers=args.num_workers,
                                 drop_last=True)

    # Load Model
    '''
    model = eraft.ERAFT(
        args,
        n_first_channels=config['data_loader']['test']['args']['num_voxel_bins']
    )
    '''
    # Create Save Folder
    save_path = helper.create_save_path(config['save_dir'].lower(), config['name'].lower())
    print('Storing output in folder {}'.format(save_path))

    num_bins = config['data_loader']['test']['args']['num_voxel_bins']
    print('num_bins is ', num_bins)

    model = torch.nn.DataParallel(bam_split.ERAFT(args, n_first_channels=num_bins))
    checkpoint = torch.load(config['test']['checkpoint'])
    # model.load_state_dict(checkpoint['model'])
    # print('Checkpoint is ', checkpoint)
    model.load_state_dict(checkpoint)
    # Get Visualizer
    visualizer = get_visualizer(args)
    # Initialize Tester
    test = initialize_tester(config)
    test = test(
        model=model,
        config=config,
        data_loader=test_set_loader,
        test_logger=None,
        save_path=save_path,
        visualizer=visualizer,
        additional_args=additional_loader_returns
    )
    _, _, _, metrics_path = test.mvsec_test()
    file = np.genfromtxt(metrics_path, delimiter=',')
    AEE = file[:, 0]
    PE1 = file[:, 1]
    PE2 = file[:, 2]
    outlier = file[:, 4]

    mean_AEE = np.mean(AEE)
    mean_1PE = np.mean(PE1)
    mean_2PE = np.mean(PE2)
    mean_outlier = np.mean(outlier)
    print('The average AEE is {}'.format(mean_AEE))
    print('The average PE1 is {}'.format(mean_1PE))
    print('The average PE2 is {}'.format(mean_2PE))
    print('The average outlier is {}'.format(mean_outlier))


if __name__ == '__main__':
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', default="/storage1/dataset/MVSEC/mvsec_45HZ", type=str, help="Dataset path")
    parser.add_argument('-d', '--dataset', default="mvsec", type=str, help="Which dataset to use: ([dsec]/mvsec)")
    parser.add_argument('-f', '--frequency', default=45, type=int,
                        help="Evaluation frequency of MVSEC dataset ([20]/45) Hz")
    parser.add_argument('-t', '--type', default='standard', type=str, help="Evaluation type ([warm_start]/standard)")
    parser.add_argument('-v', '--visualize', action='store_true',
                        help='Provide this argument s.t. DSEC results are visualized. MVSEC experiments are always visualized.')
    parser.add_argument('-n', '--num_workers', default=0, type=int,
                        help='How many sub-processes to use for data loading')

    parser.add_argument('--position_only', default=False, action='store_false',
                        help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=True, action='store_true',
                        help='use position and content-wise attention')
    parser.add_argument('--num_heads', default=4, help='number of heads in attention and aggregation')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--corr_levels', default=4)
    parser.add_argument('--corr_radius', default=4)

    args = parser.parse_args()

    # Run Test Script
    '''
    输出结果排序:'AEE', 'PE1', 'PE3', 'n_points', 'outlier'
    '''
    test(args)
