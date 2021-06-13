import sys
import os
import numpy as np
import argparse
from modules.NetManager import NetManager
from modules.util import get_seed_for_sample
import torch.nn as nn
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="/home/briardoty/Source/allen-inst-cell-types/data_mountpoint/", type=str, help="Set value for data_dir")
parser.add_argument("--dataset", type=str, required=True, help="Set dataset")
parser.add_argument("--net_name", default="vgg11", type=str, help="Set value for net_name")
parser.add_argument("--batch_size", type=int, required=True)
parser.add_argument("--net_filepath", type=str, help="Set value for net_filepath")


def main(net_filepath, data_dir, net_name, batch_size, dataset):
    
    # init net manager
    manager = NetManager(dataset, net_name, None, None, data_dir, None)
    
    # load the proper net
    manager.load_net_snapshot_from_path(net_filepath)

    # seed
    seed = get_seed_for_sample(data_dir, manager.sample)
    manager.seed_everything(seed)

    # load dataset
    manager.load_activation_dataset(batch_size)

    # training vars
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(manager.net.parameters(), lr=1e-7)

    # set output hooks
    manager.set_activation_hooks()

    # train
    manager.train_net(criterion, optimizer, None, 1.0)

    # save output
    manager.save_arr("activation_dict", manager.activation_dict, False)

    print("save_net_activations.py completed")
    return


if __name__=="__main__":
    # args = parser.parse_args()
    # print(args)
    # main(**vars(args))
    
    net_filepath = "/home/briardoty/Source/allen-inst-cell-types/data_mountpoint/nets/cifar10/sticknet8/adam-lr-avg/component-tanh/tanh1/sample-1/sticknet8_case-tanh1_sample-1_epoch-500.pt"
    data_dir = "/home/briardoty/Source/allen-inst-cell-types/data_mountpoint"
    net_name = "sticknet8"
    batch_size = 128
    dataset = "cifar10"

    main(net_filepath, data_dir, net_name, batch_size, dataset)
    