#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 17:22:40 2020

@author: briardoty
"""
import argparse
from modules.NetManager import NetManager
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="/home/briardoty/Source/neuro511-artiphysiology/data/", type=str, help="Set value for data_dir")
parser.add_argument("--net_name", default="vgg11", type=str, help="Set value for net_name")
parser.add_argument("--n_classes", default=10, type=int, help="Set value for n_classes")
parser.add_argument("--net_filepath", type=str, help="Set value for net_filepath")


def main(net_filepath, data_dir, net_name, n_classes):
    
    # init net manager
    manager = NetManager(net_name, n_classes, data_dir)
    manager.load_imagenette()
    
    # load the proper net
    manager.load_net_snapshot_from_path(net_filepath)
    
    # set response hooks
    manager.set_output_hook("relu10")
    manager.set_input_hook("dropout2")
    
    # training vars
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(manager.net.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # train
    manager.train_net(criterion, optimizer, scheduler, batches=2)
    
    # save responses
    manager.save_net_responses()
    
    print("sanity_check.py completed")
    return


if __name__=="__main__":
    args = parser.parse_args()
    print(args)
    main(**vars(args))
    
    