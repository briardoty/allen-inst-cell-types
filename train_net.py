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
parser.add_argument("--epochs", default=10, type=int, help="Set value for epochs")
parser.add_argument("--train_frac", default=1., type=float, help="Set value for train_frac")
parser.add_argument("--net_filepath", type=str, help="Set value for net_filepath")


def main(net_filepath, data_dir, net_name, n_classes, epochs, train_frac):
    
    # init net manager
    manager = NetManager(net_name, n_classes, data_dir)
    manager.load_imagenette()
    
    # load the proper net
    manager.load_net_snapshot_from_path(net_filepath)
    
    # training vars
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(manager.net.parameters(), lr=0.001)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.6)
    
    # train
    manager.run_training_loop(criterion, optimizer, exp_lr_scheduler, 
                              train_frac, n_epochs=epochs)
    
    print("net_train.py completed")
    return


if __name__=="__main__":
    args = parser.parse_args()
    print(args)
    main(**vars(args))
    
    