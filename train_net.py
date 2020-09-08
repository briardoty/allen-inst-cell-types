#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 17:22:40 2020

@author: briardoty
"""
import sys
import os
import numpy as np
import argparse
from modules.NetManager import NetManager
from modules.util import get_training_vars
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="/home/briardoty/Source/neuro511-artiphysiology/data/", type=str, help="Set value for data_dir")
parser.add_argument("--dataset", type=str, required=True, help="Set dataset")
parser.add_argument("--net_name", default="vgg11", type=str, help="Set value for net_name")
parser.add_argument("--n_classes", default=10, type=int, help="Set value for n_classes")
parser.add_argument("--epochs", default=10, type=int, help="Set value for epochs")
parser.add_argument("--train_frac", default=1., type=float, help="Set value for train_frac")
parser.add_argument("--lr", type=float)
parser.add_argument("--momentum", type=float, required=True)
parser.add_argument("--lr_step_size", type=int, required=True)
parser.add_argument("--lr_gamma", type=float, required=True)
parser.add_argument("--batch_size", type=int, required=True)
parser.add_argument("--net_filepath", type=str, help="Set value for net_filepath")
parser.add_argument("--scheme", type=str, help="Set scheme", required=True)


def main(net_filepath, data_dir, net_name, n_classes, epochs, train_frac,
         lr, lr_step_size, lr_gamma, batch_size, scheme, dataset, momentum):
    
    # init net manager
    manager = NetManager(dataset, net_name, n_classes, data_dir, scheme)
    manager.load_dataset(batch_size)
    
    # load the proper net
    manager.load_net_snapshot_from_path(net_filepath)

    # set initial lr
    if manager.initial_lr is not None:
        initial_lr = manager.initial_lr
        print(f"Using computed initial LR of {initial_lr}")
    else:
        initial_lr = lr

    (criterion, optimizer, scheduler) = get_training_vars(scheme, 
        manager, initial_lr, lr_step_size, lr_gamma, momentum)

    # train
    manager.run_training_loop(criterion, optimizer, scheduler, train_frac, 
        end_epoch=epochs)
    
    print("net_train.py completed")
    return


if __name__=="__main__":
    args = parser.parse_args()
    print(args)
    main(**vars(args))
    
    