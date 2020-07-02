#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 17:22:40 2020

@author: briardoty
"""
import argparse
from NetManager import NetManager
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="/home/briardoty/Source/neuro511-artiphysiology/data/", type=str, help="Set value for data_dir")
parser.add_argument("--net_name", default="vgg11", type=str, help="Set value for net_name")
parser.add_argument("--n_classes", default=10, type=int, help="Set value for n_classes")
parser.add_argument("--case_id", type=str, help="Set value for case_id")
parser.add_argument("--sample", type=int, help="Set value for sample")


def main(case_id, sample, data_dir="/home/briardoty/Source/neuro511-artiphysiology/data/", 
         net_name="vgg11", n_classes=10):
    
    # init net manager
    manager = NetManager(net_name, n_classes, data_dir)
    manager.load_imagenette()
    
    # load the proper net
    # TODO: parametrize epoch to load
    manager.load_net_snapshot(case_id, sample, epoch=0)
    
    # training vars
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(manager.net.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # train
    epochs = 10 # TODO: parametize?
    manager.run_training_loop(criterion, optimizer, exp_lr_scheduler, 
                              n_epochs=epochs, n_snapshots=epochs)
    
    print("net_train.py completed")
    return


if __name__=="__main__":
    args = parser.parse_args()
    print(args)
    main(**vars(args))
    
    