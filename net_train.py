#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 17:22:40 2020

@author: briardoty
"""
import argparse
from NetManager import NetManager
from MixedActivationLayer import MixedActivationLayer
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="/home/briardoty/Source/neuro511-artiphysiology/data/", type=str, help="Set value for data_dir")
parser.add_argument("--net_name", default="vgg11", type=str, help="Set value for net_name")
parser.add_argument("--n_classes", default=10, type=int, help="Set value for n_classes")

def main(data_dir="/home/briardoty/Source/neuro511-artiphysiology/data/", 
         net_name="vgg11", n_classes=10):

    print(data_dir)
    print(net_name)
    print(str(n_classes))
    
    # init net manager
    manager = NetManager(net_name, n_classes, data_dir, pretrained=True)
    manager.load_imagenette()
    
    # training vars
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(manager.net.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # evaluate
    manager.evaluate_net(criterion)

    # modify activation functions
    layer_to_modify = 19
    manager.replace_layer(layer_to_modify)
    
    # evaluate
    manager.evaluate_net(criterion)
    
    # train
    manager.run_training_loop(0, criterion, optimizer, exp_lr_scheduler, 10)
    
    # evaluate
    manager.evaluate_net(criterion)
    
    print("Test run successful!!!")
    return


if __name__=="__main__":
    args = parser.parse_args()
    print(args)
    main(**vars(args))
    
    