#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 12:11:26 2020

@author: briardoty
"""
import argparse
from NetManager import NetManager
from MixedActivationLayer import MixedActivationLayer

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="/home/briardoty/Source/neuro511-artiphysiology/data/", type=str, help="Set value for data_dir")
parser.add_argument("--net_name", default="vgg11", type=str, help="Set value for net_name")
parser.add_argument("--n_classes", default=10, type=int, help="Set value for n_classes")
parser.add_argument("--n_samples", default=10, type=int, help="Set value for n_samples")


def main(data_dir="/home/briardoty/Source/neuro511-artiphysiology/data/", 
         net_name="vgg11", n_classes=10, n_samples=10):
    
    # init net manager
    manager = NetManager(net_name, n_classes, data_dir, pretrained=True)
    
    case_id = "control"
    
    # build and save nets
    for i in range(n_samples):
        manager.init_net(case_id, i+1)
        manager.save_net_snapshot()
    
    print("candidate_net_gen.py completed")
    return

if __name__=="__main__":
    args = parser.parse_args()
    print(args)
    main(**vars(args))