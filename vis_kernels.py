#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 14:24:38 2020

@author: briardoty
"""
import argparse
from modules.Visualizer import Visualizer
from modules.NetManager import nets

# general params with defaults
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="/home/briardoty/Source/neuro511-artiphysiology/data/", type=str, help="Set value for data_dir")
parser.add_argument("--net_name", default="vgg11", type=str, help="Set value for net_name")
parser.add_argument("--n_classes", default=10, type=int, help="Set value for n_classes")

# required params
parser.add_argument("--case_id", type=str, help="Set value for case_id")
parser.add_argument("--layer_name", type=str, help="Set value for layer_name")


def main(case_id, layer_name,
         data_dir="/home/briardoty/Source/neuro511-artiphysiology/data/", 
         net_name="vgg11", n_classes=10):
    
    # init visualizer
    visualizer = Visualizer(data_dir, net_name, n_classes, True)
    
    # plot
    visualizer.plot_filters(nets[net_name]["layers_of_interest"][layer_name],
                            case_id, sample=1, epoch=0)
    visualizer.plot_filters(nets[net_name]["layers_of_interest"][layer_name],
                            case_id, sample=1, epoch=9)
    
    print(f"vis_kernels.py completed")
    return

if __name__=="__main__":
    args = parser.parse_args()
    print(args)
    main(**vars(args))

