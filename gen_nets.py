#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 12:11:26 2020

@author: briardoty
"""
import argparse
from modules.NetManager import NetManager

# general params with defaults
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="/home/briardoty/Source/allen-inst-cell-types/data/", type=str, help="Set value for data_dir")
parser.add_argument("--net_name", default="vgg11", type=str, help="Set value for net_name")
parser.add_argument("--n_classes", default=10, type=int, help="Set value for n_classes")
parser.add_argument("--n_samples", default=10, type=int, help="Set value for n_samples")

# required config params
parser.add_argument("--case_id", type=str, help="Set value for case_id")
parser.add_argument("--layer_name", type=str, help="Set value for layer_name")
parser.add_argument("--n_repeat_arr", type=int, nargs="+", help="Set value for n_repeat_arr")
parser.add_argument("--act_fns", type=str, nargs="+", help="Set value for act_fns")
parser.add_argument("--act_fn_params", type=str, nargs="+", help="Set value for act_fn_params")


def main(case_id, layer_name, n_repeat_arr, act_fns, act_fn_params, 
         data_dir="/home/briardoty/Source/allen-inst-cell-types/data/", 
         net_name="vgg11", n_classes=10, n_samples=10):
    
    # init net manager
    manager = NetManager(net_name, n_classes, data_dir, pretrained=True)
    
    # build and save nets
    for n_repeat in n_repeat_arr:
        
        # each n_repeat deserves its own case
        case_id_nr = f"{case_id}_nr-{n_repeat}"
        
        for i in range(n_samples):
            
            # init net
            manager.init_net(case_id_nr, i+1)
            
            # modify
            manager.replace_layer(layer_name, n_repeat, act_fns, act_fn_params)
            
            # save
            manager.save_net_snapshot()
    
    print(f"gen_nets.py completed case {case_id}")
    return

if __name__=="__main__":
    args = parser.parse_args()
    print(args)
    main(**vars(args))








