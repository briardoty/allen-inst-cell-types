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
parser.add_argument("--data_dir", default="/home/briardoty/Source/allen-inst-cell-types/data/", 
                    type=str, help="Set value for data_dir")
parser.add_argument("--net_name", default="vgg11", type=str, help="Set value for net_name")
parser.add_argument("--n_classes", default=10, type=int, help="Set value for n_classes")
parser.add_argument("--n_samples", default=10, type=int, help="Set value for n_samples")

# config params without defaults
parser.add_argument("--case", type=str, help="Set value for case")
parser.add_argument("--layer_names", type=str, nargs="+", help="Set value for layer_names")
parser.add_argument("--n_repeat_arr", type=int, nargs="+", help="Set value for n_repeat_arr")
parser.add_argument("--act_fns", type=str, nargs="+", help="Set value for act_fns")
parser.add_argument("--act_fn_params", type=str, nargs="+", help="Set value for act_fn_params")


def main(case, layer_names, n_repeat_arr, act_fns, act_fn_params, data_dir, 
         net_name, n_classes, n_samples):
    
    # init net manager
    manager = NetManager(net_name, n_classes, data_dir, pretrained=True)
    
    # build and save nets
    if (case.startswith("control")):
        # control nets are unmodified
        for i in range(n_samples):
            gen_control_net(manager, case, i)
    
    else:
        # mixed nets require modification
        for n_repeat in n_repeat_arr:
            
            layers = "-".join(layer_names)
            
            # each layer/n_repeat deserves its own case id
            case_id = f"{case}_{layers}_nr-{n_repeat}"
            
            for i in range(n_samples):
                gen_net(manager, case_id, i, layer_names, n_repeat, act_fns, 
                        act_fn_params)
            
    print(f"gen_nets.py completed case {case}")
    return

def gen_control_net(manager, case, i):
    # init net
    manager.init_net(case, i)
    
    # save
    manager.save_net_snapshot()

def gen_net(manager, case_id, i, layer_names, n_repeat, act_fns, act_fn_params):
    # init net
    manager.init_net(case_id, i)
    
    # modify
    manager.replace_layers(layer_names, n_repeat, act_fns, act_fn_params)
    
    # save
    manager.save_net_snapshot()

if __name__=="__main__":
    args = parser.parse_args()
    print(args)
    main(**vars(args))








