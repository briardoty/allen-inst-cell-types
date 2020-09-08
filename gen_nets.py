"""
Created on Wed Jul  1 12:11:26 2020

@author: briardoty
"""
import argparse
from modules.NetManager import NetManager
from modules.util import get_training_vars
import numpy as np


# general params
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="/home/briardoty/Source/allen-inst-cell-types/data/", 
                    type=str, help="Set value for data_dir")
parser.add_argument("--net_name", type=str, help="Set value for net_name", required=True)
parser.add_argument("--n_classes", default=10, type=int, help="Set value for n_classes")
parser.add_argument("--n_samples", default=10, type=int, help="Set value for n_samples")
parser.add_argument("--scheme", type=str, help="Set value for scheme", required=True)

# config params
parser.add_argument("--case", type=str, help="Set value for case")
parser.add_argument("--layer_names", type=str, nargs="+", help="Set value for layer_names")
parser.add_argument("--n_repeat", type=int)
parser.add_argument("--act_fns", type=str, nargs="+", help="Set value for act_fns")
parser.add_argument("--act_fn_params", type=str, nargs="+", help="Set value for act_fn_params")
parser.add_argument("--dataset", type=str, required=True, help="Set dataset")
parser.add_argument("--spatial", dest="spatial", action="store_true")
parser.set_defaults(spatial=False)

# pretrained is a PITA since it's a bool
pretrained_parser = parser.add_mutually_exclusive_group(required=False)
pretrained_parser.add_argument('--pretrained', dest='pretrained', action='store_true')
pretrained_parser.add_argument('--untrained', dest='pretrained', action='store_false')
parser.set_defaults(pretrained=False)

def main(case, layer_names, n_repeat, act_fns, act_fn_params, data_dir, 
         net_name, n_classes, n_samples, pretrained, scheme, dataset,
         spatial):
    
    # init net manager
    manager = NetManager(dataset, net_name, n_classes, data_dir, scheme, 
        pretrained)
    manager.load_dataset()
    
    # build and save nets
    lr_arr = []
    net_filepaths = []
    for i in range(n_samples):

        # init net
        manager.init_net(case, i)
        
        # modify layers
        if (act_fns is not None and len(act_fns) > 0):
            manager.replace_act_layers(n_repeat, act_fns, act_fn_params, spatial)
        
        # find initial learning rate
        lr_low = 1e-7
        lr_high = 0.01
        (criterion, optimizer, _) = get_training_vars(scheme, manager, 
            lr_low)
        found_lr = manager.find_initial_lr(criterion, optimizer, lr_low, lr_high)
        lr_arr.append(found_lr)

        # save
        manager.save_net_snapshot()
        net_filepaths.append(manager.get_net_filepath())

    # determine mean starting lr, add it to network snapshots
    mean_lr = np.mean(lr_arr)
    std_dev_lr = np.std(lr_arr)
    print(f"Mean initial LR of {mean_lr} has std dev of {std_dev_lr}.")
    for net_filepath in net_filepaths:
        
        # load snapshot
        manager.load_net_snapshot_from_path(net_filepath)

        # append lr
        manager.initial_lr = mean_lr

        # re-save snapshot
        manager.save_net_snapshot()

    print(f"gen_nets.py completed case {case}")
    return   

if __name__=="__main__":
    args = parser.parse_args()
    print(args)
    main(**vars(args))








