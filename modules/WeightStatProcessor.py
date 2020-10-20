# -*- coding: utf-8 -*-
import torch
import os
import pandas as pd
import re
import numpy as np
import json
from scipy.stats import ttest_ind
import math

try:
    from .NetManager import NetManager, nets
except:
    from NetManager import NetManager, nets

try:
    from .MixedActivationLayer import generate_masks
except:
    from MixedActivationLayer import generate_masks

try:
    from .util import ensure_sub_dir
except:
    from util import ensure_sub_dir

def get_epoch_from_filename(filename):
    
    epoch = re.search(r"\d+\.pt$", filename)
    epoch = int(epoch.group().split(".")[0]) if epoch else None
    
    return epoch
    
def get_first_epoch(net_filenames):
    
    for filename in net_filenames:
        
        epoch = get_epoch_from_filename(filename)
        if epoch == 0:
            return filename

def get_last_epoch(net_filenames):
    
    max_epoch = -1
    last_net_filename = None
    
    for filename in net_filenames:
        
        epoch = get_epoch_from_filename(filename)

        if epoch is None:
            continue

        if epoch > max_epoch:
            max_epoch = epoch
            last_net_filename = filename

    return last_net_filename


class WeightStatProcessor():
    """
    Class to handle processing network snapshots into meaningful statistics.
    """
    
    def __init__(self, data_dir, n_classes):
        
        self.data_dir = data_dir
        self.n_classes = n_classes
        self.exclude_slug = "(exclude)"
    
    def load_weight_df(self, net_name, case, train_schemes):
        """
        Loads a dataframe containing the mean absolute weights for each
        cell type across layers
        """
        # for loading nets/metadata
        mgr = NetManager(net_name, self.data_dir, None)

        weights_arr = []
        state_keys = list(nets["vgg11"]["state_keys"].keys())

        # walk dir for final snapshots
        net_dir = os.path.join(self.data_dir, f"nets/{net_name}")
        for root, dirs, files in os.walk(net_dir):
        
            # only interested in locations files (nets) are saved
            if len(files) <= 0:
                continue

            slugs = root.split("/")

            # only interested in the given training schemes
            if not any(t in slugs for t in train_schemes):
                continue

            # only interested in the given case
            if not case in slugs:
                continue
            
            # get final snapshot
            last_net_filename = get_last_epoch(files)
            if last_net_filename is None:
                continue
            
            last_net_path = os.path.join(root, last_net_filename)
            last_net = mgr.load_snapshot_metadata(last_net_path, True)
            
            # filter to just the weights we want
            last_net["state_dict"] = { my_key: last_net["state_dict"][my_key] for my_key in state_keys }
            
            # separate by cell type
            sample = last_net.get("sample")
            modified_layers = last_net["modified_layers"]
            n_fns = len(modified_layers["act_fns"])
            n_repeat = modified_layers["n_repeat"]

            for layer_name, layer in last_net["state_dict"].items():
                
                n_features = len(layer) # TODO: this will not work for mixing within spatial dim of feature map
                masks = generate_masks(n_features, n_fns, n_repeat)

                for mask, act_fn in zip(masks, modified_layers["act_fns"]):

                    # weight stats
                    cells = layer[mask]
                    avg_weight = torch.mean(torch.abs(cells)).item()
                    sem_weight = torch.std(torch.abs(cells)).item() / np.sqrt(cells.numel())

                    # add to data array
                    weights_arr.append([sample, layer_name, act_fn, avg_weight, sem_weight])

        # make df
        cols = ["sample", "layer", "act_fn", "avg_weight", "sem_weight"]
        df = pd.DataFrame(weights_arr, columns=cols)

        # group and compute stats
        df.set_index(["layer", "act_fn"], inplace=True)
        df_groups = df.groupby(["layer", "act_fn"])
        df_stats = df_groups.agg({ "avg_weight": np.mean, "sem_weight": np.mean })

        return df_stats

    def load_weight_change_df(self, net_name, case_ids, train_schemes):
        """
        Loads a dataframe containing the mean, absolute weight changes
        over training for each layer. 

        Args:
            case_ids (list): Experimental cases to include in figure.

        Returns:
            weight_change_df (dataframe).

        """
        # for loading nets/metadata
        mgr = NetManager(net_name, self.data_dir, None)

        weight_change_arr = []
        
        # walk dir looking for net snapshots
        net_dir = os.path.join(self.data_dir, f"nets/{net_name}")
        for root, dirs, files in os.walk(net_dir):
            
            # only interested in locations files (nets) are saved
            if len(files) <= 0:
                continue

            slugs = root.split("/")

            # only interested in the given training schemes
            if not any(t in slugs for t in train_schemes):
                continue

            # only interested in the given cases
            if not any(c in slugs for c in case_ids):
                continue
            
            # nets are all from one sample, get just the first and last
            first_net_filename = get_first_epoch(files)
            last_net_filename = get_last_epoch(files)
            
            if first_net_filename is None or last_net_filename is None:
                continue

            first_net_path = os.path.join(root, first_net_filename)
            last_net_path = os.path.join(root, last_net_filename)
            
            first_net = mgr.load_snapshot_metadata(first_net_path, True)
            last_net = mgr.load_snapshot_metadata(last_net_path, True)
            
            # filter to just the weights we want
            layer_keys = [k for k in first_net["state_dict"].keys() if k.endswith(".weight")]
            first_net["state_dict"] = { k: first_net["state_dict"][k] for k in layer_keys }
            last_net["state_dict"] = { k: last_net["state_dict"][k] for k in layer_keys }
            
            # diff weights
            diff_net = [last_net["state_dict"][layer] - first_net["state_dict"][layer] for layer in layer_keys]

            # avg weights
            avg_weights = [torch.mean(torch.abs(layer)).item() for layer in diff_net]
            sem_weights = [torch.std(torch.abs(layer)).item() / np.sqrt(layer.numel()) for layer in diff_net]

            # add to arr
            case = first_net.get("case") if first_net.get("case") is not None else "control"
            sample = first_net.get("sample")
            scheme = first_net.get("train_scheme") if first_net.get("train_scheme") is not None else "sgd"
            weight_change_arr.append([scheme, case, sample] + avg_weights + sem_weights)

        # make df
        layer_keys = layer_keys + [f"{key}.sem" for key in layer_keys]
        cols = ["train_scheme", "case", "sample"] + layer_keys
        weight_change_df = pd.DataFrame(weight_change_arr, columns=cols)

        # group and compute stats
        weight_change_df.set_index(["train_scheme", "case"], inplace=True)
        df_groups = weight_change_df.groupby(["train_scheme", "case"])
        df_stats = df_groups[layer_keys].agg("mean")

        return df_stats
    
    
    
    
    
    
    
