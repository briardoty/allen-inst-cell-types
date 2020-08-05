# -*- coding: utf-8 -*-
import torch
import os
import pandas as pd
import re
import numpy as np

try:
    from .NetManager import NetManager, nets
except:
    from NetManager import NetManager, nets

try:
    from .MixedActivationLayer import generate_masks
except:
    from MixedActivationLayer import generate_masks

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
    
class StatsProcessor(NetManager):
    """
    Class to handle processing network snapshots into meaningful statistics.
    Subclass of NetManager to inherit snapshot loading functionality.
    """
    
    def __init__(self, net_name, n_classes, data_dir, train_scheme, pretrained=False):
        
        super(StatsProcessor, self).__init__(net_name, n_classes, data_dir, 
            train_scheme, pretrained)
    
    def load_weight_df(self, case):
        """
        Loads a dataframe containing the mean absolute weights for each
        cell type across layers
        """
        weights_arr = []
        state_keys = list(nets["vgg11"]["state_keys"].keys())

        # walk dir for final snapshots
        net_dir = os.path.join(self.data_dir, f"nets/{self.net_name}/{self.train_scheme}/{case}")
        for root, dirs, files in os.walk(net_dir):
        
            # only interested in locations files (nets) are saved
            if len(files) <= 0:
                continue
            
            # get final snapshot
            last_net_filename = get_last_epoch(files)
            if last_net_filename is None:
                continue
            
            last_net_path = os.path.join(root, last_net_filename)
            last_net = self.load_snapshot_metadata(last_net_path, True)
            
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

    def load_weight_change_df(self, case_ids):
        """
        Loads a dataframe containing the mean, absolute weight changes
        over training for each layer. 

        Args:
            case_ids (list): Experimental cases to include in figure.

        Returns:
            weight_change_df (dataframe).

        """
        weight_change_arr = []
        state_keys = list(nets["vgg11"]["state_keys"].keys())
        
        # walk dir looking for net snapshots
        net_dir = os.path.join(self.data_dir, f"nets/{self.net_name}/{self.train_scheme}")
        for root, dirs, files in os.walk(net_dir):
            
            # only interested in locations files (nets) are saved
            if len(files) <= 0:
                continue
            
            # only interested in the given cases
            slugs = root.split("/")
            if not any(c in slugs for c in case_ids):
                continue
            
            # nets are all from one sample, get just the first and last
            first_net_filename = get_first_epoch(files)
            last_net_filename = get_last_epoch(files)
            
            if first_net_filename is None or last_net_filename is None:
                continue

            first_net_path = os.path.join(root, first_net_filename)
            last_net_path = os.path.join(root, last_net_filename)
            
            first_net = self.load_snapshot_metadata(first_net_path, True)
            last_net = self.load_snapshot_metadata(last_net_path, True)
            
            # filter to just the weights we want
            first_net["state_dict"] = { my_key: first_net["state_dict"][my_key] for my_key in state_keys }
            last_net["state_dict"] = { my_key: last_net["state_dict"][my_key] for my_key in state_keys }
            
            # diff weights
            diff_net = [last_net["state_dict"][layer] - first_net["state_dict"][layer] for layer in state_keys]

            # avg weights
            avg_weights = [torch.mean(torch.abs(layer)).item() for layer in diff_net]
            sem_weights = [torch.std(torch.abs(layer)).item() / np.sqrt(layer.numel()) for layer in diff_net]

            # add to arr
            case = first_net.get("case") if first_net.get("case") is not None else "control"
            sample = first_net.get("sample")
            weight_change_arr.append([case, sample] + avg_weights + sem_weights)

        # make df
        layer_keys = state_keys + [f"{key}.sem" for key in state_keys]
        cols = ["case", "sample"] + layer_keys
        weight_change_df = pd.DataFrame(weight_change_arr, columns=cols)

        # group and compute stats
        weight_change_df.set_index("case", inplace=True)
        df_groups = weight_change_df.groupby("case")
        df_stats = df_groups[layer_keys].agg("mean")

        return df_stats
    
    def load_final_acc_df(self, cases):
        """
        Loads dataframe with final validation accuracy for different 
        experimental cases.

        Args:
            cases (list): Experimental cases to include in figure.

        Returns:
            acc_df (dataframe): Dataframe containing final accuracy.
        """

        acc_arr = []
        act_fn_param_dict = dict()

        # walk dir looking for saved net stats
        net_dir = os.path.join(self.data_dir, f"nets/{self.net_name}/{self.train_scheme}")
        for root, dirs, files in os.walk(net_dir):
            
            # only interested in locations files are saved
            if len(files) <= 0:
                continue
            
            # only interested in the given cases
            slugs = root.split("/")
            if not any(c in slugs for c in cases):
                continue
            
            # consider all files...
            for filename in files:

                # ...as long as they are perf_stats
                if not "perf_stats" in filename:
                    continue
                
                filepath = os.path.join(root, filename)
                stats_dict = np.load(filepath, allow_pickle=True).item()
                
                case = stats_dict.get("case")
                sample = stats_dict.get("sample")
                modified_layers = stats_dict.get("modified_layers")
                act_fn_params = modified_layers.get("act_fn_params")
                act_fn_param_dict[case] = act_fn_params

                perf_stats = stats_dict.get("perf_stats")
                (val_acc, val_loss, train_acc, train_loss) = perf_stats[-1]
                acc_arr.append([case, sample, val_acc])
                
        # make dataframe
        acc_df = pd.DataFrame(acc_arr, columns=["case", "sample", "final_val_acc"])
        
        # aggregate
        acc_df.set_index("case", inplace=True)
        df_groups = acc_df.groupby("case")
        df_stats = df_groups.agg({ "final_val_acc": [np.mean, np.std] })

        return df_stats, act_fn_param_dict

    def load_accuracy_df(self, case_ids, train_schemes):
        """
        Loads dataframe with accuracy over training for different experimental 
        cases.

        Args:
            case_ids (list): Experimental cases to include in figure.
            train_schemes (list): Valid training schemes to include in figure.

        Returns:
            acc_df (dataframe): Dataframe containing validation accuracy.
        """
        acc_arr = []
            
        # walk dir looking for saved net stats
        net_dir = os.path.join(self.data_dir, f"nets/{self.net_name}")
        for root, dirs, files in os.walk(net_dir):
            
            # only interested in locations files are saved
            if len(files) <= 0:
                continue
            
            slugs = root.split("/")

            # only interested in the given training schemes
            if not any(t in slugs for t in train_schemes):
                continue

            # only interested in the given cases
            if not any(c in slugs for c in case_ids):
                continue
            
            # consider all files...
            for filename in files:

                # ...as long as they are perf_stats
                if not "perf_stats" in filename:
                    continue
                
                filepath = os.path.join(root, filename)
                stats_dict = np.load(filepath, allow_pickle=True).item()
                
                train_scheme = stats_dict.get("train_scheme") if stats_dict.get("train_scheme") is not None else "sgd"
                case = stats_dict.get("case")
                sample = stats_dict.get("sample")

                perf_stats = stats_dict.get("perf_stats")
                for epoch in range(len(perf_stats)):
                    (val_acc, val_loss, train_acc, train_loss) = perf_stats[epoch]
                    acc_arr.append([train_scheme, case, sample, epoch, val_acc])
                
        # make dataframe
        acc_df = pd.DataFrame(acc_arr, columns=["train_scheme", "case", "sample", "epoch", "acc"])  
        return acc_df

if __name__=="__main__":
    
    processor = StatsProcessor("vgg11", 10, "/home/briardoty/Source/allen-inst-cell-types/data", "sgd")
    
    processor.load_accuracy_df(["control2"])
    # processor.load_weight_change_df(["control1"])
    
    
    
    
    
    
    
    
