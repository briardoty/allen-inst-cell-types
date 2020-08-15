# -*- coding: utf-8 -*-
import torch
import os
import pandas as pd
import re
import numpy as np
import json

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
    
def get_component_cases(case_dict, case):
    """
    Returns the names of cases that compose the given mixed case

    Args:
        case_dict (dict)
        case: the mixed case 
    """

    # identify "component" cases...
    def param_to_float(p):
        return float(p) if p != "None" else p

    z = list(zip(case_dict[case]["act_fns"], [param_to_float(p) for p in case_dict[case]["act_fn_params"]]))
    component_cases = []

    for k, v in case_dict.items():

        if len(component_cases) >= len(z):
            return component_cases
        
        if (len(v["act_fns"]) == 1 
            and (v["act_fns"][0], param_to_float(v["act_fn_params"][0])) in z
            and "_" not in k): # THIS IS A HACK TO GET RID OF OLD CASES
            component_cases.append(k)

    return component_cases

class StatsProcessor():
    """
    Class to handle processing network snapshots into meaningful statistics.
    """
    
    def __init__(self, data_dir, n_classes):
        
        self.data_dir = data_dir
        self.n_classes = n_classes
    
    def load_weight_df(self, net_name, case, train_schemes):
        """
        Loads a dataframe containing the mean absolute weights for each
        cell type across layers
        """
        # for loading nets/metadata
        mgr = NetManager(net_name, self.n_classes, self.data_dir, None)

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
        mgr = NetManager(net_name, self.n_classes, self.data_dir, None)

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
    
    def load_final_acc_df(self, refresh_df=True):
        """
        Loads dataframe with final validation accuracy for different 
        experimental cases.

        Args:
            refresh

        Returns:
            df_stats (dataframe): Dataframe containing final accuracy.
            case_dict (dict): Dict() of act fn names to their params
        """

        # optional refresh
        if refresh_df:
            self.refresh_final_acc_df()

        # load
        sub_dir = os.path.join(self.data_dir, "dataframes/")
        acc_df = pd.read_csv(os.path.join(sub_dir, "final_acc_df.csv"))
        with open(os.path.join(sub_dir, "case_dict.json"), "r") as json_file:
            case_dict = json.load(json_file)

        # process
        # 1. mark mixed nets
        acc_df.drop(columns="Unnamed: 0", inplace=True)
        acc_df["is_mixed"] = [len(case_dict[c]["act_fns"]) > 1 if case_dict.get(c) is not None else False for c in acc_df["case"]]

        # 2. aggregate
        idx_cols = ["dataset", "net_name", "train_scheme", "case", "is_mixed"]
        df_stats = acc_df.groupby(idx_cols).agg(
            { "final_val_acc": [np.mean, np.std] })
        df_groups = df_stats.groupby(idx_cols)

        # 3. predictions
        linear_preds = []
        linear_stds = []
        max_preds = []
        max_stds = []
        for g in df_groups.groups:
            d, n, s, c, m = g

            # only predict for mixed cases
            if not m:
                linear_preds.append(None)
                max_preds.append(None)
                linear_stds.append(None)
                max_stds.append(None)
                continue
            
            # get component cases and their stats
            component_cases = get_component_cases(case_dict, c)
            component_accs = df_stats["final_val_acc"]["mean"][d][n][s].get(component_cases)
            component_stds = df_stats["final_val_acc"]["std"][d][n][s].get(component_cases)

            # this shouldn't happen much
            if component_accs is None or len(component_cases) == 0 or len(component_accs) == 0:
                linear_preds.append(None)
                max_preds.append(None)
                linear_stds.append(None)
                max_stds.append(None)
                print(f"Component case accuracies do not exist for: {d} {n} {s} {c}")
                continue
            
            # predictions!
            linear_preds.append(component_accs.mean())
            max_preds.append(component_accs.max())
            linear_stds.append(component_stds.mean())
            max_stds.append(component_stds[component_accs.to_list().index(component_accs.max())])

        df_stats["linear_pred"] = linear_preds
        df_stats["max_pred"] = max_preds
        df_stats["linear_std"] = linear_stds
        df_stats["max_std"] = max_stds

        return df_stats, case_dict, idx_cols

    def refresh_final_acc_df(self):
        """
        Refreshes dataframe with final validation accuracy.
        """

        acc_arr = []
        case_dict = dict()

        # walk dir looking for saved net stats
        net_dir = os.path.join(self.data_dir, f"nets/")
        for root, dirs, files in os.walk(net_dir):
            
            # only interested in locations files are saved
            if len(files) <= 0:
                continue
            
            # consider all files...
            for filename in files:

                # ...as long as they are perf_stats
                if not "perf_stats" in filename:
                    continue
                
                filepath = os.path.join(root, filename)
                stats_dict = np.load(filepath, allow_pickle=True).item()
                
                # extract data
                dataset = stats_dict.get("dataset") if stats_dict.get("dataset") is not None else "imagenette2"
                net_name = stats_dict.get("net_name")
                train_scheme = stats_dict.get("train_scheme") if stats_dict.get("train_scheme") is not None else "sgd"
                case = stats_dict.get("case")
                sample = stats_dict.get("sample")
                modified_layers = stats_dict.get("modified_layers")
                if modified_layers is not None:
                    case_dict[case] = {
                        "act_fns": modified_layers.get("act_fns"),
                        "act_fn_params": modified_layers.get("act_fn_params")
                    }

                perf_stats = stats_dict.get("perf_stats")
                try:
                    (val_acc, val_loss, train_acc, train_loss) = perf_stats[-1]
                    acc_arr.append([dataset, net_name, train_scheme, case, sample, val_acc])
                except ValueError:
                    print(f"Final entry in {case} {sample} perf_stats did not match expectations.")
                    continue

        # make dataframe
        acc_df = pd.DataFrame(acc_arr, columns=["dataset", "net_name", "train_scheme", "case", "sample", "final_val_acc"])

        self.save_df("final_acc_df.csv", acc_df)
        self.save_json("case_dict.json", case_dict)

    def save_df(self, name, df):

        sub_dir = ensure_sub_dir(self.data_dir, f"dataframes/")
        filename = os.path.join(sub_dir, name)
        df.to_csv(filename, header=True, columns=df.columns)

    def save_json(self, name, blob):

        sub_dir = ensure_sub_dir(self.data_dir, f"dataframes/")
        filename = os.path.join(sub_dir, name)
        json_obj = json.dumps(blob)
        with open(filename, "w") as json_file:
            json_file.write(json_obj)

    def load_accuracy_df(self, dataset, net_name, cases, train_schemes):
        """
        Loads dataframe with accuracy over training for different experimental 
        cases.

        Args:
            cases (list): Experimental cases to include in figure.
            train_schemes (list): Valid training schemes to include in figure.

        Returns:
            acc_df (dataframe): Dataframe containing validation accuracy.
        """
        acc_arr = []
            
        # walk dir looking for saved net stats
        net_dir = os.path.join(self.data_dir, f"nets/{dataset}/{net_name}")
        for root, dirs, files in os.walk(net_dir):
            
            # only interested in locations files are saved
            if len(files) <= 0:
                continue
            
            slugs = root.split("/")

            # only interested in the given training schemes
            if not any(t in slugs for t in train_schemes):
                continue

            # only interested in the given cases
            if not any(c in slugs for c in cases):
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
    
    
    
    
    
    
    
    
