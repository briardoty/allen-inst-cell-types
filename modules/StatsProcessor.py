# -*- coding: utf-8 -*-
import torch
import os
import pandas as pd
import re
import numpy as np
import json
from scipy.stats import ttest_ind
import math
from statsmodels.stats.multitest import multipletests

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
        self.exclude_slug = "(exclude)"
    
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
    
    def load_max_acc_df_ungrouped(self, refresh=True):

        # optional refresh
        if refresh:
            self.refresh_max_acc_df()

        # load
        sub_dir = os.path.join(self.data_dir, "dataframes/")
        df = pd.read_csv(os.path.join(sub_dir, "max_acc_df.csv"))
        with open(os.path.join(sub_dir, "case_dict.json"), "r") as json_file:
            case_dict = json.load(json_file)

        # ??
        gidx_cols = ["dataset", "net_name", "train_scheme", "case", "sample"]
        df.set_index(gidx_cols, inplace=True)

        return df, case_dict, gidx_cols

    def load_max_acc_df(self, refresh_df=True):
        """
        Loads dataframe with max validation accuracy for different 
        experimental cases.

        Args:
            refresh

        Returns:
            df_stats (dataframe): Dataframe containing max accuracy.
            case_dict (dict): Dict() of act fn names to their params
        """

        # optional refresh
        if refresh_df:
            self.refresh_max_acc_df()

        # load
        sub_dir = os.path.join(self.data_dir, "dataframes/")
        acc_df = pd.read_csv(os.path.join(sub_dir, "max_acc_df.csv"))
        with open(os.path.join(sub_dir, "case_dict.json"), "r") as json_file:
            case_dict = json.load(json_file)

        # aggregate
        gidx_cols = ["dataset", "net_name", "train_scheme", "case", "is_mixed", "cross_fam"]
        df_stats = acc_df.groupby(gidx_cols).agg(
            { "max_val_acc": [np.mean, np.std],
              "max_pred": [np.mean],
              "linear_pred": [np.mean],
              "max_pred_p_val": np.mean,
              "linear_pred_p_val": np.mean })

        # benjamini-hochberg correction
        mixed_idx = df_stats.query("is_mixed == True").index
        pvals = df_stats.loc[mixed_idx]["max_pred_p_val","mean"].values
        rej_h0, pval_corr = multipletests(pvals, alpha=0.05, method="fdr_bh")[:2]
        df_stats.loc[mixed_idx, "max_pred_p_val_corr"] = pval_corr
        df_stats.loc[mixed_idx, "max_pred_rej_h0"] = rej_h0

        pvals = df_stats.loc[mixed_idx]["linear_pred_p_val","mean"].values
        rej_h0, pval_corr = multipletests(pvals, alpha=0.05, method="fdr_bh")[:2]
        df_stats.loc[mixed_idx, "linear_pred_p_val_corr"] = pval_corr
        df_stats.loc[mixed_idx, "linear_pred_rej_h0"] = rej_h0

        df_stats.drop(columns=["max_pred_p_val", "linear_pred_p_val"], inplace=True)

        return df_stats, case_dict, gidx_cols

    def refresh_max_acc_df(self):
        """
        Refreshes dataframe with max validation accuracy.
        """

        acc_arr = []
        case_dict = dict()

        # walk dir looking for saved net stats
        net_dir = os.path.join(self.data_dir, f"nets/")
        for root, _, files in os.walk(net_dir):
            
            # only interested in locations files are saved
            if len(files) <= 0:
                continue
            
            slugs = root.split("/")

            # exclude some dirs...
            if any(self.exclude_slug in slug for slug in slugs):
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
                    i_max = np.argmax(perf_stats[:,0])
                    (val_acc, val_loss, train_acc, train_loss) = perf_stats[i_max]
                    acc_arr.append([dataset, net_name, train_scheme, case, sample, val_acc])
                except ValueError:
                    print(f"Max entry in {case} {sample} perf_stats did not match expectations.")
                    continue

        # make dataframe
        acc_df = pd.DataFrame(acc_arr, columns=["dataset", "net_name", "train_scheme", "case", "sample", "max_val_acc"])

        # process
        # 1. mark mixed nets
        acc_df["is_mixed"] = [len(case_dict[c]["act_fns"]) > 1 if case_dict.get(c) is not None else False for c in acc_df["case"]]
        acc_df["cross_fam"] = [len(case_dict[c]["act_fns"]) == len(set(case_dict[c]["act_fns"])) if case_dict.get(c) is not None else False for c in acc_df["case"]]

        # 2. add columns
        acc_df["max_pred"] = np.nan
        acc_df["linear_pred"] = np.nan
        acc_df["max_pred_p_val"] = np.nan
        acc_df["linear_pred_p_val"] = np.nan

        # 2.9. multi-index
        midx_cols = ["dataset", "net_name", "train_scheme", "case", "sample"]
        acc_df.set_index(midx_cols, inplace=True)

        # 3. predictions for mixed cases
        for midx in acc_df.query("is_mixed == True").index.values:

            # break up multi-index
            d, n, sch, c, s = midx
            
            # skip if already predicted
            if not math.isnan(acc_df.at[midx, "max_pred"]):
                continue

            # get rows in this mixed case
            mixed_case_rows = acc_df.loc[(d, n, sch, c)]
            
            # get component case rows
            component_cases = get_component_cases(case_dict, c)
            component_rows = acc_df.query(f"is_mixed == False") \
                .query(f"dataset == '{d}'") \
                .query(f"net_name == '{n}'") \
                .query(f"train_scheme == '{sch}'") \
                .query(f"case in {component_cases}")

            # flag to indicate whether row used in prediction yet
            component_rows["used"] = False

            # make a prediction for each sample in this mixed case
            for i in range(len(mixed_case_rows)):
                mixed_case_row = mixed_case_rows.iloc[i]

                # choose component row accs
                c_accs = []
                for cc in component_cases:
                    c_row = component_rows \
                        .query(f"case == '{cc}'") \
                        .query(f"used == False")
                    
                    if len(c_row) == 0:
                        break
                    c_row = c_row.sample()
                    c_accs.append(c_row.max_val_acc.values[0])

                    # mark component row as used in prediction
                    component_rows.at[c_row.index.values[0], "used"] = True

                if len(c_accs) == 0:
                    break

                acc_df.at[(d, n, sch, c, mixed_case_row.name), "max_pred"] = np.max(c_accs)
                acc_df.at[(d, n, sch, c, mixed_case_row.name), "linear_pred"] = np.mean(c_accs)

            # significance
            t, p = ttest_ind(acc_df.at[(d, n, sch, c), "max_val_acc"], acc_df.at[(d, n, sch, c), "max_pred"])
            if t < 0:
                p = 1. - p / 2.
            else:
                p = p / 2.
            acc_df.loc[(d, n, sch, c), "max_pred_p_val"] = p

            t, p = ttest_ind(acc_df.at[(d, n, sch, c), "max_val_acc"], acc_df.at[(d, n, sch, c), "linear_pred"])
            if t < 0:
                p = 1. - p / 2.
            else:
                p = p / 2.
            acc_df.loc[(d, n, sch, c), "linear_pred_p_val"] = p

        # save things
        self.save_df("max_acc_df.csv", acc_df)
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

    def load_accuracy_df(self, dataset, net_name, schemes, cases, refresh):
        """
        Loads dataframe with validation accuracy for different 
        experimental cases over epochs.

        Args:
            refresh

        Returns:
            df (dataframe): Dataframe containing accuracy trajectories
        """
        # optional refresh
        if refresh:
            self.refresh_accuracy_df()

        # load
        sub_dir = os.path.join(self.data_dir, "dataframes/")
        df = pd.read_csv(os.path.join(sub_dir, "acc_df.csv"))

        # filter
        df.drop(columns="Unnamed: 0", inplace=True)
        df = df.query(f"dataset == '{dataset}'") 
        df = df.query(f"net_name == '{net_name}'")
        df = df.query(f"train_scheme in {schemes}")
        df = df.query(f"case in {cases}")

        return df

    def refresh_accuracy_df(self):
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
        net_dir = os.path.join(self.data_dir, f"nets/")
        for root, _, files in os.walk(net_dir):
            
            # only interested in locations files are saved
            if len(files) <= 0:
                continue
            
            slugs = root.split("/")

            # exclude some dirs...
            if any(self.exclude_slug in slug for slug in slugs):
                continue
            
            # consider all files...
            for filename in files:

                # ...as long as they are perf_stats
                if not "perf_stats" in filename:
                    continue
                
                filepath = os.path.join(root, filename)
                stats_dict = np.load(filepath, allow_pickle=True).item()
                
                dataset = stats_dict.get("dataset") if stats_dict.get("dataset") is not None else "imagenette2"
                net_name = stats_dict.get("net_name")
                train_scheme = stats_dict.get("train_scheme") if stats_dict.get("train_scheme") is not None else "sgd"
                case = stats_dict.get("case")
                sample = stats_dict.get("sample")

                perf_stats = stats_dict.get("perf_stats")
                for epoch in range(len(perf_stats)):
                    (val_acc, val_loss, train_acc, train_loss) = perf_stats[epoch]
                    acc_arr.append([dataset, net_name, train_scheme, case, sample, epoch, val_acc, train_acc])
                
        # make dataframe
        acc_df = pd.DataFrame(acc_arr, columns=["dataset", "net_name", "train_scheme", "case", "sample", "epoch", "val_acc", "train_acc"])
        
        # save df
        self.save_df("acc_df.csv", acc_df)

    def reduce_snapshots(self):

        # walk networks directory
        net_dir = os.path.join(self.data_dir, f"nets/")
        for root, _, files in os.walk(net_dir):
            
            # only interested in locations files are saved
            if len(files) <= 0:
                continue
            
            slugs = root.split("/")
            
            # consider all files...
            for filename in files:

                # ...as long as they are snapshots
                if not filename.endswith(".pt"):
                    continue
                
                epoch = get_epoch_from_filename(filename)

                if epoch is None:
                    continue

                if epoch % 10 == 0:
                    continue
                else:
                    # delete
                    filepath = os.path.join(root, filename)
                    print(f"Deleting {filepath}")
                    os.remove(filepath)

if __name__=="__main__":
    
    processor = StatsProcessor("/home/briardoty/Source/allen-inst-cell-types/data_mountpoint", 10)
    
    # processor.reduce_snapshots()

    # processor.load_accuracy_df(["control2"])
    df, _, _ = processor.load_max_acc_df(refresh_df=True)
    # processor.load_weight_change_df(["control1"])
    
    
    
    
    
    
    
    
