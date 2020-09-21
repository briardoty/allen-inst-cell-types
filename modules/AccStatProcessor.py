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

class AccStatProcessor():
    """
    Class to handle processing network snapshots into meaningful statistics.
    """
    
    def __init__(self, data_dir, n_classes):
        
        self.data_dir = data_dir
        self.n_classes = n_classes
        self.exclude_slug = "(exclude)"
        self.pct = 90
    
    def load_learning_df(self, pct, refresh=True):
        """

        """
        # optional refresh
        if refresh:
            self.refresh_accuracy_df()

        # load
        sub_dir = os.path.join(self.data_dir, "dataframes/")
        acc_df = pd.read_csv(os.path.join(sub_dir, "acc_df.csv"))
        acc_df.drop(columns="Unnamed: 0", inplace=True)

        # TODO: separate the refresh code for this from max_acc_df???
        with open(os.path.join(sub_dir, "case_dict.json"), "r") as json_file:
            case_dict = json.load(json_file)

        # optional refresh
        if refresh:
            self.refresh_learning_df(acc_df, pct)

        # load
        learning_df = pd.read_csv(os.path.join(sub_dir, "learning_df.csv"))
        learning_df.drop(columns="Unnamed: 0", inplace=True)

        # TODO: process? aggregate? etc.

        return learning_df, case_dict
        
    def refresh_learning_df(self, acc_df, pct):

        learning_arr = []

        index_cols = ["dataset", "net_name", "train_scheme", "case", "sample"]
        acc_df.set_index(index_cols, inplace=True)

        for idx in acc_df.index.unique():

            # identify first epoch to surpass pct% of peak acc
            epochs = acc_df.loc[idx]
            peak_acc = max(epochs["val_acc"])
            pct_acc = (pct / 100.) * peak_acc
            
            # find that epoch
            i_first = next(x for x, val in enumerate(epochs["val_acc"]) if val > pct_acc)

            # append to df array
            learning_arr.append(idx + (peak_acc, i_first))

        # make and save df
        learning_df = pd.DataFrame(learning_arr, 
            columns=["dataset", "net_name", "train_scheme", "case", "sample", "max_val_acc", "epoch_past_pct"])
        self.save_df("learning_df.csv", learning_df)

    def load_max_acc_df_ungrouped(self, refresh=True):

        # optional refresh
        if refresh:
            self.refresh_max_acc_df()

        # load
        sub_dir = os.path.join(self.data_dir, "dataframes/")
        df = pd.read_csv(os.path.join(sub_dir, "max_acc_df.csv"))
        with open(os.path.join(sub_dir, "case_dict.json"), "r") as json_file:
            case_dict = json.load(json_file)

        # index
        idx_cols = ["dataset", "net_name", "train_scheme", "case", "sample"]
        df.set_index(idx_cols, inplace=True)

        return df, case_dict, idx_cols

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
              "max_pred": [np.mean, np.std],
              "linear_pred": [np.mean, np.std],
              "max_pred_p_val": np.mean,
              "linear_pred_p_val": np.mean,

              "epochs_past": [np.mean, np.std],
              "max_pred_epochs_past": [np.mean, np.std],
              "linear_pred_epochs_past": [np.mean, np.std],
              "max_pred_epochs_past_p_val": np.mean,
              "linear_pred_epochs_past_p_val": np.mean })

        # benjamini-hochberg correction
        to_correct_arr = ["max_pred", "linear_pred", "max_pred_epochs_past", "linear_pred_epochs_past"]
        mixed_idx = df_stats.query("is_mixed == True").index
        for to_correct in to_correct_arr:

            pvals = df_stats.loc[mixed_idx][f"{to_correct}_p_val","mean"].values
            rej_h0, pval_corr = multipletests(pvals, alpha=0.05, method="fdr_bh")[:2]
            df_stats.loc[mixed_idx, f"{to_correct}_p_val_corr"] = pval_corr
            df_stats.loc[mixed_idx, f"{to_correct}_rej_h0"] = rej_h0

        df_stats.drop(columns=[f"{to_correct}_p_val" for to_correct in to_correct_arr], inplace=True)

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

                # array containing acc/loss
                perf_stats = stats_dict.get("perf_stats")

                # find peak accuracy
                try:
                    i_max = np.argmax(perf_stats[:,0])
                    (val_acc, val_loss, train_acc, train_loss) = perf_stats[i_max]

                    # for learning speed
                    pct_acc = (self.pct / 100.) * val_acc
                    i_first = next(x for x, val in enumerate(perf_stats[:,0]) if val > pct_acc)

                    acc_arr.append([dataset, net_name, train_scheme, case, sample, val_acc, i_first])
                except ValueError:
                    print(f"Max entry in {case} {sample} perf_stats did not match expectations.")
                    continue

        # make dataframe
        acc_df = pd.DataFrame(acc_arr, columns=["dataset", "net_name", "train_scheme", "case", "sample", "max_val_acc", "epochs_past"])

        # process
        # 1. mark mixed nets
        acc_df["is_mixed"] = [len(case_dict[c]["act_fns"]) > 1 if case_dict.get(c) is not None else False for c in acc_df["case"]]
        acc_df["cross_fam"] = [len(case_dict[c]["act_fns"]) == len(set(case_dict[c]["act_fns"])) if case_dict.get(c) is not None else False for c in acc_df["case"]]

        # 2. add columns
        acc_df["max_pred"] = np.nan
        acc_df["linear_pred"] = np.nan
        acc_df["max_pred_p_val"] = np.nan
        acc_df["linear_pred_p_val"] = np.nan
        acc_df["max_pred_epochs_past_p_val"] = np.nan
        acc_df["linear_pred_epochs_past_p_val"] = np.nan

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

                # choose component row accs/learning epochs
                c_accs = []
                c_epochs = []
                for cc in component_cases:
                    c_row = component_rows \
                        .query(f"case == '{cc}'") \
                        .query(f"used == False")
                    
                    if len(c_row) == 0:
                        break
                    c_row = c_row.sample()
                    c_accs.append(c_row.max_val_acc.values[0])
                    c_epochs.append(c_row.epochs_past.values[0])

                    # mark component row as used in prediction
                    component_rows.at[c_row.index.values[0], "used"] = True

                if len(c_accs) == 0:
                    break

                acc_df.at[(d, n, sch, c, mixed_case_row.name), "max_pred"] = np.max(c_accs)
                acc_df.at[(d, n, sch, c, mixed_case_row.name), "linear_pred"] = np.mean(c_accs)
                
                acc_df.at[(d, n, sch, c, mixed_case_row.name), "max_pred_epochs_past"] = np.max(c_epochs)
                acc_df.at[(d, n, sch, c, mixed_case_row.name), "linear_pred_epochs_past"] = np.mean(c_epochs)

            # significance
            metrics = ["max_val_acc", "max_val_acc", "epochs_past", "epochs_past"]
            pred_cols = ["max_pred", "linear_pred", "max_pred_epochs_past", "linear_pred_epochs_past"]
            for metric, col in zip(metrics, pred_cols):

                t, p = ttest_ind(acc_df.at[(d, n, sch, c), metric], acc_df.at[(d, n, sch, c), col])
                if t < 0:
                    p = 1. - p / 2.
                else:
                    p = p / 2.
                acc_df.loc[(d, n, sch, c), f"{col}_p_val"] = p

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
    
    processor = AccStatProcessor("/home/briardoty/Source/allen-inst-cell-types/data_mountpoint", 10)
    
    # processor.reduce_snapshots()

    # processor.load_accuracy_df(["control2"])
    df, _, _ = processor.load_max_acc_df(refresh_df=True)
    # processor.load_weight_change_df(["control1"])
    
    
    
    
    
    
    
    
