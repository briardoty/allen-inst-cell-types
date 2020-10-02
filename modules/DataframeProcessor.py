import torch
import os
import pandas as pd
import re
import numpy as np
import json
from scipy.stats import ttest_ind
import math

try:
    from .util import ensure_sub_dir
except:
    from util import ensure_sub_dir

class DataframeProcessor():
    """
    Class to handle processing network snapshots into meaningful statistics.
    """
    
    def __init__(self, data_dir):
        
        self.data_dir = data_dir
        self.exclude_slug = "(exclude)"
        self.pct = 90

        self.net_idx_cols = ["dataset", "net_name", "train_scheme", "group", "case", "sample"]
    
    def refresh_learning_df(self, acc_df, pct):

        learning_arr = []
        acc_df.set_index(self.net_idx_cols, inplace=True)

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
            columns=self.net_idx_cols+["max_val_acc", "epoch_past_pct"])
        self.save_df("learning_df.csv", learning_df)

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
                group = stats_dict.get("group")
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

                    acc_arr.append([dataset, net_name, train_scheme, group, case, sample, val_acc, i_first])
                except ValueError:
                    print(f"Max entry in {case} {sample} perf_stats did not match expectations.")
                    continue

        # make dataframe
        acc_df = pd.DataFrame(acc_arr, columns=self.net_idx_cols+["max_val_acc", "epochs_past"])

        # process
        # 1. mark mixed nets
        acc_df["is_mixed"] = [len(case_dict[c]["act_fns"]) > 1 if case_dict.get(c) is not None else False for c in acc_df["case"]]
        acc_df["cross_fam"] = [len(case_dict[c]["act_fns"]) == len(set(case_dict[c]["act_fns"])) if case_dict.get(c) is not None else False for c in acc_df["case"]]

        # 2. add columns
        acc_df["max_pred"] = np.nan
        acc_df["linear_pred"] = np.nan
        acc_df["max_pred_p_val"] = np.nan
        acc_df["linear_pred_p_val"] = np.nan
        acc_df["min_pred_epochs_past_p_val"] = np.nan
        acc_df["linear_pred_epochs_past_p_val"] = np.nan

        # 2.9. multi-index
        acc_df.set_index(self.net_idx_cols, inplace=True)

        # 3. predictions for mixed cases
        for midx in acc_df.query("is_mixed == True").index.values:

            # break up multi-index
            d, n, sch, g, c, s = midx
            
            # skip if already predicted
            if not math.isnan(acc_df.at[midx, "max_pred"]):
                continue

            # get rows in this mixed case
            mixed_case_rows = acc_df.loc[(d, n, sch, g, c)]
            
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
                
                acc_df.at[(d, n, sch, c, mixed_case_row.name), "min_pred_epochs_past"] = np.min(c_epochs)
                acc_df.at[(d, n, sch, c, mixed_case_row.name), "linear_pred_epochs_past"] = np.mean(c_epochs)

            # significance
            upper_dists = ["max_val_acc", "max_val_acc", "min_pred_epochs_past", "linear_pred_epochs_past"]
            lower_dists = ["max_pred", "linear_pred", "epochs_past", "epochs_past"]
            cols = ["max_pred", "linear_pred", "min_pred_epochs_past", "linear_pred_epochs_past"]
            for upper, lower, col in zip(upper_dists, lower_dists, cols):

                t, p = ttest_ind(acc_df.at[(d, n, sch, c), upper], acc_df.at[(d, n, sch, c), lower])
                if t < 0:
                    p = 1. - p / 2.
                else:
                    p = p / 2.
                acc_df.loc[(d, n, sch, c), f"{col}_p_val"] = p

        # save things
        self.save_df("max_acc_df.csv", acc_df)

        # TODO: separate the refresh code for this from max_acc_df???
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
        acc_df = pd.DataFrame(acc_arr, columns=self.net_idx_cols+["epoch", "val_acc", "train_acc"])
        
        # save df
        self.save_df("acc_df.csv", acc_df)
  
    
    
    
