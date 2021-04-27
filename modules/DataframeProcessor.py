import torch
import os
import pandas as pd
import re
import numpy as np
import json
from scipy.stats import ttest_ind
import math
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

try:
    from .util import ensure_sub_dir, get_component_cases
except:
    from util import ensure_sub_dir, get_component_cases

class DataframeProcessor():
    """
    Class to handle processing network snapshots into meaningful statistics.
    """
    
    def __init__(self, data_dir):
        
        self.data_dir = data_dir
        self.exclude_slug = "(exclude)"
        self.pct = 90

        self.df_sub_dir = os.path.join(self.data_dir, "dataframes/")
        self.net_idx_cols = ["dataset", "net_name", "train_scheme", "group", "case", "epoch", "sample"]
        self.acc_idx_cols = ["dataset", "net_name", "train_scheme", "group", "case", "sample", "epoch"]
    
    def refresh_learning_df(self, acc_df, pct):

        learning_arr = []
        acc_df.set_index(self.acc_idx_cols, inplace=True)

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
            columns=self.net_idx_cols+["val_acc", "epoch_past_pct"])
        self.save_df("learning_df.csv", learning_df)

    def add_group_to_df(self):

        # build case -> group dict
        group_dict = dict()
        with open(os.path.join(self.data_dir, "net_configs.json"), "r") as json_file:
            net_configs = json.load(json_file)

        for g in net_configs.keys():
            cases = net_configs[g]
            case_names = cases.keys()
            
            for c in case_names:

                group_dict[c] = g

        # load current df
        df_name = "final_acc_df.csv"
        df = pd.read_csv(os.path.join(self.df_sub_dir, df_name))
        df.drop(columns="Unnamed: 0", inplace=True)

        # update
        df["group"] = None
        for idx, row in df.iterrows():
            group = group_dict.get(row["case"])
            df.at[idx, "group"] = group

        # save
        self.save_df(df_name, df)

    def refresh_activation_df(self):
        """

        """
        return


    def refresh_final_acc_df(self, report_peak_acc=False):
        """
        Refreshes dataframe with max validation accuracy.
        """

        # build case -> group dict
        group_dict = dict()
        with open(os.path.join(self.data_dir, "net_configs.json"), "r") as json_file:
            net_configs = json.load(json_file)

        for g in net_configs.keys():
            cases = net_configs[g]
            case_names = cases.keys()
            
            for c in case_names:

                group_dict[c] = g

        # load current df if exists
        df_name = "final_acc_df.csv"
        # curr_df = pd.read_csv(os.path.join(self.df_sub_dir, df_name))
        # curr_df.drop(columns="Unnamed: 0", inplace=True)

        acc_arr = []
        case_dict = dict()
        with open(os.path.join(self.df_sub_dir, "case_dict.json"), "r") as json_file:
            case_dict = json.load(json_file)

        # walk dir looking for saved net stats
        net_dir = os.path.join(self.data_dir, f"nets")
        for root, _, files in os.walk(net_dir):
            
            # only interested in locations files are saved
            if len(files) <= 0:
                continue
            
            slugs = root.split("/")

            # exclude some dirs...
            if any(self.exclude_slug in slug for slug in slugs):
                continue

            # only latest results
            if not "adam_lravg_nosplit" in slugs:
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
                initial_lr = stats_dict.get("initial_lr") if stats_dict.get("initial_lr") is not None else -1
                case = stats_dict.get("case")
                sample = stats_dict.get("sample")
                group = stats_dict.get("group")
                if group is None:
                    group = group_dict.get(case)
                modified_layers = stats_dict.get("modified_layers")
                if modified_layers is not None:
                    case_dict[case] = {
                        "act_fns": modified_layers.get("act_fns"),
                        "act_fn_params": modified_layers.get("act_fn_params")
                    }

                # array containing acc/loss
                perf_stats = np.array([s for s in stats_dict.get("perf_stats") if s is not None])
                if len(perf_stats) == 0:
                    continue

                # find peak accuracy?
                try:

                    if report_peak_acc:
                        i_acc = np.argmax(perf_stats[:,0])
                    else:
                        i_acc = -1
                    (val_acc, val_loss, train_acc, train_loss) = perf_stats[i_acc]

                    # for learning speed
                    pct_acc = (self.pct / 100.) * val_acc
                    i_first = next(x for x, val in enumerate(perf_stats[:,0]) if val > pct_acc)
                    
                    test_acc = stats_dict.get("test_acc")

                    acc_arr.append([dataset, net_name, train_scheme, group, case, i_acc, sample, val_acc, test_acc, i_first, initial_lr])

                    # by epoch
                    n_epoch_samples = 31
                    epochs = [10*i for i in range(n_epoch_samples)]
                    epochs = epochs[:-1] + [int(x) for x in np.linspace(epochs[-1], len(perf_stats)-1, 5)]
                    epochs = list(set(epochs))
                    for epoch in epochs:
                        
                        try:
                            (val_acc, val_loss, train_acc, train_loss) = perf_stats[epoch]
                            acc_arr.append([dataset, net_name, train_scheme, group, case, epoch, sample, val_acc, None, None, initial_lr])
                        except IndexError:
                            break

                except ValueError:
                    print(f"Max entry in {case} {sample} perf_stats did not match expectations.")
                    continue

        # make dataframe
        acc_df = pd.DataFrame(acc_arr, columns=self.net_idx_cols+["val_acc", "test_acc", "epochs_past", "initial_lr"])

        # process
        # 1. mark mixed nets
        acc_df["is_mixed"] = [len(case_dict[c]["act_fns"]) > 1 if case_dict.get(c) is not None else False for c in acc_df["case"]]
        acc_df["cross_fam"] = [len(case_dict[c]["act_fns"]) == len(set(case_dict[c]["act_fns"])) if case_dict.get(c) is not None else False for c in acc_df["case"]]

        # 2. add columns for predictions
        acc_df["max_pred_val_acc"] = np.nan
        acc_df["linear_pred_val_acc"] = np.nan
        acc_df["max_pred_val_acc_p_val"] = np.nan
        acc_df["linear_pred_val_acc_p_val"] = np.nan

        acc_df["max_pred_test_acc"] = np.nan
        acc_df["linear_pred_test_acc"] = np.nan
        acc_df["max_pred_test_acc_p_val"] = np.nan
        acc_df["linear_pred_test_acc_p_val"] = np.nan

        # index new and old without group
        # idx_no_group = list(self.net_idx_cols + ["epoch"])
        # idx_no_group.remove("group")
        # curr_df.set_index(idx_no_group, inplace=True)
        # acc_df.set_index(idx_no_group, inplace=True)

        # merge new and old, preferring new
        # ndf = pd.concat([curr_df[~curr_df.index.isin(acc_df.index)], acc_df])

        # port over group from old df where appropriate
        # ndf[ndf.index.isin(curr_df.index)]["group"] = curr_df["group"]

        # 2.9. index with group
        ndf = acc_df
        ndf.reset_index(drop=False, inplace=True)
        ndf.set_index(self.net_idx_cols, inplace=True)

        # 3. predictions for mixed cases
        mixed_df = ndf.query("is_mixed == True")
        for epoch in mixed_df.index.unique(level=5):

            for midx in mixed_df.query(f"epoch == {epoch}").index.values:

                # break up multi-index
                d, n, sch, g, c, e, s = midx
                
                # skip if already predicted
                try:
                    prediction = ndf.at[midx, "max_pred_val_acc"]
                    if not math.isnan(prediction):
                        continue
                except:
                    print(f"Prediction did not match expectations at: {midx} - {prediction}")
                    continue

                # get rows in this mixed case
                mixed_case_rows = ndf.loc[(d, n, sch, g, c, e)]
                
                # get component case rows
                component_cases = get_component_cases(case_dict, c)
                component_rows = ndf.query(f"is_mixed == False") \
                    .query(f"dataset == '{d}'") \
                    .query(f"net_name == '{n}'") \
                    .query(f"train_scheme == '{sch}'") \
                    .query(f"case in {component_cases}") \
                    .query(f"epoch == {e}")

                # flag to indicate whether row used in prediction yet
                component_rows["used"] = False

                # make a prediction for each sample in this mixed case
                for i in range(len(mixed_case_rows)):
                    mixed_case_row = mixed_case_rows.iloc[i]

                    # choose component row accs/learning epochs
                    c_accs = []
                    c_accs_test = []
                    # c_epochs = []
                    for cc in component_cases:
                        c_row = component_rows \
                            .query(f"case == '{cc}'") \
                            .query(f"used == False")
                        
                        if len(c_row) == 0:
                            break
                        c_row = c_row.sample()
                        c_accs.append(c_row.val_acc.values[0])
                        c_accs_test.append(c_row.test_acc.values[0])
                        # c_epochs.append(c_row.epochs_past.values[0])

                        # mark component row as used in prediction
                        component_rows.at[c_row.index.values[0], "used"] = True

                    if len(c_accs) == 0:
                        break

                    max_pred = np.max(c_accs)
                    lin_pred = np.mean(c_accs)

                    ndf.at[(d, n, sch, g, c, e, mixed_case_row.name), "max_pred_val_acc"] = max_pred
                    ndf.at[(d, n, sch, g, c, e, mixed_case_row.name), "linear_pred_val_acc"] = lin_pred
                    
                    if len(c_accs_test) == 0:
                        continue

                    ndf.at[(d, n, sch, g, c, e, mixed_case_row.name), "max_pred_test_acc"] = np.max(c_accs_test)
                    ndf.at[(d, n, sch, g, c, e, mixed_case_row.name), "linear_pred_test_acc"] = np.mean(c_accs_test)

                # significance
                upper_dists = ["val_acc", "val_acc", "test_acc", "test_acc"]
                lower_dists = ["max_pred_val_acc", "linear_pred_val_acc", "max_pred_test_acc", "linear_pred_test_acc"]
                cols = ["max_pred_val_acc", "linear_pred_val_acc", "max_pred_test_acc", "linear_pred_test_acc"]
                for upper, lower, col in zip(upper_dists, lower_dists, cols):

                    t, p = ttest_ind(ndf.at[(d, n, sch, g, c, e), upper], ndf.at[(d, n, sch, g, c), lower])
                    if t < 0:
                        p = 1. - p / 2.
                    else:
                        p = p / 2.
                    ndf.loc[(d, n, sch, g, c, e), f"{col}_p_val"] = p

        # save things
        self.save_df(df_name, ndf)

        # TODO: separate the refresh code for this from final_acc_df???
        self.save_json("case_dict.json", case_dict)

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
            
        # load in current df if it exists

        # walk dir looking for saved net stats
        net_dir = os.path.join(self.data_dir, f"nets")
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
                group = stats_dict.get("group")
                case = stats_dict.get("case")
                sample = stats_dict.get("sample")

                perf_stats = np.array([s for s in stats_dict.get("perf_stats") if s is not None])
                for epoch in range(len(perf_stats)):
                    try:
                        (val_acc, val_loss, train_acc, train_loss) = perf_stats[epoch]
                    except TypeError:
                        print(f"Entry in perf_stats did not match expectations. Dataset: {dataset}; Scheme: {train_scheme}; Case {case}; Sample: {sample}; Epoch: {epoch}")
                        continue
                    acc_arr.append([dataset, net_name, train_scheme, group, case, sample, epoch, val_acc, val_loss, train_acc])
                
        # make dataframe
        acc_df = pd.DataFrame(acc_arr, columns=self.net_idx_cols+["val_acc", "val_loss", "train_acc"])
        
        # save df
        self.save_df("acc_df.csv", acc_df)

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
  
if __name__=="__main__":
    
    data_dir = "/home/briardoty/Source/allen-inst-cell-types/data_mountpoint"
    proc = DataframeProcessor(data_dir)
    proc.refresh_final_acc_df()
    # proc.refresh_accuracy_df()
    # proc.add_group_to_df()

    
    
