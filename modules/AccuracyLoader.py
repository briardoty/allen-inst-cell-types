import torch
import os
import pandas as pd
import re
import numpy as np
import json
from scipy.stats import ttest_ind
import math
from statsmodels.stats.multitest import multipletests


class AccuracyLoader():
    """
    Class to handle processing network snapshots into meaningful statistics.
    """
    
    def __init__(self, data_dir):
        
        self.data_dir = data_dir
        self.df_sub_dir = os.path.join(self.data_dir, "dataframes/")
        self.exclude_slug = "(exclude)"
        self.pct = 90

        self.net_idx_cols = ["dataset", "net_name", "train_scheme", "group", "case", "sample"]
    
    def load_learning_df(self, pct):
        """

        """

        with open(os.path.join(self.df_sub_dir, "case_dict.json"), "r") as json_file:
            case_dict = json.load(json_file)

        # load
        learning_df = pd.read_csv(os.path.join(self.df_sub_dir, "learning_df.csv"))
        learning_df.drop(columns="Unnamed: 0", inplace=True)

        # TODO: process? aggregate? etc.

        return learning_df, case_dict

    def load_max_acc_df_ungrouped(self):

        # load
        df = pd.read_csv(os.path.join(self.df_sub_dir, "max_acc_df.csv"))
        with open(os.path.join(self.df_sub_dir, "case_dict.json"), "r") as json_file:
            case_dict = json.load(json_file)

        # index
        df.set_index(self.net_idx_cols, inplace=True)

        return df, case_dict, self.net_idx_cols

    def load_max_acc_df(self):
        """
        Loads dataframe with max validation accuracy for different 
        experimental cases.

        Returns:
            df_stats (dataframe): Dataframe containing max accuracy.
            case_dict (dict): Dict() of act fn names to their params
        """

        # load
        acc_df = pd.read_csv(os.path.join(self.df_sub_dir, "max_acc_df.csv"))
        with open(os.path.join(self.df_sub_dir, "case_dict.json"), "r") as json_file:
            case_dict = json.load(json_file)

        # aggregate
        gidx_cols = self.net_idx_cols + ["is_mixed", "cross_fam"]
        df_stats = acc_df.groupby(gidx_cols).agg(
            { "max_val_acc": [np.mean, np.std],
              "max_pred": [np.mean, np.std],
              "linear_pred": [np.mean, np.std],
              "max_pred_p_val": np.mean,
              "linear_pred_p_val": np.mean,

              "epochs_past": [np.mean, np.std],
              "min_pred_epochs_past": [np.mean, np.std],
              "linear_pred_epochs_past": [np.mean, np.std],
              "min_pred_epochs_past_p_val": np.mean,
              "linear_pred_epochs_past_p_val": np.mean })

        # benjamini-hochberg correction
        to_correct_arr = ["max_pred", "linear_pred", "min_pred_epochs_past", "linear_pred_epochs_past"]
        mixed_idx = df_stats.query("is_mixed == True").index
        for to_correct in to_correct_arr:

            pvals = df_stats.loc[mixed_idx][f"{to_correct}_p_val","mean"].values
            rej_h0, pval_corr = multipletests(pvals, alpha=0.05, method="fdr_bh")[:2]
            df_stats.loc[mixed_idx, f"{to_correct}_p_val_corr"] = pval_corr
            df_stats.loc[mixed_idx, f"{to_correct}_rej_h0"] = rej_h0

        df_stats.drop(columns=[f"{to_correct}_p_val" for to_correct in to_correct_arr], inplace=True)

        return df_stats, case_dict, gidx_cols

    def load_accuracy_df(self, dataset, net_name, schemes, cases):
        """
        Loads dataframe with validation accuracy for different 
        experimental cases over epochs.

        Returns:
            df (dataframe): Dataframe containing accuracy trajectories
        """

        # load
        df = pd.read_csv(os.path.join(self.df_sub_dir, "acc_df.csv"))

        # filter
        df.drop(columns="Unnamed: 0", inplace=True)
        df = df.query(f"dataset == '{dataset}'") 
        df = df.query(f"net_name == '{net_name}'")
        df = df.query(f"train_scheme in {schemes}")
        df = df.query(f"case in {cases}")

        return df


if __name__=="__main__":
    
    processor = AccuracyLoader("/home/briardoty/Source/allen-inst-cell-types/data_mountpoint")
    
    # processor.reduce_snapshots()

    # processor.load_accuracy_df(["control2"])
    df, _, _ = processor.load_max_acc_df()
    # processor.load_weight_change_df(["control1"])
    
    
    
    
    
    
    
    
