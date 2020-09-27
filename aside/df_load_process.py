import os
import pandas as pd
import numpy as np
import json
from scipy.stats import ttest_ind
import math
from statsmodels.stats.multitest import multipletests

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