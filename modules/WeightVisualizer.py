#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 14:22:11 2020

@author: briardoty
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
try:
    from .StatsProcessor import StatsProcessor
except:
    from StatsProcessor import StatsProcessor

try:
    from .NetManager import nets
except:
    from NetManager import nets

try:
    from .ActivationFunctions import *
except:
    from ActivationFunctions import *

try:
    from .util import ensure_sub_dir
except:
    from util import ensure_sub_dir

import matplotlib
matplotlib.rc("xtick", labelsize=14) 
matplotlib.rc("ytick", labelsize=14) 


class WeightVisualizer():
    
    def __init__(self, data_dir, n_classes=10, save_fig=False, refresh=False):
        
        self.data_dir = data_dir
        self.save_fig = save_fig
        self.refresh = refresh
        
        self.stats_processor = StatsProcessor(data_dir, n_classes)
        
    def plot_type_specific_weights(self, net_name, case):
        """
        Plots mean absolute weights for each cell type across layers 
        """

        # pull data
        df = self.stats_processor.load_weight_df(net_name, case)

        # plot
        state_keys = list(nets[net_name]["state_keys"].keys())
        x = np.array([i * 1.25 for i in range(len(state_keys))])
        n_act_fns = len(df.index.levels[1])
        width = 1.0 / n_act_fns
        err_kw = dict(lw=1, capsize=3, capthick=1)

        fig, ax = plt.subplots(figsize=(14,8))
        clrs = sns.color_palette("hls", n_act_fns)

        for i in range(n_act_fns):

            act_fn = df.index.levels[1][i]

            yvals = df["avg_weight"][:, act_fn][state_keys]
            yerr = df["sem_weight"][:, act_fn][state_keys]

            ax.bar(x, yvals, width, yerr=yerr, label=act_fn, error_kw=err_kw, 
                color=clrs[i])

            # update bar locations for next group
            x = [loc + width for loc in x]

        ax.set_title("Weight distribution across layers after training")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean abs weight per layer")
        ax.legend()

        loc = (n_act_fns - 1) / (2. * n_act_fns)
        ax.set_xticks([loc + i * 1.25 for i in range(len(state_keys))])
        labels = list(nets[net_name]["state_keys"].values())
        ax.set_xticklabels(labels)

        # optional saving
        if not self.save_fig:
            print("Not saving.")
            plt.show()
            return

        sub_dir = ensure_sub_dir(self.data_dir, f"figures/{net_name}/weight distr/")
        filename = f"{case} weight distr.svg"
        filename = os.path.join(sub_dir, filename)
        print(f"Saving... {filename}")
        plt.savefig(filename, dpi=300)  


    def plot_weight_changes(self, net_name, cases, train_schemes):
        """
        Plots average change in weights over training for the given
        experimental cases.

        Args:
            cases (list): Experimental cases to include in figure.

        Returns:
            None.

        """
        # pull data
        df = self.stats_processor.load_weight_change_df(net_name, cases, train_schemes)

        state_keys = df.columns.to_list()
        sem_cols = list(filter(lambda x: x.endswith(".sem"), df.columns))
        df_groups = df.groupby(["train_scheme", "case"])

        # plot
        x = np.array([i * 1.25 for i in range(len(state_keys))])
        width = 1.0 / len(cases)
        err_kw = dict(lw=1, capsize=3, capthick=1)

        fig, ax = plt.subplots(figsize=(14,8))
        clrs = sns.color_palette("hls", len(cases))

        for i in range(len(cases)):

            case = cases[i]
            group = df_groups.get_group(case)
            yvals = group[state_keys].values[0]
            yerr = group[sem_cols].values[0]

            ax.bar(x, yvals, width, yerr=yerr, label=case, error_kw=err_kw, 
                color=clrs[i])

            # update bar locations for next group
            x = [loc + width for loc in x]

        ax.set_title("Weight changes by layer during training")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean abs weight change per layer")
        ax.legend()

        loc = (len(cases) - 1) / (2. * len(cases))
        ax.set_xticks([loc + i * 1.25 for i in range(len(state_keys))])
        labels = [k[:-7] for k in df.columns if k.endswith(".weight")]
        ax.set_xticklabels(labels)

        # optional saving
        if not self.save_fig:
            print("Not saving.")
            plt.show()
            return

        sub_dir = ensure_sub_dir(self.data_dir, f"figures/{net_name}/weight change/")
        cases = " & ".join(cases)
        filename = f"{cases} weight.svg"
        filename = os.path.join(sub_dir, filename)
        print(f"Saving... {filename}")
        plt.savefig(filename, dpi=300)
        

if __name__=="__main__":
    
    visualizer = WeightVisualizer("/home/briardoty/Source/allen-inst-cell-types/data_mountpoint", 
        10, save_fig=True, refresh=False)
    
    # visualizer.plot_type_specific_weights("swish10-tanhe1-relu")

    # visualizer.plot_weight_changes(["unmodified"], ["adam"])
