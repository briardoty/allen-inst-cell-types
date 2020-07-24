#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 14:22:11 2020

@author: briardoty
"""
import os
import numpy as np
import matplotlib.pyplot as plt
try:
    from .StatsProcessor import StatsProcessor
except:
    from StatsProcessor import StatsProcessor
try:
    from .NetManager import nets
except:
    from NetManager import nets

class Visualizer():
    
    def __init__(self, data_dir, net_name, n_classes=10, save_fig=False):
        
        self.data_dir = data_dir
        self.save_fig = save_fig
        
        self.stats_processor = StatsProcessor(net_name, n_classes, data_dir)
        
    def plot_accuracy(self, case_ids):
        """
        Plots accuracy over training for different experimental cases.

        Args:
            cases (list): Experimental cases to include in figure.

        Returns:
            None.

        """
        # pull data
        acc_df = self.stats_processor.load_accuracy_df(case_ids)

        # group and compute stats        
        acc_df.set_index(["case", "epoch"], inplace=True)
        acc_df_groups = acc_df.groupby(["case", "epoch"])
        acc_df_stats = acc_df_groups.agg({ "acc": [np.mean, np.std] })
        acc_df_stats_groups = acc_df_stats.groupby("case")
        
        # plot
        fig, ax = plt.subplots(figsize=(7,5))
        
        for name, group in acc_df_stats_groups:
            
            # error bars = 2 standard devs
            yvals = group["acc"]["mean"].values
            yerr = group["acc"]["std"].values * 2
            ax.errorbar(range(len(yvals)), yvals, yerr=yerr, label=name, 
                        capsize=3, elinewidth=1)
            
        ax.set_title("Classification accuracy during training")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation accuracy")
        ax.legend()
        step = 5
        ax.set_xticks([i * step for i in range(int((len(yvals) + 1)/step))])
        
        # optional saving
        if not self.save_fig:
            print("Not saving.")
            plt.show()
            return
        
        sub_dir = self.sub_dir(f"figures/{self.stats_processor.net_name}/accuracy/")
        cases = " & ".join(case_ids)
        filename = f"{cases} accuracy.png"
        filename = os.path.join(sub_dir, filename)
        print(f"Saving... {filename}")
        plt.savefig(filename, dpi=300)  
        
    def plot_weight_changes(self, case_ids):
        """
        Plots average change in weights over training for the given
        experimental cases.

        Args:
            cases (list): Experimental cases to include in figure.

        Returns:
            None.

        """
        # pull data
        df = self.stats_processor.load_weight_change_df(case_ids)

        sem_cols = list(filter(lambda x: x.endswith(".sem"), df.columns))
        df_groups = df.groupby("case")
        state_keys = list(nets["vgg11"]["state_keys"].keys())

        # plot
        x = np.arange(len(state_keys))
        width = 1.0 / len(case_ids)
        err_kw = dict(lw=1, capsize=3, capthick=1)

        fig, ax = plt.subplots(figsize=(14,8))
        for name, group in df_groups:

            yvals = group[state_keys].values[0]
            yerr = group[sem_cols].values[0]

            ax.bar(x, yvals, width, yerr=yerr, label=name, error_kw=err_kw)

            # update bar locations for next group
            x = [loc + 2 * width for loc in x]

        ax.set_title("Weight changes by layer during training")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean abs weight change per layer")
        ax.legend()

        ax.set_xticks([r + width for r in range(len(state_keys))])
        labels = list(nets["vgg11"]["state_keys"].values())
        ax.set_xticklabels(labels)

        # optional saving
        if not self.save_fig:
            print("Not saving.")
            plt.show()
            return

        sub_dir = self.sub_dir(f"figures/{self.stats_processor.net_name}/weight change/")
        cases = " & ".join(case_ids)
        filename = f"{cases} weight.png"
        filename = os.path.join(sub_dir, filename)
        print(f"Saving... {filename}")
        plt.savefig(filename, dpi=300)  
        
    def sub_dir(self, sub_dir):
        """
        Ensures existence of sub directory of self.data_dir and 
        returns its absolute path.

        Args:
            sub_dir (TYPE): DESCRIPTION.

        Returns:
            sub_dir (TYPE): DESCRIPTION.

        """
        sub_dir = os.path.join(self.data_dir, sub_dir)
        
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
            
        return sub_dir
        

if __name__=="__main__":
    
    visualizer = Visualizer("/home/briardoty/Source/allen-inst-cell-types/data", "vgg11", 10, False)
    
    # visualizer.plot_weight_changes(["control", "mixed-2_relu10_nr-1"])
    visualizer.plot_accuracy(["control2"])
