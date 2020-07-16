#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 14:22:11 2020

@author: briardoty
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from .StatsProcessor import StatsProcessor



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
        fig, ax = plt.subplots()
        
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
        
        # optional saving
        if not self.save_fig:
            print("Not saving.")
            plt.show()
            return
        
        sub_dir = self.sub_dir(f"figures/{self.stats_processor.net_name}/")
        cases = " & ".join(case_ids)
        filename = f"{cases} accuracy.png"
        filename = os.path.join(sub_dir, filename)
        print(f"Saving... {filename}")
        plt.savefig(filename, dpi=100)  
        
    def plot_weight_changes(self, case_ids):
        """
        Plots average change in weights over training for the given
        experimental cases.

        Args:
            cases (list): Experimental cases to include in figure.

        Returns:
            None.

        """
        
        
        return   
        
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
        


