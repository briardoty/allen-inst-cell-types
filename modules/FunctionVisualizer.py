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

class FunctionVisualizer():
    
    def __init__(self, data_dir, n_classes=10, save_fig=False, refresh=False):
        
        self.data_dir = data_dir
        self.save_fig = save_fig
        self.refresh = refresh
        
        self.stats_processor = StatsProcessor(data_dir, n_classes)
        
    def plot_activation_fns(self, act_fns):
        """
        Plots the given activation functions on the same figure
        """

        x = np.linspace(-5, 5, 50)
        x = torch.tensor(x)
        fig, ax = plt.subplots(figsize=(7,5))

        for fn in act_fns:
            y = fn(x)
            ax.plot(x, y, label=str(fn))

        ax.legend()

        # optional saving
        if not self.save_fig:
            print("Not saving.")
            plt.show()
            return
        
        sub_dir = ensure_sub_dir(self.data_dir, f"figures/act_fns/")
        fn_names = " & ".join([str(fn) for fn in act_fns])
        filename = f"{fn_names}.svg"
        filename = os.path.join(sub_dir, filename)
        print(f"Saving... {filename}")
        plt.savefig(filename, dpi=300)


if __name__=="__main__":
    
    visualizer = FunctionVisualizer("/home/briardoty/Source/allen-inst-cell-types/data_mountpoint", 
        10, save_fig=True, refresh=False)

    # visualizer.plot_activation_fns([Sigfreud(1), Sigfreud(1.5), Sigfreud(2.), Sigfreud(4.)])
    # visualizer.plot_activation_fns([Swish(3), Swish(5), Swish(10)])
    # visualizer.plot_activation_fns([Tanhe(0.1), Tanhe(0.5), Tanhe(1), Tanhe(10)])
    # visualizer.plot_activation_fns([Renlu(0.5), Renlu(1), Renlu(1.5)])
