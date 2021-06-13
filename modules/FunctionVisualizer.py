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
import torch

try:
    from .ActivationFunctions import *
except:
    from ActivationFunctions import *

try:
    from .util import ensure_sub_dir
except:
    from util import ensure_sub_dir

import matplotlib
large_font_size = 20
small_font_size = 16
matplotlib.rc("xtick", labelsize=small_font_size) 
matplotlib.rc("ytick", labelsize=small_font_size)

class FunctionVisualizer():
    
    def __init__(self, data_dir, save_fig=False):
        
        self.data_dir = data_dir
        self.save_fig = save_fig
        
    def plot_activation_fns(self, act_fns, clr_set="husl"):
        """
        Plots the given activation functions on the same figure
        """

        x = np.linspace(-100, 100, 10000)
        x = torch.tensor(x)
        fig, ax = plt.subplots(figsize=(5,5))
        clrs = sns.color_palette(clr_set, len(act_fns))

        for i in range(len(act_fns)):
            fn = act_fns[i]
            y = fn(x)
            normalized = y / max(y)
            label = str(fn)
            ax.plot(x, y, label=label, c=clrs[i], linewidth=3)
            # ax.plot(x, normalized, label=f"{str(fn)} norm")

        # axes
        ax.axhline(y=0, color="k", linestyle="--", alpha=0.2)
        ax.axvline(x=0, color="k", linestyle="--", alpha=0.2)

        ax.set_xticks([-1, 0, 1])
        ax.set_xticklabels([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels([-1, 0, 1])
        ax.set_xlim([-2, 2])
        ax.set_ylim([-1, 2])
        # ax.axis("equal")
        ax.set_aspect("equal", "box")
        ax.set_xlabel("Input", fontsize=large_font_size)
        ax.set_ylabel("Activation", fontsize=large_font_size)
        ax.legend(fontsize=small_font_size, loc="upper left")
        plt.tight_layout()

        # optional saving
        if not self.save_fig:
            print("Not saving.")
            plt.show()
            return
        
        sub_dir = ensure_sub_dir(self.data_dir, f"figures/act_fns/")
        fn_names = " & ".join([str(fn) for fn in act_fns])
        filename = f"{fn_names}"
        print(f"Saving... {filename}")
        plt.savefig(os.path.join(sub_dir, f"{filename}.svg"))
        plt.savefig(os.path.join(sub_dir, f"{filename}.png"), dpi=300)

    def plot_act_fn_mapping(self, act_fn1, act_fn2):
        """
        Visualize the expressivity of mixed activation functions
        """

        # plot input space
        fig, ax = plt.subplots(figsize=(7,5))
        circle = plt.Circle((0,0),1, color="k", fill=False, linewidth=2)
        ax.add_artist(circle)
        ax.axis("equal")
        ax.set(xlim=(-2,2), ylim=(-2,2))
        ax.axvline(0, linestyle="--", alpha=0.25,color="k")
        ax.axhline(0, linestyle="--", alpha=0.25,color="k")
        # plt.savefig("unit_circle.svg")

        # plot output space
        fig, ax = plt.subplots(figsize=(7,5))
        ax.axis("equal")
        ax.set(xlim=(-2,2), ylim=(-2,2))
        ax.axvline(0, linestyle="--", alpha=0.25,color="k")
        ax.axhline(0, linestyle="--", alpha=0.25,color="k")
        x = np.arange(0,2*np.pi, 1/100)
        x1 = torch.tensor(np.sin(x))
        x2 = torch.tensor(np.cos(x))
        ax.plot(x1, x2, "k:", linewidth=2, label="Input")

        # output space
        ax.plot(act_fn1(x1), act_fn1(x2), "b--", linewidth=2, label="act_fn1")
        ax.plot(act_fn2(x1), act_fn2(x2), "g--", linewidth=2, label="act_fn2")

        # mixed space
        ax.plot(act_fn1(x1), act_fn2(x2), "r", linewidth=2, label="Mixed")
        plt.legend()

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
    
    visualizer = FunctionVisualizer(
        "/home/briardoty/Source/allen-inst-cell-types/data_mountpoint", 
        save_fig=True)

    # visualizer.plot_activation_fns([Tanh(1), Swish(1), Relu()])
    # visualizer.plot_activation_fns([Swish(1), Swish(2), Swish(10)])
    visualizer.plot_activation_fns([PTanh(0.1), PTanh(0.5), PTanh(1)], clr_set="Set2")
    visualizer.plot_activation_fns([PTanh(1), Swish(7.5)])

    # visualizer.plot_act_fn_mapping(Swish(1), torch.tanh)
