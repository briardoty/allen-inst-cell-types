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


class AccuracyVisualizer():
    
    def __init__(self, data_dir, n_classes=10, save_fig=False, refresh=False):
        
        self.data_dir = data_dir
        self.save_fig = save_fig
        self.refresh = refresh
        
        self.stats_processor = StatsProcessor(data_dir, n_classes)

    def plot_predictions(self, dataset, net_names, schemes, pred_type="max"):
        """
        Plot a single axis figure of offset from predicted final accuracy for
        the given mixed cases.
        """

        # pull data
        df, _, _ = self.stats_processor.load_final_acc_df(self.refresh)

        # performance relative to predictions
        df["acc_vs_linear"] = df["final_val_acc"]["mean"] - df["linear_pred"]
        df["acc_vs_max"] = df["final_val_acc"]["mean"] - df["max_pred"]

        sort_df = df.query("is_mixed == True").sort_values(["net_name", f"acc_vs_{pred_type}"])
        unique_nets = df.query(f"is_mixed & dataset == '{dataset}' & net_name in {net_names} & train_scheme in {schemes}").index.unique(level=1).tolist()
        
        # plot
        fig = plt.figure(figsize=(16,12))
        plt.gca().axvline(0, color='k', linestyle='--')
        clrs = sns.color_palette("husl", len(unique_nets))

        ylabels = []
        handles = dict()
        i = 0
        for midx in sort_df.index.values:

            # dataset, net, scheme, case, mixed
            d, n, s, c, m = midx
            if (not m 
                or not d == dataset 
                or not n in net_names
                or not s in schemes):
                continue

            # prettify
            if np.mod(i, 2) == 0:
                plt.gca().axhspan(i-.5, i+.5, alpha = 0.1, color="k")
            
            perf = sort_df.loc[midx][f"acc_vs_{pred_type}"].values[0]
            err = sort_df.loc[midx]["final_val_acc"]["std"] * 1.98

            clr = clrs[unique_nets.index(n)]
            # plot good and bad
            if perf - err > 0:
                plt.plot([perf - err, perf + err], [i,i], linestyle="-", 
                    c=clr, linewidth=6, alpha=.6)
                h = plt.plot(perf, i, c=clr, marker="o")
                handles[n] = h[0]
            else:
                plt.plot([perf - err, perf + err], [i,i], linestyle="-", 
                    c=clr, linewidth=6, alpha=.2)
                plt.plot([perf - err, perf + err], [i,i], linestyle="-", 
                    linewidth=6, c='k', alpha=.1)
                h = plt.plot(perf, i, c=clr, marker="o", alpha=0.5)
                if handles.get(n) is None:
                    handles[n] = h[0]

            i += 1
            ylabels.append(f"{n} {s} {c}")

        # set figure text
        plt.title("Mixed network performance relative to predicted performance", 
            fontsize=20, pad=20)
        plt.xlabel(f"Accuracy relative to {pred_type} prediction", 
            fontsize=16, labelpad=10)
        plt.ylabel("Network configuration", fontsize=16, labelpad=10)
        plt.yticks(np.arange(0, i, 1), ylabels)
        plt.ylim(-0.5, i - 0.5)
        # plt.xlim(-0.1, 0.3)
        plt.legend(handles.values(), handles.keys())
        plt.tight_layout()
        # yax = plt.gca().get_yaxis()
        # yax.set_tick_params(pad=100)

        # optional saving
        if not self.save_fig:
            print("Not saving.")
            plt.show()
            return

        sub_dir = ensure_sub_dir(self.data_dir, f"figures/prediction/{dataset}")
        net_names = ", ".join(net_names)
        schemes = ", ".join(schemes)
        filename = f"{dataset}_{net_names}_{schemes}_{pred_type}-prediction.svg"
        filename = os.path.join(sub_dir, filename)
        print(f"Saving... {filename}")
        plt.savefig(filename, dpi=300)  

    def scatter_final_acc(self, dataset, net_names, schemes, pred_type="linear"):
        """
        Plot a scatter plot of predicted vs actual final accuracy for the 
        given mixed cases.

        Args:
            net_names
            schemes
        """

        # pull data
        df, case_dict, index_cols = self.stats_processor.load_final_acc_df(self.refresh)
        df_groups = df.groupby(index_cols)
        mixed_cases = df.query("is_mixed == True").index.unique(level=3).tolist()
        n_mixed = len(mixed_cases)

        # plot
        fig, ax = plt.subplots(figsize=(14,14))
        fmts = [".", "^"]
        mfcs = ["None", None]
        clrs = sns.color_palette("husl", n_mixed)

        # plot mixed cases
        i = 0
        for g in df_groups.groups:

            # dataset, net, scheme, case, mixed
            d, n, s, c, m = g
            if (not m 
                or not d == dataset 
                or not n in net_names
                or not s in schemes):
                continue

            g_data = df_groups.get_group(g)
            fmt = fmts[net_names.index(n)]
            mfc = mfcs[schemes.index(s)]
            clr = clrs[mixed_cases.index(c)]

            # actual
            y_act = g_data["final_val_acc"]["mean"].values[0]
            y_err = g_data["final_val_acc"]["std"].values[0] * 1.98

            # prediction
            x_pred = g_data[f"{pred_type}_pred"].values[0]
            x_err = g_data[f"{pred_type}_std"].values[0] * 1.98
            
            # plot
            ax.errorbar(x_pred, y_act, xerr = x_err, yerr=y_err, 
                label=f"{n} {s} {c}", 
                elinewidth=1, c=clr, fmt=fmt, markersize=10,
                markerfacecolor=mfc)

            i += 1

        # plot reference line
        x = np.linspace(0, 1, 50)
        ax.plot(x, x, c=(0.5, 0.5, 0.5, 0.25), dashes=[6,2])

        # set figure text
        ax.set_title(f"Linear predicted vs actual mixed case final accuracy - {dataset}", fontsize=18)
        ax.set_xlabel("Predicted", fontsize=16)
        ax.set_ylabel("Actual", fontsize=16)
        ax.set_xlim([0.1, 1])
        ax.set_ylim([0.1, 1])
        ax.set_aspect("equal", "box")
        ax.legend()
         
        # optional saving
        if not self.save_fig:
            print("Not saving.")
            plt.show()
            return

        sub_dir = ensure_sub_dir(self.data_dir, f"figures/scatter/{dataset}")
        net_names = ", ".join(net_names)
        schemes = ", ".join(schemes)
        filename = f"{dataset}_{net_names}_{schemes}_scatter.svg"
        filename = os.path.join(sub_dir, filename)
        print(f"Saving... {filename}")
        plt.savefig(filename, dpi=300)  

    def plot_final_accuracy(self, net_name, control_cases, mixed_cases):
        """
        Plot accuracy at the end of training for given control cases
        and mixed case, including predicted mixed case accuracy based
        on linear combination of control cases
        """

        # pull data
        acc_df, case_dict = self.stats_processor.load_final_acc_df(
            net_name, control_cases + mixed_cases)
        acc_df_groups = acc_df.groupby("case")

        # plot...
        handles = []
        labels = []

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14,8), sharey=True)
        fig.subplots_adjust(wspace=0)
        clrs = sns.color_palette("hls", len(control_cases) + 2 * len(mixed_cases))
        
        for i in range(len(control_cases)):

            case = control_cases[i]
            group = acc_df_groups.get_group(case)
            p = float(case_dict[case][0])

            # error bars = 2 standard devs
            yvals = group["final_val_acc"]["mean"].values
            yerr = group["final_val_acc"]["std"].values * 1.98
            h = axes[0].errorbar(p, yvals[0], yerr=yerr, label=case,
                capsize=3, elinewidth=1, c=clrs[i], fmt=".")
            
            handles.append(h)
            labels.append(case)
            
        # plot mixed case
        for i in range(len(mixed_cases)):

            mixed_case = mixed_cases[i]

            # actual
            group = acc_df_groups.get_group(mixed_case)
            y_act = group["final_val_acc"]["mean"].values[0]
            y_err = group["final_val_acc"]["std"].values * 1.98
            l = f"{mixed_case} actual"
            h = axes[1].errorbar(i, y_act, yerr=y_err, label=l,
                capsize=3, elinewidth=1, c=clrs[len(control_cases) + i], fmt=".")
            
            labels.append(l)
            handles.append(h)

            # predicted
            ps = [p for p in case_dict[mixed_case]]
            component_cases = [k for k, v in case_dict.items() if len(v) == 1 and v[0] in ps]
            y_pred = acc_df["final_val_acc"]["mean"][component_cases].mean()
            l = f"{mixed_case} prediction"
            h = axes[1].plot(i, y_pred, "x", label=l,
                c=clrs[len(control_cases) + i + 1])

            labels.append(l)
            handles.append(h)

        fig.suptitle("Final accuracy")
        axes[0].set_xlabel("Activation function parameter value")
        axes[1].set_xlabel("Mixed cases")
        axes[0].set_ylabel("Final validation accuracy")
        axes[1].xaxis.set_ticks([])

        # shrink second axis by 20%
        box = axes[1].get_position()
        axes[1].set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # append legend to second axis
        axes[1].legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5))
         
        # optional saving
        if not self.save_fig:
            print("Not saving.")
            plt.show()
            return

        sub_dir = ensure_sub_dir(self.data_dir, f"figures/{net_name}/final accuracy/")
        cases = " & ".join(mixed_cases)
        filename = f"{cases} final acc.svg"
        filename = os.path.join(sub_dir, filename)
        print(f"Saving... {filename}")
        plt.savefig(filename, dpi=300)  

    def plot_accuracy(self, dataset, net_name, schemes, cases):
        """
        Plots accuracy over training for different experimental cases.

        Args:
            dataset
            net_name
            schemes
            cases (list): Experimental cases to include in figure.

        Returns:
            None.

        """
        # pull data
        acc_df = self.stats_processor.load_accuracy_df(dataset, net_name, 
            cases, schemes)

        # group and compute stats
        acc_df.set_index(["train_scheme", "case", "epoch"], inplace=True)
        acc_df_groups = acc_df.groupby(["train_scheme", "case", "epoch"])
        acc_df_stats = acc_df_groups.agg({ "acc": [np.mean, np.std] })
        acc_df_stats_groups = acc_df_stats.groupby(["train_scheme", "case"])
        
        # plot
        fig, ax = plt.subplots(figsize=(14,8))
        clrs = sns.color_palette("hls", len(acc_df_stats_groups.groups))
        
        for group, clr in zip(acc_df_stats_groups.groups, clrs):

            scheme, case = group
            group_data = acc_df_stats_groups.get_group((scheme, case))

            # error bars = 2 standard devs
            yvals = group_data["acc"]["mean"].values
            yerr = group_data["acc"]["std"].values * 1.98
            ax.plot(range(len(yvals)), yvals, label=f"{scheme} {case}", c=clr)
            ax.fill_between(range(len(yvals)), yvals - yerr, yvals + yerr,
                    alpha=0.1, facecolor=clr)
            
        ax.set_title(f"Classification accuracy during training: {net_name}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation accuracy")
        ax.set_ylim([0.1, 1])
        ax.legend()
        
        # optional saving
        if not self.save_fig:
            print("Not saving.")
            plt.show()
            return
        
        sub_dir = ensure_sub_dir(self.data_dir, f"figures/{dataset}/{net_name}/accuracy/")
        case_names = " & ".join(cases)
        filename = f"{case_names} accuracy.svg"
        filename = os.path.join(sub_dir, filename)
        print(f"Saving... {filename}")
        plt.savefig(filename, dpi=300)
        

if __name__=="__main__":
    
    visualizer = AccuracyVisualizer("/home/briardoty/Source/allen-inst-cell-types/data_mountpoint", 
        10, save_fig=True, refresh=False)
    
    # visualizer.plot_final_accuracy(["swish_0.5", "swish_1", "swish_3", "swish_5", "swish_10"], ["swish_1-3", "swish_5-10"])

    # visualizer.plot_accuracy("cifar10", "sticknet8", ["adam"], ["relu", "tanhe0.01", "tanhe0.1", "tanhe0.5", "tanhe1", "tanhe2", "tanhe5"])
    # visualizer.plot_accuracy("cifar10", "sticknet8", ["adam"], ["relu", "swish0.1", "swish0.5", "swish1", "swish2", "swish5", "swish7.5", "swish10"])
    # visualizer.plot_accuracy("cifar10", "sticknet8", ["adam"], ["relu", "swish10", "tanhe1", "swish10-tanhe1"])
    # visualizer.plot_accuracy("cifar10", "sticknet8", ["adam"], ["relu", "swish5", "tanhe0.5", "swish5-tanhe0.5"])
    # visualizer.plot_accuracy("cifar10", "sticknet8", ["adam"], ["relu", "swish0.1", "tanhe5", "swish0.1-tanhe5"])
    # visualizer.plot_accuracy("cifar10", "sticknet8", ["adam"], ["relu", "swish1", "tanhe1", "swish1-tanhe1"])
    # visualizer.plot_accuracy("cifar10", "vgg11", ["adam"], ["relu", "swish10-tanhe1", "relu-spatial", "swish10-tanhe1-spatial"])
    
    visualizer.plot_predictions("cifar10",
        ["vgg11", "sticknet8"],
        ["adam"], "linear")

    # visualizer.scatter_final_acc("cifar10", 
    #     ["vgg11", "sticknet8"], 
    #     ["adam"])
