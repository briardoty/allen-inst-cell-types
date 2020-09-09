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

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

try:
    from .StatsProcessor import StatsProcessor
except:
    from StatsProcessor import StatsProcessor

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

    def plot_predictions(self, dataset, net_names, schemes, excl_arr, 
        pred_type="max", cross_family=None):
        """
        Plot a single axis figure of offset from predicted max accuracy for
        the given mixed cases.
        """

        # pull data
        df, _, _ = self.stats_processor.load_max_acc_df(self.refresh)

        # performance relative to predictions
        df["acc_vs_linear"] = df["max_val_acc"]["mean"] - df["linear_pred"]["mean"]
        df["acc_vs_max"] = df["max_val_acc"]["mean"] - df["max_pred"]["mean"]

        # filter dataframe
        df = df.query(f"is_mixed") \
            .query(f"dataset == '{dataset}'") \
            .query(f"net_name in {net_names}") \
            .query(f"train_scheme in {schemes}")
        if cross_family is not None:
            df = df.query(f"cross_fam == {cross_family}")
        for excl in excl_arr:
            df = df.query(f"not case.str.contains('{excl}')", engine="python")
        sort_df = df.sort_values(["net_name", f"acc_vs_{pred_type}"])
        
        # determine each label length for alignment
        lengths = {}
        for i in range(4):
            lengths[i] = np.max([len(x) for x in sort_df.index.unique(level=i)]) + 2

        # plot
        plt.figure(figsize=(16,16))
        plt.gca().axvline(0, color='k', linestyle='--')
        clrs = sns.color_palette("husl", len(net_names))

        ylabels = dict()
        handles = dict()
        sig_arr = list()
        i = 0
        xmax = 0
        xmin = 0
        for midx in sort_df.index.values:

            # dataset, net, scheme, case, mixed, cross-family
            d, n, s, c, m, cf = midx
            clr = clrs[net_names.index(n)]

            # prettify
            if np.mod(i, 2) == 0:
                plt.gca().axhspan(i-.5, i+.5, alpha = 0.1, color="k")
            
            # stats
            perf = sort_df.loc[midx][f"acc_vs_{pred_type}"].values[0] * 100
            err = sort_df.loc[midx]["max_val_acc"]["std"] * 1.98 * 100

            xmin = min(xmin, perf - err)
            xmax = max(xmax, perf + err)

            # plot "good" and "bad"
            if perf - err > 0:
                if cf or cross_family is not None:
                    plt.plot([perf - err, perf + err], [i,i], linestyle="-", 
                        c=clr, linewidth=6, alpha=.8)
                else:
                    plt.plot([perf - err, perf + err], [i,i], linestyle=":", 
                        c=clr, linewidth=6, alpha=.8)
                h = plt.plot(perf, i, c=clr, marker="o")
                handles[n] = h[0]
            else:
                if cf or cross_family is not None:
                    plt.plot([perf - err, perf + err], [i,i], linestyle="-", 
                        c=clr, linewidth=6, alpha=.2)
                    plt.plot([perf - err, perf + err], [i,i], linestyle="-", 
                        linewidth=6, c='k', alpha=.1)
                else:
                    plt.plot([perf - err, perf + err], [i,i], linestyle=":", 
                        c=clr, linewidth=6, alpha=.2)
                    plt.plot([perf - err, perf + err], [i,i], linestyle=":", 
                        linewidth=6, c='k', alpha=.1)
                h = plt.plot(perf, i, c=clr, marker="o", alpha=0.5)
                if handles.get(n) is None:
                    handles[n] = h[0]

            # BH corrected significance
            sig_arr.append(sort_df.loc[midx, f"{pred_type}_pred_rej_h0"].values[0])

            # make an aligned label
            aligned = d.ljust(lengths[0]) + n.ljust(lengths[1]) +\
                s.ljust(lengths[2]) + c.ljust(lengths[3])
            ylabels[i] = aligned

            # track vars
            i += 1

        # indicate BH corrected significance
        h = plt.plot(i+100, 0, "k*", alpha=0.5)
        handles["p < 0.05"] = h[0]
        for i in range(len(sig_arr)):
            if sig_arr[i]:
                plt.plot(xmax + xmax/12., i, "k*", alpha=0.5)

        # determine padding for labels
        max_length = np.max([len(l) for l in ylabels.values()])

        # add handles
        if cross_family is None:
            h1 = plt.gca().axhline(i+100, color="k", linestyle="-", alpha=0.5)
            h2 = plt.gca().axhline(i+100, color="k", linestyle=":", alpha=0.5)
            handles["cross-family"] = h1
            handles["within-family"] = h2

        # set figure text
        plt.title("Mixed network performance relative to predicted performance", 
            fontsize=20, pad=20)
        plt.xlabel(f"Accuracy relative to {pred_type} prediction (%)", 
            fontsize=16, labelpad=10)
        plt.ylabel("Network configuration", fontsize=16, labelpad=10)
        plt.yticks(list(ylabels.keys()), ylabels.values(), ha="left")
        plt.ylim(-0.5, i - 0.5)
        plt.legend(handles.values(), handles.keys(), fontsize=14, loc="upper left")
        plt.xlim([xmin - xmax/10., xmax + xmax/10.])
        yax = plt.gca().get_yaxis()
        yax.set_tick_params(pad=max_length*7)
        plt.tight_layout()

        # optional saving
        if not self.save_fig:
            print("Not saving.")
            plt.show()
            return

        sub_dir = ensure_sub_dir(self.data_dir, f"figures/{dataset}/prediction")
        net_names = ", ".join(net_names)
        schemes = ", ".join(schemes)
        filename = f"{dataset}_{net_names}_{schemes}_{pred_type}-prediction"
        if cross_family == True:
            filename += "_xfam"
        elif cross_family == False:
            filename += "_infam"
        filename = os.path.join(sub_dir, filename)
        print(f"Saving... {filename}")
        plt.savefig(f"{filename}.svg")  
        plt.savefig(f"{filename}.png", dpi=300)  

    def scatter_acc(self, dataset, net_names, schemes, excl_arr, 
        pred_type="max", cross_family=None):
        """
        Plot a scatter plot of predicted vs actual max accuracy for the 
        given mixed cases.
        """

        # pull data
        df, _, _ = self.stats_processor.load_max_acc_df(self.refresh)

        # filter dataframe
        df = df.query(f"is_mixed")
        df = df.query(f"dataset == '{dataset}'") 
        df = df.query(f"net_name in {net_names}")
        df = df.query(f"train_scheme in {schemes}")
        if cross_family is not None:
            df = df.query(f"cross_fam == {cross_family}")
        for excl in excl_arr:
            df = df.query(f"not case.str.contains('{excl}')", engine="python")

        # plot
        fig, ax = plt.subplots(figsize=(12,12))

        # identifiers
        fmts = [".", "^"]
        clrs = sns.color_palette("husl", len(net_names))

        # plot mixed cases
        i = 0
        xmin, ymin, xmax, ymax = 100, 100, 10, 10
        for midx in df.index.values:

            # dataset, net, scheme, case, mixed
            d, n, s, c, m, cf = midx

            # pick identifiers
            clr = clrs[net_names.index(n)] # net
            mfc = None if cf else "None" # xfam vs within fam
            fmt = fmts[0]

            # actual
            y_act = df.loc[midx]["max_val_acc"]["mean"] * 100
            y_err = df.loc[midx]["max_val_acc"]["std"] * 1.98 * 100

            # prediction
            x_pred = df.loc[midx][f"{pred_type}_pred"].values[0] * 100
            x_err = df.loc[midx][f"{pred_type}_std"].values[0] * 1.98 * 100
            
            # plot
            h = ax.plot(x_pred, y_act, c=clr, marker=fmt, markersize=10,
                markerfacecolor=mfc)

            facecolor = clr if cf else "None"
            alpha = 0.15 if cf else 0.4
            lipse = matplotlib.patches.Ellipse((x_pred, y_act), x_err, y_err, 
                facecolor=facecolor, edgecolor=clr, alpha=alpha)
            ax.add_patch(lipse)

            # update limits
            xmin = min(x_pred, xmin)
            ymin = min(y_act, ymin)
            xmax = max(x_pred, xmax)
            ymax = max(y_act, ymax)

            i += 1

        # plot reference line
        x = np.linspace(0, 100, 500)
        ax.plot(x, x, c=(0.5, 0.5, 0.5, 0.25), dashes=[6,2])

        # build legend handles
        handles = dict()
        for n in net_names:
            clr = clrs[net_names.index(n)]
            h = ax.plot(1000, 1000, c=clr, marker=fmt, markersize=20)
            handles[n] = h[0]

        if cross_family is None:
            gray = (0.5, 0.5, 0.5)
            h1 = ax.plot(1000, 1000, c=gray, marker=fmt, markersize=20, mfc=None)
            h2 = ax.plot(1000, 1000, c=gray, marker=fmt, markersize=20, mfc="None")
            handles["cross-family"] = h1[0]
            handles["within-family"] = h2[0]

        # set figure text
        ax.set_title(f"{pred_type.capitalize()} prediction vs actual mixed network accuracy", 
            fontsize=20, pad=20)
        ax.set_xlabel(f"{pred_type.capitalize()} predicted accuracy (%)", fontsize=16)
        ax.set_ylabel("Actual accuracy (%)", fontsize=16)
        
        ax.set_xlim([xmin - 1, xmax + 1])
        ax.set_ylim([ymin - 1, ymax + 1])
        ax.set_aspect("equal", "box")
        ax.legend(handles.values(), handles.keys(), fontsize=14)

        plt.tight_layout()
         
        # optional saving
        if not self.save_fig:
            print("Not saving.")
            plt.show()
            return

        sub_dir = ensure_sub_dir(self.data_dir, f"figures/{dataset}/scatter")
        net_names = ", ".join(net_names)
        schemes = ", ".join(schemes)
        filename = f"{dataset}_{net_names}_{schemes}_{pred_type}-scatter"
        if cross_family == True:
            filename += "_xfam"
        elif cross_family == False:
            filename += "_infam"
        filename = os.path.join(sub_dir, filename)
        print(f"Saving... {filename}")
        plt.savefig(f"{filename}.svg")  
        plt.savefig(f"{filename}.png", dpi=300)  

    def plot_single_accuracy(self, dataset, net_name, scheme, case, sample=0):
        """
        Plot single net accuracy trajectory with windowed z score

        Args:
            dataset
            net_name
            scheme
            case
            sample
        """

        # pull data
        acc_df = self.stats_processor.load_accuracy_df(dataset, net_name, 
            [case], [scheme], self.refresh)

        # filter more
        acc_df = acc_df.query(f"sample == {sample}")

        # windowed z score at each epoch
        window = 10
        acc_df["z"] = 0.
        for idx in acc_df.index.values:

            # decompose
            row = acc_df.loc[idx]
            d, n, sch, c, s, e, acc, z = row

            # get window
            w_start = max(0, e - window + 1)
            w_idx = acc_df.epoch[w_start:e + 1].index

            # compute window stats
            w_mean = np.mean(acc_df.loc[w_idx, "acc"])
            w_std = np.std(acc_df.loc[w_idx, "acc"])

            z_score = (acc - w_mean) / w_std if w_std != 0 else 0

            # update df
            acc_df.at[idx, "z"] = z_score

        # plot...
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14,8), sharex=True)
        fig.subplots_adjust(hspace=0)
        clrs = sns.color_palette("hls", 2)

        # plot acc
        yvals = acc_df["acc"].values
        axes[0].plot(range(len(yvals)), yvals, c=clrs[0])

        # plot z score
        yvals = acc_df["z"].values
        axes[1].plot(range(len(yvals)), yvals, c=clrs[1])
        
        fig.suptitle(f"Classification accuracy during training: {net_name} on {dataset}", fontsize=20)
        # ax.set_xlabel("Epoch", fontsize=16)
        # ax.set_ylabel("Validation accuracy (%)", fontsize=16)
        # ax.set_ylim([10, 100])
        # ax.legend(fontsize=14)
        
        plt.tight_layout()
        
        # optional saving
        if not self.save_fig:
            print("Not saving.")
            plt.show()
            return
        
        sub_dir = ensure_sub_dir(self.data_dir, f"figures/{dataset}/{net_name}/accuracy/")
        case_names = ", ".join(cases)
        schemes = ", ".join(schemes)
        filename = f"{dataset}_{net_name}_{schemes}_{case_names} accuracy"
        filename = os.path.join(sub_dir, filename)
        print(f"Saving... {filename}")
        plt.savefig(f"{filename}.svg")
        plt.savefig(f"{filename}.png", dpi=300)

    def plot_accuracy(self, dataset, net_name, schemes, cases, inset=True):
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
            cases, schemes, self.refresh)

        # process a bit
        index_cols = ["dataset", "net_name", "train_scheme", "case", "epoch"]
        acc_df.set_index(index_cols, inplace=True)
        df_stats = acc_df.groupby(index_cols).agg({ "acc": [np.mean, np.std] })

        # group
        df_groups = df_stats.groupby(index_cols[:-1])
        
        # plot
        fig, ax = plt.subplots(figsize=(14,8))
        clrs = sns.color_palette("hls", len(df_groups.groups))
        
        y_arr = []
        yerr_arr = []
        for group, clr in zip(df_groups.groups, clrs):

            d, n, s, c = group
            group_data = df_groups.get_group(group)

            # error bars = 2 standard devs
            yvals = group_data["acc"]["mean"].values * 100
            yerr = group_data["acc"]["std"].values * 1.98 * 100
            ax.plot(range(len(yvals)), yvals, label=f"{s} {c}", c=clr)
            ax.fill_between(range(len(yvals)), yvals - yerr, yvals + yerr,
                    alpha=0.1, facecolor=clr)

            # for the insert...
            y_arr.append(yvals)
            yerr_arr.append(yerr)
            
        # zoomed inset
        if inset:
            axins = zoomed_inset_axes(ax, zoom=10, loc=8)
            for yvals, yerr, clr in zip(y_arr, yerr_arr, clrs):
                nlast = 10
                x = [i for i in range(len(yvals) - nlast, len(yvals))]
                y_end = yvals[-nlast:]
                yerr_end = yerr[-nlast:]
                axins.plot(x, y_end, c=clr)
                axins.fill_between(x, y_end - yerr_end, y_end + yerr_end,
                        alpha=0.1, facecolor=clr)

            mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
            axins.xaxis.set_ticks([])

        ax.set_title(f"Classification accuracy during training: {net_name} on {dataset}", fontsize=20)
        ax.set_xlabel("Epoch", fontsize=16)
        ax.set_ylabel("Validation accuracy (%)", fontsize=16)
        ax.set_ylim([10, 100])
        ax.legend(fontsize=14)
        
        plt.tight_layout()
        
        # optional saving
        if not self.save_fig:
            print("Not saving.")
            plt.show()
            return
        
        sub_dir = ensure_sub_dir(self.data_dir, f"figures/{dataset}/{net_name}/accuracy/")
        case_names = ", ".join(cases)
        schemes = ", ".join(schemes)
        filename = f"{dataset}_{net_name}_{schemes}_{case_names} accuracy"
        filename = os.path.join(sub_dir, filename)
        print(f"Saving... {filename}")
        plt.savefig(f"{filename}.svg")
        plt.savefig(f"{filename}.png", dpi=300)
        

if __name__=="__main__":
    
    visualizer = AccuracyVisualizer("/home/briardoty/Source/allen-inst-cell-types/data_mountpoint", 
        10, save_fig=False, refresh=False)
    
    # visualizer.plot_max_accuracy(["swish_0.5", "swish_1", "swish_3", "swish_5", "swish_10"], ["swish_1-3", "swish_5-10"])

    # visualizer.plot_accuracy("cifar10", "vgg11", ["adam"], ["tanh0.01", "tanh0.1", "tanh0.5", "tanh1", "tanh2", "tanh5", "tanh10"], inset=False)
    # visualizer.plot_accuracy("cifar10", "vgg11", ["adam"], ["swish0.1", "swish0.5", "swish1", "swish2", "swish5", "swish7.5", "swish10"])
    # visualizer.plot_accuracy("cifar10", "vgg11", ["adam"], ["swish1", "tanhe1", "swish1-tanhe1"])
    # visualizer.plot_accuracy("cifar10", "sticknet8", ["adam"], ["relu", "swish5", "tanhe0.5", "swish5-tanhe0.5"])
    # visualizer.plot_accuracy("cifar10", "sticknet8", ["adam"], ["relu", "swish0.1", "tanhe5", "swish0.1-tanhe5"])
    # visualizer.plot_accuracy("cifar10", "sticknet8", ["adam"], ["relu", "swish1", "tanhe1", "swish1-tanhe1"])
    # visualizer.plot_accuracy("cifar10", "vgg11", ["adam"], ["relu", "swish10-tanhe1", "relu-spatial", "swish10-tanhe1-spatial"])
    # visualizer.plot_accuracy("cifar10", "vgg11", ["adam"], ["swish1", "swish2", "swish5", "swish7.5", "swish10", "swish1-2", "swish5-7.5", "swish5-10", "swish1-10"])
    # visualizer.plot_accuracy("cifar10", "vgg11", ["adam", "sgd"], ["relu"], inset=False)
    # visualizer.plot_accuracy("cifar10", "vgg11", ["adam"], ["swish10", "tanhe0.5", "swish10-tanhe0.5"], inset=True)
    # visualizer.plot_accuracy("cifar10", "vgg11", ["adam"], ["testrelu", "testswish10", "testswish1", "testtanh0.1", "testtanh2"], inset=False)

    visualizer.plot_single_accuracy("cifar10", "vgg11", "adam", "swish10", sample=0)

    # visualizer.plot_predictions("cifar10",
    #     ["vgg11", "sticknet8"],
    #     ["adam"], 
    #     excl_arr=["spatial", "tanhe5", "tanhe0.1-5"],
    #     pred_type="max",
    #     cross_family=True)

    # visualizer.scatter_acc("cifar10",
    #     ["vgg11", "sticknet8"],
    #     ["adam"], 
    #     excl_arr=["spatial", "tanhe5", "tanhe0.1-5"],
    #     pred_type="max",
    #     cross_family=True)
