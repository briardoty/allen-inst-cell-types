#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 14:22:11 2020

@author: briardoty
"""
import os
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

def get_key(dct, val):

    for k, v in dct.items():
        if val == v:
            return k
        return None

class Visualizer():
    
    def __init__(self, data_dir, net_name, train_scheme, n_classes=10, save_fig=False):
        
        self.data_dir = data_dir
        self.train_scheme = train_scheme
        self.save_fig = save_fig
        
        self.stats_processor = StatsProcessor(net_name, n_classes, data_dir, train_scheme)
        
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
        filename = f"{fn_names}.png"
        filename = os.path.join(sub_dir, filename)
        print(f"Saving... {filename}")
        plt.savefig(filename, dpi=300)

    def plot_final_accuracy(self, control_cases, mixed_cases):
        """
        Plot accuracy at the end of training for given control cases
        and mixed case, including predicted mixed case accuracy based
        on linear combination of control cases
        """

        # pull data
        acc_df, act_fn_param_dict = self.stats_processor.load_final_acc_df(
            control_cases + mixed_cases)
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
            p = float(act_fn_param_dict[case][0])

            # error bars = 2 standard devs
            yvals = group["final_val_acc"]["mean"].values
            yerr = group["final_val_acc"]["std"].values * 2
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
            y_err = group["final_val_acc"]["std"].values * 2
            l = f"{mixed_case} actual"
            h = axes[1].errorbar(i, y_act, yerr=y_err, label=l,
                capsize=3, elinewidth=1, c=clrs[len(control_cases) + i], fmt=".")
            
            labels.append(l)
            handles.append(h)

            # predicted
            ps = [p for p in act_fn_param_dict[mixed_case]]
            component_cases = [k for k, v in act_fn_param_dict.items() if len(v) == 1 and v[0] in ps]
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
        axes[1].legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))
         
        # optional saving
        if not self.save_fig:
            print("Not saving.")
            plt.show()
            return

        sub_dir = ensure_sub_dir(self.data_dir, f"figures/{self.stats_processor.net_name}/{self.train_scheme}/final accuracy/")
        cases = " & ".join(mixed_cases)
        filename = f"{cases} final acc.png"
        filename = os.path.join(sub_dir, filename)
        print(f"Saving... {filename}")
        plt.savefig(filename, dpi=300)  

    def plot_accuracy(self, case_ids, train_schemes):
        """
        Plots accuracy over training for different experimental cases.

        Args:
            cases (list): Experimental cases to include in figure.

        Returns:
            None.

        """
        # pull data
        acc_df = self.stats_processor.load_accuracy_df(case_ids, train_schemes)

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
            yerr = group_data["acc"]["std"].values * 2
            ax.plot(range(len(yvals)), yvals, label=f"{scheme} {case}", c=clr)
            ax.fill_between(range(len(yvals)), yvals - yerr, yvals + yerr,
                    alpha=0.2, facecolor=clr)
            
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
        
        sub_dir = ensure_sub_dir(self.data_dir, f"figures/{self.stats_processor.net_name}/{self.train_scheme}/accuracy/")
        cases = " & ".join(case_ids)
        filename = f"{cases} accuracy.png"
        filename = os.path.join(sub_dir, filename)
        print(f"Saving... {filename}")
        plt.savefig(filename, dpi=300)  
        
    def plot_type_specific_weights(self, case):
        """
        Plots mean absolute weights for each cell type across layers 
        """

        # pull data
        df = self.stats_processor.load_weight_df(case)

        # plot
        state_keys = list(nets[self.stats_processor.net_name]["state_keys"].keys())
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
        labels = list(nets[self.stats_processor.net_name]["state_keys"].values())
        ax.set_xticklabels(labels)

        # optional saving
        if not self.save_fig:
            print("Not saving.")
            plt.show()
            return

        sub_dir = ensure_sub_dir(self.data_dir, f"figures/{self.stats_processor.net_name}/{self.train_scheme}/weight distr/")
        filename = f"{case} weight distr.png"
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
        state_keys = list(nets[self.stats_processor.net_name]["state_keys"].keys())

        # plot
        x = np.array([i * 1.25 for i in range(len(state_keys))])
        width = 1.0 / len(case_ids)
        err_kw = dict(lw=1, capsize=3, capthick=1)

        fig, ax = plt.subplots(figsize=(14,8))
        clrs = sns.color_palette("hls", len(case_ids))

        for i in range(len(case_ids)):

            case = case_ids[i]
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

        loc = (len(case_ids) - 1) / (2. * len(case_ids))
        ax.set_xticks([loc + i * 1.25 for i in range(len(state_keys))])
        labels = list(nets[self.stats_processor.net_name]["state_keys"].values())
        ax.set_xticklabels(labels)

        # optional saving
        if not self.save_fig:
            print("Not saving.")
            plt.show()
            return

        sub_dir = ensure_sub_dir(self.data_dir, f"figures/{self.stats_processor.net_name}/{self.train_scheme}/weight change/")
        cases = " & ".join(case_ids)
        filename = f"{cases} weight.png"
        filename = os.path.join(sub_dir, filename)
        print(f"Saving... {filename}")
        plt.savefig(filename, dpi=300)
        

if __name__=="__main__":
    
    visualizer = Visualizer("/home/briardoty/Source/allen-inst-cell-types/data_mountpoint", "vgg11", 10, False)
    
    # visualizer.plot_type_specific_weights("swish10-tanhe1-relu")

    # visualizer.plot_final_accuracy(["swish_0.5", "swish_1", "swish_3", "swish_5", "swish_10"], ["swish_1-3", "swish_5-10"])

    # visualizer.plot_weight_changes(["control2", "mixed-2_relu10_nr-1"])
    
    visualizer.plot_accuracy(["unmodified"], ["sgd", "adam"])
    
    # visualizer.plot_activation_fns([Sigfreud(1), Sigfreud(1.5), Sigfreud(2.), Sigfreud(4.)])
    # visualizer.plot_activation_fns([Swish(3), Swish(5), Swish(10)])
    # visualizer.plot_activation_fns([Tanhe(0.1), Tanhe(0.5), Tanhe(1)])
    # visualizer.plot_activation_fns([Renlu(0.5), Renlu(1), Renlu(1.5)])
