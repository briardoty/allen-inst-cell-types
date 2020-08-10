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
    
    def __init__(self, data_dir, n_classes=10, save_fig=False):
        
        self.data_dir = data_dir
        self.save_fig = save_fig
        
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
        filename = f"{fn_names}.png"
        filename = os.path.join(sub_dir, filename)
        print(f"Saving... {filename}")
        plt.savefig(filename, dpi=300)

    def scatter_final_acc(self, net_names, schemes, cases):
        """
        Plot a scatter plot of predicted vs actual final accuracy for the 
        given mixed cases.

        Args:
            net_names
            schemes
            cases
        """

        # pull data
        df, act_fn_param_dict = self.stats_processor.load_final_acc_df(net_names, cases, schemes)
        df_groups = df.groupby(["net_name", "train_scheme", "case"])

        # plot
        mixed_cases = [a for a in act_fn_param_dict.keys() if len(act_fn_param_dict[a]) > 1]
        fig, axes = plt.subplots(figsize=(14,14))
        clrs = sns.color_palette("hls", len(net_names)*len(schemes)*len(mixed_cases) + 1)
        
        # plot mixed cases
        i = 0
        for g in df_groups.groups:

            net, scheme, case = g
            if not case in mixed_cases:
                continue

            g_data = df_groups.get_group((net, scheme, case))
            clr = clrs[i]

            # actual
            y_act = g_data["final_val_acc"]["mean"].values[0]
            y_err = g_data["final_val_acc"]["std"].values[0] * 2

            # prediction
            act_fn_params = [p for p in act_fn_param_dict[case]]
            component_cases = [k for k, v in act_fn_param_dict.items() if len(v) == 1 and v[0] in act_fn_params]
            x_pred = df["final_val_acc"]["mean"][net][scheme][component_cases].mean()
            
            # TODO: factor in x error?

            # plot
            axes.errorbar(x_pred, y_act, yerr=y_err, label=f"{net} {scheme} {case}",
                capsize=3, elinewidth=1, c=clr, fmt=".")

            i += 1

        # plot reference line
        x = np.linspace(0, 1, 50)
        axes.plot(x, x, c=clrs[-1], dashes=[6,2])

        # set figure text
        axes.set_title("Linear predicted vs actual mixed case final accuracy")
        axes.set_xlabel("Predicted")
        axes.set_ylabel("Actual")
        axes.legend()
        axes.set_xlim([0, 1])
        axes.set_ylim([0, 1])
         
        # optional saving
        if not self.save_fig:
            print("Not saving.")
            plt.show()
            return

        sub_dir = ensure_sub_dir(self.data_dir, f"figures/scatter/")
        net_names = ", ".join(net_names)
        schemes = ", ".join(schemes)
        cases = ", ".join(cases)
        filename = f"{net_names}_{schemes}_{cases}_scatter.png"
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
        acc_df, act_fn_param_dict = self.stats_processor.load_final_acc_df(
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

        sub_dir = ensure_sub_dir(self.data_dir, f"figures/{net_name}/final accuracy/")
        cases = " & ".join(mixed_cases)
        filename = f"{cases} final acc.png"
        filename = os.path.join(sub_dir, filename)
        print(f"Saving... {filename}")
        plt.savefig(filename, dpi=300)  

    def plot_accuracy(self, net_name, cases, train_schemes):
        """
        Plots accuracy over training for different experimental cases.

        Args:
            cases (list): Experimental cases to include in figure.

        Returns:
            None.

        """
        # pull data
        acc_df = self.stats_processor.load_accuracy_df(net_name, cases, 
            train_schemes)

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
        
        sub_dir = ensure_sub_dir(self.data_dir, f"figures/{net_name}/accuracy/")
        case_names = " & ".join(cases)
        filename = f"{case_names} accuracy.png"
        filename = os.path.join(sub_dir, filename)
        print(f"Saving... {filename}")
        plt.savefig(filename, dpi=300)  
        
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
        filename = f"{case} weight distr.png"
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
        filename = f"{cases} weight.png"
        filename = os.path.join(sub_dir, filename)
        print(f"Saving... {filename}")
        plt.savefig(filename, dpi=300)
        

if __name__=="__main__":
    
    visualizer = Visualizer("/home/briardoty/Source/allen-inst-cell-types/data_mountpoint", 10, False)
    
    # visualizer.plot_type_specific_weights("swish10-tanhe1-relu")

    # visualizer.plot_final_accuracy(["swish_0.5", "swish_1", "swish_3", "swish_5", "swish_10"], ["swish_1-3", "swish_5-10"])

    # visualizer.plot_weight_changes(["unmodified"], ["adam"])
    
    # visualizer.plot_accuracy("sticknet8", ["unmodified"], ["sgd"])
    
    visualizer.scatter_final_acc(["vgg11", "sticknet"], ["adam", "sgd"], 
        ["swish_10", "tanhe_1.0", "swish10-tanhe1"])

    # visualizer.plot_activation_fns([Sigfreud(1), Sigfreud(1.5), Sigfreud(2.), Sigfreud(4.)])
    # visualizer.plot_activation_fns([Swish(3), Swish(5), Swish(10)])
    # visualizer.plot_activation_fns([Tanhe(0.1), Tanhe(0.5), Tanhe(1)])
    # visualizer.plot_activation_fns([Renlu(0.5), Renlu(1), Renlu(1.5)])
