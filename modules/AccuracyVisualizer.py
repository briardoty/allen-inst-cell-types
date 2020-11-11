#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 14:22:11 2020

@author: briardoty
"""
import os
import sys
import functools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
import json

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

try:
    from .AccuracyLoader import AccuracyLoader
except:
    from AccuracyLoader import AccuracyLoader

try:
    from .util import ensure_sub_dir, get_component_cases
except:
    from util import ensure_sub_dir, get_component_cases

import matplotlib
large_font_size = 16
small_font_size = 14
matplotlib.rc("xtick", labelsize=16) 
matplotlib.rc("ytick", labelsize=16) 


class AccuracyVisualizer():
    
    def __init__(self, data_dir, save_fig=False, save_png=False, sub_dir_name=None):
        
        self.data_dir = data_dir
        self.save_fig = save_fig
        self.save_png = save_png
        self.sub_dir_name = sub_dir_name
        
        self.stats_processor = AccuracyLoader(data_dir)

        with open(os.path.join(self.data_dir, "net_configs.json"), "r") as json_file:
            self.net_configs = json.load(json_file)

    def plot_final_acc_decomp(self, dataset, net_name, scheme, mixed_case):
        """
        Plot accuracy at the end of training for mixed case, 
        including predicted mixed case accuracy based
        on combination of component cases
        """

        # pull data
        df, case_dict, idx_cols = self.stats_processor.load_max_acc_df_ungrouped()
        component_cases = get_component_cases(case_dict, mixed_case)

        # filter dataframe
        df = df.query(f"dataset == '{dataset}'") \
            .query(f"net_name == '{net_name}'") \
            .query(f"train_scheme == '{scheme}'")

        # plot...
        markersize = 18
        c_labels = dict()
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5,4), sharey=True)
        fig.subplots_adjust(wspace=0)
        c_clrs = sns.color_palette("hls", len(component_cases))
        c_clrs.reverse()
        
        # plot component nets
        x = [-2]
        width = 0.35
        lw = 4
        for i in range(len(component_cases)):

            cc = component_cases[i]
            cc_rows = df.query(f"case == '{cc}'")
            yvals = cc_rows["max_val_acc"].values * 100
            start = x[-1] + 2
            x = [i for i in range(start, start + len(yvals))]

            axes[0].plot([i] * len(yvals), yvals, ".", label=cc,
                c=c_clrs[i], markersize=markersize, alpha=0.6)
            axes[0].plot([i-width, i+width], [np.mean(yvals), np.mean(yvals)], 
                linestyle=":", label=cc, c=c_clrs[i], linewidth=lw)
            
            cc = cc.replace("tanh", "ptanh")
            c_labels[i] = cc
            
        # plot mixed case
        # actual
        mwidth = width + 0.09
        m_clrs = sns.color_palette("hls", len(component_cases) + 3)
        m_rows = df.query(f"case == '{mixed_case}'")
        yact = m_rows["max_val_acc"].values * 100
        axes[1].plot([0] * len(yact), yact, ".", label=cc,
            c=m_clrs[len(component_cases)], markersize=markersize, alpha=0.6)
        axes[1].plot([-mwidth, mwidth], [np.mean(yact), np.mean(yact)], 
                linestyle=":", c=m_clrs[len(component_cases)], linewidth=lw)

        # predicted
        pred_types = ["max", "linear"]
        handles = dict()
        for pred_type, clr in zip(pred_types, m_clrs[-2:]):
            ypred = m_rows[f"{pred_type}_pred"].mean() * 100
            h = axes[1].plot([-width, width], [ypred, ypred], 
                label=cc, c=clr, linewidth=lw)

            # legend stuff
            handles[pred_type] = h[0]

        # legend stuff
        handles["mean"] = axes[0].plot(-100, ypred, "k:", 
            linewidth=lw, alpha=0.5)[0]

        # set figure text
        axes[0].set_xlabel("Component", fontsize=16, 
            labelpad=10)
        axes[1].set_xlabel("Mixed", fontsize=large_font_size, 
            labelpad=10)
        axes[0].set_ylabel("Final validation accuracy (%)", 
            fontsize=large_font_size, labelpad=10)
        axes[0].set_xlim([-0.5, len(component_cases) - 0.5])
        axes[1].set_xlim([-0.5, 0.5])
        axes[0].set_xticks(list(c_labels.keys()))
        axes[0].tick_params(axis="y", labelsize=12)
        axes[0].set_xticklabels(list(c_labels.values()), fontsize=12)
        axes[1].set_xticks([0])
        l = mixed_case.replace("tanh", "ptanh")
        axes[1].set_xticklabels([l], fontsize=12)
        plt.tight_layout()
        # plt.tight_layout(rect=[0, 0, 1, 0.92])

        # shrink axes...
        box1 = axes[0].get_position()
        axes[0].set_position([box1.x0, box1.y0, box1.width * 0.8, box1.height])
        box2 = axes[1].get_position()
        axes[1].set_position([box1.x0 + box1.width * 1.2, box2.y0, box2.width * 0.3, box2.height])

        # append legend to second axis
        axes[1].legend(handles.values(), handles.keys(), 
            fontsize=12, loc="center left", bbox_to_anchor=(1, 0.5))
         
        # optional saving
        if not self.save_fig:
            print("Not saving.")
            plt.show()
            return

        filename = f"{mixed_case} comparison"
        self.save("decomposition", filename)

    def get_prediction_df(self, datasets, net_names, schemes, cases, excl_arr, 
        pred_type="max", cross_family=None, mixed=True):

        # pull data
        df, case_dict, _ = self.stats_processor.load_max_acc_df()

        # performance relative to predictions
        df["acc_vs_linear"] = df["max_val_acc"]["mean"] - df["linear_pred"]["mean"]
        df["acc_vs_max"] = df["max_val_acc"]["mean"] - df["max_pred"]["mean"]

        # filter dataframe
        if mixed:
            df = df.query(f"is_mixed")
        df = df.query(f"dataset in {datasets}") \
            .query(f"net_name in {net_names}") \
            .query(f"train_scheme in {schemes}")
        if len(cases) > 0:
            df = df.query(f"case in {cases}")
        if cross_family is not None:
            df = df.query(f"cross_fam == {cross_family}")
        for excl in excl_arr:
            df = df.query(f"not case.str.contains('{excl}')", engine="python")

        # filter vgg tanh2 because it's terrible
        # df = df.query(f"not (net_name == 'vgg11' and (case.str.contains('tanh2') or (case.str.startswith('tanh') and case.str.endswith('-2'))))",
        #     engine="python")

        sort_df = df.sort_values(["net_name", f"acc_vs_{pred_type}"])

        return sort_df, case_dict

    def plot_ratio_group(self, dataset, net_name, scheme, ratio_group):

        # build dict for looking up cases in group
        ratio_cases = list(self.net_configs[ratio_group].keys())
        
        # pull data
        sort_df, case_dict = self.get_prediction_df([dataset], [net_name], [scheme], ratio_cases, 
            [], "max", None)
        
        # get component cases as either extreme
        cc_names = get_component_cases(case_dict, ratio_cases[0])
        cases = [cc_names[0]] + ratio_cases + [cc_names[1]]

        sort_df, case_dict = self.get_prediction_df([dataset], [net_name], [scheme], cases, 
            [], "max", None, mixed=False)

        # plot
        fig, ax = plt.subplots(figsize=(6,5))

        x = [-2]
        clrs = sns.color_palette("Set2", 2)
        lw = 4
        handles = dict()
        xlabels = dict()
        for i in range(len(cases)):

            c = cases[i]
            yval = sort_df.query(f"case == '{c}'")["max_val_acc"]["mean"][0] * 100
            yerr = sort_df.query(f"case == '{c}'")["max_val_acc"]["std"][0] * 1.98 * 100

            try:
                xlabels[i] = ":".join([str(n) for n in self.net_configs[ratio_group][c]["n_repeat"]])
                ax.plot([i, i], [yval - yerr, yval + yerr], linestyle="-", 
                    c=clrs[0], linewidth=lw, alpha=.8)
                h = ax.plot(i, yval, c=clrs[0], marker="o")
                handles["mixed"] = h[0]
            except KeyError:
                xlabels[i] = c
                ax.plot([i, i], [yval - yerr, yval + yerr], linestyle="-", 
                    c=clrs[1], linewidth=lw, alpha=.8)
                h = ax.plot(i, yval, c=clrs[1], marker="o")
                handles["component"] = h[0]

        # plot predictions
        ypred = sort_df.query(f"case == '{ratio_cases[0]}'")["max_pred"]["mean"][0] * 100
        h = ax.axhline(ypred, color="k", linestyle='--', alpha=0.5)
        handles["max pred"] = h

        ypred1 = sort_df.query(f"case == '{cases[0]}'")["max_val_acc"]["mean"][0] * 100
        ypred2 = sort_df.query(f"case == '{cases[-1]}'")["max_val_acc"]["mean"][0] * 100
        h = ax.plot([0, len(cases) - 1], [ypred1, ypred2], linestyle=":", c="k", 
            alpha=.5)
        handles["linear pred"] = h[0]

        # set figure text
        ax.set_xlabel("Net ratio", fontsize=16, labelpad=-10)
        ax.set_ylabel("Final validation accuracy (%)", 
            fontsize=large_font_size, labelpad=10)
        ax.set_xticks(list(xlabels.keys()))
        ax.set_xticklabels(list(xlabels.values()), fontsize=12, rotation=45)
        ax.tick_params(axis="y", labelsize=12)
        plt.tight_layout()

        # append legend
        ax.legend(handles.values(), handles.keys(), fontsize=12)
         
        # optional saving
        if not self.save_fig:
            print("Not saving.")
            plt.show()
            return

        filename = f"{ratio_group}"
        self.save("ratio group", filename)
        

    def plot_predictions(self, dataset, net_names, schemes, cases=[], excl_arr=[], 
        pred_type="max", cross_family=None, pred_std=False, small=False, filename=None):
        """
        Plot a single axis figure of offset from predicted max accuracy for
        the given mixed cases.
        """

        # pull data
        sort_df, _ = self.get_prediction_df([dataset], net_names, schemes, cases, 
            excl_arr, pred_type, cross_family)
        
        # determine each label length for alignment
        lengths = {}
        label_idxs = [3]
        for i in label_idxs:
            lengths[i] = np.max([len(x) for x in sort_df.index.unique(level=i)]) + 2

        # plot
        if small:
            plt.figure(figsize=(10,16))
        else:
            plt.figure(figsize=(16,16))
        plt.gca().axvline(0, color='k', linestyle='--')
        clrs = sns.color_palette("husl", len(net_names))

        ylabels = dict()
        handles = dict()
        sig_arr = list()
        i = 0
        xmax = 0
        xmin = 0
        lw = 600 / len(sort_df)
        ms = lw * 0.5
        for midx, row in sort_df.iterrows():

            # stats
            perf = row[f"acc_vs_{pred_type}"].values[0] * 100
            err = row["max_val_acc"]["std"] * 1.98 * 100

            xmin = min(xmin, perf - err)
            xmax = max(xmax, perf + err)

            # dataset, net, scheme, case, mixed, cross-family
            d, n, s, g, c, m, cf = midx
            clr = clrs[net_names.index(n)]
            
            # prettify	
            if np.mod(i, 2) == 0:	
                plt.gca().axhspan(i-.5, i+.5, alpha = 0.1, color="k")

            # BH corrected significance
            bh_sig = row[f"{pred_type}_pred_rej_h0"].values[0]
            sig_arr.append(bh_sig)
            if bh_sig:
                if cf or cross_family is not None:
                    plt.plot([perf - err, perf + err], [i,i], linestyle="-", 
                        c=clr, linewidth=lw, alpha=.8)
                else:
                    plt.plot([perf - err, perf + err], [i,i], linestyle=":", 
                        c=clr, linewidth=lw, alpha=.8)
                h = plt.plot(perf, i, c=clr, marker="o", ms=ms)
                handles[n] = h[0]
            else:
                if cf or cross_family is not None:
                    plt.plot([perf - err, perf + err], [i,i], linestyle="-", 
                        c=clr, linewidth=lw, alpha=.2)
                    plt.plot([perf - err, perf + err], [i,i], linestyle="-", 
                        linewidth=lw, c="k", alpha=.1)
                else:
                    plt.plot([perf - err, perf + err], [i,i], linestyle=":", 
                        c=clr, linewidth=lw, alpha=.2)
                    plt.plot([perf - err, perf + err], [i,i], linestyle=":", 
                        linewidth=lw, c="k", alpha=.1)
                h = plt.plot(perf, i, c=clr, marker="o", ms=ms, alpha=0.5)
                if handles.get(n) is None:
                    handles[n] = h[0]

            # optionally, plot the 95% ci for the prediction
            if pred_std:
                pred_err = row[f"{pred_type}_pred"]["std"] * 1.98 * 100
                plt.plot([-pred_err, pred_err], [i,i], linestyle="-", 
                        c="k", linewidth=lw, alpha=.2)

            # make an aligned label
            c = c.replace("tanh", "ptanh")
            label_arr = [d, n, s, c]
            aligned = "".join([label_arr[i].ljust(lengths[i]) for i in label_idxs])
            ylabels[i] = aligned

            # track vars
            i += 1

        # indicate BH corrected significance
        h = plt.plot(i+100, 0, "k*", alpha=0.5, ms=ms)
        handles["p < 0.05"] = h[0]
        xstar = xmax + xmax/14. if small else xmax + xmax/20.
        for i in range(len(sig_arr)):
            if sig_arr[i]:
                plt.plot(xstar, i, "k*", alpha=0.5, ms=ms)

        # determine padding for labels
        max_length = np.max([len(l) for l in ylabels.values()])

        # add handles
        if cross_family is None:
            h1 = plt.gca().axhline(i+100, color="k", linestyle="-", alpha=0.5)
            h2 = plt.gca().axhline(i+100, color="k", linestyle=":", alpha=0.5)
            handles["cross-family"] = h1
            handles["within-family"] = h2

        # set figure text
        plt.xlabel(f"Acc. relative to {pred_type} prediction (%)", 
            fontsize=28, labelpad=10)
        plt.ylabel("Network configuration", fontsize=28, labelpad=10)
        plt.yticks(list(ylabels.keys()), ylabels.values(), ha="left")
        yax = plt.gca().get_yaxis()
        yax.set_tick_params(pad=max_length*9)
        plt.ylim(-0.5, i + 0.5)
        plt.legend(handles.values(), handles.keys(), fontsize=20 if small else 22, loc="upper left")
        plt.xlim([xmin - xmax/10., xmax + xmax/10.])
        
        plt.tight_layout()

        # optional saving
        if not self.save_fig:
            print("Not saving.")
            plt.show()
            return

        if filename is None:
            net_names = ", ".join(net_names)
            schemes = ", ".join(schemes)
            filename = f"{dataset}_{net_names}_{schemes}_{pred_type}-prediction"
        self.save("prediction", filename)

    def plot_family_supplement(self, dataset, net_names, schemes, excl_arr, 
        pred_type="max", cross_family=None):

        # pull data
        df, _ = self.get_prediction_df([dataset], net_names, schemes, [], excl_arr, 
            pred_type, cross_family)

        # filter to just p-val < 0.05
        df = df[df.max_pred_rej_h0 == True]

        # plot
        fig, ax = plt.subplots(figsize=(8,5))
        net_vals = list(df.index.unique(level=1))
        net_vals.reverse()
        # x = np.array([i * 1.25 for i in range(len(net_vals))])
        x = 0

        cf_vals = list(df.index.unique(level=6))
        cf_vals.reverse()
        width = 1.0 / len(cf_vals)
        clrs = sns.color_palette("husl", len(cf_vals))

        err_kw = dict(lw=1, capsize=15, capthick=1)

        handles = dict()
        for i in range(len(net_vals)):

            net = net_vals[i]

            for j in range(len(net_vals)):

                cf = cf_vals[j]
                rows = df.query(f"net_name == '{net}'") \
                    .query(f"cross_fam == {cf}")[f"acc_vs_{pred_type}"]
                yval = np.mean(rows) * 100

                # plot
                label = "Cross-family" if cf else "Within-family"
                clr = clrs[i]
                alpha = 1 if cf else 0.4
                ax.bar(x, yval, width, label=label, 
                    color=clr, alpha=alpha)

                x += width

            x += 0.25
            
        # legend
        gray = (0.5, 0.5, 0.5)
        h2 = ax.bar(100, yval, width, label=label, 
            color=gray, alpha=0.4)
        handles["Within-family"] = h2
        h1 = ax.bar(100, yval, width, label=label, 
            color=gray, alpha=1)
        handles["Cross-family"] = h1

        # plot text
        loc = (len(cf_vals) - 1) / (2. * len(cf_vals))
        ax.set_xticks([loc + i * 1.25 for i in range(len(net_vals))])
        ax.set_xticklabels(net_vals)
        ax.set_xlim([-3/8., x - width + 1/8.])

        ax.axhline(y=0, color="k", linestyle="--", alpha=0.2)
        ax.set_xlabel("Network", fontsize=large_font_size + 2)
        ax.set_ylabel(f"Mean relative acc. (%)", fontsize=large_font_size + 2)
        ax.legend(handles.values(), handles.keys(), fontsize=18)
        plt.tight_layout()

        self.save("supplementary", "family")

    def plot_network_supplement(self, dataset, net_names, schemes, excl_arr, 
        pred_type="max", cross_family=None):

        # pull data
        df, _ = self.get_prediction_df([dataset], net_names, schemes, [], excl_arr, 
            pred_type, cross_family)

        # filter to just p-val < 0.05
        df = df[df.max_pred_rej_h0 == True]

        # plot
        fig, ax = plt.subplots(figsize=(5,5))
        x = 0
        net_vals = list(df.index.unique(level=1))
        net_vals.reverse()
        clrs = sns.color_palette("husl", len(net_vals))
        labels = []
        ticks = []
        for i in range(len(net_vals)):

            net = net_vals[i]
            yvals = df.query(f"net_name == '{net}'")[f"acc_vs_{pred_type}"]
            ymean = np.mean(yvals) * 100
            labels.append(net)
            ticks.append(x)
            ax.bar(x, ymean, 1/len(net_vals), label=net, color=clrs[i])
            x += 0.5

        ax.axhline(y=0, color="k", linestyle="--", alpha=0.2)
        ax.set_xlabel("Network", fontsize=large_font_size + 2)
        ax.set_ylabel(f"Mean relative acc. (%)", fontsize=large_font_size + 2)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        plt.tight_layout()

        self.save("supplementary", "network")

    def plot_dataset_supplement(self, datasets, net_names, schemes, excl_arr, 
        pred_type="max", cross_family=None):

        # pull data
        df, _ = self.get_prediction_df(datasets, net_names, schemes, [], excl_arr, 
            pred_type, cross_family)

        # filter to just p-val < 0.05
        df = df[df.max_pred_rej_h0 == True]

        # plot
        fig, ax = plt.subplots(figsize=(5,5))
        x = 0
        dataset_vals = list(df.index.unique(level=0))
        clrs = sns.color_palette("Set1", len(dataset_vals))
        labels = []
        ticks = []
        for i in range(len(dataset_vals)):

            dataset = dataset_vals[i]
            yvals = df.query(f"dataset == '{dataset}'")[f"acc_vs_{pred_type}"]
            ymean = np.mean(yvals) * 100
            labels.append(dataset)
            ticks.append(x)
            ax.bar(x, ymean, 1/len(dataset_vals), label=dataset, color=clrs[i])
            x += 0.5

        ax.axhline(y=0, color="k", linestyle="--", alpha=0.2)
        ax.set_xlabel("Dataset", fontsize=large_font_size + 2)
        ax.set_ylabel(f"Mean relative acc. (%)", fontsize=large_font_size + 2)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        plt.tight_layout()

        self.save("supplementary", "dataset")

    def scatter_acc(self, datasets, net_names, schemes, excl_arr, 
        pred_type="max", cross_family=None):
        """
        Plot a scatter plot of predicted vs actual max accuracy for the 
        given mixed cases.
        """

        # pull data
        df, _, _ = self.stats_processor.load_max_acc_df()

        # filter dataframe
        df = df.query(f"is_mixed")
        df = df.query(f"dataset in {datasets}") 
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
        clrs = sns.color_palette("husl", len(datasets))
        ms = 10

        # plot mixed cases
        i = 0
        xmin, ymin, xmax, ymax = 100, 100, 10, 10
        for midx in df.index.values:

            # dataset, net, scheme, case, mixed
            d, n, sch, g, c, m, cf = midx

            # pick identifiers
            clr = clrs[datasets.index(d)] # dataset
            mfc = None if cf else "None" # xfam vs within fam
            fmt = fmts[net_names.index(n)] # net
            alpha = 0.15 if cf else 0.4

            # actual
            y_act = df.loc[midx]["max_val_acc"]["mean"] * 100
            y_err = df.loc[midx]["max_val_acc"]["std"] * 1.98 * 100

            # prediction
            x_pred = df.loc[midx][f"{pred_type}_pred"]["mean"] * 100
            x_err = df.loc[midx][f"{pred_type}_pred"]["std"] * 100 * 1.98
            
            # plot
            h = ax.plot(x_pred, y_act, c=clr, marker=fmt, markersize=ms,
                markerfacecolor=mfc, alpha=alpha)

            facecolor = clr if cf else "None"
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
        gray = (0.5, 0.5, 0.5)
        ms = ms * 1.5
        for n in net_names:
            fmt = fmts[net_names.index(n)] # net
            h = ax.plot(1000, 1000, c=gray, marker=fmt, markersize=ms)
            handles[n] = h[0]
        
        for d in datasets:
            clr = clrs[datasets.index(d)] # dataset
            h = ax.plot(1000, 1000, c=clr, marker=".", markersize=ms)
            handles[d] = h[0]

        if cross_family is None:
            h1 = ax.plot(1000, 1000, c=gray, marker=".", markersize=ms, mfc=None)
            h2 = ax.plot(1000, 1000, c=gray, marker=".", markersize=ms, mfc="None")
            handles["cross-family"] = h1[0]
            handles["within-family"] = h2[0]

        # set figure text
        ax.set_xlabel(f"{pred_type.capitalize()} predicted accuracy (%)", fontsize=20)
        ax.set_ylabel("Actual accuracy (%)", fontsize=20)
        
        ax.set_xlim([xmin - 1, xmax + 1])
        ax.set_ylim([ymin - 1, ymax + 1])
        ax.set_aspect("equal", "box")
        ax.legend(handles.values(), handles.keys(), fontsize=18)

        plt.tight_layout()
         
        # optional saving
        if not self.save_fig:
            plt.show()
            return

        net_names = ", ".join(net_names)
        datasets = ", ".join(datasets)
        schemes = ", ".join(schemes)
        filename = f"{datasets}_{net_names}_{schemes}_{pred_type}-scatter"
        self.save("scatter", filename)

    def save(self, sub_dir_name, filename):

        if self.sub_dir_name is not None:
            sub_dir_name = self.sub_dir_name
        sub_dir = ensure_sub_dir(self.data_dir, f"figures/{sub_dir_name}")

        filename = os.path.join(sub_dir, filename)

        print(f"Saving... {filename}")

        plt.savefig(f"{filename}.svg")  

        if self.save_png:
            plt.savefig(f"{filename}.png", dpi=300)

    def ratio_heatmap(self, dataset, net_name, scheme, metric="acc_vs_max", cmap="RdBu_r"):

        # get all ratio groups
        ratio_groups = [g for g in self.net_configs.keys() if g.startswith("ratios-")]

        # build ratio matrix
        ratio_matrix = dict()
        vmin = 100
        vmax = -100
        for rg in ratio_groups:

            # get cases in this rg
            ratio_cases = list(self.net_configs[rg].keys())

            # pull data
            sort_df, case_dict = self.get_prediction_df([dataset], [net_name], [scheme], ratio_cases, 
                [], "max", None)
            
            # get component cases as either extreme
            cc_names = get_component_cases(case_dict, ratio_cases[0])
            cases = [cc_names[0]] + ratio_cases + [cc_names[1]]

            sort_df, case_dict = self.get_prediction_df([dataset], [net_name], [scheme], cases, 
                [], "max", None, mixed=False)

            # update preds on component cases
            if metric == "acc_vs_max":
                rows = sort_df.query(f"case in {cc_names}")
                max_pred = max(rows["max_val_acc"]["mean"])
                for row in rows.iterrows():
                    acc = sort_df.loc[row[0]]["max_val_acc"]["mean"]
                    sort_df.loc[row[0], "acc_vs_max"] = acc - max_pred

            # build array of relative accuracies for this rg
            rg_arr = list() 
            for c in cases:
                row = sort_df.query(f"case == '{c}'")
                if len(row) == 0:
                    break
                try:
                    yval = float(row[metric]["mean"][0]) * 100
                except:
                    yval = float(row[metric]) * 100
                rg_arr.append(yval)

                if yval == yval:
                    vmin = min(yval, vmin)
                    vmax = max(yval, vmax)
            
            if len(rg_arr) > 1:
                ratio_matrix[rg] = rg_arr

        # plot
        if metric == "acc_vs_max":
            metric = "Relative accuracy (%)"
            vabs = max(abs(vmin), abs(vmax))
            vmin, vmax = -vabs, vabs
        else:
            metric = "Peak accuracy (%)"

        ratio_matrix = sorted(list(ratio_matrix.items()), key=lambda kvp: max(kvp[1][1:-1]), reverse=True)
        M = np.array([v for k, v in ratio_matrix])

        plt.figure()
        im = plt.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax)
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax)

        # cbar = plt.gcf().colorbar(im, ax=plt.gca())
        cbar.ax.set_ylabel(metric, fontsize=large_font_size)
        cbar.ax.tick_params(labelsize=12)

        ax.set_xlabel("Net ratio", fontsize=large_font_size)
        ax.set_ylabel("Net config", fontsize=large_font_size)
        xlabels = [f"{10-i}:{i}" for i in range(11)]
        xticks = [i for i in range(len(xlabels))]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, fontsize=12, rotation=45)
        yticks = [i for i in range(len(ratio_matrix))]
        ylabels = [rg[len("ratios-"):]for rg, v in ratio_matrix]
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, fontsize=12)

        plt.tight_layout()

        if self.save_fig:
            filename = f"Ratio groups {net_name} {metric}"
            self.save("heatmaps", filename)
        else:
            plt.show()

    def heatmap_acc(self, dataset, net_name, scheme, metric="acc_vs_max", 
        cmap="RdBu_r", v_min=None, v_max=None):

        # pull data
        df, case_dict = self.get_prediction_df([dataset], [net_name], [scheme], list(), 
            [], "max", cross_family=None, mixed=False)
        
        # build parameter matrices
        p_dict = dict()
        for k in df.index.unique(level=df.index.names.index("case")):
            v = case_dict[k]
            for fn, p in zip(v["act_fns"], v["act_fn_params"]):
                if p == "None": continue
                if p_dict.get(fn) is None:
                    p_dict[fn] = set()
                p_dict[fn].add(float(p))
        p_dict = { k: sorted(v) for k, v in p_dict.items() }

        within_mats, cross_mats = dict(), dict()
        fn_keys = list(p_dict.keys())
        for k1, v1 in p_dict.items():
            within_mats[k1] = np.zeros((len(v1), len(v1)))
            fn_keys.remove(k1)
            for k2 in fn_keys:
                multikey = sorted([k1, k2])
                cross_mats[tuple(multikey)] = np.zeros((len(p_dict[multikey[0]]), len(p_dict[multikey[1]])))

        vmin = 100
        vmax = -100
        vm_dict = { "xf": {"vmin": vmin, "vmax": vmax}, "swish": {"vmin": vmin, "vmax": vmax}, "tanh": {"vmin": vmin, "vmax": vmax} }
        for midx, row in df.iterrows():
            d, n, sch, g, c, m, xf = midx


            cc_arr = get_component_cases(case_dict, c)
            cc_fns = tuple([case_dict[cc]["act_fns"][0] for cc in cc_arr])

            try:
                metric_val = float(row[metric]["mean"]) * 100
            except:
                metric_val = float(row[metric]) * 100

            if m and xf:
                multikey = tuple(sorted(cc_fns))
                try:
                    i = p_dict[multikey[0]].index(float(case_dict[sorted(cc_arr)[0]]["act_fn_params"][0]))
                    j = p_dict[multikey[1]].index(float(case_dict[sorted(cc_arr)[1]]["act_fn_params"][0]))
                except KeyError:
                    continue
                cross_mats[multikey][i, j] = metric_val
                vm_dict["xf"]["vmin"] = min(metric_val, vm_dict["xf"]["vmin"], vmin)
                vm_dict["xf"]["vmax"] = max(metric_val, vm_dict["xf"]["vmax"], vmax)
            elif m:
                key = cc_fns[0]
                i = p_dict[key].index(float(case_dict[sorted(cc_arr)[0]]["act_fn_params"][0]))
                j = p_dict[key].index(float(case_dict[sorted(cc_arr)[1]]["act_fn_params"][0]))

                within_mats[key][min(i, j), max(i, j)] = metric_val
                within_mats[key][max(i, j), min(i, j)] = np.nan
                vm_dict[key]["vmin"] = min(metric_val, vm_dict[key]["vmin"], vmin)
                vm_dict[key]["vmax"] = max(metric_val, vm_dict[key]["vmax"], vmax)
            else:
                try:
                    key = cc_fns[0]
                    i = p_dict[key].index(float(case_dict[sorted(cc_arr)[0]]["act_fn_params"][0]))
                    within_mats[key][i, i] = metric_val
                    vm_dict[key]["vmin"] = min(metric_val, vm_dict[key]["vmin"], vmin)
                    vm_dict[key]["vmax"] = max(metric_val, vm_dict[key]["vmax"], vmax)
                except:
                    print(f"Not plotting {key}")
                    continue
        # plots
        if metric == "acc_vs_max":
            metric = "Relative accuracy (%)"

            for key in vm_dict.keys():
                vabs = max(abs(vm_dict[key]["vmin"]), abs(vm_dict[key]["vmax"]))
                vm_dict[key]["vmin"], vm_dict[key]["vmax"] = -vabs, vabs

        else:
            metric = "Peak accuracy (%)"
        for k, v in within_mats.items():
            plt.figure()
            current_cmap = matplotlib.cm.get_cmap()
            current_cmap.set_bad(color="black")
            v_min = vm_dict[k]["vmin"] if v_min is None else v_min
            v_max = vm_dict[k]["vmax"] if v_max is None else v_max
            im = plt.imshow(np.flip(v, axis=1), cmap=cmap, vmin=v_min, vmax=v_max)
            
            ax = plt.gca()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = plt.colorbar(im, cax=cax)
            cbar.ax.set_ylabel(metric, fontsize=large_font_size)

            l = k if k == "swish" else "ptanh"
            ax.set_ylabel(rf"{l} $\beta$", fontsize=large_font_size)
            ax.set_xlabel(rf"{l} $\beta$", fontsize=large_font_size)
            tlabels = p_dict[k]
            tticks = [i for i in range(len(tlabels))]
            ax.set_xticks(tticks)
            ax.set_xticklabels(tlabels, fontsize=small_font_size)
            ax.set_yticks(tticks)
            ax.set_yticklabels(list(reversed(tlabels)), fontsize=small_font_size)
            
            title = f"Within-family {net_name}"
            ax.set_title(title, fontsize=large_font_size)
            plt.tight_layout()

            if self.save_fig:
                filename = f"Within-family {net_name} {metric} {k}"
                self.save("heatmaps", filename)
            else:
                plt.show()
        
        for k, v in cross_mats.items():
            plt.figure()
            v_min = vm_dict["xf"]["vmin"] if v_min is None else v_min
            v_max = vm_dict["xf"]["vmax"] if v_max is None else v_max
            im = plt.imshow(np.flip(v, axis=0), cmap=cmap, vmin=v_min, vmax=v_max)
            ax = plt.gca()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = plt.colorbar(im, cax=cax)
            cbar.ax.set_ylabel(metric, fontsize=large_font_size)

            l = [l if l == "swish" else "ptanh" for l in k]
            ax.set_xlabel(rf"{l[1]} $\beta$", fontsize=large_font_size)
            ax.set_ylabel(rf"{l[0]} $\beta$", fontsize=large_font_size)
            xtlabels = p_dict[k[1]]
            xticks = [i for i in range(len(xtlabels))]
            ytlabels = p_dict[k[0]]
            yticks = [i for i in range(len(ytlabels))]
            ax.set_xticks(xticks)
            ax.set_xticklabels(xtlabels, fontsize=small_font_size)
            ax.set_yticks(yticks)
            ax.set_yticklabels(list(reversed(ytlabels)), fontsize=small_font_size)
            title = f"Cross-family {net_name}"
            ax.set_title(title, fontsize=large_font_size)
            plt.tight_layout()

            if self.save_fig:
                filename = f"Cross-family {net_name} {metric} {k}"
                self.save("heatmaps", filename)
            else:
                plt.show()

    def plot_all_samples_accuracy(self, dataset, net_name, scheme, case, acc_type="val"):

        # pull data
        acc_df = self.stats_processor.load_accuracy_df(dataset, net_name, 
            [scheme], [case])

        # process a bit
        index_cols = ["dataset", "net_name", "train_scheme", "case", "sample"]
        acc_df.set_index(index_cols, inplace=True)


        # plot...
        fig, ax = plt.subplots(figsize=(8,5))
        idxs = acc_df.index.unique()
        clrs = sns.color_palette("hls", len(idxs))

        # plot acc
        for i in range(len(idxs)):

            idx = idxs[i]
            epochs = acc_df.loc[idx]
            
            yvals = epochs[f"{acc_type}_acc"].values * 100
            ax.plot(range(len(yvals)), yvals, c=clrs[i], label=i)
        
        # figure text
        # fig.suptitle(f"Classification accuracy during training: {net_name} on {dataset}", fontsize=large_font_size)
        ax.set_xlabel("Epoch", fontsize=large_font_size)
        ax.set_ylabel(f"{acc_type} accuracy (%)", fontsize=large_font_size)
        ax.set_ylim([10, 100])
        ax.legend(fontsize=small_font_size)
        
        plt.tight_layout()
        
        # optional saving
        if not self.save_fig:
            print("Not saving.")
            plt.show()
            return
        
        sub_dir = ensure_sub_dir(self.data_dir, f"figures/accuracy/")
        filename = f"{dataset}_{net_name}_{scheme}_{case} accuracy"
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
            [scheme], [case])

        # filter more
        acc_df = acc_df.query(f"sample == {sample}")

        # windowed z score at each epoch
        window = 2
        acc_df["z"] = 0.
        for idx in acc_df.index.values:

            # decompose
            row = acc_df.loc[idx]
            d, n, sch, g, c, s, e, val_acc, train_acc, z = row

            # get window
            w_start = max(0, e - window)
            w_idx = acc_df.epoch[w_start:e].index

            # compute window stats
            w_mean = np.mean(acc_df.loc[w_idx, "val_acc"])
            w_std = np.std(acc_df.loc[w_idx, "val_acc"])

            z_score = (val_acc - w_mean) / w_std if w_std != 0 else 0

            # update df
            acc_df.at[idx, "z"] = z_score

        # plot...
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14,8), sharex=True)
        fig.subplots_adjust(hspace=0)
        clrs = sns.color_palette("Set2", 3)

        # plot acc
        yvals = acc_df["val_acc"].values * 100
        axes[0].plot(range(len(yvals)), yvals, c=clrs[0])

        # highlight peak acc
        ipeak = np.argmax(yvals)
        axes[0].plot(ipeak, yvals[ipeak], marker="*", c=clrs[2])

        # plot z score
        zvals = acc_df["z"].values
        axes[1].plot(range(len(zvals)), zvals, c=clrs[1])
        axes[1].axhline(0, color="k", linestyle="--", alpha=0.5)
        
        # highlight consecutive negative z scores
        consec = 5
        for i in range(len(zvals)):

            prev_consec_neg = [True if z < 0 else False for z in zvals[i-consec+1:i+1]]
            if len(prev_consec_neg) == 0:
                continue
            prev_consec_neg = functools.reduce(lambda a,b : a and b, prev_consec_neg)
            if (prev_consec_neg):
                axes[1].plot(i, zvals[i], marker="*", c=clrs[2])
        
        fig.suptitle(f"Classification accuracy during training: {net_name} on {dataset}, sample {sample}", fontsize=20)
        axes[1].set_xlabel("Epoch", fontsize=16)
        axes[0].set_ylabel("Validation accuracy (%)", fontsize=16)
        axes[1].set_ylabel(f"Acc z-score over {window} epochs", fontsize=16)
        axes[0].set_ylim([10, 100])
        plt.tight_layout()
        
        # optional saving
        if not self.save_fig:
            print("Not saving.")
            plt.show()
            return
        
        sub_dir = ensure_sub_dir(self.data_dir, f"figures/accuracy/")
        filename = f"{dataset}_{net_name}_{scheme}_{case}_{sample} accuracy"
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
            schemes, cases)

        # process a bit
        index_cols = ["dataset", "net_name", "train_scheme", "case", "epoch"]
        acc_df.set_index(index_cols, inplace=True)
        df_stats = acc_df.groupby(index_cols).agg({ "val_acc": [np.mean, np.std] })

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
            yvals = group_data["val_acc"]["mean"].values * 100
            yerr = group_data["val_acc"]["std"].values * 1.98 * 100
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
        
        sub_dir = ensure_sub_dir(self.data_dir, f"figures/accuracy/")
        case_names = ", ".join(cases)
        schemes = ", ".join(schemes)
        filename = f"{dataset}_{net_name}_{schemes}_{case_names} accuracy"
        filename = os.path.join(sub_dir, filename)
        print(f"Saving... {filename}")
        plt.savefig(f"{filename}.svg")
        plt.savefig(f"{filename}.png", dpi=300)
        

if __name__=="__main__":
    
    vis = AccuracyVisualizer(
        "/home/briardoty/Source/allen-inst-cell-types/data_mountpoint", 
        save_fig=False,
        save_png=True
        )
    
    # vis.plot_ratio_group("cifar10", "sticknet8", "adam", "ratios-swish2-tanh2")

    # vis.plot_final_acc_decomp("cifar10", "vgg11", "adam", "swish5-tanh0.5")

    # vis.plot_all_samples_accuracy("cifar10", "vgg11", "adam", "testswish10c", acc_type="val")

    vis.plot_single_accuracy("cifar10", "vgg11", "adam", "swish10", sample=0)

    # vis.plot_accuracy("cifar10", 
    #     "vgg11", 
    #     ["adam"], 
    #     ["swish5", "swish5a", "swish5b", "swish5c"],
    #     inset=False)
    # vis.plot_accuracy("cifar10", 
    #     "vgg11", 
    #     ["adam"], 
    #     ["tanh0.5", "tanh0.5a", "tanh0.5b", "tanh0.5c"],
    #     inset=False)
    # vis.plot_accuracy("cifar10", 
    #     "vgg11", 
    #     ["adam"], 
    #     ["swish5-tanh0.5", "swish5-tanh0.5-alla", "swish5-tanh0.5-allb", "swish5-tanh0.5-allc"],
    #     inset=False)




