import os
import sys
import functools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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


class PredictionVisualizer():
    
    def __init__(self, data_dir, save_fig=False, save_png=False, sub_dir_name=None):
        
        self.data_dir = data_dir
        self.save_fig = save_fig
        self.save_png = save_png
        self.sub_dir_name = sub_dir_name
        
        self.accuracy_loader = AccuracyLoader(data_dir)

        with open("/home/briardoty/Source/allen-inst-cell-types/hpc-jobs/net_configs.json", "r") as json_file:
            self.net_configs = json.load(json_file)

    def get_prediction_df(self, datasets, net_names, schemes, cases, excl_arr, 
        pred_type="max", cross_family=None, mixed=True, metric="val_acc"):

        # pull data
        df, case_dict, _ = self.accuracy_loader.load_max_acc_df()

        # performance relative to prediction
        df[f"{metric}_vs_{pred_type}"] = df[f"{metric}"]["mean"] - df[f"{pred_type}_pred_{metric}"]["mean"]

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
        df = df.query(f"not (net_name == 'vgg11' and (case.str.contains('tanh2') or (case.str.startswith('tanh') and case.str.endswith('-2'))))",
            engine="python")

        sort_df = df.sort_values(["net_name", f"{metric}_vs_{pred_type}"])

        return sort_df, case_dict

    def plot_final_acc_decomp(self, dataset, net_name, scheme, mixed_case):
        """
        Plot accuracy at the end of training for mixed case, 
        including predicted mixed case accuracy based
        on combination of component cases
        """

        # pull data
        df, case_dict, idx_cols = self.accuracy_loader.load_max_acc_df_ungrouped()
        component_cases = get_component_cases(case_dict, mixed_case)

        # filter dataframe
        df = df.query(f"dataset == '{dataset}'") \
            .query(f"net_name == '{net_name}'") \
            .query(f"train_scheme == '{scheme}'") \
            .query(f"epoch == -1")

        # plot...
        markersize = 18
        c_labels = dict()
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5,4), sharey=True)
        fig.subplots_adjust(wspace=0)
        c_clrs = sns.color_palette("husl", len(component_cases))
        c_clrs.reverse()
        
        # plot component nets
        width = 0.35
        lw = 4
        for i in range(len(component_cases)):

            cc = component_cases[i]
            cc_rows = df.query(f"case == '{cc}'")
            yvals = cc_rows["val_acc"].values * 100

            axes[0].plot([i] * len(yvals), yvals, ".", label=cc,
                c=c_clrs[i], markersize=markersize, alpha=0.6)
            axes[0].plot([i-width, i+width], [np.mean(yvals), np.mean(yvals)], 
                linestyle=":", label=cc, c=c_clrs[i], linewidth=lw)
            
            cc = cc.replace("tanh", "ptanh")
            c_labels[i] = cc
            
        # plot mixed case
        # actual
        mwidth = width + 0.09
        m_clrs = sns.color_palette("husl", len(component_cases) + 3)
        m_rows = df.query(f"case == '{mixed_case}'")
        yact = m_rows["val_acc"].values * 100
        axes[1].plot([0] * len(yact), yact, ".", label=cc,
            c=m_clrs[len(component_cases)], markersize=markersize, alpha=0.6)
        axes[1].plot([-mwidth, mwidth], [np.mean(yact), np.mean(yact)], 
                linestyle=":", c=m_clrs[len(component_cases)], linewidth=lw)

        # predicted
        pred_types = ["max", "linear"]
        handles = dict()
        for pred_type, clr in zip(pred_types, m_clrs[-2:]):
            ypred = m_rows[f"{pred_type}_pred_val_acc"].mean() * 100
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

        
    def plot_pred_vs_lr(self, dataset, net_name, schemes, excl_arr=[], pred_type="max",
        metric="val_acc", filename=None):
        """
        Scatter plot comparing relative accuracy with initial learning rate
        """

        # pull data
        sort_df, _ = self.get_prediction_df([dataset], [net_name], schemes, [], 
            excl_arr, pred_type, None, metric)

        # plot
        fig, ax = plt.subplots(figsize=(6,6))

        # identifiers
        fmts = [".", "^"]
        clrs = sns.color_palette("husl", len(schemes))
        ms = 10

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
            rel_acc = row[f"{metric}_vs_{pred_type}"].values[0] * 100
            rel_acc_err = row[f"{metric}"]["std"] * 1.98 * 100

            initial_lr = row["initial_lr"]["mean"]
            initial_lr_err = row["initial_lr"]["std"]

            xmin = min(xmin, rel_acc - rel_acc_err)
            xmax = max(xmax, rel_acc + rel_acc_err)

            # dataset, net, scheme, case, mixed, cross-family
            d, n, sch, g, c, m, cf = midx
            clr = clrs[schemes.index(sch)]
            fmt = fmts[0]

            # plot
            h = ax.plot(initial_lr, rel_acc, c=clr, marker=fmt, alpha=0.8)
            handles[sch] = h[0]

        # set figure text
        ax.set_ylabel(f"{metric} relative to {pred_type} prediction (%)", fontsize=20)
        ax.set_xlabel("Initial LR", fontsize=20)
        
        # ax.set_xlim([xmin - 1, xmax + 1])
        # ax.set_ylim([ymin - 1, ymax + 1])
        # ax.set_aspect("equal", "box")
        ax.legend(handles.values(), handles.keys(), fontsize=18)

        plt.tight_layout()
         
        # optional saving
        if not self.save_fig:
            plt.show()
            return

        filename = f"{dataset}_{net_name}_{pred_type}-lr-scatter"
        self.save("lr-scatter", filename)


    def plot_epoch_prediction_scatter(self, dataset, net_name, scheme, epochs, 
        cases=[], excl_arr=[], pred_type="max", metric="val_acc"):
        """
        Make a scatterplot of training epoch vs. accuracy relative to prediction.
        """

        # pull data
        df, _ = self.get_prediction_df([dataset], [net_name], [scheme], cases, 
            excl_arr=excl_arr, pred_type=pred_type, cross_family=None, 
            mixed=True, metric=metric)

        # create figure
        _, ax = plt.subplots(figsize=(6,6))

        # identifiers
        clrs = sns.color_palette("Set2", 2)
        handles = dict()

        xfam_arr = [True, False]
        for xfam in xfam_arr:

            net_df = df \
                .query(f"net_name == '{net_name}'") \
                .query(f"cross_fam == {xfam}")
            bh_sig = [net_df.query(f"epoch == {epoch}")[f"{pred_type}_pred_{metric}_rej_h0"].values for epoch in epochs]
            data = [net_df.query(f"epoch == {epoch}")[f"{metric}_vs_{pred_type}"].values * 100 for epoch in epochs]

            clr = clrs[xfam_arr.index(xfam)]
            handles["Cross-family" if xfam else "Within-family"] = mpatches.Patch(color=clr)
            parts = ax.violinplot(data,
                showmeans=False,
                showmedians=True)
            for pc in parts['bodies']:
                pc.set_color(clr)
                pc.set_facecolor(clr)
                pc.set_edgecolor(clr)
                pc.set_alpha(0.4)
            
            for pc in [parts["cbars"], parts["cmaxes"], parts["cmins"], parts["cmedians"]]:
                pc.set_color(clr)
                pc.set_facecolor(clr)
                pc.set_edgecolor(clr)
                pc.set_alpha(0.8)


        # for epoch in epochs:
            
        #     sort_df = df.query(f"epoch == {epoch}")

        #     # plot
        #     lw = 600 / len(sort_df)
        #     ms = lw * 0.5
        #     for midx, row in sort_df.iterrows():

        #         # stats
        #         perf = row[f"{metric}_vs_{pred_type}"].values[0] * 100
        #         err = row[f"{metric}"]["std"] * 1.98 * 100

        #         xmin = min(xmin, perf)
        #         xmax = max(xmax, perf)

        #         # dataset, net, scheme, case, mixed, cross-family
        #         d, n, sch, g, c, e, m, cf = midx
        #         if (e != epoch):
        #             continue
        #         clr = clrs[net_names.index(n)]
        #         fmt = fmts[0]

        #         # BH corrected significance
        #         bh_sig = row[f"{pred_type}_pred_{metric}_rej_h0"].values[0]
        #         sig_arr.append(bh_sig)

        #         # 
        #         ax.plot(epoch, perf, c=clr, marker=fmt, alpha=0.8)

        # set figure text
        plt.gca().axhline(0, color="k", linestyle="-", alpha=0.5)
        metric_label = "Val acc" if metric=="val_acc" else "Test acc"
        plt.xlabel("Epoch", fontsize=18, labelpad=10)
        plt.ylabel(f"{metric_label} relative to {pred_type} (%)", fontsize=18, labelpad=10)
        ax.set_xticks([e+1 for e in range(len(epochs))])
        ax.set_xticklabels(epochs)
        plt.legend(handles.values(), handles.keys(), fontsize=18, loc="upper left")
        # plt.xlim([min(epochs) - 5, max(epochs) + 5])
        # plt.ylim([xmin - xmax/10., xmax + xmax/10.])
        
        plt.tight_layout()

        # optional saving
        if not self.save_fig:
            print("Not saving.")
            plt.show()
            return

        filename = f"{dataset}_{net_name}_{pred_type}_{metric}"
        self.save("epoch pred scatter", filename)


    def plot_predictions(self, dataset, net_names, schemes, cases=[], excl_arr=[], 
        pred_type="max", metric="val_acc", cross_family=None, pred_std=False, small=False, filename=None):
        """
        Plot a single axis figure of offset from predicted max accuracy for
        the given mixed cases.
        """

        # pull data
        sort_df, _ = self.get_prediction_df([dataset], net_names, schemes, cases, 
            excl_arr=excl_arr, pred_type=pred_type, cross_family=cross_family, 
            mixed=True, metric=metric)
        
        # final accuracy
        sort_df = sort_df.query("epoch == -1")

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
            perf = row[f"{metric}_vs_{pred_type}"].values[0] * 100
            err = row[f"{metric}"]["std"] * 1.98 * 100

            xmin = min(xmin, perf - err)
            xmax = max(xmax, perf + err)

            # dataset, net, scheme, case, mixed, cross-family
            d, n, s, g, c, e, m, cf = midx
            clr = clrs[net_names.index(n)]
            
            # prettify	
            if np.mod(i, 2) == 0:	
                plt.gca().axhspan(i-.5, i+.5, alpha = 0.1, color="k")

            # BH corrected significance
            bh_sig = row[f"{pred_type}_pred_{metric}_rej_h0"].values[0]
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
                pred_err = row[f"{pred_type}_pred_{metric}"]["std"] * 1.98 * 100
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
        metric = "Val acc" if metric=="val_acc" else "Test acc"
        plt.xlabel(f"{metric} relative to {pred_type} (%)", 
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
            filename = f"{dataset}_{net_names}_{schemes}_{pred_type}_{metric}"
        self.save("prediction", filename)


    def plot_predictions_over_epochs(self, dataset, net_names, schemes, cases=[], excl_arr=[], 
        pred_type="max", metric="val_acc", cross_family=None, pred_std=False, small=False):
        """
        Plot a single axis figure of offset from predicted max accuracy for
        the given mixed cases.
        """

        # pull data
        df, _ = self.get_prediction_df([dataset], net_names, schemes, cases, 
            excl_arr=excl_arr, pred_type=pred_type, cross_family=cross_family, 
            mixed=True, metric=metric)
        
        net_names = ", ".join(net_names)
        schemes = ", ".join(schemes)

        # determine each label length for alignment
        lengths = {}
        label_idxs = [3]
        for i in label_idxs:
            lengths[i] = np.max([len(x) for x in df.index.unique(level=i)]) + 2

        epochs = list(df.index.unique(level=5).values)
        epochs.sort()
        epochs.append(epochs.pop(0))

        for epoch in epochs:
            
            sort_df = df.query(f"epoch == {epoch}")

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
                perf = row[f"{metric}_vs_{pred_type}"].values[0] * 100
                err = row[f"{metric}"]["std"] * 1.98 * 100

                xmin = min(xmin, perf - err)
                xmax = max(xmax, perf + err)

                # dataset, net, scheme, case, mixed, cross-family
                d, n, s, g, c, e, m, cf = midx
                if (e != epoch):
                    continue
                clr = clrs[net_names.index(n)]
                
                # prettify	
                if np.mod(i, 2) == 0:	
                    plt.gca().axhspan(i-.5, i+.5, alpha = 0.1, color="k")

                # BH corrected significance
                bh_sig = row[f"{pred_type}_pred_{metric}_rej_h0"].values[0]
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
                    pred_err = row[f"{pred_type}_pred_{metric}"]["std"] * 1.98 * 100
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
            metric_label = "Val acc" if metric=="val_acc" else "Test acc"
            plt.xlabel(f"{metric_label} relative to {pred_type} (%)", 
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

            filename = f"{dataset}_{net_names}_{schemes}_{pred_type}_{metric}_{epoch}"
            self.save("prediction over epochs", filename)


    def save(self, sub_dir_name, filename):

        if self.sub_dir_name is not None:
            sub_dir_name = self.sub_dir_name
        sub_dir = ensure_sub_dir(self.data_dir, f"figures/{sub_dir_name}")

        filename = os.path.join(sub_dir, filename)

        print(f"Saving... {filename}")

        plt.savefig(f"{filename}.svg")  

        if self.save_png:
            plt.savefig(f"{filename}.png", dpi=300)


if __name__=="__main__":
    
    data_dir = "/home/briardoty/Source/allen-inst-cell-types/data_mountpoint"
    scheme = "adam_lravg_nosplit"
    metric = "val_acc"

    # build group: case dict
    group_dict = dict()
    config_path = "/home/briardoty/Source/allen-inst-cell-types/hpc-jobs/net_configs.json"
    with open(config_path, "r") as json_file:
        net_configs = json.load(json_file)

    for g in net_configs.keys():
        cases = net_configs[g]
        case_names = cases.keys()
        group_dict[g] = list(case_names)

    # init vis
    vis = PredictionVisualizer(
        data_dir, 
        save_fig=True,
        save_png=True
        )

    # plot
    vis.plot_final_acc_decomp("cifar10", "vgg11", scheme, "swish7.5-tanh1")
    cases = group_dict["cross-swish-tanh"]
    # vis.plot_predictions_over_epochs("cifar10",
    #     ["vgg11", "sticknet8"],
    #     [scheme],
    #     cases=cases,
    #     excl_arr=["spatial", "test", "ratio", "tanh0.01", "swish0.1"],
    #     pred_type="max",
    #     metric=metric,
    #     cross_family=True,
    #     pred_std=False,
    #     small=False
    # )

    # vis.plot_epoch_prediction_scatter("cifar10", 
    #     "sticknet8", 
    #     scheme, 
    #     [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    #     cases=[], 
    #     excl_arr=["spatial", "test", "ratio", "tanh0.01", "swish0.1"], 
    #     pred_type="max", 
    #     metric=metric
    # )

    vis.plot_predictions("cifar10",
        ["vgg11", "sticknet8"],
        [scheme],
        cases=[],
        excl_arr=["spatial", "test", "ratio", "tanh0.01", "swish0.1"],
        pred_type="max",
        metric=metric,
        cross_family=True,
        pred_std=False,
        small=False,
    )
