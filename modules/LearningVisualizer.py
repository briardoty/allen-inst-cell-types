import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

try:
    from .AccStatProcessor import AccStatProcessor, get_component_cases
except:
    from AccStatProcessor import AccStatProcessor, get_component_cases

try:
    from .util import ensure_sub_dir
except:
    from util import ensure_sub_dir

import matplotlib
large_font_size = 16
small_font_size = 14
matplotlib.rc("xtick", labelsize=small_font_size) 
matplotlib.rc("ytick", labelsize=small_font_size) 


class LearningVisualizer():
    
    def __init__(self, data_dir, n_classes=10, save_fig=False, refresh=False):
        
        self.data_dir = data_dir
        self.save_fig = save_fig
        self.refresh = refresh
        
        self.stats_processor = AccStatProcessor(data_dir, n_classes)

    def plot_learning_speed(self, dataset, net_name, scheme, mixed_case, pct=90):
        """
        For the given mixed network and its component nets, plot the number
        of epochs it takes each net to get to pct% of its peak accuracy
        """

        # pull data
        df, case_dict = self.stats_processor.load_learning_df(self.refresh)

        # plot


    def plot_final_acc_decomp(self, dataset, net_name, scheme, mixed_case):
        """
        Plot accuracy at the end of training for mixed case, 
        including predicted mixed case accuracy based
        on combination of component cases
        """

        # pull data
        df, case_dict, idx_cols = self.stats_processor.load_max_acc_df_ungrouped(self.refresh)
        component_cases = get_component_cases(case_dict, mixed_case)

        # filter dataframe
        df = df.query(f"dataset == '{dataset}'") \
            .query(f"net_name == '{net_name}'") \
            .query(f"train_scheme == '{scheme}'")

        # plot...
        markersize = 18
        c_labels = dict()
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5,5), sharey=True)
        fig.subplots_adjust(wspace=0)
        c_clrs = sns.color_palette("hls", len(component_cases))
        c_clrs.reverse()
        
        # plot component nets
        x = [-2]
        width = 0.35
        lw = 4
        for i in range(len(component_cases)):

            cc = component_cases[i]
            cc_rows = df.loc[(dataset, net_name, scheme, cc)]
            yvals = cc_rows["max_val_acc"].values * 100
            start = x[-1] + 2
            x = [i for i in range(start, start + len(yvals))]

            axes[0].plot([i] * len(yvals), yvals, ".", label=cc,
                c=c_clrs[i], markersize=markersize, alpha=0.6)
            axes[0].plot([i-width, i+width], [np.mean(yvals), np.mean(yvals)], 
                linestyle=":", label=cc, c=c_clrs[i], linewidth=lw)
            
            c_labels[i] = cc
            
        # plot mixed case
        # actual
        m_clrs = sns.color_palette("hls", len(component_cases) + 3)
        m_rows = df.loc[(dataset, net_name, scheme, mixed_case)]
        yact = m_rows["max_val_acc"].values * 100
        axes[1].plot([0] * len(yact), yact, ".", label=cc,
            c=m_clrs[len(component_cases)], markersize=markersize, alpha=0.6)

        # predicted
        pred_types = ["max", "linear"]
        handles = dict()
        for pred_type, clr in zip(pred_types, m_clrs[-2:]):
            ypred = df.loc[(dataset, net_name, scheme, mixed_case)][f"{pred_type}_pred"].mean() * 100
            h = axes[1].plot([-width, width], [ypred, ypred], 
                label=cc, c=clr, linewidth=lw)

            # legend stuff
            handles[pred_type] = h[0]

        # legend stuff
        handles["actual"] = axes[0].plot(-100, ypred, "k.", 
            markersize=markersize, alpha=0.5)[0]
        handles["mean"] = axes[0].plot(-100, ypred, "k:", 
            linewidth=lw, alpha=0.5)[0]

        # set figure text
        # fig.suptitle(f"Component and mixed network performance comparison\n {net_name} on {dataset}",
        #     fontsize=20)
        matplotlib.rc("xtick", labelsize=small_font_size)
        matplotlib.rc("ytick", labelsize=small_font_size)
        axes[0].set_xlabel("Component", fontsize=16, 
            labelpad=10)
        axes[1].set_xlabel("Mixed", fontsize=large_font_size, 
            labelpad=10)
        axes[0].set_ylabel("Final validation accuracy (%)", 
            fontsize=large_font_size, labelpad=10)
        axes[0].set_xlim([-0.5, len(component_cases) - 0.5])
        axes[1].set_xlim([-0.5, 0.5])
        axes[0].set_xticks(list(c_labels.keys()))
        axes[0].set_yticklabels(list(axes[0].get_yticks()), fontsize=small_font_size)
        axes[0].set_xticklabels(list(c_labels.values()), fontsize=small_font_size)
        axes[1].set_xticks([0])
        axes[1].set_xticklabels([mixed_case], fontsize=small_font_size)
        plt.tight_layout()
        # plt.tight_layout(rect=[0, 0, 1, 0.92])

        # shrink axes...
        box1 = axes[0].get_position()
        axes[0].set_position([box1.x0, box1.y0, box1.width * 0.8, box1.height])
        box2 = axes[1].get_position()
        axes[1].set_position([box1.x0 + box1.width, box2.y0, box2.width * 0.4, box2.height])

        # append legend to second axis
        axes[1].legend(handles.values(), handles.keys(), 
            fontsize=small_font_size, loc="center left", bbox_to_anchor=(1, 0.5))
         
        # optional saving
        if not self.save_fig:
            print("Not saving.")
            plt.show()
            return

        sub_dir = ensure_sub_dir(self.data_dir, f"figures/final acc comparison/")
        filename = f"{mixed_case} comparison"
        filename = os.path.join(sub_dir, filename)
        print(f"Saving... {filename}")
        plt.savefig(f"{filename}.svg")
        plt.savefig(f"{filename}.png", dpi=300)

    def plot_predictions(self, dataset, net_names, schemes, excl_arr, 
        pred_type="max", cross_family=None, pred_std=False):
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
        label_idxs = [3]
        for i in label_idxs:
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
                        linewidth=6, c="k", alpha=.1)
                else:
                    plt.plot([perf - err, perf + err], [i,i], linestyle=":", 
                        c=clr, linewidth=6, alpha=.2)
                    plt.plot([perf - err, perf + err], [i,i], linestyle=":", 
                        linewidth=6, c="k", alpha=.1)
                h = plt.plot(perf, i, c=clr, marker="o", alpha=0.5)
                if handles.get(n) is None:
                    handles[n] = h[0]

            # optionally, plot the 95% ci for the prediction
            if pred_std:
                pred_err = sort_df.loc[midx][f"{pred_type}_pred"]["std"] * 1.98 * 100
                plt.plot([-pred_err, pred_err], [i,i], linestyle="-", 
                        c="k", linewidth=6, alpha=.2)

            # BH corrected significance
            sig_arr.append(sort_df.loc[midx, f"{pred_type}_pred_rej_h0"].values[0])

            # make an aligned label
            label_arr = [d, n, s, c]
            aligned = "".join([label_arr[i].ljust(lengths[i]) for i in label_idxs])
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
        # plt.title("Mixed network performance relative to predicted performance", 
        #     fontsize=20, pad=20)
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

        sub_dir = ensure_sub_dir(self.data_dir, f"figures/prediction")
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
        

if __name__=="__main__":
    
    visualizer = LearningVisualizer("/home/briardoty/Source/allen-inst-cell-types/data_mountpoint", 
        10, save_fig=True, refresh=False)



