import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

try:
    from .AccuracyLoader import AccuracyLoader
except:
    from AccuracyLoader import AccuracyLoader

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
    
    def __init__(self, data_dir, n_classes=10, save_fig=False):
        
        self.data_dir = data_dir
        self.save_fig = save_fig
        
        self.stats_processor = AccuracyLoader(data_dir, n_classes)

    def plot_learning_speed(self, dataset, net_names, schemes, mixed_case, pct=90):
        """
        For the given mixed network and its component nets, plot the number
        of epochs it takes each net to get to pct% of its peak accuracy
        """

        # plot
        x = 1

    def plot_predictions(self, dataset, net_names, schemes, excl_arr,
        pred_type="min", cross_family=None, pred_std=False):
        """
        Plot a single axis figure of offset from predicted number of epochs
        it takes a net to get to 90% of its peak accuracy.
        """

        # pull data
        df, case_dict, idx_cols = self.stats_processor.load_max_acc_df()

        # performance relative to predictions
        vs = "epochs_past_vs"
        df[f"{vs}_linear"] = df["epochs_past"]["mean"] - df["linear_pred_epochs_past"]["mean"]
        df[f"{vs}_min"] = df["epochs_past"]["mean"] - df["min_pred_epochs_past"]["mean"]

        # filter dataframe
        df = df.query(f"is_mixed") \
            .query(f"dataset == '{dataset}'") \
            .query(f"net_name in {net_names}") \
            .query(f"train_scheme in {schemes}")
        if cross_family is not None:
            df = df.query(f"cross_fam == {cross_family}")
        for excl in excl_arr:
            df = df.query(f"not case.str.contains('{excl}')", engine="python")
        sort_df = df.sort_values(["net_name", f"{vs}_{pred_type}"], ascending=False)

        # determine each label length for alignment
        lengths = {}
        label_idxs = [3]
        for i in label_idxs:
            lengths[i] = np.max([len(x) for x in sort_df.index.unique(level=i)]) + 2

        # plot
        plt.figure(figsize=(16,16))
        plt.gca().axvline(0, color='k', linestyle='--')
        clrs = sns.color_palette("Set2", len(net_names))

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
            perf = sort_df.loc[midx][f"{vs}_{pred_type}"].values[0]
            err = sort_df.loc[midx]["epochs_past"]["std"] * 1.98

            xmin = min(xmin, perf - err)
            xmax = max(xmax, perf + err)

            # plot "good" and "bad"
            if perf + err < 0:
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
                pred_err = sort_df.loc[midx][f"{pred_type}_pred_epochs_past"]["std"] * 1.98
                plt.plot([-pred_err, pred_err], [i,i], linestyle="-", 
                        c="k", linewidth=6, alpha=.2)

            # BH corrected significance
            sig_arr.append(sort_df.loc[midx, f"{pred_type}_pred_epochs_past_rej_h0"].values[0])

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
        plt.xlabel(f"N epochs to reach {self.stats_processor.pct}% peak accuracy relative to {pred_type} prediction", 
            fontsize=16, labelpad=10)
        plt.ylabel("Network configuration", fontsize=16, labelpad=10)
        plt.yticks(list(ylabels.keys()), ylabels.values(), ha="left")
        plt.ylim(-0.5, i + 0.5)
        plt.legend(handles.values(), handles.keys(), fontsize=14, loc="lower left")
        plt.xlim([xmin - xmax/10., xmax + xmax/10.])
        yax = plt.gca().get_yaxis()
        yax.set_tick_params(pad=max_length*7)
        plt.tight_layout()

        # optional saving
        if not self.save_fig:
            print("Not saving.")
            plt.show()
            return

        sub_dir = ensure_sub_dir(self.data_dir, f"figures/learning prediction")
        net_names = ", ".join(net_names)
        schemes = ", ".join(schemes)
        filename = f"{dataset}_{net_names}_{schemes}_{pred_type}-learning-prediction"
        if cross_family == True:
            filename += "_xfam"
        elif cross_family == False:
            filename += "_infam"
        filename = os.path.join(sub_dir, filename)
        print(f"Saving... {filename}")
        plt.savefig(f"{filename}.svg")  
        plt.savefig(f"{filename}.png", dpi=300) 
        

if __name__=="__main__":
    
    vis = LearningVisualizer("/home/briardoty/Source/allen-inst-cell-types/data_mountpoint", 
        10, save_fig=True)

    # vis.plot_learning_speed("cifar10", "sticknet8", "adam", "swish7.5-tanh0.5", pct=90)

    vis.plot_predictions("cifar10",
        ["sticknet8", "vgg11"],
        ["adam"],
        excl_arr=["spatial", "tanhe5", "tanhe0.1-5", "test"],
        pred_type="linear",
        cross_family=True,
        pred_std=True)

