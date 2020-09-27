def plot_predictions(self, dataset, net_names, schemes, excl_arr, 
        pred_type="max", cross_family=None, pred_std=False, overlap=False):
        """
        Plot a single axis figure of offset from predicted max accuracy for
        the given mixed cases.
        """

        # pull data
        sort_df = self.get_prediction_df(dataset, net_names, schemes, excl_arr, 
            pred_type, cross_family)
        
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
        ydict = dict()
        for midx, row in sort_df.iterrows():

            # stats
            perf = row[f"acc_vs_{pred_type}"].values[0] * 100
            err = row["max_val_acc"]["std"] * 1.98 * 100

            xmin = min(xmin, perf - err)
            xmax = max(xmax, perf + err)

            # dataset, net, scheme, case, mixed, cross-family
            d, n, s, c, m, cf = midx
            clr = clrs[net_names.index(n)]

            # locate row on y axis
            if ydict.get(c) is None:
                ydict[c] = i

                # prettify new rows
                if np.mod(i, 2) == 0:	
                    plt.gca().axhspan(i-.5, i+.5, alpha = 0.1, color="k")

            y = ydict[c] if overlap else i
            
            # plot "good" and "bad"
            lw = 6
            if perf - err > 0:
                if cf or cross_family is not None:
                    plt.plot([perf - err, perf + err], [y,y], linestyle="-", 
                        c=clr, linewidth=lw, alpha=.8)
                else:
                    plt.plot([perf - err, perf + err], [y,y], linestyle=":", 
                        c=clr, linewidth=lw, alpha=.8)
                h = plt.plot(perf, y, c=clr, marker="o")
                handles[n] = h[0]
            else:
                if cf or cross_family is not None:
                    plt.plot([perf - err, perf + err], [y,y], linestyle="-", 
                        c=clr, linewidth=lw, alpha=.2)
                    plt.plot([perf - err, perf + err], [y,y], linestyle="-", 
                        linewidth=lw, c="k", alpha=.1)
                else:
                    plt.plot([perf - err, perf + err], [y,y], linestyle=":", 
                        c=clr, linewidth=lw, alpha=.2)
                    plt.plot([perf - err, perf + err], [y,y], linestyle=":", 
                        linewidth=lw, c="k", alpha=.1)
                h = plt.plot(perf, y, c=clr, marker="o", alpha=0.5)
                if handles.get(n) is None:
                    handles[n] = h[0]

            # optionally, plot the 95% ci for the prediction
            if pred_std:
                pred_err = row[f"{pred_type}_pred"]["std"] * 1.98 * 100
                plt.plot([-pred_err, pred_err], [y,y], linestyle="-", 
                        c="k", linewidth=6, alpha=.2)

            # BH corrected significance
            sig_arr.append(row[f"{pred_type}_pred_rej_h0"].values[0])

            # make an aligned label
            label_arr = [d, n, s, c]
            aligned = "".join([label_arr[i].ljust(lengths[i]) for i in label_idxs])
            ylabels[y] = aligned

            # track vars
            i += 1

        # indicate BH corrected significance
        # h = plt.plot(i+100, 0, "k*", alpha=0.5)
        # handles["p < 0.05"] = h[0]
        # for i in range(len(sig_arr)):
        #     if sig_arr[i]:
        #         plt.plot(xmax + xmax/14., i, "k*", alpha=0.5)

        # determine padding for labels
        max_length = np.max([len(l) for l in ylabels.values()])

        # add handles
        if cross_family is None:
            h1 = plt.gca().axhline(i+100, color="k", linestyle="-", alpha=0.5)
            h2 = plt.gca().axhline(i+100, color="k", linestyle=":", alpha=0.5)
            handles["cross-family"] = h1
            handles["within-family"] = h2

        # set figure text
        plt.xlabel(f"Accuracy relative to {pred_type} prediction (%)", 
            fontsize=20, labelpad=10)
        plt.ylabel("Network configuration", fontsize=20, labelpad=10)
        plt.yticks(list(ylabels.keys()), ylabels.values(), ha="left")
        yax = plt.gca().get_yaxis()
        yax.set_tick_params(pad=max_length*7)
        plt.ylim(-0.5, max(ydict.values()) + 0.5)
        plt.legend(handles.values(), handles.keys(), fontsize=18, loc="upper left")
        plt.xlim([xmin - xmax/10., xmax + xmax/10.])
        
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