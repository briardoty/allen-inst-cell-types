from modules.AccuracyVisualizer import AccuracyVisualizer

# init visualizer
data_dir = "/home/briardoty/Source/allen-inst-cell-types/data_mountpoint"
vis = AccuracyVisualizer(
    data_dir, 
    save_fig=True,
    save_png=True
    )

# all nets
nets = ["sticknet8", "vgg11"]
for n in nets:
    # vis.ratio_heatmap("cifar10", n, "adam", metric="max_val_acc", cmap="Reds")
    vis.ratio_heatmap("cifar10", n, "adam", metric="acc_vs_max", cmap="RdBu_r")

