from modules.AccuracyVisualizer import AccuracyVisualizer

# init visualizer
data_dir = "/home/briardoty/Source/allen-inst-cell-types/data_mountpoint"
vis = AccuracyVisualizer(
    data_dir, 
    save_fig=True,
    save_png=True
    )

vis.heatmap_acc("cifar10", 
    "vgg11", 
    "adam", 
    metric="max_val_acc",
    cmap="Reds"
    )

vis.heatmap_acc("cifar10", 
    "vgg11", 
    "adam", 
    metric="acc_vs_max"
    )

vis.heatmap_acc("cifar10", 
    "sticknet8", 
    "adam", 
    metric="max_val_acc",
    cmap="Reds"
    )

vis.heatmap_acc("cifar10", 
    "sticknet8", 
    "adam", 
    metric="acc_vs_max"
    )


