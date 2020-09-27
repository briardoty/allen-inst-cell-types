from modules.AccuracyVisualizer import AccuracyVisualizer

vis = AccuracyVisualizer("/home/briardoty/Source/allen-inst-cell-types/data_mountpoint", 
    10, save_fig=True, refresh=False)

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


