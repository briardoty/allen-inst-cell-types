from modules.AccuracyVisualizer import AccuracyVisualizer

vis = AccuracyVisualizer(
    "/home/briardoty/Source/allen-inst-cell-types/data_mountpoint", 
    save_fig=True
    )

# plot a ratio group
vis.plot_predictions("cifar10",
    ["vgg11"],
    ["adam"],
    groups=["ratios-swish5-tanh0.5"],
    excl_arr=["spatial", "test"],
    pred_type="max",
    cross_family=None,
    pred_std=True,
    small=False
    )