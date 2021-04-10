from modules.AccuracyVisualizer import AccuracyVisualizer
import os
import json

data_dir = "/home/briardoty/Source/allen-inst-cell-types/data_mountpoint"
scheme = "adam-lr-avg"
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

# init visualizer
vis = AccuracyVisualizer(
    data_dir, 
    save_fig=True,
    save_png=False,
    sub_dir_name="manuscript fig2"
)

# summary
# cases = group_dict["cross-swish-tanh"]
# vis.plot_predictions("cifar10",
#     ["vgg11", "sticknet8"],
#     [scheme],
#     cases=cases,
#     excl_arr=["spatial", "test", "ratio", "tanh0.01", "swish0.1"],
#     pred_type="max",
#     metric=metric,
#     cross_family=True,
#     pred_std=False,
#     small=True,
#     filename=f"manuscript fig2 prediction"
# )

# histograms
pred_types = ["max", "linear"]
for pred in pred_types:
    vis.plot_family_supplement("cifar10",
        ["vgg11", "sticknet8"],
        [scheme],
        excl_arr=["spatial", "test", "ratio"],
        pred_type=pred,
        cross_family=None,
        ymax=3.,
        ymin=-1.
    )

# heatmaps
vis.heatmap_acc("cifar10", 
    "vgg11", 
    scheme, 
    metric=metric,
    cmap="Reds",
    v_min=85,
    v_max=89
)
vis.heatmap_acc("cifar10", 
    "sticknet8", 
    scheme, 
    metric=metric,
    cmap="Reds",
    v_min=58,
    v_max=64
)
