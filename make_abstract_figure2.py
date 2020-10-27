from modules.AccuracyVisualizer import AccuracyVisualizer
import os
import json

data_dir = "/home/briardoty/Source/allen-inst-cell-types/data_mountpoint"

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
    sub_dir_name="abstract figure 2"
)

# summary
# cases = group_dict["cross-swish-tanh"]
# vis.plot_predictions("cifar10",
#     ["vgg11", "sticknet8"],
#     ["adam"],
#     cases=cases,
#     excl_arr=["spatial", "test", "ratio", "tanh0.01", "swish0.1"],
#     pred_type="max",
#     cross_family=True,
#     pred_std=False,
#     small=True,
#     filename=f"abstract fig2 prediction"
# )

# histograms
vis.plot_family_supplement("cifar10",
    ["vgg11"],
    ["adam"],
    excl_arr=["spatial", "test", "ratio"],
    pred_type="max",
    cross_family=None
)

vis.plot_network_supplement("cifar10",
    ["vgg11", "sticknet8"],
    ["adam"],
    excl_arr=["spatial", "test", "ratio"],
    pred_type="max",
    cross_family=None
)

# heatmaps
vis.heatmap_acc("cifar10", 
    "vgg11", 
    "adam", 
    metric="acc_vs_max",
    v_min=-0.5,
    v_max=0.5
)
vis.heatmap_acc("cifar10", 
    "sticknet8", 
    "adam", 
    metric="acc_vs_max",
    v_min=-2,
    v_max=2
)
