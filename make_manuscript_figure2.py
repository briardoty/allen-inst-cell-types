from modules.PredictionVisualizer import PredictionVisualizer
from modules.AccuracyVisualizer import AccuracyVisualizer
import os
import json

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

# init visualizer
# vis = PredictionVisualizer(
#     data_dir, 
#     save_fig=True,
#     save_png=False,
#     sub_dir_name="manuscript fig2"
# )

# # summary
# # cases = group_dict["cross-swish-tanh"]
# cases = []
# vis.plot_predictions("cifar10",
#     ["vgg11", "sticknet8"],
#     [scheme],
#     cases=cases,
#     excl_arr=["spatial", "test", "ratio"],
#     pred_type="linear",
#     metric=metric,
#     cross_family=None,
#     pred_std=False,
#     small=True,
#     filename=f"manuscript fig2 prediction - all"
# )

# # decomp
# vis.plot_final_acc_decomp("cifar10", "vgg11", scheme, "swish7.5-tanh1")

vis = AccuracyVisualizer(
    data_dir, 
    save_fig=True,
    save_png=True,
    sub_dir_name="manuscript fig2"
)

# histograms
pred_types = ["max", "linear"]
for pred in pred_types:
    vis.plot_family_supplement("cifar10",
        ["vgg11", "sticknet8"],
        [scheme],
        excl_arr=["spatial", "test", "ratio"],
        pred_type=pred,
        cross_family=None,
        ymax=5.,
        ymin=-2.
    )

# heatmaps
# vis.heatmap_acc("cifar10", 
#     "vgg11", 
#     scheme, 
#     metric=metric,
#     cmap="Reds",
#     v_min=85.5,
#     v_max=89.5
# )
# vis.heatmap_acc("cifar10", 
#     "sticknet8", 
#     scheme, 
#     metric=metric,
#     cmap="Reds",
#     v_min=61,
#     v_max=69
# )
