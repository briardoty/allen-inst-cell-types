from modules.AccuracyVisualizer import AccuracyVisualizer
import os
import json

data_dir = "/home/briardoty/Source/allen-inst-cell-types/data_mountpoint"

# build group: case dict
group_dict = dict()
with open(os.path.join(data_dir, "net_configs.json"), "r") as json_file:
    net_configs = json.load(json_file)

for g in net_configs.keys():
    cases = net_configs[g]
    case_names = cases.keys()
    group_dict[g] = list(case_names)

# init visualizer
vis = AccuracyVisualizer(
    data_dir, 
    save_fig=True,
    save_png=True
    )

# all nets, all datasets
nets = ["sticknet8", "vgg11"]
datasets = ["cifar10", "fashionmnist", "cifar100"]
for n in nets:
    for d in datasets:
        vis.plot_predictions(d,
            [n],
            ["adam"],
            cases=[],
            excl_arr=["spatial", "test", "ratio"],
            pred_type="max",
            cross_family=None,
            pred_std=True,
            small=False,
            filename=f"{n} {d} prediction"
            )

# plot ratio groups
ratio_groups = ["ratios-swish5-tanh0.5", "ratios-swish2-tanh2", "ratios-relu-tanh1"]
for rg in ratio_groups:
    cases = group_dict[rg] + [rg[len("ratios-"):]]
    vis.plot_predictions("cifar10",
        ["vgg11", "sticknet8"],
        ["adam"],
        cases=cases,
        excl_arr=["spatial", "test"],
        pred_type="max",
        cross_family=None,
        pred_std=True,
        small=False,
        filename=f"{rg} prediction"
        )
