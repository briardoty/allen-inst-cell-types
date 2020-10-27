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

# supplementary plots
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

vis.plot_dataset_supplement(["cifar10", "cifar100", "fashionmnist"],
    ["vgg11", "sticknet8"],
    ["adam"],
    excl_arr=["spatial", "test", "ratio"],
    pred_type="max",
    cross_family=None
)