from modules.AccuracyVisualizer import AccuracyVisualizer
import os
import json

data_dir = "/home/briardoty/Source/allen-inst-cell-types/data_mountpoint"

# init visualizer
vis = AccuracyVisualizer(
    data_dir, 
    save_fig=True,
    save_png=False
)

dataset = "cifar10"
net = "sticknet8"
scheme = "adam"
case = "swish5-tanh0.5"
metric = "alex"

for s in range(10):
    vis.plot_single_accuracy(dataset, net, scheme, case, 
    metric=metric,
    sample=s)
