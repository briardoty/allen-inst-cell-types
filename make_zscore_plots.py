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
net = "vgg11"
scheme = "adam"
case = "swish10"

for s in range(10):
    vis.plot_single_accuracy(dataset, net, scheme, case, 
    metric="deriv",
    sample=s)
