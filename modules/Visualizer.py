#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 14:22:11 2020

@author: briardoty
"""
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from .NetManager import get_net_tag, NetManager



class Visualizer():
    
    def __init__(self, data_dir, net_name, n_classes=10, save_fig=False):
        
        self.data_dir = data_dir
        self.save_fig = save_fig
        
        self.net_manager = NetManager(net_name, n_classes, data_dir)
        
    def plot_accuracy(self, case_ids):
        """
        Plots accuracy over training for different experimental cases.

        Args:
            cases (list): Experimental cases to include in figure.

        Returns:
            None.

        """
        # pull data
        acc_df = self.net_manager.load_accuracy_df(case_ids)

        # group and compute stats        
        acc_df.set_index(["case", "epoch"], inplace=True)
        acc_df_groups = acc_df.groupby(["case", "epoch"])
        acc_df_stats = acc_df_groups.agg({ "acc": [np.mean, np.std] })
        acc_df_stats_groups = acc_df_stats.groupby("case")
        
        # plot
        fig, ax = plt.subplots()
        
        for name, group in acc_df_stats_groups:
            ax.plot(group["acc"]["mean"].values, label=name)
            
        ax.set_title("Classification accuracy during training")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation accuracy")
        ax.legend()
        
        # optional saving
        if not self.save_fig:
            print("Not saving.")
            plt.show()
            return
        
        sub_dir = self.sub_dir(f"figures/{self.net_manager.net_name}/")
        cases = " & ".join(case_ids)
        filename = f"{cases} accuracy.png"
        filename = os.path.join(sub_dir, filename)
        print(f"Saving... {filename}")
        plt.savefig(filename, dpi=100)  
                    
        
    def plot_filters(self, i_layer, case_id, sample, epoch):
        """
        Plots visualizations of filters

        Args:
            layer_name (str): Name of layer.
            epochs (list): Epochs to plot filters over.

        Returns:
            None.

        """
        # TODO: loop over multiple epochs?
        
        # load net
        net = self.net_manager.load_net_snapshot(case_id, sample, epoch)
        
        # extract kernels
        layer = net.features[i_layer]
        kernels = layer.weight.data
        if not isinstance(layer, nn.Conv2d):
            print()
            return
        n_kernels = kernels.shape[0]
        
        # define sub plots
        n_cols = 20
        n_rows = n_kernels
        
        # create fig
        fig = plt.figure(figsize=(n_cols, n_rows))
        
        # iterate over kernels
        for i in range(n_kernels):
            ax = fig.add_subplot(n_rows, n_cols, i+1)
            
            # denormalize and convert
            npimg = np.array(kernels[i].numpy(), np.float32)
            npimg = (npimg - np.mean(npimg)) / np.std(npimg)
            npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
            npimg = npimg.transpose((1, 2, 0))
            
            # plot
            ax.imshow(npimg)
            ax.axis('off')
            ax.set_title(str(i))
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        
        plt.tight_layout()
        plt.show()

        # below here it's all about saving
        if not self.save_fig:
            print("Not saving.")
            return
        
        sub_dir = self.sub_dir(f"figures/{self.net_manager.net_name}/{self.net_manager.case_id}/sample-{self.net_manager.sample}/")
        net_tag = get_net_tag(self.net_name, self.case_id, self.sample, 
                              self.net_manager.epoch)
        filename = f"{net_tag}.png"
        filename = os.path.join(sub_dir, filename)
        print(f"Saving... {filename}")
        plt.savefig(filename, dpi=100)    
        
    def sub_dir(self, sub_dir):
        """
        Ensures existence of sub directory of self.data_dir and 
        returns its absolute path.

        Args:
            sub_dir (TYPE): DESCRIPTION.

        Returns:
            sub_dir (TYPE): DESCRIPTION.

        """
        sub_dir = os.path.join(self.data_dir, sub_dir)
        
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
            
        return sub_dir
        
        
if __name__=="__main__":
    # init visualizer
    visualizer = Visualizer(data_dir, net_name, n_classes, True)
    
    # plot
    visualizer.plot_accuracy(case_ids)

