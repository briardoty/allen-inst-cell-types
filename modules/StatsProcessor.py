# -*- coding: utf-8 -*-
import torch
import os
import pandas as pd

try:
    from .NetManager import NetManager
except:
    from NetManager import NetManager
    
def get_earliest_nets(net_filenames):
    return

def get_latest_nets(net_filenames):
    return

class StatsProcessor(NetManager):
    """
    Class to handle processing network snapshots into meaningful statistics.
    Subclass of NetManager to inherit snapshot loading functionality.
    """
    
    def __init__(self, net_name, n_classes, data_dir, pretrained=False):
        
        super(StatsProcessor, self).__init__(net_name, n_classes, data_dir, 
                                             pretrained)
    
    def load__weight_df(self, case_ids):
        """
        Loads a dataframe containing the mean weight 

        Args:
            case_ids (list): Experimental cases to include in figure.

        Returns:
            None.

        """
        
        # walk dir looking for net snapshots
        net_dir = os.path.join(self.data_dir, f"nets/{self.net_name}")
        for root, dirs, files in os.walk(net_dir):
            
            # only interested in locations files (nets) are saved
            if len(files) <= 0:
                continue
            
            # only interested in the given cases
            if not any(c in root for c in case_ids):
                continue
            
            # consider just nets that...
            for net_filename in files:
                net_filepath = os.path.join(root, net_filename)
                net_metadata = self.load_snapshot_metadata(net_filepath)
    
    def load_accuracy_df(self, case_ids):
        """
        Loads dataframe with accuracy over training for different experimental 
        cases.

        Args:
            case_ids (list): Experimental cases to include in figure.

        Returns:
            acc_df (dataframe): Dataframe containing training accuracy.

        """
        acc_arr = []
            
        # walk dir looking for net snapshots
        net_dir = os.path.join(self.data_dir, f"nets/{self.net_name}")
        for root, dirs, files in os.walk(net_dir):
            
            # only interested in locations files (nets) are saved
            if len(files) <= 0:
                continue
            
            # only interested in the given cases
            if not any(c in root for c in case_ids):
                continue
            
            # consider all nets...
            for net_filename in files:
                
                net_filepath = os.path.join(root, net_filename)
                net_metadata = self.load_snapshot_metadata(net_filepath)
                
                sample = net_metadata.get("sample")
                epoch = net_metadata.get("epoch")
                val_acc = net_metadata.get("val_acc")
                if torch.is_tensor(val_acc):
                    val_acc = val_acc.item()
                case = net_metadata.get("case")
                
                acc_arr.append([case, sample, epoch, val_acc])
                
        # make dataframe
        acc_df = pd.DataFrame(acc_arr, columns=["case", "sample", "epoch", "acc"])  
        return acc_df