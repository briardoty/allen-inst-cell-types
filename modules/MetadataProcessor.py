import torch
import os
import pandas as pd
import re
import numpy as np
import json
from scipy.stats import ttest_ind
import math


try:
    from .util import *
except:
    from util import *

class MetadataProcessor():
    """
    Class to handle processing network snapshots into meaningful statistics.
    """
    
    def __init__(self, data_dir):
        
        self.data_dir = data_dir
        self.exclude_slug = "(exclude)"
        self.pct = 90
    
    def reduce_snapshots(self):

        # walk networks directory
        net_dir = os.path.join(self.data_dir, f"nets/")
        for root, _, files in os.walk(net_dir):
            
            # only interested in locations files are saved
            if len(files) <= 0:
                continue
            
            slugs = root.split("/")
            
            # consider all files...
            for filename in files:

                # ...as long as they are snapshots
                if not filename.endswith(".pt"):
                    continue
                
                epoch = get_epoch_from_filename(filename)

                if epoch is None:
                    continue

                if epoch % 50 == 0:
                    continue
                else:
                    # delete
                    filepath = os.path.join(root, filename)
                    print(f"Deleting {filepath}")
                    os.remove(filepath)

if __name__=="__main__":
    
    processor = MetadataProcessor("/home/briardoty/Source/allen-inst-cell-types/data_mountpoint")
    
    processor.reduce_snapshots()
    
    
    
    
    
    
