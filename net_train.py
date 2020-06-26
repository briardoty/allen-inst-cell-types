#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 17:22:40 2020

@author: briardoty
"""
from NetManager import NetManager

if __name__=="__main__":
    
    # TODO: parse args
    net_name = "vgg11"
    n_classes = 10
    data_dir = "/allen/programs/braintv/workgroups/nc-ophys/briar.doty/data"
    
    # init net manager
    manager = NetManager(net_name, n_classes, data_dir, pretrained=True)
    
    print("Test run successful!!!")