#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 15:21:08 2020

@author: briardoty
"""
import argparse
import json
import sys
import os
from itertools import chain

# import pbstools (Python 3 version)
sys.path.append("/home/briar.doty/pbstools")
from pbstools import PythonJob

# paths
python_executable = "/home/briar.doty/anaconda3/envs/dlct/bin/python"
conda_env = "/home/briar.doty/anaconda3/envs/dlct"
job_dir = "/allen/programs/braintv/workgroups/nc-ophys/briar.doty/log_files/"

# args
parser = argparse.ArgumentParser()
parser.add_argument("--case_id", type=str, help="Set value for case_id", required=True)

def main(case_id):
    
    job_title = "train_net"
    
    # script, run_params and job_settings
    with open("job_params.json", "r") as json_file:
        job_params = json.load(json_file)
    
    job_params = job_params[job_title]
    script = job_params["script"]
    run_params = job_params["run_params"]   
    job_settings = job_params["job_settings"]
    
    # walk dir looking for nets to train
    net_dir = os.path.join(run_params["data_dir"], f"nets/{run_params['net_name']}")
    for root, dirs, files in os.walk(net_dir):
        
        # only interested in locatons files (nets) are saved
        if len(files) <= 0:
            continue

        # only interested in the given case
        if not f"case-{case_id}" in root:
            continue
        
        # consider all nets...
        for net_filename in files:
            
            # ...that are saved at epoch 0
            if not net_filename.endswith("epoch-0.pt"):
                continue
            
            # and submit a training job for them
            net_filepath = os.path.join(root, net_filename)
            run_params["net_filepath"] = net_filepath
            
            # prepare args
            params_list = list(chain.from_iterable((f"--{k}", str(run_params[k])) for k in run_params))
            params_string = " ".join(params_list)
            
            # kick off HPC job
            PythonJob(
                script,
                python_executable,
                conda_env = conda_env,
                python_args = params_string,
                jobname = job_title + f" {net_filename}",
                jobdir = job_dir,
                **job_settings
            ).run(dryrun=False)


if __name__=="__main__":
    args = parser.parse_args()
    print(args)
    main(**vars(args))
