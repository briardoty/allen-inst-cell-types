#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 15:21:08 2020

@author: briardoty
"""
import argparse
import json
import sys
import re
import os
from itertools import chain

# import pbstools (Python 3 version)
sys.path.append("/home/briar.doty/pbstools")
from pbstools import PythonJob

# paths
python_executable = "/allen/programs/braintv/workgroups/nc-ophys/briar.doty/anaconda3/envs/dlct2/bin/python"
conda_env = "/allen/programs/braintv/workgroups/nc-ophys/briar.doty/anaconda3/envs/dlct2"
job_dir = "/allen/programs/braintv/workgroups/nc-ophys/briar.doty/log_files/"

# args
parser = argparse.ArgumentParser()
parser.add_argument("--cases", type=str, nargs="+", help="Set value for cases", required=True)
parser.add_argument("--train_scheme", type=str, help="Set train_scheme", required=True)
parser.add_argument("--resume", dest="resume", action="store_true")
parser.set_defaults(resume=False)

def main(cases, train_scheme, resume):
    
    job_title = "train_net"
    
    # script, run_params and job_settings
    with open("job_params.json", "r") as json_file:
        job_params = json.load(json_file)
    
    job_params = job_params[job_title]
    script = job_params["script"]
    run_params = job_params["run_params"]
    run_params["train_scheme"] = train_scheme
    job_settings = job_params["job_settings"]
    
    # set to avoid submitting jobs for the same net twice
    net_filepaths = set()

    # walk dir looking for nets to train
    net_dir = os.path.join(run_params["data_dir"], f"nets/{run_params['net_name']}")
    for root, dirs, files in os.walk(net_dir):
        
        # only interested in locations files (nets) are saved
        if len(files) <= 0:
            continue
        
        slugs = root.split("/")

        # only interested in the given training scheme
        if train_scheme in slugs:
            continue

        # only interested in the given case
        if not any(c in slugs for c in cases):
            continue
        
        # start from first or last epoch
        if resume:
            net_filename = get_last_epoch(files)
            print(f"Submitting job to resume training of {net_filename}.")
        else:
            net_filename = get_first_epoch(files)
            print(f"Job will begin from initial snapshot {net_filename}.")

        # and add it to the training job set
        net_filepath = os.path.join(root, net_filename)
        net_filepaths.add(net_filepath)

    # loop over set, submitting jobs
    for net_filepath in net_filepaths:

        # update param
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

def get_epoch_from_filename(filename):
    
    epoch = re.search(r"\d+\.pt$", filename)
    epoch = int(epoch.group().split(".")[0]) if epoch else None
    
    return epoch
    
def get_first_epoch(net_filenames):
    
    for filename in net_filenames:
        
        epoch = get_epoch_from_filename(filename)
        if epoch == 0:
            return filename

def get_last_epoch(net_filenames):
    
    max_epoch = -1
    last_net_filename = None
    
    for filename in net_filenames:
        
        epoch = get_epoch_from_filename(filename)

        if epoch is None:
            continue

        if epoch > max_epoch:
            max_epoch = epoch
            last_net_filename = filename

    return last_net_filename

if __name__=="__main__":
    args = parser.parse_args()
    print(args)
    main(**vars(args))
