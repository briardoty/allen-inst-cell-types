#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 17:01:34 2020

@author: briardoty
"""
import os
import json
import sys

# import pbstools (Python 3 version)
sys.path.append("/home/briar.doty/pbstools")
from pbstools import PythonJob

# path to python executable
python_executable = "/home/briar.doty/anaconda3/envs/dlct/bin/python"
conda_env = "/home/briar.doty/anaconda3/envs/dlct"
train_script = "/allen/programs/braintv/workgroups/nc-ophys/briar.doty/Source/allen-inst-cell-types/net_train.py"
job_dir = "/allen/programs/braintv/workgroups/nc-ophys/briar.doty/log_files/"

# params
run_params = {
    "data_dir": "/allen/programs/braintv/workgroups/nc-ophys/briar.doty/data/",
    "net_name": "vgg11",
    "n_classes": 10
}

# job settings
job_settings = {
    "queue": "braintv",
    "mem": "2g",
    "walltime": "1:00:00",
    "ppn": 16,
    "nodes": 1,
    "gpus": 1,
}

if __name__=="__main__":
    # prepare args
    run_params = [str(run_params)]
    params_string = " ".join(run_params)
    job_title = "test job"
    
    # kick off HPC job
    PythonJob(
        train_script,
        python_executable,
        conda_env = conda_env,
        python_args = params_string,
        jobname = job_title,
        jobdir = job_dir,
        **job_settings
    ).run(dryrun=False)
