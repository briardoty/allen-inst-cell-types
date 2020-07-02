#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 15:21:08 2020

@author: briardoty
"""
import argparse
import json
import sys
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
parser.add_argument("--n_samples", type=int, help="Set value for n_samples")
parser.add_argument("--case_id", type=str, help="Set value for case_id")

def main(n_samples, case_id):
    
    job_title = "train_net"
    
    # script, run_params and job_settings
    with open("job_params.json", "r") as json_file:
        job_params = json.load(json_file)
    
    job_params = job_params[job_title]
    script = job_params["script"]
    run_params = job_params["run_params"]   
    job_settings = job_params["job_settings"]
    
    # kick off a job for each net
    for i in range(n_samples):
        
        # update params for this net
        run_params["sample"] = i+1
        run_params["case_id"] = case_id
        
        # prepare args
        params_list = list(chain.from_iterable((f"--{k}", str(run_params[k])) for k in run_params))
        params_string = " ".join(params_list)
        
        # kick off HPC job
        PythonJob(
            script,
            python_executable,
            conda_env = conda_env,
            python_args = params_string,
            jobname = job_title + f" c-{case_id} s-{sample}",
            jobdir = job_dir,
            **job_settings
        ).run(dryrun=False)

if __name__=="__main__":
    args = parser.parse_args()
    print(args)
    main(**vars(args))