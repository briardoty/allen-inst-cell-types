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

def main():
    
    job_title = "gen_nets"
    
    # script, run_params and job_settings
    with open("job_params.json", "r") as json_file:
        job_params = json.load(json_file)
    
    job_params = job_params[job_title]
    script = job_params["script"]
    run_params = job_params["run_params"]
    net_configs = job_params["configs"]
    job_settings = job_params["job_settings"]
    
    # kick off set of jobs for each net configuration
    for config in net_configs:
        
        case_id = config["case_id"]
        
        # update params for this net config
        run_params["case_id"] = case_id
        run_params["layer_name"] = config["layer_name"]
        run_params["n_repeat"] = config["n_repeat"]
        run_params["act_fns"] = " ".join(config["act_fns"])
        run_params["act_fn_params"] = " ".join(str(p) for p in config["act_fn_params"])
        
        # prepare args
        params_list = list(chain.from_iterable((f"--{k}", str(run_params[k])) for k in run_params))
        params_string = " ".join(params_list)
        
        # kick off HPC job
        PythonJob(
            script,
            python_executable,
            conda_env = conda_env,
            python_args = params_string,
            jobname = job_title + f" c-{case_id}",
            jobdir = job_dir,
            **job_settings
        ).run(dryrun=False)
        
        

if __name__=="__main__":
    args = parser.parse_args()
    print(args)
    main(**vars(args))
