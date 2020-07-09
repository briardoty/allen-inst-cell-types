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
        
    with open("net_configs.json", "r") as json_file:
        net_configs = json.load(json_file)
    
    job_params = job_params[job_title]
    script = job_params["script"]
    run_params = job_params["run_params"]
    cases = job_params["cases"]
    job_settings = job_params["job_settings"]
    
    # kick off job for each net configuration
    for case in cases:
        
        # get net config
        config = net_configs[case]
        
        # update params for this net config
        run_params["case"] = case
        
        if config.get("layer_name") is not None:
            run_params["layer_name"] = config.get("layer_name")
            
        if config.get("n_repeat_arr") is not None:
            run_params["n_repeat_arr"] = param_arr_helper(config.get("n_repeat_arr"))
        
        if config.get("act_fns") is not None:
            run_params["act_fns"] = param_arr_helper(config.get("act_fns"))
        
        if config.get("act_fn_params") is not None:
            run_params["act_fn_params"] = param_arr_helper(config.get("act_fn_params"))
        
        # prepare args
        params_list = list(chain.from_iterable((f"--{k}", str(run_params[k])) for k in run_params))
        params_string = " ".join(params_list)
        
        # kick off HPC job
        PythonJob(
            script,
            python_executable,
            conda_env = conda_env,
            python_args = params_string,
            jobname = job_title + f" c-{case}",
            jobdir = job_dir,
            **job_settings
        ).run(dryrun=False)
      
def param_arr_helper(param_arr):
    
    if param_arr is None or len(param_arr) == 0:
        return None
    
    return " ".join(str(p) for p in param_arr)
        

if __name__=="__main__":
    args = parser.parse_args()
    print(args)
    main(**vars(args))
