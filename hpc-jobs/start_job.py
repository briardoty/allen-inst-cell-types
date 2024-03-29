#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 17:01:34 2020

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
python_executable = "/allen/programs/braintv/workgroups/nc-ophys/briar.doty/anaconda3/envs/dlct2/bin/python"
conda_env = "/allen/programs/braintv/workgroups/nc-ophys/briar.doty/anaconda3/envs/dlct2"
job_dir = "/allen/programs/braintv/workgroups/nc-ophys/briar.doty/log_files/"

# args
parser = argparse.ArgumentParser()
parser.add_argument("--job_title", type=str, help="Set value for job_title")
parser.add_argument("--case", default=None, type=str)
parser.add_argument("--schemes", default=[], nargs="+", type=str)
parser.add_argument("--net_name", default=None, type=str)
parser.add_argument("--cases", default=[], nargs="+", type=str)


def main(job_title, cases, case, net_name, schemes):
    
    # script, run_params and job_settings
    with open("job_params.json", "r") as json_file:
        job_params = json.load(json_file)
    
    job_params = job_params[job_title]
    script = job_params["script"]
    run_params = job_params["run_params"]   
    job_settings = job_params["job_settings"]
    
    if len(cases) > 0:
        run_params["cases"] = param_arr_helper(cases)

    if len(schemes) > 0:
        run_params["schemes"] = param_arr_helper(schemes)

    if case is not None:
        run_params["case"] = case

    if net_name is not None:
        run_params["net_name"] = net_name
    
    # prepare args
    params_list = list(chain.from_iterable((f"--{k}", str(run_params[k])) for k in run_params))
    params_string = " ".join(params_list)
    
    # kick off HPC job
    PythonJob(
        script,
        python_executable,
        conda_env = conda_env,
        python_args = params_string,
        jobname = job_title,
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
