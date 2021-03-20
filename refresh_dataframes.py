import sys
import os
import numpy as np
import argparse
import pandas as pd
from modules.DataframeProcessor import DataframeProcessor

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="/home/briardoty/Source/neuro511-artiphysiology/data/", type=str, help="Set value for data_dir")


def main(data_dir):
    
    # init processor
    proc = DataframeProcessor(data_dir)
    
    # refresh various dataframes
    # proc.refresh_accuracy_df()
    proc.refresh_final_acc_df()

    # need to load acc df to refresh learning
    # df_sub_dir = os.path.join(data_dir, "dataframes/")
    # acc_df = pd.read_csv(os.path.join(df_sub_dir, "acc_df.csv"))
    # acc_df.drop(columns="Unnamed: 0", inplace=True)

    # proc.refresh_learning_df(acc_df, pct=90)

    print("refresh_dataframes.py completed")
    return


if __name__=="__main__":
    args = parser.parse_args()
    print(args)
    main(**vars(args))
    
    