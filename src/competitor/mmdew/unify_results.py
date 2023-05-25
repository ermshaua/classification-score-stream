import argparse
import pandas as pd
import os
from ast import literal_eval
import matplotlib
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from mmdew import metrics

def load_data(folder):
    df = pd.DataFrame()
    files = os.listdir(folder)
    for f in files:
        df = pd.concat([df, pd.read_csv(folder + "/" + f, index_col=0)])

    df = df.reset_index(drop=True)
    df = df.drop(df[df["timeout"] == True].index)
    df = df.drop("timeout", axis=1)
    df = df.replace({"GasSensors" : "Gas", "CIFAR10" : "CIF", "TrafficUnif": "Traf"})

    for col in ["actual_cps", "detected_cps_at", "detected_cps"]:
        df.loc[:,col] = df.loc[:,col].apply(lambda x : literal_eval(x))
        df.loc[:,col] = df.loc[:,col].apply(lambda z : [i for i in z if i > 40]) # 40 to account for TrafficUnif
        


    df = df.fillna(0)
    df.loc[:,"percent_changes_detected"] = df.apply(lambda x : metrics.percent_changes_detected(x.actual_cps, x.detected_cps_at), axis=1)
    return df


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("norm", type=int,
                    help="width of the window")
    parser.add_argument("folder",
                    help="folder to process")
    args = parser.parse_args()

    norm = args.norm
    df = load_data(args.folder)

    Ts = { "MNIST" : 7000/norm,
           "CIF" : 6000/norm,
           "FMNIST" : 7000/norm,
           "Gas" : 1159*2/norm,
           "HAR" : 858*2/norm,
           "Traf" : 40/norm,
    }

    for k,v in Ts.items():
        df.loc[df["dataset"] == k,"f1_detected_cps_at"] = df.apply(lambda x : metrics.fb_score(x.actual_cps, x.detected_cps_at, T=v, beta=1), axis = 1)
        df.loc[df["dataset"] == k,"precision"] = df.apply(lambda x : metrics.prec_full(x.actual_cps, x.detected_cps_at, T=v), axis = 1)
        df.loc[df["dataset"] == k,"recall"] = df.apply(lambda x : metrics.rec_full(x.actual_cps, x.detected_cps_at, T=v), axis = 1)



    avg_results = df.groupby(["dataset", "algorithm", "config"]).mean().reset_index().fillna(0)
    best_configs = avg_results.loc[avg_results.groupby(["dataset", "algorithm"])["f1_detected_cps_at"].idxmax()]
    data = pd.merge(best_configs, df, how="left", on=["dataset","algorithm","config"])


    data["mean_until_detection"] = data.apply(lambda x : metrics.mean_until_detection(x.actual_cps, x.detected_cps_at), axis=1)
    
    data.to_csv(f"../results/results_combined_{norm}.csv")
