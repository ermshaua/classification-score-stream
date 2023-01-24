import os
ABS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd


def load_dataset(dataset, selection=None):
    desc_filename = ABS_PATH + f"/datasets/{dataset}/desc.txt"
    desc_file = []

    with open(desc_filename, 'r') as file:
        for line in file.readlines(): desc_file.append(line.split(","))

    df = []

    for idx, row in enumerate(desc_file):
        if selection is not None and idx not in selection: continue
        (ts_name, window_size), change_points = row[:2], row[2:]
        if len(change_points) == 1 and change_points[0] == "\n": change_points = list()
        path = ABS_PATH + f'/datasets/{dataset}/'

        if os.path.exists(path + ts_name + ".txt"):
            ts = np.loadtxt(fname=path + ts_name + ".txt", dtype=np.float64)
        else:
            ts = np.load(file=path + "data.npz")[ts_name]

        df.append((ts_name, int(window_size), np.array([int(_) for _ in change_points]), ts))

    return pd.DataFrame.from_records(df, columns=["name", "window_size", "change_points", "time_series"])


def load_train_dataset():
    train_names = [
        'DodgerLoopDay',
        'EEGRat',
        'EEGRat2',
        'FaceFour',
        'GrandMalSeizures2',
        'GreatBarbet1',
        'Herring',
        'InlineSkate',
        'InsectEPG1',
        'MelbournePedestrian',
        'NogunGun',
        'NonInvasiveFetalECGThorax1',
        'ShapesAll',
        'TiltECG',
        'ToeSegmentation1',
        'ToeSegmentation2',
        'Trace',
        'UWaveGestureLibraryY',
        'UWaveGestureLibraryZ',
        'WordSynonyms',
        'Yoga'
    ]

    df = pd.concat([load_dataset("UTSA"), load_dataset("TSSB")])
    df = df[df["name"].isin(train_names)]

    return df.sort_values(by="name")


def load_benchmark_dataset():
    df = pd.concat([load_dataset("UTSA"), load_dataset("TSSB")])
    df.sort_values(by="name", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def load_archives_dataset():
    df = pd.concat([
        load_dataset("PAMAP"),
        load_dataset("mHealth"),
        load_dataset("WESAD"),
        load_dataset("MIT-BIH-VE"),
        load_dataset("MIT-BIH-Arr"),
        load_dataset("SleepDB"),
    ])
    df.sort_values(by="name", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def load_combined_dataset():
    df = pd.concat([
        load_benchmark_dataset(),
        load_archives_dataset()
    ])
    df.sort_values(by="name", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
