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

        ts = np.loadtxt(fname=os.path.join(ABS_PATH + f'/datasets/{dataset}/', ts_name + '.txt'), dtype=np.float64)
        df.append((ts_name, int(window_size), np.array([int(_) for _ in change_points]), ts))

    return pd.DataFrame.from_records(df, columns=["name", "window_size", "change_points", "time_series"])
