import sys, logging
sys.path.insert(0, "../")

import numpy as np
import pandas as pd
np.random.seed(1379)
import matplotlib.pyplot as plt

from src.clazz.segmentation import ClaSPSegmetationStream
from src.competitor.FLOSS import FLOSS
from src.utils import load_dataset
from benchmark.metrics import covering
from src.visualizer import plot_clasp_with_ts
from src.realtime_animation import ClaSPSegmetationStreamViewer


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    selection = 59 # 16 27 59

    df = load_dataset("TSSB", [selection]) #
    name, w, cps, ts = df.iloc[0, :].tolist()

    css = ClaSPSegmetationStream(n_timepoints=min(ts.shape[0], 10_000), verbose=ts.shape[0]) #
    # css = FLOSS(n_timepoints=min(ts.shape[0], 10_000), window_size=w, threshold=.2, verbose=ts.shape[0])
    # csv = ClaSPSegmetationStreamViewer(name, css, frame_rate=w)

    global_profile = np.full(shape=ts.shape[0], fill_value=-np.inf, dtype=np.float64)

    for dx, timepoint in enumerate(ts):
        profile = css.update(timepoint)

        global_profile[max(0, dx-profile.shape[0]):dx] = np.max([
            global_profile[max(0, dx-profile.shape[0]):dx],
            profile[max(0, profile.shape[0]-dx):]
        ], axis=0)

    profile, window_size, found_cps, scores = css.profile, css.window_size, np.array(css.global_change_points), np.array(css.scores)

    profile = global_profile 

    if profile.shape[0] < ts.shape[0]:
        profile = np.hstack((np.full(shape=ts.shape[0]-profile.shape[0], fill_value=np.min(profile), dtype=np.float64), profile))

    plot_clasp_with_ts(
        name,
        ts,
        profile,
        cps,
        found_cps, #
        show=False,
        save_path="../tmp/simple_test.pdf"
    )

    covering_score = covering({0 : cps}, found_cps, ts.shape[0]) #
    print(f"{name}: Window Size: {window_size} True Change Points: {cps}, Found Change Points: {found_cps}, Scores: {np.round(scores, 3)}, Covering-Score: {np.round(covering_score, 3)}")