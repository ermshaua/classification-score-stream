import sys, logging
sys.path.insert(0, "../")

import numpy as np
import pandas as pd
np.random.seed(1379)
import matplotlib.pyplot as plt

from src.clazz.segmentation import ClaSPSegmetationStream
from src.competitor.FLOSS import FLOSS
from src.competitor.Window import Window
from src.competitor.BOCD import BOCD
from src.utils import load_dataset
from benchmark.metrics import covering
from src.visualizer import plot_clasp_with_ts
from src.realtime_animation import ClaSPSegmetationStreamViewer
from src.dominat_period import dominant_fourier_freq


def compute_profile(stream, ts, aggregate=np.max, interpolate=True):
    fill_value = -np.inf if aggregate is np.max else np.inf
    profile = np.full(shape=ts.shape[0], fill_value=fill_value, dtype=np.float64)

    for dx, timepoint in enumerate(ts):
        window_profile = stream.update(timepoint)

        if window_profile.shape[0] > profile.shape[0]:
            window_profile = window_profile[-profile.shape[0]:]

        profile[max(0, dx-window_profile.shape[0]):dx] = aggregate([
            profile[max(0, dx-window_profile.shape[0]):dx],
            window_profile[max(0, window_profile.shape[0]-dx):]
        ], axis=0)

    if profile.shape[0] < ts.shape[0]:
        profile = np.hstack((np.full(shape=ts.shape[0]-profile.shape[0], fill_value=np.min(profile), dtype=np.float64), profile))

    if interpolate is True:
        profile[np.isinf(profile)] = np.nan
        profile = pd.Series(profile).interpolate(limit_direction="both").to_numpy()

    return profile


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    selection = 2 # 16 27 59

    df = load_dataset("TSSB", [selection]) #
    name, w, cps, ts = df.iloc[0, :].tolist()

    # stream = ClaSPSegmetationStream(n_timepoints=10_000, n_prerun=min(10_000, ts.shape[0]), verbose=ts.shape[0]) #
    # stream = FLOSS(n_timepoints=10_000, n_prerun=min(10_000, ts.shape[0]), window_size=w, threshold=.3, verbose=ts.shape[0])
    stream = Window(10_000, window_size=w*5, cost_func="ar", threshold=10, verbose=ts.shape[0])
    # stream = BOCD(n_timepoints=10_000, threshold=-50, verbose=ts.shape[0])

    # stream = ClaSPSegmetationStreamViewer(name, css, frame_rate=w)

    profile = compute_profile(stream, ts, aggregate=np.max)
    found_cps, scores = stream.change_points, stream.scores

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
    print(f"{name}: True Change Points: {cps}, Found Change Points: {found_cps}, Scores: {np.round(scores, 3)}, Covering-Score: {np.round(covering_score, 3)}")