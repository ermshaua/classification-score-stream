import sys, logging, time
sys.path.insert(0, "../")

import numpy as np
import pandas as pd
np.random.seed(1379)
import matplotlib.pyplot as plt

from src.clazz.segmentation import ClaSS
from src.competitor.FLOSS import FLOSS
from src.competitor.Window import Window
from src.competitor.BOCD import BOCD
from src.utils import load_dataset
from benchmark.metrics import covering
from src.visualizer import plot_clasp_with_ts
from src.realtime_animation import ClaSPSegmetationStreamViewer
from src.dominat_period import dominant_fourier_freq
from benchmark.utils import run_stream





if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    selection = 28 # 16 27 59

    df = load_dataset("UTSA", [selection]) #
    name, w, cps, ts = df.iloc[0, :].tolist()

    stream = ClaSS(n_timepoints=10_000, k_neighbours=3, n_prerun=min(10_000, ts.shape[0]), jump=None, verbose=ts.shape[0]) #
    # stream = FLOSS(n_timepoints=10_000, n_prerun=min(10_000, ts.shape[0]), window_size=w, threshold=.3, verbose=ts.shape[0])
    # stream = Window(10_000, window_size=100, cost_func="ar", threshold=10, verbose=ts.shape[0])
    # stream = BOCD(n_timepoints=10_000, threshold=-50, verbose=ts.shape[0])

    # stream = ClaSPSegmetationStreamViewer(name, css, frame_rate=w)

    profile, runtimes = run_stream(stream, ts, aggregate_profile=np.max)

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
    print(f"{name}: True Change Points: {cps}, Found Change Points: {found_cps}, Scores: {np.round(scores, 3)}, Runtime: {np.round(runtimes.sum(), 3)} Covering-Score: {np.round(covering_score, 3)}")