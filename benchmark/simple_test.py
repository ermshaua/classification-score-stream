import sys, logging, time
sys.path.insert(0, "../")

import numpy as np
import pandas as pd
np.random.seed(1379)
import matplotlib.pyplot as plt

from src.clazz.segmentation import ClaSS
from src.competitor.FLOSS import FLOSS
from src.competitor.Window import Window
# from src.competitor.BOCD import BOCD
from src.utils import load_dataset
from benchmark.metrics import covering
from src.visualizer import plot_clasp_with_ts
from src.realtime_animation import ClaSPSegmetationStreamViewer
from benchmark.utils import run_stream





if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    selection = 59 # 16 27 59

    df = load_dataset("TSSB", [selection]) #
    name, w, cps, ts = df.iloc[0, :].tolist()

    from src.clazz.profile import binary_acc_score



    # stream = ClaSS(n_timepoints=10_000, n_prerun=min(10_000, ts.shape[0]), verbose=ts.shape[0]) #
    # stream = FLOSS(n_timepoints=10_000, n_prerun=min(10_000, ts.shape[0]), threshold=.5, verbose=ts.shape[0])
    stream = Window(n_timepoints=10_000, n_prerun=min(10_000, ts.shape[0]), cost_func="rank", threshold=.05, verbose=ts.shape[0])
    # stream = BOCD(n_timepoints=10_000, threshold=-50, verbose=ts.shape[0])

    # stream = ClaSPSegmetationStreamViewer(name, stream, frame_rate=w)

    profile, runtimes, found_cps, found_cps_dx = run_stream(stream, ts, aggregate_profile=np.max)

    scores = stream.scores

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