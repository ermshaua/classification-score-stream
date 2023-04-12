import logging
import sys

sys.path.insert(0, "../")

from src.clazz.segmentation import ClaSS
from src.realtime_animation import ClaSSViewer

import numpy as np

np.random.seed(1379)

from src.utils import load_dataset
from benchmark.metrics import covering
from src.profile_visualization import plot_profile_with_ts
from benchmark.utils import run_stream

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    selection = 59

    df = load_dataset("TSSB", [selection])
    ts_name, w, cps, ts = df.iloc[0, :].tolist()

    # df = pd.read_csv("../tmp/penguin.txt", sep="\t", header=None)
    # name, w, cps, ts = "X-Acc of Penguin Movement", None, np.array([]), df.iloc[:,0]

    stream = ClaSS(n_timepoints=10_000, n_prerun=min(10_000, ts.shape[0]), verbose=ts.shape[0])  #
    # stream = ClaSSViewer(ts_name, stream)

    profile, runtimes, found_cps, found_cps_dx = run_stream(stream, ts, aggregate_profile=np.max)
    scores = stream.scores

    plot_profile_with_ts(
        ts_name,
        ts,
        profile,
        cps,
        found_cps,  #
        show=False,
        save_path="../tmp/simple_test.pdf"
    )

    covering_score = covering({0: cps}, found_cps, ts.shape[0])  #
    print(
        f"{ts_name}: True Change Points: {cps}, Found Change Points: {found_cps}, Scores: {np.round(scores, 3)}, Runtime: {np.round(runtimes.sum(), 3)} Covering-Score: {np.round(covering_score, 3)}")
