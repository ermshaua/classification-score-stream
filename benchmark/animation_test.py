import logging
import sys

sys.path.insert(0, "../")

import numpy as np

np.random.seed(1379)

from src.clazz.segmentation import ClaSS
from src.utils import load_dataset
from benchmark.metrics import covering
from src.profile_animation import ClaSSAnimator

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    selection = 59

    df = load_dataset("TSSB", [selection])
    ts_name, w, cps, ts = df.iloc[0, :].tolist()

    clazz = ClaSS(n_timepoints=10_000, n_prerun=min(10_000, ts.shape[0]), verbose=ts.shape[0])

    clazz_animator = ClaSSAnimator(
        ts_name,
        ts,
        clazz,
        true_cps=cps,
        file_path="../tmp/animation_test.mp4", score="ClaSP Score"
    )
    clazz_animator.run_animation()

    found_cps, scores = clazz.change_points, clazz.scores

    covering_score = covering({0: cps}, found_cps, ts.shape[0])  #
    print(
        f"{ts_name}: True Change Points: {cps}, Found Change Points: {found_cps}, Scores: {np.round(scores, 3)}, Covering-Score: {np.round(covering_score, 3)}")
