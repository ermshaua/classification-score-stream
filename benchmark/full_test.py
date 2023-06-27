import os
import sys

sys.path.insert(0, "../")

import numpy as np

np.random.seed(1379)

from benchmark.utils import evaluate_class, evaluate_candidate

if __name__ == '__main__':
    exp_path = "../tmp/"
    n_jobs, verbose = -1, 1

    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    df = evaluate_candidate(
        "full_test",
        "train",
        eval_func=evaluate_class,
        n_jobs=n_jobs,
        verbose=verbose
    )

    df.to_csv(f"{exp_path}/full_test.csv")
