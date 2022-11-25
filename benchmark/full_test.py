import sys, os, shutil
sys.path.insert(0, "../")

import numpy as np
np.random.seed(1379)

from benchmark.utils import evaluate_class, evaluate_floss, evaluate_candidate

if __name__ == '__main__':
    exp_path = "../tmp/"
    n_jobs, verbose = 60, 1

    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    df = evaluate_candidate(
        "full_test",
        "train",
        eval_func=evaluate_floss,
        n_jobs=n_jobs,
        verbose=verbose,
        threshold=.45
    )

    df.to_csv(f"{exp_path}/full_test.csv")