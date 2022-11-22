import sys, os, shutil
sys.path.insert(0, "../")

import numpy as np
np.random.seed(1379)

from benchmark.utils import evaluate_class, evaluate_candidate


def evaluate_k_neighbours_parameter(exp_path, n_jobs, verbose):
    name = "k_neighbours"

    if os.path.exists(exp_path + name):
        shutil.rmtree(exp_path + name)

    os.mkdir(exp_path + name)

    k_neighbours = (1,3,5)

    for nn in k_neighbours:
        candidate_name = f"{nn}-NN"
        print(f"Evaluating parameter candidate: {candidate_name}")

        df = evaluate_candidate(
            candidate_name,
            "train",
            eval_func=evaluate_class,
            n_jobs=n_jobs,
            verbose=verbose,
            k_neighbours=nn
        )

        df.to_csv(f"{exp_path}{name}/{candidate_name}.csv")


if __name__ == '__main__':
    exp_path = "../experiments/"
    n_jobs, verbose = 60, 0

    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    evaluate_k_neighbours_parameter(exp_path, n_jobs, verbose)