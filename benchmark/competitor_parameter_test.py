import sys, os, shutil
sys.path.insert(0, "../")

import numpy as np
np.random.seed(1379)

from itertools import product
from src.clazz.profile import binary_f1_score, binary_acc_score
from src.clazz.window_size import dominant_fourier_freq, highest_autocorrelation, suss, mwf
from benchmark.utils import evaluate_floss, evaluate_window, evaluate_candidate, evaluate_bocd


def evaluate_floss_threshold(exp_path, n_jobs, verbose):
    name = "floss_threshold"

    if os.path.exists(exp_path + name):
        shutil.rmtree(exp_path + name)

    os.mkdir(exp_path + name)

    thresholds = np.round(np.arange(.05, 1., .05), 2)

    for t in thresholds:
        candidate_name = f"{t}-threshold"
        print(f"Evaluating parameter candidate: {candidate_name}")

        df = evaluate_candidate(
            candidate_name,
            "train",
            eval_func=evaluate_floss,
            n_jobs=n_jobs,
            verbose=verbose,
            threshold=t,
        )

        df.to_csv(f"{exp_path}{name}/{candidate_name}.csv")


def evaluate_window_cost_threshold(exp_path, n_jobs, verbose):
    name = "window_cost_threshold"

    if os.path.exists(exp_path + name):
        shutil.rmtree(exp_path + name)

    os.mkdir(exp_path + name)

    costs = ("l1", "l2", "normal", "ar", "rank", "mahalanobis")
    # thresholds = np.round(np.arange(.01, .11, .01), 2)
    thresholds = np.round(np.arange(.05, 1., .05), 2)

    for cost, t in product(costs, thresholds):
        candidate_name = f"{cost}-cost-{t}-threshold"
        print(f"Evaluating parameter candidate: {candidate_name}")

        df = evaluate_candidate(
            candidate_name,
            "train",
            eval_func=evaluate_window,
            n_jobs=n_jobs,
            verbose=verbose,
            cost_func=cost,
            threshold=t
        )

        df.to_csv(f"{exp_path}{name}/{candidate_name}.csv")


def evaluate_bocd_threshold(exp_path, n_jobs, verbose):
    name = "bocd_threshold"

    if os.path.exists(exp_path + name):
        shutil.rmtree(exp_path + name)

    os.mkdir(exp_path + name)

    thresholds = -np.arange(50,500+1,50)

    for t in thresholds:
        candidate_name = f"{t}-threshold"
        print(f"Evaluating parameter candidate: {candidate_name}")

        df = evaluate_candidate(
            candidate_name,
            "train",
            eval_func=evaluate_bocd,
            n_jobs=n_jobs,
            verbose=verbose,
            threshold=t
        )

        df.to_csv(f"{exp_path}{name}/{candidate_name}.csv")


if __name__ == '__main__':
    exp_path = "../experiments/"
    n_jobs, verbose = 60, 0

    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    # evaluate_floss_threshold(exp_path, n_jobs, verbose)
    evaluate_window_cost_threshold(exp_path, n_jobs, verbose)
    # evaluate_bocd_threshold(exp_path, n_jobs, verbose)


