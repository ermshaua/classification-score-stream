import os
import shutil
import sys

sys.path.insert(0, "../")

import numpy as np

np.random.seed(1379)

from itertools import product
from benchmark.utils import evaluate_floss, evaluate_window, evaluate_candidate, evaluate_bocd, evaluate_adwin, \
    evaluate_ddm, evaluate_hddm, evaluate_change_finder, evaluate_newma


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

    thresholds = -np.arange(50, 500 + 1, 50)

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


def evaluate_adwin_delta(exp_path, n_jobs, verbose):
    name = "adwin_delta"

    if os.path.exists(exp_path + name):
        shutil.rmtree(exp_path + name)

    os.mkdir(exp_path + name)

    deltas = [0.002, 0.004, 0.006, 0.008, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1]

    for d in deltas:
        candidate_name = f"{d}-delta"
        print(f"Evaluating parameter candidate: {candidate_name}")

        df = evaluate_candidate(
            candidate_name,
            "train",
            eval_func=evaluate_adwin,
            n_jobs=n_jobs,
            verbose=verbose,
            delta=d
        )

        df.to_csv(f"{exp_path}{name}/{candidate_name}.csv")


def evaluate_ddm_out_control(exp_path, n_jobs, verbose):
    name = "ddm_out_control"

    if os.path.exists(exp_path + name):
        shutil.rmtree(exp_path + name)

    os.mkdir(exp_path + name)

    out_control_levels = list(range(1, 30 + 1))

    for o in out_control_levels:
        candidate_name = f"{o}-out_control_level"
        print(f"Evaluating parameter candidate: {candidate_name}")

        df = evaluate_candidate(
            candidate_name,
            "train",
            eval_func=evaluate_ddm,
            n_jobs=n_jobs,
            verbose=verbose,
            out_control_level=o
        )

        df.to_csv(f"{exp_path}{name}/{candidate_name}.csv")


def evaluate_hddm_drift_confidence(exp_path, n_jobs, verbose):
    name = "hddm_drift_confidence"

    if os.path.exists(exp_path + name):
        shutil.rmtree(exp_path + name)

    os.mkdir(exp_path + name)

    confs = (1e-10, 1e-20, 1e-30, 1e-40, 1e-50, 1e-60, 1e-70, 1e-80, 1e-90, 1e-100)

    for c in confs:
        candidate_name = f"{c}-drift_confidence"
        print(f"Evaluating parameter candidate: {candidate_name}")

        df = evaluate_candidate(
            candidate_name,
            "train",
            eval_func=evaluate_hddm,
            n_jobs=n_jobs,
            verbose=verbose,
            drift_confidence=c
        )

        df.to_csv(f"{exp_path}{name}/{candidate_name}.csv")


def evaluate_change_finder_threshold(exp_path, n_jobs, verbose):
    name = "change_finder_threshold"

    if os.path.exists(exp_path + name):
        shutil.rmtree(exp_path + name)

    os.mkdir(exp_path + name)

    thresholds = list(range(10, 100 + 1, 10))

    for t in thresholds:
        candidate_name = f"{t}-threshold"
        print(f"Evaluating parameter candidate: {candidate_name}")

        df = evaluate_candidate(
            candidate_name,
            "train",
            eval_func=evaluate_change_finder,
            n_jobs=n_jobs,
            verbose=verbose,
            threshold=t
        )

        df.to_csv(f"{exp_path}{name}/{candidate_name}.csv")


def evaluate_newma_thresholding_quantile(exp_path, n_jobs, verbose):
    name = "newma_thresholding_quantile"

    if os.path.exists(exp_path + name):
        shutil.rmtree(exp_path + name)

    os.mkdir(exp_path + name)

    thresholds = (0.95, 0.96, 0.97, 0.98, 0.99, 1.)

    for t in thresholds:
        candidate_name = f"{t}-threshold"
        print(f"Evaluating parameter candidate: {candidate_name}")

        df = evaluate_candidate(
            candidate_name,
            "train",
            eval_func=evaluate_newma,
            n_jobs=n_jobs,
            verbose=verbose,
            thresholding_quantile=t
        )

        df.to_csv(f"{exp_path}{name}/{candidate_name}.csv")


if __name__ == '__main__':
    exp_path = "../experiments/"
    n_jobs, verbose = -1, 0

    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    evaluate_floss_threshold(exp_path, n_jobs, verbose)
    evaluate_window_cost_threshold(exp_path, n_jobs, verbose)
    evaluate_bocd_threshold(exp_path, n_jobs, verbose)
    evaluate_adwin_delta(exp_path, n_jobs, verbose)
    evaluate_ddm_out_control(exp_path, n_jobs, verbose)
    evaluate_hddm_drift_confidence(exp_path, n_jobs, verbose)
    evaluate_change_finder_threshold(exp_path, n_jobs, verbose)
    evaluate_newma_thresholding_quantile(exp_path, n_jobs, verbose)
