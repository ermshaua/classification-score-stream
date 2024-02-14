import os
import shutil
import sys

sys.path.insert(0, "../")

import numpy as np

np.random.seed(1379)

from src.clazz.profile import binary_f1_score, binary_acc_score
from src.clazz.window_size import dominant_fourier_freq, highest_autocorrelation, suss, mwf
from benchmark.utils import evaluate_class, evaluate_candidate


def evaluate_sliding_window_parameter(exp_path, n_jobs, verbose):
    name = "sliding_window"

    if os.path.exists(exp_path + name):
       shutil.rmtree(exp_path + name)

    os.mkdir(exp_path + name)

    ds = np.arange(1_000, 20_000 + 1, 1_000)
    datasets = ["train"] # "UTSA", "TSSB", "PAMAP", "mHealth", "WESAD", "MIT-BIH-VE", "MIT-BIH-Arr", "SleepDB"

    for dataset in datasets:
        for n_timepoints in ds:
            candidate_name = f"{n_timepoints}-timepoints"
            print(f"Evaluating parameter candidate: {candidate_name}")

            df = evaluate_candidate(
                candidate_name,
                dataset,
                eval_func=evaluate_class,
                n_jobs=n_jobs,
                verbose=verbose,
                n_timepoints=n_timepoints
            )

            df.to_csv(f"{exp_path}{name}/{dataset}_{candidate_name}.csv")


def evaluate_k_neighbours_parameter(exp_path, n_jobs, verbose):
    name = "k_neighbours"

    if os.path.exists(exp_path + name):
        shutil.rmtree(exp_path + name)

    os.mkdir(exp_path + name)

    k_neighbours = (1, 3, 5, 7)

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


def evaluate_score_parameter(exp_path, n_jobs, verbose):
    name = "score"

    if os.path.exists(exp_path + name):
        shutil.rmtree(exp_path + name)

    os.mkdir(exp_path + name)

    scores = [
        ("F1", binary_f1_score),
        ("Accuracy", binary_acc_score)
    ]

    for candidate_name, score in scores:
        print(f"Evaluating parameter candidate: {candidate_name}")

        df = evaluate_candidate(
            candidate_name,
            "train",
            eval_func=evaluate_class,
            n_jobs=n_jobs,
            verbose=verbose,
            score=score
        )

        df.to_csv(f"{exp_path}{name}/{candidate_name}.csv")


def evaluate_window_size_parameter(exp_path, n_jobs, verbose):
    name = "window_size"

    if os.path.exists(exp_path + name):
        shutil.rmtree(exp_path + name)

    os.mkdir(exp_path + name)

    window_sizes = [
        ("predefined", "predefined"),
        ("FFT", dominant_fourier_freq),
        ("ACF", highest_autocorrelation),
        ("SuSS", suss),
        ("MWF", mwf),
    ]

    for candidate_name, window_size in window_sizes:
        print(f"Evaluating parameter candidate: {candidate_name}")

        df = evaluate_candidate(
            candidate_name,
            "train",
            eval_func=evaluate_class,
            n_jobs=n_jobs,
            verbose=verbose,
            window_size=window_size
        )

        df.to_csv(f"{exp_path}{name}/{candidate_name}.csv")


def evaluate_similarity_parameter(exp_path, n_jobs, verbose):
    name = "similarity"

    if os.path.exists(exp_path + name):
        shutil.rmtree(exp_path + name)

    os.mkdir(exp_path + name)

    similarities = ["pearson", "ed", "cid"]

    for candidate_name in similarities:
        print(f"Evaluating parameter candidate: {candidate_name}")

        df = evaluate_candidate(
            candidate_name,
            "train",
            eval_func=evaluate_class,
            n_jobs=n_jobs,
            verbose=verbose,
            similarity=candidate_name
        )

        df.to_csv(f"{exp_path}{name}/{candidate_name}.csv")


def evaluate_p_value_parameter(exp_path, n_jobs, verbose):
    name = "p_value"

    if os.path.exists(exp_path + name):
        shutil.rmtree(exp_path + name)

    os.mkdir(exp_path + name)

    p_values = (1e-10, 1e-20, 1e-30, 1e-40, 1e-50, 1e-60, 1e-70, 1e-80, 1e-90, 1e-100)

    for p_value in p_values:
        candidate_name = f"{p_value}-p_value"
        print(f"Evaluating parameter candidate: {candidate_name}")

        df = evaluate_candidate(
            candidate_name,
            "train",
            eval_func=evaluate_class,
            n_jobs=n_jobs,
            verbose=verbose,
            p_value=p_value
        )

        df.to_csv(f"{exp_path}{name}/{candidate_name}.csv")


def evaluate_sample_size_parameter(exp_path, n_jobs, verbose):
    name = "sample_size"

    if os.path.exists(exp_path + name):
        shutil.rmtree(exp_path + name)

    os.mkdir(exp_path + name)

    sample_sizes = [None, 10, 100, 1_000, 10_000]  #

    for sample_size in sample_sizes:
        candidate_name = f"{sample_size}-sample_size"
        print(f"Evaluating parameter candidate: {candidate_name}")

        df = evaluate_candidate(
            candidate_name,
            "train",
            eval_func=evaluate_class,
            n_jobs=n_jobs,
            verbose=verbose,
            sample_size=sample_size
        )

        df.to_csv(f"{exp_path}{name}/{candidate_name}.csv")


# change components in ClaSS to run this method
def evaluate_different_components(exp_path, n_jobs, verbose):
    name = "components"

    if os.path.exists(exp_path + name):
      shutil.rmtree(exp_path + name)

    os.mkdir(exp_path + name)

    datasets = ["train", "UTSA", "TSSB", "PAMAP", "mHealth", "WESAD", "MIT-BIH-VE", "MIT-BIH-Arr", "SleepDB"] #

    for dataset in datasets:
        candidate_name = f"{dataset}-scoring-clasp"
        print(f"Evaluating parameter candidate: {candidate_name}")

        df = evaluate_candidate(
            candidate_name,
            dataset,
            eval_func=evaluate_class,
            columns=["dataset", "true_cps", "found_cps", "found_cps_dx", "f1_score", "covering_score", "profile", "runtimes", "knn_runtimes", "scoring_runtimes"],
            n_jobs=n_jobs,
            verbose=verbose,
        )

        df.to_csv(f"{exp_path}{name}/{candidate_name}.csv")


if __name__ == '__main__':
    exp_path = "../experiments/"
    n_jobs, verbose = -1, 0

    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    evaluate_sliding_window_parameter(exp_path, n_jobs, verbose)
    evaluate_k_neighbours_parameter(exp_path, n_jobs, verbose)
    evaluate_score_parameter(exp_path, n_jobs, verbose)
    evaluate_window_size_parameter(exp_path, n_jobs, verbose)
    evaluate_similarity_parameter(exp_path, n_jobs, verbose)
    evaluate_p_value_parameter(exp_path, n_jobs, verbose)
    evaluate_sample_size_parameter(exp_path, n_jobs, verbose)
    # evaluate_different_components(exp_path, n_jobs, verbose)
