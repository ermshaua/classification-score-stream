import os
import shutil
import sys

sys.path.insert(0, "../")

import numpy as np

np.random.seed(1379)

from benchmark.utils import evaluate_candidate, evaluate_class, evaluate_floss, evaluate_window, evaluate_adwin, \
    evaluate_ddm, evaluate_hddm, evaluate_change_finder, evaluate_newma, evaluate_bocd


def evaluate_competitor_dataset(dataset_name, exp_path, n_jobs, verbose):
    name = f"competitor_{dataset_name}"

    if os.path.exists(exp_path + name):
        shutil.rmtree(exp_path + name)

    os.mkdir(exp_path + name)

    competitors = [
        ("ClaSS", evaluate_class),
        ("FLOSS", evaluate_floss),
        ("Window", evaluate_window),
        ("BOCD", evaluate_bocd),
        ("ADWIN", evaluate_adwin),
        ("DDM", evaluate_ddm),
        ("HDDM", evaluate_hddm),
        ("ChangeFinder", evaluate_change_finder),
        ("NEWMA", evaluate_newma)
    ]

    for candidate_name, eval_func in competitors:
        print(f"Evaluating parameter candidate: {candidate_name}")

        df = evaluate_candidate(
            candidate_name,
            dataset_name,
            eval_func=eval_func,
            n_jobs=n_jobs,
            verbose=verbose,
        )

        df.to_csv(f"{exp_path}{name}/{candidate_name}.csv.gz", compression='gzip')


if __name__ == '__main__':
    exp_path = "../experiments/"
    n_jobs, verbose = -1, 0

    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    evaluate_competitor_dataset("UTSA", exp_path, n_jobs, verbose)
    evaluate_competitor_dataset("TSSB", exp_path, n_jobs, verbose)

    evaluate_competitor_dataset("PAMAP", exp_path, n_jobs, verbose)
    evaluate_competitor_dataset("mHealth", exp_path, n_jobs, verbose)
    evaluate_competitor_dataset("WESAD", exp_path, n_jobs, verbose)
    evaluate_competitor_dataset("MIT-BIH-VE", exp_path, n_jobs, verbose)
    evaluate_competitor_dataset("MIT-BIH-Arr", exp_path, n_jobs, verbose)
    evaluate_competitor_dataset("SleepDB", exp_path, n_jobs, verbose)
