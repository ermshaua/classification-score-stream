import time

import numpy as np
import pandas as pd
import daproli as dp

from src.competitor.HDDM import HDDM
from src.utils import load_dataset, load_train_dataset, load_benchmark_dataset
from src.clazz.segmentation import ClaSS
from src.competitor.FLOSS import FLOSS
from src.competitor.Window import Window
from src.competitor.BOCD import BOCD
from src.competitor.ADWIN import ADWIN
from src.competitor.DDM import DDM
from benchmark.metrics import f_measure, covering
from tqdm import tqdm


def run_stream(stream, ts, aggregate_profile=np.max, interpolate_profile=True):
    fill_value = -np.inf if aggregate_profile is np.max else np.inf
    profile = np.full(shape=ts.shape[0], fill_value=fill_value, dtype=np.float64)
    runtimes = np.full(shape=ts.shape[0], fill_value=np.nan, dtype=np.float64)
    found_cps, found_cps_dx = [], []

    for dx, timepoint in enumerate(ts):
        runtime = time.process_time()
        window_profile = stream.update(timepoint)
        runtime = time.process_time() - runtime

        # update profile
        if window_profile.shape[0] > profile.shape[0]:
            window_profile = window_profile[-profile.shape[0]:]

        profile[max(0, dx-window_profile.shape[0]+1):dx+1] = aggregate_profile([
            profile[max(0, dx-window_profile.shape[0]+1):dx+1],
            window_profile[max(0, window_profile.shape[0]-dx-1):]
        ], axis=0)

        # store runtime
        runtimes[dx] = runtime

        # store CPs
        while len(stream.change_points) > len(found_cps):
            found_cps.append(stream.change_points[len(found_cps)])
            found_cps_dx.append(dx)

    if profile.shape[0] < ts.shape[0]:
        profile = np.hstack((np.full(shape=ts.shape[0]-profile.shape[0], fill_value=np.min(profile), dtype=np.float64), profile))

    if interpolate_profile is True:
        profile[np.isinf(profile)] = np.nan
        profile = pd.Series(profile).interpolate(limit_direction="both").to_numpy()

    return profile, np.round(runtimes, 5), found_cps, found_cps_dx


def evaluate_class(name, w, cps, ts, **seg_kwargs):
    if "n_timepoints" in seg_kwargs:
        n_prerun = min(seg_kwargs["n_prerun"], ts.shape[0])
    else:
        n_prerun = min(10_000, ts.shape[0])

    if "window_size" in seg_kwargs and seg_kwargs["window_size"] == "predefined":
        seg_kwargs["window_size"] = w

    stream = ClaSS(n_prerun=n_prerun, verbose=0, **seg_kwargs)
    profile, runtimes, found_cps, found_cps_dx = run_stream(stream, ts, aggregate_profile=np.max)

    f1_score = np.round(f_measure({0: cps}, found_cps, margin=int(ts.shape[0] * .01)), 3)
    covering_score = np.round(covering({0: cps}, found_cps, ts.shape[0]), 3)

    # print(f"{name}: Found Change Points: {found_cps}, F1-Score: {f1_score}, Covering-Score: {covering_score}")
    return name, cps.tolist(), found_cps, found_cps_dx, f1_score, covering_score, profile.tolist(), runtimes.tolist()


def evaluate_floss(name, w, cps, ts, **seg_kwargs):
    if "n_timepoints" in seg_kwargs:
        n_prerun = min(seg_kwargs["n_prerun"], ts.shape[0])
    else:
        n_prerun = min(10_000, ts.shape[0])

    stream = FLOSS(window_size=w, n_prerun=n_prerun, verbose=0, **seg_kwargs)
    profile, runtimes, found_cps, found_cps_dx = run_stream(stream, ts, aggregate_profile=np.min)

    f1_score = np.round(f_measure({0: cps}, found_cps, margin=int(ts.shape[0] * .01)), 3)
    covering_score = np.round(covering({0: cps}, found_cps, ts.shape[0]), 3)

    # print(f"{name}: Found Change Points: {found_cps}, F1-Score: {f1_score}, Covering-Score: {covering_score}")
    return name, cps.tolist(), found_cps, found_cps_dx, f1_score, covering_score, profile.tolist(), runtimes.tolist()


def evaluate_window(name, w, cps, ts, **seg_kwargs):
    stream = Window(window_size=w, verbose=0, **seg_kwargs)
    profile, runtimes, found_cps, found_cps_dx = run_stream(stream, ts, aggregate_profile=np.max)

    f1_score = np.round(f_measure({0: cps}, found_cps, margin=int(ts.shape[0] * .01)), 3)
    covering_score = np.round(covering({0: cps}, found_cps, ts.shape[0]), 3)

    # print(f"{name}: Found Change Points: {found_cps}, F1-Score: {f1_score}, Covering-Score: {covering_score}")
    return name, cps.tolist(), found_cps, found_cps_dx, f1_score, covering_score, profile.tolist(), runtimes.tolist()


def evaluate_bocd(name, w, cps, ts, **seg_kwargs):
    stream = BOCD(verbose=0, **seg_kwargs)
    profile, runtimes, found_cps, found_cps_dx = run_stream(stream, ts, aggregate_profile=np.min)

    f1_score = np.round(f_measure({0: cps}, found_cps, margin=int(ts.shape[0] * .01)), 3)
    covering_score = np.round(covering({0: cps}, found_cps, ts.shape[0]), 3)

    # print(f"{name}: Found Change Points: {found_cps}, F1-Score: {f1_score}, Covering-Score: {covering_score}")
    return name, cps.tolist(), found_cps, found_cps_dx, f1_score, covering_score, profile.tolist(), runtimes.tolist()


def evaluate_adwin(name, w, cps, ts, **seg_kwargs):
    stream = ADWIN(verbose=0, **seg_kwargs)
    profile, runtimes, found_cps, found_cps_dx = run_stream(stream, ts)

    f1_score = np.round(f_measure({0: cps}, found_cps, margin=int(ts.shape[0] * .01)), 3)
    covering_score = np.round(covering({0: cps}, found_cps, ts.shape[0]), 3)

    # print(f"{name}: Found Change Points: {found_cps}, F1-Score: {f1_score}, Covering-Score: {covering_score}")
    return name, cps.tolist(), found_cps, found_cps_dx, f1_score, covering_score, profile.tolist(), runtimes.tolist()


def evaluate_ddm(name, w, cps, ts, **seg_kwargs):
    stream = DDM(verbose=0, **seg_kwargs)
    profile, runtimes, found_cps, found_cps_dx = run_stream(stream, ts)

    f1_score = np.round(f_measure({0: cps}, found_cps, margin=int(ts.shape[0] * .01)), 3)
    covering_score = np.round(covering({0: cps}, found_cps, ts.shape[0]), 3)

    # print(f"{name}: Found Change Points: {found_cps}, F1-Score: {f1_score}, Covering-Score: {covering_score}")
    return name, cps.tolist(), found_cps, found_cps_dx, f1_score, covering_score, profile.tolist(), runtimes.tolist()


def evaluate_hddm(name, w, cps, ts, **seg_kwargs):
    stream = HDDM(verbose=0, **seg_kwargs)
    profile, runtimes, found_cps, found_cps_dx = run_stream(stream, ts)

    f1_score = np.round(f_measure({0: cps}, found_cps, margin=int(ts.shape[0] * .01)), 3)
    covering_score = np.round(covering({0: cps}, found_cps, ts.shape[0]), 3)

    # print(f"{name}: Found Change Points: {found_cps}, F1-Score: {f1_score}, Covering-Score: {covering_score}")
    return name, cps.tolist(), found_cps, found_cps_dx, f1_score, covering_score, profile.tolist(), runtimes.tolist()


def evaluate_candidate(candidate_name, dataset_name, eval_func, columns=None, n_jobs=1, verbose=0, **seg_kwargs):
    if dataset_name == "train":
        df_data = load_train_dataset()
    elif dataset_name == "benchmark":
        df_data = load_benchmark_dataset()
    else:
        df_data = load_dataset(dataset_name)

    df_res = dp.map(
        lambda _, args: eval_func(*args, **seg_kwargs),
        tqdm(list(df_data.iterrows()), disable=verbose<1),
        ret_type=list,
        verbose=0,
        n_jobs=n_jobs,
    )

    if columns is None:
        columns = ["dataset", "true_cps", "found_cps", "found_cps_dx", "f1_score", "covering_score", "profile", "runtimes"]

    df_res = pd.DataFrame.from_records(
        df_res,
        index="dataset",
        columns=columns,
    )

    print(f"{candidate_name}: mean_covering_score={np.round(df_res.covering_score.mean(), 3)}")
    return df_res