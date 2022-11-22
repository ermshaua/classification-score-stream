import time

import numpy as np
import pandas as pd
import daproli as dp

from src.utils import load_dataset
from src.clazz.segmentation import ClaSS
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

        profile[max(0, dx-window_profile.shape[0]):dx] = aggregate_profile([
            profile[max(0, dx-window_profile.shape[0]):dx],
            window_profile[max(0, window_profile.shape[0]-dx):]
        ], axis=0)

        # store runtime
        runtimes[dx] = runtime

        # store CPs
        if len(stream.change_points) > len(found_cps):
            found_cps.append(stream.change_points[-1])
            found_cps_dx.append(dx)

    if profile.shape[0] < ts.shape[0]:
        profile = np.hstack((np.full(shape=ts.shape[0]-profile.shape[0], fill_value=np.min(profile), dtype=np.float64), profile))

    if interpolate_profile is True:
        profile[np.isinf(profile)] = np.nan
        profile = pd.Series(profile).interpolate(limit_direction="both").to_numpy()

    return profile, np.round(runtimes, 5), found_cps, found_cps_dx


def evaluate_class(name, _, cps, ts, **seg_kwargs):
    if "n_timepoints" in seg_kwargs:
        n_prerun = min(seg_kwargs["n_prerun"], ts.shape[0])
    else:
        n_prerun = min(10_000, ts.shape[0])

    stream = ClaSS(n_prerun=n_prerun, verbose=0, **seg_kwargs)
    profile, runtimes, found_cps, found_cps_dx = run_stream(stream, ts, aggregate_profile=np.max)

    f1_score = np.round(f_measure({0: cps}, found_cps, margin=int(ts.shape[0] * .01)), 3)
    covering_score = np.round(covering({0: cps}, found_cps, ts.shape[0]), 3)

    print(f"{name}: F1-Score: {f1_score}, Covering-Score: {covering_score}")
    return name, cps.tolist(), found_cps, found_cps_dx, f1_score, covering_score, profile.tolist(), runtimes.tolist(), ts.tolist()


def evaluate_candidate(candidate_name, dataset_name, eval_func, columns=None, n_jobs=1, verbose=0, **seg_kwargs):
    if dataset_name == "train":
        pass
        # todo: implent train data set
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
        columns = ["dataset", "true_cps", "found_cps", "found_cps_dx", "f1_score", "covering_score", "profile", "runtimes", "time_series"]

    df_res = pd.DataFrame.from_records(
        df_res,
        index="dataset",
        columns=columns,
    )

    print(f"{candidate_name}: mean_covering_score={np.round(df_res.covering_score.mean(), 3)}")

    return df_res