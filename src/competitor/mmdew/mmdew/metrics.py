import numpy as np
import pandas as pd

"""In all functions, we assume that `true_cps` and `reported_cps` are sorted from lowest to highest."""


def percent_changes_detected(true_cps, reported_cps):
    return ratio_changes_detected(true_cps, reported_cps) * 100


def ratio_changes_detected(true_cps, reported_cps):
    if len(true_cps) == 0:
        return np.nan
    return len(reported_cps) / len(true_cps)


def mean_until_detection(true_cps, reported_cps):
    reported_cps = reported_cps.copy()
    dist = 0
    detected_cps = 0
    for cpi, true_cp in enumerate(true_cps):
        next_cp = true_cps[cpi + 1] if cpi < len(true_cps) - 1 else np.infty
        for reported_cp in reported_cps:
            if reported_cp <= true_cp:
                continue
            if reported_cp >= next_cp:
                continue
            dist += reported_cp - true_cp
            detected_cps += 1
            reported_cps.remove(reported_cp)
            break
    return dist / detected_cps if detected_cps > 0 else np.nan


def mean_cp_detection_time_error(true_cps, reported_cps, reported_detection_delays):
    reported_cps = reported_cps.copy()
    delay_errors = []
    for cpi, true_cp in enumerate(true_cps):
        next_cp = true_cps[cpi + 1] if cpi < len(true_cps) - 1 else np.infty
        for rcpi, reported_cp in enumerate(reported_cps):
            if reported_cp <= true_cp:
                continue
            if reported_cp >= next_cp:
                continue
            delay = reported_detection_delays[rcpi]
            if pd.isna(delay):
                delay_errors.append(np.nan)
            else:
                error = reported_cp - true_cp - delay
                delay_errors.append(abs(error))
                reported_cps.remove(reported_cp)
            break
    return np.nanmean(delay_errors)


def true_positives(true_cps, reported_cps, T=10):
    true_cps = true_cps.copy()
    tps = 0
    for reported_cp in reported_cps:
        for true_cp in true_cps:
            if abs(true_cp - reported_cp) < T and reported_cp >= true_cp:
                tps += 1
                true_cps.remove(true_cp)
                break
    return tps


def false_positives(true_cps, reported_cps, T=10):
    tps = true_positives(true_cps, reported_cps, T)
    return len(reported_cps) - tps


def false_negatives(true_cps, reported_cps, T=10):
    reported_cps = reported_cps.copy()
    fns = len(true_cps)
    for true_cp in true_cps:
        for reported_cp in reported_cps:
            if abs(true_cp - reported_cp) < T and reported_cp >= true_cp:
                fns -= 1
                reported_cps.remove(reported_cp)
                break
    return fns


def precision(tp, fp, fn):
    if tp + fp == 0:
        return np.nan
    return tp / (tp + fp)


def recall(tp, fp, fn):
    if tp + fn == 0:
        return np.nan
    return tp / (tp + fn)


def jaccard(a, b):
    union = np.union1d(a, b)
    intersect = np.intersect1d(a, b)
    if len(union) == 0:
        return np.nan
    return len(intersect) / len(union)


def prec_full(true_cps, reported_cps, T=10):
    tps = true_positives(true_cps, reported_cps, T)
    fps = false_positives(true_cps, reported_cps, T)
    fns = false_negatives(true_cps, reported_cps, T)
    return precision(tps, fps, fns)


def rec_full(true_cps, reported_cps, T=10):
    tps = true_positives(true_cps, reported_cps, T)
    fps = false_positives(true_cps, reported_cps, T)
    fns = false_negatives(true_cps, reported_cps, T)
    return recall(tps, fps, fns)


def fb_score(true_cps, reported_cps, T=10, beta=1):
    tps = true_positives(true_cps, reported_cps, T)
    fps = false_positives(true_cps, reported_cps, T)
    fns = false_negatives(true_cps, reported_cps, T)
    prec = precision(tps, fps, fns)
    rec = recall(tps, fps, fns)
    if prec == 0:
        return np.nan
    return (1 + beta**2) * (prec * rec) / ((beta**2 * prec) + rec)


def test_true_positives():
    true_cps = [99, 200, 400]
    reported_cps = [100, 150, 200, 250, 300, 400, 500]
    assert true_positives(true_cps, reported_cps) == 3


def test_true_positives2():
    true_cps = [99, 200, 400]
    reported_cps = [50, 400]
    assert true_positives(true_cps, reported_cps) == 1


def test_true_positives3():
    true_cps = [99, 102]
    reported_cps = [50, 100, 200]
    assert true_positives(true_cps, reported_cps) == 1


def test_true_positives4():
    true_cps = [70, 102]
    reported_cps = [100, 101]
    assert true_positives(true_cps, reported_cps) == 0


def test_false_negatives():
    true_cps = [99, 102]
    reported_cps = [50]
    assert false_negatives(true_cps, reported_cps) == 2


def test_false_negatives2():
    true_cps = [99, 102]
    reported_cps = [50, 100]
    assert false_negatives(true_cps, reported_cps) == 1


def test_false_negatives3():
    true_cps = [102]
    reported_cps = [50, 100, 101]
    assert false_negatives(true_cps, reported_cps) == 1


def test_fb():
    true_cps = [102]
    reported_cps = [50, 100, 101]
    tps = true_positives(true_cps, reported_cps)
    fps = false_positives(true_cps, reported_cps)
    fns = false_negatives(true_cps, reported_cps)
    assert np.isnan(fb_score(true_cps, reported_cps))


def test_fb2():
    true_cps = [102]
    reported_cps = [50, 100, 102]
    tps = true_positives(true_cps, reported_cps)
    fps = false_positives(true_cps, reported_cps)
    fns = false_negatives(true_cps, reported_cps)
    assert fb_score(true_cps, reported_cps) == tps / (tps + 0.5 * (fps + fns))


def test_mean_until_detection():
    true_cps = [100]
    reported_cps = [101]
    assert mean_until_detection(true_cps, reported_cps) == 1


def test_mean_until_detection2():
    true_cps = [100, 200]
    reported_cps = [101, 204]
    assert mean_until_detection(true_cps, reported_cps) == 5 / 2


def test_mean_until_detection3():
    true_cps = [100, 200]
    reported_cps = [101, 150, 160, 180, 210]
    assert mean_until_detection(true_cps, reported_cps) == 11 / 2


def test_jaccard1():
    a = [1, 2, 3]
    b = a
    assert jaccard(a, b) == 1


def test_jaccard2():
    a = np.array([1, 2, 3])
    b = -a
    assert jaccard(a, b) == 0
