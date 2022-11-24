import numpy as np
import pandas as pd

from statsmodels.tsa.stattools import acf
from scipy.signal import find_peaks

import math


def dominant_fourier_freq(ts, min_size=10, max_size=1000): #
    fourier = np.fft.fft(ts)
    freq = np.fft.fftfreq(ts.shape[0], 1)

    magnitudes = []
    window_sizes = []

    for coef, freq in zip(fourier, freq):
        if coef and freq > 0:
            window_size = int(1 / freq)
            mag = math.sqrt(coef.real * coef.real + coef.imag * coef.imag)

            if window_size >= min_size and window_size < max_size:
                window_sizes.append(window_size)
                magnitudes.append(mag)

    return window_sizes[np.argmax(magnitudes)]


def highest_autocorrelation(ts, min_size=10, max_size=1000):
    acf_values = acf(ts, fft=True, nlags=int(ts.shape[0]/2))

    peaks, _ = find_peaks(acf_values)
    peaks = peaks[np.logical_and(peaks >= min_size, peaks < max_size)]
    corrs = acf_values[peaks]

    if peaks.shape[0] == 0:
        return -1

    return peaks[np.argmax(corrs)]


def suss_score(time_series, window_size, stats):
    roll = pd.Series(time_series).rolling(window_size)
    ts_mean, ts_std, ts_min_max = stats

    roll_mean = roll.mean().to_numpy()[window_size:]
    roll_std = roll.std(ddof=0).to_numpy()[window_size:]
    roll_min = roll.min().to_numpy()[window_size:]
    roll_max = roll.max().to_numpy()[window_size:]

    X = np.array([
        roll_mean - ts_mean,
        roll_std - ts_std,
        (roll_max - roll_min) - ts_min_max
    ])

    X = np.sqrt(np.sum(np.square(X), axis=0)) / np.sqrt(window_size)
    return np.mean(X)


def suss(time_series, lbound=10, threshold=.89):
    if time_series.max() == time_series.min(): return lbound
    time_series = (time_series - time_series.min()) / (time_series.max() - time_series.min())

    ts_mean = np.mean(time_series)
    ts_std = np.std(time_series)
    ts_min_max = np.max(time_series) - np.min(time_series)

    stats = (ts_mean, ts_std, ts_min_max)

    max_score = suss_score(time_series, 1, stats)
    min_score = suss_score(time_series, time_series.shape[0]-1, stats)

    exp = 0

    # exponential search (to find window size interval)
    while True:
        window_size = 2 ** exp

        if window_size < lbound:
            exp += 1
            continue

        score = 1 - (suss_score(time_series, window_size, stats) - min_score) / (max_score - min_score)

        if score > threshold:
            break

        exp += 1

    lbound, ubound = max(lbound, 2 ** (exp - 1)), 2 ** exp + 1

    # binary search (to find window size in interval)
    while lbound <= ubound:
        window_size = int((lbound + ubound) / 2)
        score = 1 - (suss_score(time_series, window_size, stats) - min_score) / (max_score - min_score)

        if score < threshold:
            lbound = window_size+1
        elif score > threshold:
            ubound = window_size-1
        else:
            break

    return 2*lbound


def moving_mean(ts, w):
    moving_avg = np.cumsum(ts, dtype=float)
    moving_avg[w:] = moving_avg[w:] - moving_avg[:-w]
    return moving_avg[w - 1:] / w


def mwf(ts, lbound=10, ubound=1_000):
    all_averages = []
    window_sizes = []

    for w in range(lbound, ubound, 1):
        movingAvg = np.array(moving_mean(ts, w))
        all_averages.append(movingAvg)
        window_sizes.append(w)

    movingAvgResiduals = []

    for i, w in enumerate(window_sizes):
        moving_avg = all_averages[i][:len(all_averages[-1])]
        movingAvgResidual = np.log(abs(moving_avg - (moving_avg).mean()).sum())
        movingAvgResiduals.append(movingAvgResidual)

    # local min
    b = (np.diff(np.sign(np.diff(movingAvgResiduals))) > 0).nonzero()[0] + 1

    if len(b) == 0: return -1
    if len(b) < 3: return window_sizes[b[0]]

    reswin = np.array([window_sizes[b[i]] / (i + 1) for i in range(3)])
    w = np.mean(reswin)

    return int(w)