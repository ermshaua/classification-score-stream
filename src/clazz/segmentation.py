import logging

import numpy as np
from tqdm import tqdm

from src.clazz.knn import TimeSeriesStream
from src.clazz.penalty import rank_sums_test
from src.clazz.profile import binary_f1_score
from src.clazz.profile import calc_class
from src.clazz.window_size import suss


class ClaSS:

    def __init__(self, n_timepoints=10_000, n_prerun=None, window_size=suss,
                 k_neighbours=3, score=binary_f1_score, jump=5, p_value=1e-50,
                 sample_size=1_000, similarity="pearson",
                 profile_mode="global", verbose=0):
        if n_prerun is None: n_prerun = n_timepoints

        self.n_timepoints = n_timepoints
        self.n_prerun = n_prerun if callable(window_size) else 1
        self.window_size = window_size
        self.k_neighbours = k_neighbours
        self.score = score
        self.jump = jump
        self.p_value = p_value
        self.sample_size = sample_size
        self.similarity = similarity
        self.profile_mode = profile_mode
        self.verbose = verbose

        if verbose == 1:
            self.p_bar = tqdm()
        elif verbose > 1:
            self.p_bar = tqdm(total=verbose)
        else:
            self.p_bar = None

        self.prerun_ts = np.full(shape=self.n_prerun, fill_value=-np.inf, dtype=np.float64)
        self.profile = np.full(shape=n_timepoints, fill_value=-np.inf, dtype=np.float64)

        self.change_points = list()
        self.local_change_points = list()
        self.scores = list()
        self.ps = list()

        self.last_cp = 0

        self.ingested = 0
        self.ts_stream_lag = 0
        self.lag = -1
        self.prerun_counter = 0


    def prerun(self, timepoint):
        # update prerun ts
        self.prerun_counter += 1

        self.prerun_ts = np.roll(self.prerun_ts, -1)
        self.prerun_ts[-1] = timepoint

        if self.prerun_counter != self.n_prerun:
            return self.profile

        # determine window size
        if callable(self.window_size):
            self.window_size = self.window_size(self.prerun_ts)

        # determine jump size
        if self.jump is None:
            self.jump = int(self.window_size/2)

        if self.verbose > 0:
            logging.info(f"Using window size: {self.window_size}")
            # logging.info(f"Using similarity: {self.similarity}")

        self.min_seg_size = 5 * self.window_size

        # create ts stream
        self.ts_stream = TimeSeriesStream(self.window_size, self.n_timepoints, self.k_neighbours, self.similarity)
        self.ts_stream_lag = self.ts_stream.window_size + self.ts_stream.exclusion_radius + self.ts_stream.k_neighbours

        # update ts stream with prerun
        for timepoint in self.prerun_ts:
            self.run(timepoint)

        return self.profile

    def run(self, timepoint):
        # log p_bar if verbose > 0
        if self.p_bar is not None: self.p_bar.update(1)

        # log how much data was ingested
        self.ingested += 1

        # close p_bar if ingested == verbose
        if self.ingested == self.verbose and self.verbose > 1: self.p_bar.close()

        # update time series stream (and knn)
        self.ts_stream.lbound = self.ts_stream.knn_insert_idx - self.ts_stream.knn_fill + 1 + self.last_cp
        self.ts_stream.update(timepoint)

        # update profile stream
        self.profile = np.roll(self.profile, -1, axis=0)
        self.profile[-1] = -np.inf

        if self.ingested < self.min_seg_size * 2:
            return self.profile

        if self.ts_stream.knn_insert_idx - self.ts_stream.knn_fill == 0:
            self.last_cp = max(0, self.last_cp - 1)

        profile_start, profile_end = self.ts_stream.lbound, self.ts_stream.knn_insert_idx

        if profile_end - profile_start < 2 * self.min_seg_size or self.ingested % self.jump != 0:
            return self.profile

        offset = self.min_seg_size
        profile, knn = calc_class(self.ts_stream, self.score, offset, return_knn=True)

        not_ninf = np.logical_not(profile == -np.inf)

        tc = profile[not_ninf].shape[0] / self.n_timepoints
        profile[not_ninf] = (2 * profile[not_ninf] + tc) / 3

        cp, score = np.argmax(profile) + self.window_size, np.max(profile)

        if cp < offset or profile.shape[0] - cp < offset:
            return self.profile

        if profile[cp:-offset].shape[0] == 0:
            return self.profile

        if self.profile_mode == "global":
            self.profile[profile_start:profile_end] = np.max([profile, self.profile[profile_start:profile_end]], axis=0)
        elif self.profile_mode == "local":
            self.profile[profile_start:profile_end] = profile

        p, passed = rank_sums_test(knn, cp, self.window_size, sample_size=self.sample_size, threshold=self.p_value) #

        if passed:
            global_cp = self.ingested - self.ts_stream_lag - (profile_end - profile_start) + cp
            self.change_points.append(global_cp)
            self.local_change_points.append(cp)
            self.scores.append(score)
            self.ps.append(p)
            self.last_cp += self.local_change_points[-1]

        return self.profile

    def update(self, timepoint):
        if self.prerun_counter < self.n_prerun:
            return self.prerun(timepoint)

        return self.run(timepoint)


