import numpy as np

from tqdm import tqdm
from ruptures.costs import cost_factory
from src.clazz.window_size import suss


class Window:

    def __init__(self, n_timepoints=10_000, window_size=suss, cost_func=None, n_prerun=None, threshold=None, excl_factor=5, verbose=0):
        if n_prerun is None: n_prerun = n_timepoints

        self.n_timepoints = n_timepoints
        self.window_size = window_size
        self.cost_func = cost_factory(model=cost_func)
        self.threshold = threshold
        self.excl_factor = excl_factor
        self.verbose = verbose

        self.profile = np.full(shape=self.n_timepoints, fill_value=-np.inf, dtype=np.float64)
        self.change_points = []
        self.scores = []

        if verbose == 1:
            self.p_bar = tqdm()
        elif verbose > 1:
            self.p_bar = tqdm(total=verbose)
        else:
            self.p_bar = None

        self.prerun_counter = 0
        self.ingested = 0

        self.n_prerun = n_prerun
        self.prerun_ts = np.full(shape=self.n_prerun, fill_value=-np.inf, dtype=np.float64)

        self.sliding_window = np.full(shape=self.n_timepoints, fill_value=-np.inf, dtype=np.float64)

    def _prerun(self, timepoint):
        # update prerun ts
        self.prerun_counter += 1

        self.prerun_ts = np.roll(self.prerun_ts, -1)
        self.prerun_ts[-1] = timepoint

        if self.prerun_counter != self.n_prerun:
            return self.profile

        # determine window size
        if callable(self.window_size):
            self.window_size = self.excl_factor * self.window_size(self.prerun_ts)

        # update ts stream with prerun
        for timepoint in self.prerun_ts:
            self._run(timepoint)

        return self.profile

    def _run(self, timepoint):
        # log p_bar if verbose > 0
        if self.p_bar is not None: self.p_bar.update(1)

        # log how much data was ingested
        self.ingested += 1

        # close p_bar if ingested == verbose
        if self.ingested == self.verbose and self.verbose > 1: self.p_bar.close()

        self.sliding_window = np.roll(self.sliding_window, -1)
        self.sliding_window[-1] = timepoint

        if self.ingested < 2*self.window_size:
            return self.profile

        self.profile = np.roll(self.profile, -1)
        self.profile[-1] = -np.inf

        self.cost_func.fit(self.sliding_window[self.n_timepoints-2*self.window_size:])

        start, middle, end = 0, self.window_size, 2*self.window_size
        cost_all = self.cost_func.error(start, end) / self.window_size

        idx = self.n_timepoints - self.window_size

        # assert that the gain is not NaN
        if cost_all == cost_all:
            cost_seg = (self.cost_func.error(start, middle) + self.cost_func.error(middle, end)) / self.window_size
            res = cost_all - cost_seg
            self.profile[idx] = res

        if self.profile[idx] < self.threshold:
            return self.profile

        global_pos = self.ingested - self.window_size

        if len(self.change_points) == 0:
            self.change_points.append(global_pos)
            self.scores.append(self.profile[idx])
            return self.profile

        if all(np.abs(cp_ - global_pos) > self.window_size for cp_ in self.change_points):
            self.change_points.append(global_pos)
            self.scores.append(self.profile[idx])

        return self.profile

    def update(self, timepoint):
        if self.prerun_counter < self.n_prerun:
            return self._prerun(timepoint)

        return self._run(timepoint)