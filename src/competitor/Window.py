import numpy as np

from tqdm import tqdm
from ruptures.costs import cost_factory


class Window:

    def __init__(self, n_timepoints, window_size, cost_func, threshold, excl_factor=5, verbose=0):
        self.window_size = window_size
        self.n_timepoints = n_timepoints
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

        self.sliding_window = np.full(shape=self.n_timepoints, fill_value=-np.inf, dtype=np.float64)

    def _prerun(self, timepoint):
        # log p_bar if verbose > 0
        if self.p_bar is not None: self.p_bar.update(1)

        # log how much data was ingested
        self.ingested += 1

        # close p_bar if ingested == verbose
        if self.ingested == self.verbose and self.verbose > 1: self.p_bar.close()

        # update prerun ts
        self.prerun_counter += 1

        self.sliding_window = np.roll(self.sliding_window, -1)
        self.sliding_window[-1] = timepoint

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

        self.profile = np.roll(self.profile, -1)
        self.profile[-1] = -np.inf

        self.cost_func.fit(self.sliding_window[self.n_timepoints-2*self.window_size:])

        start, middle, end = 0, self.window_size, 2*self.window_size
        gain = self.cost_func.error(start, end)
        idx = self.n_timepoints - self.window_size

        # assert that the gain is not NaN
        if gain == gain:
            self.profile[idx] = gain - (self.cost_func.error(start, middle) + self.cost_func.error(middle, end))

        if self.profile[idx] < self.threshold:
            return self.profile

        global_pos = self.ingested - self.window_size

        if len(self.change_points) == 0:
            self.change_points.append(global_pos)
            self.scores.append(self.profile[idx])
            return self.profile

        if np.abs(self.change_points[-1] - global_pos) > self.excl_factor * self.window_size:
            self.change_points.append(global_pos)
            self.scores.append(self.profile[idx])

        return self.profile

    def update(self, timepoint):
        if self.prerun_counter < 2*self.window_size-1:
            return self._prerun(timepoint)

        return self._run(timepoint)