import numpy as np

from tqdm import tqdm
from ruptures.costs import cost_factory

import ruptures as rpt


def window(ts, window_size, cost_func="mahalanobis", n_cps=None, offset=0.05):
    transformer = rpt.Window(width=max(window_size, 3), model=cost_func, min_size=int(ts.shape[0] * offset)).fit(ts)
    return np.array(transformer.predict(n_bkps=n_cps)[:-1], dtype=np.int64)


class Window:

    def __init__(self, n_timepoints, cost_func, threshold, jump=5, excl_factor=.05, verbose=0):
        self.n_timepoints = n_timepoints
        self.cost_func = cost_factory(model=cost_func)
        self.threshold = threshold
        self.jump = jump
        self.excl_factor = excl_factor
        self.verbose = verbose

        self.profile = np.full(shape=n_timepoints, fill_value=-np.inf, dtype=np.float64)
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

        self.cost_func.fit(self.sliding_window)
        self.profile = np.full(shape=self.n_timepoints, fill_value=-np.inf, dtype=np.float64)

        start, end = 0, self.n_timepoints
        gain = self.cost_func.error(start, end)

        excl_zone = int(self.excl_factor * self.n_timepoints)

        for idx in np.arange(excl_zone, self.profile.shape[0] - excl_zone, step=self.jump):
            if gain != gain:
                self.profile[idx] = 0
                continue
            self.profile[idx] = gain - (self.cost_func.error(start, idx) + self.cost_func.error(idx, end))

        cp = np.argmax(self.profile)

        if self.profile[cp] < self.threshold:
            return self.profile

        global_pos = self.ingested - self.n_timepoints + cp

        if len(self.change_points) == 0:
            self.change_points.append(global_pos)
            self.scores.append(self.profile[cp])
            return self.profile

        if np.abs(self.change_points[-1] - global_pos) > self.excl_factor * self.n_timepoints:
            self.change_points.append(global_pos)
            self.scores.append(self.profile[cp])

        return self.profile

    def update(self, timepoint):
        if self.prerun_counter < self.n_timepoints-1:
            return self._prerun(timepoint)

        return self._run(timepoint)