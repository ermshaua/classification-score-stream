import numpy as np
from onlinecp import algos

from tqdm import tqdm


class NEWMA:

    def __init__(self, n_timepoints=10_000, window_size=100, thresholding_quantile=1., excl_factor=5, verbose=0):
        self.n_timepoints = n_timepoints
        self.excl_factor = excl_factor
        self.verbose = verbose

        self.newma = algos.NEWMA(np.zeros(excl_factor * window_size), thresholding_quantile=thresholding_quantile)

        self.profile = np.full(shape=self.n_timepoints, fill_value=np.inf, dtype=np.float64)
        self.change_points = []
        self.scores = []

        if verbose == 1:
            self.p_bar = tqdm()
        elif verbose > 1:
            self.p_bar = tqdm(total=verbose)
        else:
            self.p_bar = None

        self.ingested = 0

        self.sliding_window = np.full(shape=self.n_timepoints, fill_value=np.inf, dtype=np.float64)

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

        self.profile[-1] = self.newma.update(timepoint)

        if not self.newma.flag_sample():
            return self.profile

        global_pos = self.ingested

        if len(self.change_points) == 0:
            self.change_points.append(global_pos)
            self.scores.append(self.profile[-1])
            return self.profile

        if np.abs(self.change_points[-1] - global_pos) > self.excl_factor:
            self.change_points.append(global_pos)
            self.scores.append(self.profile[-1])

        return self.profile

    def update(self, timepoint):
        return self._run(timepoint)
