import numpy as np

from tqdm import tqdm
from stumpy import stump
from stumpy.floss import floss, _cac

class FLOSS:

    def __init__(self, n_timepoints, window_size, threshold, n_prerun=None, excl_factor=5, verbose=0):
        if n_prerun is None: n_prerun = n_timepoints

        self.n_timepoints = n_timepoints
        self.window_size = window_size
        self.threshold = threshold
        self.excl_factor = excl_factor
        self.verbose = verbose

        self.profile = np.full(shape=n_timepoints-window_size+1, fill_value=np.inf, dtype=np.float64)
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
        self.prerun_ts = np.full(shape=self.n_prerun, fill_value=np.inf, dtype=np.float64)

    def _prerun(self, timepoint):
        # log p_bar if verbose > 0
        if self.p_bar is not None: self.p_bar.update(1)

        # log how much data was ingested
        self.ingested += 1

        # close p_bar if ingested == verbose
        if self.ingested == self.verbose and self.verbose > 1: self.p_bar.close()

        # update prerun ts
        self.prerun_counter += 1

        self.prerun_ts = np.roll(self.prerun_ts, -1)
        self.prerun_ts[-1] = timepoint

        if self.prerun_counter != self.n_prerun:
            return self.profile

        mp = stump(self.prerun_ts, m=max(self.window_size, 3))

        self.stream = floss(
            mp,
            self.prerun_ts,
            m=self.window_size,
            L=self.window_size,
            excl_factor=self.excl_factor
        )

        # the cac is not computed yet
        self.profile = _cac(
            self.stream._mp[:, 3] - self.stream._n_appended - 1,
            self.stream._L,
            bidirectional=False,
            excl_factor=self.stream._excl_factor,
            custom_iac=self.stream._custom_iac,
        )

        # extract CPs from the first batch if present
        profile = np.copy(self.profile)

        while profile.min() <= self.threshold:
            cp = np.argmin(profile)
            self.change_points.append(cp)
            self.scores.append(profile[cp])
            profile[cp - self.excl_factor * self.window_size:cp + self.excl_factor * self.window_size] = np.inf

        return self.profile

    def _run(self, timepoint):
        # log p_bar if verbose > 0
        if self.p_bar is not None: self.p_bar.update(1)

        # log how much data was ingested
        self.ingested += 1

        # close p_bar if ingested == verbose
        if self.ingested == self.verbose and self.verbose > 1: self.p_bar.close()

        self.stream.update(timepoint)
        self.profile = self.stream.cac_1d_

        cp = np.argmin(self.profile)

        if self.profile[cp] > self.threshold:
            return self.profile

        global_pos = self.ingested - self.n_prerun + cp

        if len(self.change_points) == 0:
            self.change_points.append(global_pos)
            self.scores.append(self.profile[cp])
            return self.profile

        if np.abs(self.change_points[-1] - global_pos) > self.excl_factor * self.window_size:
            self.change_points.append(global_pos)
            self.scores.append(self.profile[cp])

        return self.profile

    def update(self, timepoint):
        if self.prerun_counter < self.n_prerun:
            return self._prerun(timepoint)

        return self._run(timepoint)