import numpy as np
import numpy.fft as fft

from numba import njit


def _sliding_dot(query, time_series):
    m = len(query)
    n = len(time_series)

    time_series_add = 0
    if n % 2 == 1:
        time_series = np.insert(time_series, 0, 0)
        time_series_add = 1

    q_add = 0
    if m % 2 == 1:
        query = np.insert(query, 0, 0)
        q_add = 1

    query = query[::-1]
    query = np.pad(query, (0, n - m + time_series_add - q_add), 'constant')
    trim = m - 1 + time_series_add
    dot_product = fft.irfft(fft.rfft(time_series) * fft.rfft(query))
    return dot_product[trim:]


@njit(fastmath=True, cache=True)
def argkmin(dist, k, lbound):
    args = np.zeros(shape=k, dtype=np.int64)
    vals = np.zeros(shape=k, dtype=np.float64)

    for idx in range(k):
        min_arg = np.nan
        min_val = np.inf

        for kdx in range(lbound, dist.shape[0]):
            val = dist[kdx]

            if val < min_val:
                min_val = val
                min_arg = kdx

        min_arg = np.int64(min_arg)

        args[idx] = min_arg
        vals[idx] = min_val

        dist[min_arg] = np.inf
        # dist[max(0, min_arg - excl_radius):min(min_arg + excl_radius, dist.shape[0]-1)] = np.inf

    dist[args] = vals
    return args


class TimeSeriesStream:

    def __init__(self, window_size, n_timepoints, k_neighbours=3, similarity="pearson"):
        self.window_size = window_size
        self.exclusion_radius = int(window_size/2) #

        self.n_timepoints = n_timepoints
        self.k_neighbours = k_neighbours
        self.similarity = similarity

        self.lbound = 0

        self.time_series = np.full(shape=n_timepoints, fill_value=np.nan, dtype=np.float64)

        self.l = n_timepoints - window_size + 1
        self.knn_insert_idx = self.l-self.exclusion_radius-self.k_neighbours-1

        self.csum = np.full(shape=n_timepoints+1, fill_value=0, dtype=np.float64)
        self.csumsq = np.full(shape=n_timepoints+1, fill_value=0, dtype=np.float64)
        self.dcsum = np.full(shape=n_timepoints+1, fill_value=0, dtype=np.float64)

        self.means = np.full(shape=self.l, fill_value=np.nan, dtype=np.float64)
        self.stds = np.full(shape=self.l, fill_value=np.nan, dtype=np.float64)

        self.dists = np.full(shape=(self.l, k_neighbours), fill_value=np.inf, dtype=np.float64)
        self.knns = np.full(shape=(self.l, k_neighbours), fill_value=-1, dtype=np.int64)

        self.dot_rolled = None

        self.fill = 0
        self.knn_fill = 0

    def mean(self, idx):
        window_sum = self.csum[idx+self.window_size] - self.csum[idx]
        return window_sum / self.window_size

    def std(self, idx):
        window_sum = self.csum[idx + self.window_size] - self.csum[idx]
        window_sum_sq = self.csumsq[idx + self.window_size] - self.csumsq[idx]

        movstd = window_sum_sq / self.window_size - (window_sum / self.window_size) ** 2
        movstd = np.sqrt(movstd)

        # avoid dividing by too small std, like 0
        if abs(movstd) < 1e-3:
            movstd = 1

        return movstd

    def knn(self):
        idx = self.knn_insert_idx

        start_idx = self.l - (self.fill - self.window_size + 1)
        valid_dist = slice(self.l-(self.fill - self.window_size + 1), self.l)
        dist = np.full(shape=self.l, fill_value=np.inf, dtype=np.float64)

        if self.dot_rolled is None:
            self.dot_rolled = np.full(shape=self.l, fill_value=np.inf, dtype=np.float64)
            self.dot_rolled[valid_dist] = _sliding_dot(self.time_series[idx:idx+self.window_size], self.time_series[-self.fill:])
        else:
            self.dot_rolled += self.time_series[idx + self.window_size - 1] * self.time_series[self.window_size - 1:]
            self.dot_rolled[start_idx] = np.dot(self.time_series[start_idx:start_idx + self.window_size], self.time_series[idx:idx + self.window_size])

        rolled_dist = None

        # z-normed ed is squared (we are actually using z-normed ED instead of pearson, which is the same)
        if self.similarity == "pearson":
            rolled_dist = 2 * self.window_size * (1 - (self.dot_rolled - self.window_size * self.means * self.means[idx]) / (self.window_size * self.stds * self.stds[idx]))

        # ed is squared
        if self.similarity == "ed":
            csumsq = self.csumsq[self.window_size:] - self.csumsq[:-self.window_size]
            rolled_dist = -2 * self.dot_rolled + csumsq + csumsq[idx]

        # cid is squared
        if self.similarity == "cid":
            csumsq = self.csumsq[self.window_size:] - self.csumsq[:-self.window_size]
            ed = -2 * self.dot_rolled + csumsq + csumsq[idx] # dist is squared
            ce = self.dcsum[self.window_size:] - self.dcsum[:-self.window_size] + 1e-5 # add some noise to prevent zero divisons

            last_ce = np.repeat(ce[idx], ce.shape[0])
            cf = (np.max(np.dstack((ce, last_ce)), axis=2) / np.min(np.dstack((ce, last_ce)), axis=2))[0]

            rolled_dist = ed * cf

        dist[valid_dist] = rolled_dist[valid_dist]

        excl_range = slice(max(0, idx-self.exclusion_radius), min(idx+self.exclusion_radius, self.l)) #
        dist[excl_range] = np.max(dist)
        knns = argkmin(dist, self.k_neighbours, self.lbound)

        # update dot product
        self.dot_rolled -= self.time_series[idx] * self.time_series[:self.l]

        return dist, knns

    def update(self, timepoint):
        # log how much data is ingested
        self.fill = min(self.fill + 1, self.l)

        # update time series
        self.time_series = np.roll(self.time_series, -1)
        self.time_series[-1] = timepoint

        # update cum sum
        self.csum = np.roll(self.csum, -1)
        self.csum[-1] = self.csum[-2] + timepoint

        # update cum sum squared
        self.csumsq = np.roll(self.csumsq, -1)
        self.csumsq[-1] = self.csumsq[-2] + timepoint**2

        # update diff cum sum
        if self.fill > 1:
            self.dcsum = np.roll(self.dcsum, -1)
            self.dcsum[-1] = self.dcsum[-2] + np.square(timepoint - self.time_series[-2])

        if self.fill < self.window_size:
            return self

        # update means
        self.means = np.roll(self.means, -1)
        self.means[-1] = self.mean(len(self.time_series) - self.window_size)

        # update stds
        self.stds = np.roll(self.stds, -1)
        self.stds[-1] = self.std(len(self.time_series) - self.window_size)

        if self.fill < self.window_size + self.exclusion_radius + self.k_neighbours:
            return self

        # update knn
        self._update_knn()

        return self

    def _update_knn(self):
        # roll existing indices further
        if self.knn_fill > 0:
            self.dists = np.roll(self.dists, -1, axis=0)
            self.dists[-1,:] = np.full(shape=self.dists.shape[1], fill_value=np.inf, dtype=np.float64)

            self.knns = np.roll(self.knns, -1, axis=0)
            self.knns[self.knn_insert_idx-self.knn_fill:self.knn_insert_idx] -= 1
            self.knns[-1, :] = np.full(shape=self.dists.shape[1], fill_value=-1, dtype=np.float64)

        # insert new distances and knns
        dist, knn = self.knn()

        self.dists[self.knn_insert_idx,:] = dist[knn]
        self.knns[self.knn_insert_idx,:] = knn

        idx = np.arange(self.knn_insert_idx - self.knn_fill, self.knn_insert_idx)

        change_mask = np.full(shape=self.l, fill_value=True, dtype=np.bool)

        for kdx in range(self.k_neighbours - 1):
            change_idx = dist[idx] < self.dists[idx, kdx]
            change_idx = np.logical_and(change_idx, change_mask[idx])
            change_idx = idx[change_idx]

            change_mask[change_idx] = False

            self.knns[change_idx, kdx+1:] = self.knns[change_idx, kdx:self.k_neighbours-1]
            self.knns[change_idx, kdx] = self.knn_insert_idx

            self.dists[change_idx, kdx+1:] = self.dists[change_idx, kdx:self.k_neighbours-1]
            self.dists[change_idx, kdx] = dist[change_idx]

        # decrease lbound
        self.lbound = max(0, self.lbound-1)

        # log how much knn data is ingested
        self.knn_fill = min(self.knn_fill + 1, self.knn_insert_idx)

        return self











