import numpy as np
import numpy.fft as fft

from numba import njit, objmode, bool_


@njit(fastmath=True, cache=True)
def rolling_knn(dists, knns, dist, knn, knn_insert_idx, knn_fill, l, k_neighbours, lbound):
    dists[knn_insert_idx, :] = dist[knn]
    knns[knn_insert_idx, :] = knn

    idx = np.arange(knn_insert_idx - knn_fill, knn_insert_idx)
    change_mask = np.full(shape=l, fill_value=True, dtype=bool_)

    for kdx in range(k_neighbours - 1):
        change_idx = dist[idx] < dists[idx, kdx]
        change_idx = np.logical_and(change_idx, change_mask[idx])
        change_idx = idx[change_idx]

        change_mask[change_idx] = False

        knns[change_idx, kdx + 1:] = knns[change_idx,
                                          kdx:k_neighbours - 1]
        knns[change_idx, kdx] = knn_insert_idx

        dists[change_idx, kdx + 1:] = dists[change_idx,
                                           kdx:k_neighbours - 1]
        dists[change_idx, kdx] = dist[change_idx]

    # decrease lbound
    lbound = max(0, lbound - 1)

    # log how much knn data is ingested
    knn_fill = min(knn_fill + 1, knn_insert_idx)

    return knns, dists, lbound, knn_fill


# @njit(fastmath=True, cache=True)
def knn(knn_insert_idx, l, fill,
        window_size, dot_rolled,
        time_series, means, stds, similarity,
        csumsq, dcsum, exclusion_radius, k_neighbours, lbound):
    idx = knn_insert_idx

    start_idx = l - (fill - window_size + 1)
    valid_dist = slice(l - (fill - window_size + 1), l)
    dist = np.full(shape=l, fill_value=np.inf, dtype=np.float64)

    if dot_rolled is None:
        dot_rolled = np.full(shape=l, fill_value=np.inf, dtype=np.float64)
        dot_rolled[valid_dist] = _sliding_dot(
            time_series[idx:idx + window_size],
            time_series[-fill:])
    else:
        dot_rolled = dot_rolled + time_series[idx + window_size - 1] * time_series[window_size - 1:]
        dot_rolled[start_idx] = np.dot(
            time_series[start_idx:start_idx + window_size],
            time_series[idx:idx + window_size])

    rolled_dist = None

    # z-normed ed is squared (we are actually using z-normed ED instead of pearson, which is the same)
    if similarity == "pearson":
        rolled_dist = 2 * window_size * (1 - (
                dot_rolled - window_size * means * means[
            idx]) / (window_size * stds * stds[idx]))

    # ed is squared
    if similarity == "ed":
        csumsq = csumsq[window_size:] - csumsq[:-window_size]
        rolled_dist = -2 * dot_rolled + csumsq + csumsq[idx]

    # cid is squared
    if similarity == "cid":
        csumsq = csumsq[window_size:] - csumsq[:-window_size]
        ed = -2 * dot_rolled + csumsq + csumsq[idx]  # dist is squared
        ce = dcsum[window_size:] - dcsum[:-window_size] + 1e-5  # add some noise to prevent zero divisons

        last_ce = np.repeat(ce[idx], ce.shape[0])
        cf = (np.max(np.dstack((ce, last_ce)), axis=2) / np.min(
            np.dstack((ce, last_ce)), axis=2))[0]

        rolled_dist = ed * cf

    dist[valid_dist] = rolled_dist[valid_dist]

    excl_range = slice(max(0, idx - exclusion_radius),
                       min(idx + exclusion_radius, l))  #
    dist[excl_range] = np.max(dist)
    knns = argkmin(dist, k_neighbours, lbound)

    # update dot product
    dot_rolled -= time_series[idx] * time_series[:l]

    return dot_rolled, dist, knns

@njit(fastmath=True, cache=True)
def mean(idx, csum, window_size):
    window_sum = csum[idx + window_size] - csum[idx]
    return window_sum / window_size


@njit(fastmath=True, cache=True)
def std(idx, csumsq, csum, window_size):
    window_sum = csum[idx + window_size] - csum[idx]
    window_sum_sq = csumsq[idx + window_size] - csumsq[idx]

    movstd = window_sum_sq / window_size - (window_sum / window_size) ** 2

    # should not happen, but just in case
    if movstd < 0:
        return 1

    movstd = np.sqrt(movstd)

    # avoid dividing by too small std, like 0
    if abs(movstd) < 1e-3:
        return 1

    return movstd

@njit(fastmath=True, cache=True)
def roll_numba(arr, num, fill_value=0):
    result = np.empty_like(arr)
    result[num] = fill_value # TODO?
    result[:num] = arr[-num:]
    return result

@njit(fastmath=True, cache=True)
def roll_all(time_series, timepoint,
             csum, csumsq, fill, dcsum,
             window_size, means, stds):
    # update time series
    time_series = roll_numba(time_series, -1)
    time_series[-1] = timepoint

    # update cum sum
    csum = roll_numba(csum, -1)
    csum[-1] = csum[-2] + timepoint

    # update cum sum squared
    csumsq = roll_numba(csumsq, -1)
    csumsq[-1] = csumsq[-2] + timepoint ** 2

    # update diff cum sum
    if fill > 1:
        dcsum = roll_numba(dcsum, -1)
        dcsum[-1] = dcsum[-2] + np.square(timepoint - time_series[-2])

    if fill >= window_size:
        # update means
        means = roll_numba(means, -1)
        means[-1] = mean(len(time_series) - window_size, csum, window_size)

        # update stds
        stds = roll_numba(stds, -1)
        stds[-1] = std(len(time_series) - window_size, csumsq, csum, window_size)

    return time_series, csum, csumsq, dcsum, means, stds

# @njit(fastmath=True, cache=True)
def _sliding_dot(query, time_series):
    m = len(query)
    n = len(time_series)

    time_series_add = 0
    if n % 2 == 1:
        time_series = np.concatenate((np.array([0]), time_series))
        time_series_add = 1

    q_add = 0
    if m % 2 == 1:
        query = np.concatenate((np.array([0]), query))
        q_add = 1

    query = query[::-1]

    query = np.concatenate((query, np.zeros(n - m + time_series_add - q_add)))

    trim = m - 1 + time_series_add
    with objmode(dot_product="float32[:]"):
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
        self.exclusion_radius = int(window_size / 2)  #

        self.n_timepoints = n_timepoints
        self.k_neighbours = k_neighbours
        self.similarity = similarity

        self.lbound = 0

        self.time_series = np.full(shape=n_timepoints, fill_value=np.nan,
                                   dtype=np.float64)

        self.l = n_timepoints - window_size + 1
        self.knn_insert_idx = self.l - self.exclusion_radius - self.k_neighbours - 1

        self.csum = np.full(shape=n_timepoints + 1, fill_value=0, dtype=np.float64)
        self.csumsq = np.full(shape=n_timepoints + 1, fill_value=0, dtype=np.float64)
        self.dcsum = np.full(shape=n_timepoints + 1, fill_value=0, dtype=np.float64)

        self.means = np.full(shape=self.l, fill_value=np.nan, dtype=np.float64)
        self.stds = np.full(shape=self.l, fill_value=np.nan, dtype=np.float64)

        self.dists = np.full(shape=(self.l, k_neighbours), fill_value=np.inf,
                             dtype=np.float64)
        self.knns = np.full(shape=(self.l, k_neighbours), fill_value=-1, dtype=np.int64)

        self.dot_rolled = None

        self.fill = 0
        self.knn_fill = 0

    def knn(self):
        self.dot_rolled, dist, knns = knn(self.knn_insert_idx, self.l, self.fill,
                    self.window_size, self.dot_rolled,
                    self.time_series, self.means, self.stds, self.similarity,
                    self.csumsq, self.dcsum, self.exclusion_radius,
                    self.k_neighbours, self.lbound)

        return dist, knns

    def update(self, timepoint):
        # log how much data is ingested
        self.fill = min(self.fill + 1, self.l)

        self.time_series, self.csum, self.csumsq, self.dcsum, self.means, self.stds \
            = roll_all(self.time_series, timepoint,
                 self.csum, self.csumsq, self.fill, self.dcsum,
                 self.window_size, self.means, self.stds)

        if self.fill < self.window_size + self.exclusion_radius + self.k_neighbours:
            return self

        # update knn
        self._update_knn()

        return self

    def _update_knn(self):
        # roll existing indices further
        if self.knn_fill > 0:
            self.dists = roll_numba(self.dists, -1,
                                    np.full(shape=self.dists.shape[1],
                                            fill_value=np.inf,
                                            dtype=np.float64))

            self.knns = roll_numba(self.knns, -1)
            self.knns[self.knn_insert_idx - self.knn_fill:self.knn_insert_idx] -= 1
            self.knns[-1, :] = np.full(shape=self.dists.shape[1], fill_value=-1,
                                       dtype=np.float64)

        # insert new distances and knns
        dist, knn = self.knn()

        self.knns, self.dists, self.lbound, self.knn_fill = rolling_knn(
            self.dists, self.knns, dist, knn, self.knn_insert_idx,
            self.knn_fill, self.l, self.k_neighbours, self.lbound
        )

        return self
