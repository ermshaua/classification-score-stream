import numba
import numpy as np
import pandas as pd

from numba import njit, prange
from numba import njit, prange, typeof
from numba import types
from numba.typed import Dict, List


@njit(fastmath=True, cache=True)
def _labels(knn, split_idx, window_size):
    n_timepoints, k_neighbours = knn.shape

    y_true = np.concatenate((
        np.zeros(split_idx, dtype=np.int64),
        np.ones(n_timepoints - split_idx, dtype=np.int64),
    ))

    knn_labels = np.zeros(shape=(k_neighbours, n_timepoints), dtype=np.int64)

    for i_neighbor in range(k_neighbours):
        neighbours = knn[:, i_neighbor]
        knn_labels[i_neighbor] = y_true[neighbours]

    ones = np.sum(knn_labels, axis=0)
    zeros = k_neighbours - ones
    y_pred = np.asarray(ones > zeros, dtype=np.int64)

    valid = np.full(shape=y_true.shape, fill_value=True, dtype=np.bool8)
    valid[split_idx:split_idx+window_size] = False

    return y_true[valid], y_pred[valid]


@njit(fastmath=True, cache=True)
def _calc_neigh_pos(knn):
    ind, val = np.zeros(knn.shape[0], np.int64), np.zeros(knn.shape[0]*knn.shape[1], np.int64)

    counts = np.zeros(ind.shape[0], np.int64)
    ptr = np.zeros(ind.shape[0], np.int64)

    # count knn occurences
    for idx in range(knn.shape[0]):
        for kdx in range(knn.shape[1]):
            pos = knn[idx,kdx]
            counts[pos] += 1

    # save the indices positions
    for idx in range(1, ind.shape[0]):
        ind[idx] = ind[idx-1] + counts[idx-1]

    # save actual indices at correct positions
    for idx, neigh in enumerate(knn):
        for nn in neigh:
            pos = ind[nn] + ptr[nn]
            val[pos] = idx
            ptr[nn] += 1

    return ind, val


@njit(fastmath=True, cache=True)
def _init_labels(knn, offset):
    n_timepoints, k_neighbours = knn.shape

    y_true = np.concatenate((
        np.zeros(offset, dtype=np.int64),
        np.ones(n_timepoints - offset, dtype=np.int64),
    ))

    neigh_pos = _calc_neigh_pos(knn)

    knn_labels = np.zeros(shape=(k_neighbours, n_timepoints), dtype=np.int64)

    for i_neighbor in range(k_neighbours):
        neighbours = knn[:, i_neighbor]
        knn_labels[i_neighbor] = y_true[neighbours]

    ones = np.sum(knn_labels, axis=0)
    zeros = k_neighbours - ones
    y_pred = np.asarray(ones > zeros, dtype=np.int64)

    return (zeros, ones), neigh_pos, y_true, y_pred


@njit(fastmath=True, cache=True)
def _init_conf_matrix(y_true, y_pred):
    conf_matrix = np.zeros(shape=(2,3), dtype=np.float64)

    for label in (0,1):
        tp = np.sum(np.logical_and(y_true == label, y_pred == label))
        fp = np.sum(np.logical_and(y_true != label, y_pred == label))
        fn = np.sum(np.logical_and(y_true == label, y_pred != label))

        conf_matrix[label] = np.array([tp, fp, fn])

    return conf_matrix


@njit(fastmath=True, cache=True)
def _update_conf_matrix(old_true, old_pred, new_true, new_pred, conf_matrix):
    for label in (0,1):
        if old_true == label and old_pred == label:
            conf_matrix[label][0] -= 1

        if old_true != label and old_pred == label:
            conf_matrix[label][1] -= 1

        if old_true == label and old_pred != label:
            conf_matrix[label][2] -= 1

        if new_true == label and new_pred == label:
            conf_matrix[label][0] += 1

        if new_true != label and new_pred == label:
            conf_matrix[label][1] += 1

        if new_true == label and new_pred != label:
            conf_matrix[label][2] += 1

    return conf_matrix


@njit(fastmath=True, cache=True)
def _binary_f1_score(conf_matrix):
    f1_score = 0

    for label in (0,1):
        tp, fp, fn = conf_matrix[label]

        if (tp + fp) == 0 or (tp + fn) == 0:
            return -np.inf

        pr = tp / (tp + fp)
        re = tp / (tp + fn)

        if (pr + re) == 0:
            return -np.inf

        f1 = 2 * (pr * re) / (pr + re)
        f1_score += f1

    return f1_score / 2


@njit(fastmath=True, cache=True)
def _binary_acc_score(conf_matrix):
    acc_score = 0

    for label in (0,1):
        tp, fp, fn = conf_matrix[label]

        if (tp + fp + fn) == 0:
            return -np.inf

        acc = tp / (tp + fp + fn)
        acc_score += acc

    return acc_score / 2


@njit(fastmath=True, cache=True)
def _update_labels(split_idx, excl_zone, neigh_pos, knn_counts, y_true, y_pred, conf_matrix, excl_conf_matrix):
    np_ind, np_val = neigh_pos
    excl_start, excl_end = excl_zone
    knn_zeros, knn_ones = knn_counts

    ind = np_val[np_ind[split_idx]:np_ind[split_idx+1]]

    if ind.shape[0] > 0:
        ind = np.append(ind, split_idx)
    else:
        ind = np.array([split_idx])

    for pos in ind:
        if pos != split_idx:
            knn_zeros[pos] += 1
            knn_ones[pos] -= 1

        in_excl_zone = pos >= excl_start and pos < excl_end
        zeros, ones = knn_zeros[pos], knn_ones[pos]

        if zeros >= ones:
            conf_matrix = _update_conf_matrix(y_true[pos], y_pred[pos], y_true[pos], 0, conf_matrix)

            if in_excl_zone:
                excl_conf_matrix = _update_conf_matrix(y_true[pos], y_pred[pos], y_true[pos], 0, excl_conf_matrix)

            y_pred[pos] = 0

        if zeros < ones:
            conf_matrix = _update_conf_matrix(y_true[pos], y_pred[pos], y_true[pos], 1, conf_matrix)

            if in_excl_zone:
                excl_conf_matrix = _update_conf_matrix(y_true[pos], y_pred[pos], y_true[pos], 1, excl_conf_matrix)

            y_pred[pos] = 1

    y_true[split_idx] = 0

    # update exclusion zone range
    excl_conf_matrix = _update_conf_matrix(y_true[excl_start], y_pred[excl_start], y_true[excl_end], y_pred[excl_end], excl_conf_matrix)

    return y_true, y_pred, conf_matrix, excl_conf_matrix


@njit(fastmath=True, cache=True)
def _score(split_idx, window_size, y_true, y_pred, conf_matrix):
    exclusion_zone = np.arange(split_idx - window_size, split_idx)
    tmp = y_pred[exclusion_zone].copy()

    # exclusion zone magic
    for pos in exclusion_zone:
        conf_matrix = _update_conf_matrix(y_true[pos], y_pred[pos], 0, 1, conf_matrix) # y_true[pos]
        y_pred[pos] = 1

    score = _binary_f1_score(conf_matrix)

    # exclusion zone magic
    for kdx, pos in enumerate(exclusion_zone):
        conf_matrix = _update_conf_matrix(y_true[pos], y_pred[pos], y_true[pos], tmp[kdx], conf_matrix)
        y_pred[pos] = tmp[kdx]

    return score


@njit(fastmath=True, cache=True)
def _fast_profile(knn, window_size, offset):
    n_timepoints = knn.shape[0]
    profile = np.full(shape=n_timepoints, fill_value=-np.inf, dtype=np.float64)

    knn_counts, neigh_pos, y_true, y_pred = _init_labels(knn, offset)
    conf_matrix = _init_conf_matrix(y_true, y_pred)

    excl_zone = np.array([offset, offset+window_size])
    excl_conf_matrix = _init_conf_matrix(y_true[excl_zone[0]:excl_zone[1]], y_pred[excl_zone[0]:excl_zone[1]])

    for split_idx in range(offset, n_timepoints - offset):
        profile[split_idx] = _binary_f1_score(conf_matrix-excl_conf_matrix) #

        y_true, y_pred, conf_matrix, excl_conf_matrix = _update_labels(split_idx, excl_zone, neigh_pos, knn_counts, y_true, y_pred, conf_matrix, excl_conf_matrix)
        excl_zone += 1

    return profile


def clasp(ts_stream, offset, return_knn=False, interpolate=False):
    knn = ts_stream.knns[ts_stream.lbound:ts_stream.knn_insert_idx] - ts_stream.lbound
    knn = np.clip(knn, 0, knn.shape[0] - 1)

    profile = _fast_profile(knn, ts_stream.window_size, offset)

    if interpolate is True:
        profile[np.isinf(profile)] = np.nan
        profile = pd.Series(profile).interpolate(limit_direction="both").to_numpy()

    if return_knn is True:
        return profile, knn

    return profile



