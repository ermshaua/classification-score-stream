import numpy as np
from scipy.stats import distributions

from src.clazz.profile import _labels


def rank_binary_data(data):
    zeros = data == 0
    ones = data == 1

    zero_ranks = np.arange(np.sum(zeros))
    one_ranks = np.arange(zero_ranks.shape[0], data.shape[0])

    zero_mean = np.mean(zero_ranks) + 1 if zero_ranks.shape[0] > 0 else 0
    one_mean = np.mean(one_ranks) + 1 if one_ranks.shape[0] > 0 else 0

    ranks = np.full(data.shape[0], fill_value=zero_mean, dtype=np.float64)
    ranks[ones] = one_mean

    return ranks


def rank_sums_test(knn, change_point, window_size, sample_size=None, threshold=.05, random_state=2357):
    _, y_pred = _labels(knn, change_point)
    x, y = y_pred[:change_point], y_pred[change_point + window_size:]

    if sample_size is not None:
        np.random.seed(random_state)
        x = x[np.random.choice(x.shape[0], int(sample_size * x.shape[0] / y_pred.shape[0]), replace=True)]
        y = y[np.random.choice(y.shape[0], int(sample_size * y.shape[0] / y_pred.shape[0]), replace=True)]

    n1, n2 = len(x), len(y)
    alldata = np.concatenate((x, y))
    ranked = rank_binary_data(alldata)
    x = ranked[:n1]
    s = np.sum(x, axis=0)
    expected = n1 * (n1 + n2 + 1) / 2.0
    z = (s - expected) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    p = 2 * distributions.norm.sf(np.abs(z))

    return p, p <= threshold
