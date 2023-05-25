import numpy as np
import pytest
from mmdaw.bucket_stream import BucketStream
from mmdaw.mmd import MMD


def test_XX_count_biased():
    bs = BucketStream(biased=True, capacity=2, compress=False)
    for i in range(2 ** 4):
        bs.insert(np.array([i]))
    assert bs.buckets[0].n_XX == (2 ** 4) ** 2


def test_XX_count_unbiased():
    bs = BucketStream(biased=False, capacity=2, compress=False)
    for i in range(2 ** 4):
        bs.insert(np.array([i]))
    assert bs.buckets[0].n_XX == 16 * (16 - 1)


def test_XY_count_biased():
    bs = BucketStream(biased=True, capacity=2, compress=False)
    for i in range(2 ** 4 + 2 ** 3):
        bs.insert(np.array([i]))
    assert bs.buckets[1].n_XY == 2 ** 4 * 2 ** 3


def test_XY_count_unbiased():
    bs = BucketStream(biased=False, capacity=2, compress=False)
    for i in range(2 ** 4 + 2 ** 3):
        bs.insert(np.array([i]))
    assert bs.buckets[1].n_XY == 2 ** 4 * 2 ** 3


def test_stream_MMD_equals_static_MMD_biased():
    bs = BucketStream(biased=True, capacity=2, compress=False)
    for i in range(2 ** 4 + 2 ** 3):
        bs.insert(np.array([i]))
    X = np.array([*range(2 ** 4)]).reshape(-1, 1)
    Y = np.array([*range(2 ** 4, 2 ** 4 + 2 ** 3)]).reshape(-1, 1)

    assert abs(bs.buckets[1].mmd() - MMD(biased=True, gamma=1).mmd(X, Y)) < 10e-6


def test_stream_MMD_equals_static_MMD_biased_different_gamma():
    gamma = 2
    bs = BucketStream(biased=True, gamma=gamma, capacity=2, compress=False)
    for i in range(2 ** 4 + 2 ** 3):
        bs.insert(np.array([i]))
    X = np.array([*range(2 ** 4)]).reshape(-1, 1)
    Y = np.array([*range(2 ** 4, 2 ** 4 + 2 ** 3)]).reshape(-1, 1)

    assert abs(bs.buckets[1].mmd() - MMD(biased=True, gamma=gamma).mmd(X, Y)) < 10e-6


def test_stream_MMD_equals_static_MMD_unbiased():
    bs = BucketStream(biased=False, capacity=2, compress=False)
    for i in range(2 ** 4 + 2 ** 3):
        bs.insert(np.array([i]))
    X = np.array([*range(2 ** 4)]).reshape(-1, 1)
    Y = np.array([*range(2 ** 4, 2 ** 4 + 2 ** 3)]).reshape(-1, 1)

    assert abs(bs.buckets[1].mmd() - MMD(biased=False, gamma=1).mmd(X, Y)) < 10e-6


def test_distribution_changed_biased():
    bs = BucketStream(biased=True, capacity=2, compress=False)
    rng = np.random.default_rng(1234)
    X = rng.random(size=(2 ** 7, 1))
    Y = rng.normal(size=(2 ** 4, 1))
    data = np.concatenate((X, Y))

    for i, x in enumerate(data):
        bs.insert(x)
    assert (
        len(bs.buckets[0].X) == 2 ** 4
    )  # making sure the uniform distribution got dropped


@pytest.mark.skip(reason="Taking too long.")
def test_distribution_changed_unbiased():
    bs = BucketStream(biased=False, capacity=2, compress=False)
    rng = np.random.default_rng(1234)
    X = rng.random(size=(2 ** 9, 1))  # we need lots of data for the unbiased test
    Y = rng.normal(size=(2 ** 8, 1))
    data = np.concatenate((X, Y))

    for i, x in enumerate(data):
        bs.insert(x)
    assert (
        len(bs.buckets[0].X) == 2 ** 8
    )  # making sure the uniform distribution got dropped


@pytest.mark.skip(reason="Visual comparsion only.")
def test_compression():
    n_samples = 1000  # elements in bucket must be uniformly distributed

    elements = []
    for _ in range(n_samples):
        bs = BucketStream(biased=False, capacity=2, compress=True)
        for i in range(2 ** 8):
            bs.insert(np.array([i]))
        elements = np.concatenate((elements, bs.buckets[0].X))
    import matplotlib.pyplot as plt

    plt.hist(elements, bins=2 ** 4)
    plt.show()

@pytest.mark.skip(reason="Must implement thresholding for compression.")
def test_distribution_changed_biased_with_compression():
    bs = BucketStream(biased=True, capacity=2, compress=True)
    rng = np.random.default_rng(1234)
    X = rng.random(size=(2 ** 10, 1))
    Y = rng.random(size=(2 ** 8, 1))
    data = np.concatenate((X, Y))

    for i, x in enumerate(data):
        bs.insert(x)
    import matplotlib.pyplot as plt

    plt.hist(bs.buckets[0].X)
    plt.show()
    assert len(bs.buckets[0].X) == 4  # making sure the uniform distribution got dropped

def test_counter():
    bs = BucketStream(biased=True, capacity=2, compress=False)
    for i in range(15):
        bs.insert(np.array([i]))

    assert bs.buckets[0].start == 0
    assert bs.buckets[1].start == 8
    assert bs.buckets[2].start == 12
    assert bs.buckets[3].start == 14
