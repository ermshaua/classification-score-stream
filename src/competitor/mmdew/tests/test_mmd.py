import numpy as np
from mmdaw.mmd import MMD


def test_accept_H0_biased():
    rng = np.random.default_rng(1234)
    X = rng.random(size=(1000, 1))
    Y = rng.random(size=(1000, 1))

    mmd = MMD(biased=True, gamma=1)
    assert mmd.accept_H0(X, Y, alpha=0.1)


def test_reject_H0_biased():
    rng = np.random.default_rng(1234)
    X = rng.random(size=(1000, 1))
    Y = rng.normal(size=(1000, 1))

    mmd = MMD(biased=True, gamma=1)
    assert not mmd.accept_H0(X, Y, alpha=0.1)


def test_accept_H0_unbiased():
    rng = np.random.default_rng(1234)
    X = rng.random(size=(1000, 1))
    Y = rng.random(size=(1000, 1))

    mmd = MMD(biased=False, gamma=1)
    assert mmd.accept_H0(X, Y, alpha=0.1)


def test_reject_H0_unbiased():
    rng = np.random.default_rng(1234)
    X = rng.random(size=(1000, 1))
    Y = rng.normal(size=(1000, 1))

    mmd = MMD(biased=False, gamma=1)
    assert not mmd.accept_H0(X, Y, alpha=0.1)
