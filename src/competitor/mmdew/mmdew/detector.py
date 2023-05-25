import numpy as np
from collections import deque
import time
from mmdaw.mmd import MMD
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import rbf_kernel


class Detector:
    def __init__(self, gamma, alpha=0.1, compress=True):
        """ """
        self.gamma = gamma
        self.alpha = alpha
        self.compress = compress
        self.buckets = deque()

    def insert(self, element):
        # breakpoint()
        bucket = Bucket(gamma=self.gamma, compress=self.compress)
        bucket.insert(element)
        bucket.XX = np.sum(rbf_kernel(np.array([element]), gamma=self.gamma))

        if self.buckets:
            head = self.buckets.pop()
            bucket.XY = np.sum(rbf_kernel(head.data, bucket.data, gamma=self.gamma))
            self.buckets.append(head)

        self.merge(bucket)

    def can_merge(self, b1, b2):
        return b1.capacity == b2.capacity

    def mmd(self, b1, b2):
        """b2 must have the XY terms for b1 and b2."""
        n = len(b1.data)
        m = len(b2.data)
        return 1 / n ** 2 * b1.XX + 1 / m ** 2 * b2.XX - 2 / (m * n) * b2.XY

    def threshold(self, m, n, alpha=0.1):
        K = 1

        thresh = (
            np.sqrt(K / m + K / n)
            + np.sqrt((2 * K * (m + n) * np.log(1 / alpha)) / (m * n))
        ) ** 2

        return thresh  # square to be consistent with unbiased threshold

    def merge(self, bucket):
        if self.buckets:  # deque contains buckets
            head = self.buckets.pop()
            if self.can_merge(head, bucket):
                distance = self.mmd(head, bucket)
                threshold = self.threshold(len(head.data), len(bucket.data))
                if distance > threshold:
                    print("CP!")
                    #plt.figure()
                    #print(len(head.data))
                    #print(len(bucket.data))
                    #plt.hist(head.data, density=True)
                    #plt.hist(bucket.data, density=True)
                    #plt.show()
                    
                head.data = np.vstack((head.data, bucket.data))
                head.capacity *= 2
                head.XX += 2 * bucket.XY + bucket.XX
                if self.buckets:
                    head.XY += np.sum(
                        rbf_kernel(bucket.data, self.buckets[-1].data, gamma=self.gamma)
                    )

                
                self.merge(head)
            else:
                self.buckets.append(head)
                self.buckets.append(bucket)
        else:  # deque does not contain buckets
            self.buckets.append(bucket)

    def __str__(self):
        return "\n".join([str(b) for b in self.buckets])


class Bucket:
    def __init__(self, gamma, capacity=1, compress=True):
        """ """
        self.gamma = gamma
        self.capacity = capacity
        self.compress = compress
        self.XX = 0
        self.XY = 0
        self.data = np.array([])

    def insert(self, element):
        self.data = np.array([element])

    def __str__(self):
        return "Data:\t" + str(self.data)


def test_XX_on_stream_equals_offline_computation():
    rng = np.random.default_rng(1234)
    k = 10
    X = rng.normal(size=(2 ** k, 2))
    detector = Detector(gamma=1)

    for x in X:
        detector.insert(x)

    assert abs(detector.buckets[-1].XX - np.sum(rbf_kernel(X, gamma=1))) < 10e-6


def test_XY_on_stream_equals_offline_computation():
    rng = np.random.default_rng(1234)
    k = 10
    X = rng.normal(size=(2 ** k - 1, 2))
    detector = Detector(gamma=1)

    for x in X:
        detector.insert(x)

    truth = np.sum(
        rbf_kernel(detector.buckets[0].data, detector.buckets[1].data, gamma=1)
    )

    assert abs(truth - detector.buckets[1].XY) < 10e-6


def test_mmd():
    rng = np.random.default_rng(1234)
    k = 12
    X = rng.normal(size=(2 ** k - 1, 2))
    detector = Detector(gamma=1)

    for x in X:
        detector.insert(x)

    X = detector.buckets[0].data
    Y = detector.buckets[1].data

    truth = MMD(biased=True, gamma=1).mmd(X, Y)
    b1 = detector.buckets[0]
    b2 = detector.buckets[1]
    assert abs(truth - detector.mmd(b1, b2)) < 10e-6


def main():
    rng = np.random.default_rng(1234)
    k = 13
    X1 = rng.normal(size=(2 ** k, 1))
    X2 = rng.uniform(size=(2 ** k, 1))
    X = np.vstack((X1,X2))
    gamma = MMD.estimate_gamma(X)
    detector = Detector(gamma=gamma)

    start = time.time()

    for x in X:
        detector.insert(x)

    print(f"Runtime: {time.time()- start:.4f} seconds for {2**k} elements.")


if __name__ == "__main__":
    main()
