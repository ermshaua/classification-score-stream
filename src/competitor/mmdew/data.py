from itertools import permutations
import os

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import KBinsDiscretizer
from changeds.abstract import ChangeStream
from tensorflow import keras

def get_perm_for_cd(df):
    rng = np.random.default_rng()
    classes = sorted(df["Class"].unique())
    perms = list(permutations(classes, len(classes)))
    use_perm = rng.integers(len(perms))
    mapping = dict(zip(classes, list(perms)[use_perm]))
    df = df.sort_values("Class", key=lambda series : series.apply(lambda x: mapping[x])).reset_index(drop=True)
    return df

class Constant(ChangeStream):
    def __init__(self, n, preprocess=None, max_len=None):
        self.n = n
        data = np.ones(n).reshape(-1,1)
        y = np.zeros(n)

        if max_len:
            data = data[:max_len]
            y = y[:max_len]
        if preprocess:
            data = preprocess(data)
        self._change_points = np.diff(y, prepend=y[0]).astype(bool)
        super(Constant, self).__init__(data=data, y=np.array(y))

    def id(self) -> str:
        return "Constant" + str(self.n)

    def change_points(self):
        return self._change_points

    def _is_change(self) -> bool:
        return self._change_points[self.sample_idx]

    
class TrafficUnif(ChangeStream):
    def __init__(self, preprocess=None, max_len=None):
        df = pd.read_csv("../data/traffic/traffic.csv",sep=";",decimal=",")
        est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
        est.fit(df["Slowness in traffic (%)"].values.reshape(-1,1))
        df["Class"] = est.transform(df["Slowness in traffic (%)"].values.reshape(-1,1))

        df = get_perm_for_cd(df)
         
        data = df.drop("Class", axis=1).to_numpy()
        y = df["Class"].to_numpy()

        if max_len:
            data = data[:max_len]
            y = y[:max_len]
        if preprocess:
            data = preprocess(data)
        self._change_points = np.diff(y, prepend=y[0]).astype(bool)
        super(TrafficUnif, self).__init__(data=data, y=np.array(y))

    def id(self) -> str:
        return "TrafficUnif"

    def change_points(self):
        return self._change_points

    def _is_change(self) -> bool:
        return self._change_points[self.sample_idx]

class GasSensors(ChangeStream):
    def __init__(self, preprocess=None, max_len=None):
        df = pd.read_csv("../data/gas-drift_csv.csv")
        df = get_perm_for_cd(df)
        data = df.drop("Class", axis=1).to_numpy()
        y = df["Class"].to_numpy()

        if max_len:
            data = data[:max_len]
            y = y[:max_len]
        if preprocess:
            data = preprocess(data)
        self._change_points = np.diff(y, prepend=y[0]).astype(bool)
        super(GasSensors, self).__init__(data=data, y=np.array(y))

    def id(self) -> str:
        return "GasSensors"

    def change_points(self):
        return self._change_points

    def _is_change(self) -> bool:
        return self._change_points[self.sample_idx]

class MNIST(ChangeStream):
    def __init__(self, preprocess=None, max_len = None):
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = np.reshape(x_train, newshape=(len(x_train), x_train.shape[1] * x_train.shape[2]))
        x_test = np.reshape(x_test, newshape=(len(x_test), x_test.shape[1] * x_test.shape[2]))
        x = np.vstack([x_train, x_test])
        y = np.hstack([y_train, y_test])
        df = pd.DataFrame(x)
        df["Class"] = y
        df = get_perm_for_cd(df)
        data = df.drop("Class", axis=1).to_numpy()
        y = df["Class"].to_numpy()

        if max_len:
            data = data[:max_len]
            y = y[:max_len]
        if preprocess:
            data = preprocess(data)
        self._change_points = np.diff(y, prepend=y[0]).astype(bool)
        super(MNIST, self).__init__(data=data, y=np.array(y))

    def id(self) -> str:
        return "MNIST"

    def change_points(self):
        return self._change_points

    def _is_change(self) -> bool:
        return self._change_points[self.sample_idx]

class FashionMNIST(ChangeStream):
    def __init__(self, preprocess=None, max_len=None):
        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
        x_train = np.reshape(x_train, newshape=(len(x_train), x_train.shape[1] * x_train.shape[2]))
        x_test = np.reshape(x_test, newshape=(len(x_test), x_test.shape[1] * x_test.shape[2]))
        x = np.vstack([x_train, x_test])
        y = np.hstack([y_train, y_test])


        df = pd.DataFrame(x)
        df["Class"] = y
        df = get_perm_for_cd(df)
        data = df.drop("Class", axis=1).to_numpy()
        y = df["Class"].to_numpy()

        if max_len:
            data = data[:max_len]
            y = y[:max_len]
        if preprocess:
            data = preprocess(data)
        self._change_points = np.diff(y, prepend=y[0]).astype(bool)
        super(FashionMNIST, self).__init__(data=data, y=y)

    def id(self) -> str:
        return "FMNIST"

    def change_points(self):
        return self._change_points

    def _is_change(self) -> bool:
        return self._change_points[self.sample_idx]

class HAR(ChangeStream):
    def __init__(self, preprocess=None, max_len=None):
        har_data_dir = "../data/har"
        test = pd.read_csv(os.path.join(har_data_dir, "test.csv"))
        train = pd.read_csv(os.path.join(har_data_dir, "train.csv"))
        x = pd.concat([test, train])
        x = x.sort_values(by="Activity")
        y = LabelEncoder().fit_transform(x["Activity"])
        x = x.drop(["Activity", "subject"], axis=1)

        df = pd.DataFrame(x)
        df["Class"] = y

        df = get_perm_for_cd(df)
        data = df.drop("Class", axis=1).to_numpy()
        y = df["Class"].to_numpy()

        if max_len:
            data = data[:max_len]
            y = y[:max_len]
        if preprocess:
            data = preprocess(data)
                    
        self._change_points = np.diff(y, prepend=y[0]).astype(bool)
        super(HAR, self).__init__(data=data, y=y)

    def id(self) -> str:
        return "HAR"

    def change_points(self):
        return self._change_points

    def _is_change(self) -> bool:
        return self._change_points[self.sample_idx]

class CIFAR10(ChangeStream):
    def __init__(self, preprocess=None, max_len=None):
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        x_train = x_train.dot([0.299, 0.587, 0.114])
        x_test = x_test.dot([0.299, 0.587, 0.114])
        x_train = np.reshape(x_train, newshape=(len(x_train), x_train.shape[1] * x_train.shape[2]))
        x_test = np.reshape(x_test, newshape=(len(x_test), x_test.shape[1] * x_test.shape[2]))
        x = np.vstack([x_train, x_test])
        y = np.hstack([y_train.reshape(-1), y_test.reshape(-1)])

        df = pd.DataFrame(x)
        df["Class"] = y
        df = get_perm_for_cd(df)
        data = df.drop("Class", axis=1).to_numpy()
        y = df["Class"].to_numpy()

        if max_len:
            data = data[:max_len]
            y = y[:max_len]
        if preprocess:
            data = preprocess(data)
        self._change_points = np.diff(y, prepend=y[0]).astype(bool)
        super(CIFAR10, self).__init__(data=data, y=y)

    def id(self) -> str:
        return "CIFAR10"

    def change_points(self):
        return self._change_points

    def _is_change(self) -> bool:
        return self._change_points[self.sample_idx]

