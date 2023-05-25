from abc import abstractmethod, ABCMeta, ABC
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
from skmultiflow.data import DataStream


class ChangeStream(DataStream, metaclass=ABCMeta):
    def next_sample(self, batch_size=1):
        change = self._is_change()
        x, y = super(ChangeStream, self).next_sample(batch_size)
        return x, y, change

    @abstractmethod
    def change_points(self):
        raise NotImplementedError

    @abstractmethod
    def _is_change(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def id(self) -> str:
        raise NotImplementedError

    def type(self) -> str:
        raise NotImplementedError
