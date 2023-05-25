from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import numpy as np
from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector


class DriftDetector(BaseDriftDetector, ABC):
    @abstractmethod
    def pre_train(self, data):
        raise NotImplementedError

    @abstractmethod
    def metric(self):
        raise NotImplementedError

    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def parameter_str(self) -> str:
        raise NotImplementedError


class RegionalDriftDetector(DriftDetector, ABC):
    @abstractmethod
    def get_drift_dims(self) -> np.ndarray:
        raise NotImplementedError


@runtime_checkable
class QuantifiesSeverity(Protocol):
    @abstractmethod
    def get_severity(self):
        raise NotImplementedError

