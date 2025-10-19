from typing import Dict, List, Optional
from enum import Enum, StrEnum, auto
from dataclasses import dataclass
from collections import namedtuple
import numpy as np
from tianshou.env import DummyVectorEnv, SubprocVectorEnv

DATA_FREQUENCY = 900  # 15 minutes
RNG_SOC_MIN = 0.2
RNG_SOC_MAX = 0.8

BuildEnvDataSpecs = namedtuple('BuildEnvDataSpecs', ['load', 'price', 'pv'])
BuildingEnvDF = namedtuple('BuildingEnvDF', ['df', 'min_time', 'max_time', 'building_id', 'price_id'])

class ActionSpaceType(Enum):
    DISCRETE = 0
    CONTINUOUS = 1

class ScalingType(Enum):
    NORMALIZE = 0
    STANDARDIZE = 1
    NONE = 2

class FormatType(Enum):
    CSV = 0
    FEATHER = 1

class DataColumn(StrEnum):
    INDEX = "unixtime"
    LOAD = "load"
    PRICE = "price"
    PV = "pv"

class ModelType(Enum):
    MPC = auto()
    PREDICTOR = auto()

@dataclass
class BuildingEnvObservation:
    soc: Optional[np.ndarray] = None
    bat_energy: Optional[np.ndarray] = None
    loads: Optional[np.ndarray] = None
    prices: Optional[np.ndarray] = None
    gens: Optional[np.ndarray] = None
    time_features: Optional[np.ndarray] = None

@dataclass
class PredictionModelSets:
    X_train: np.ndarray
    y_train: np.ndarray
    target_indexes: np.ndarray
    X_val: Optional[np.ndarray] = None
    y_val: Optional[np.ndarray] = None
    X_test: Optional[np.ndarray] = None
    y_test: Optional[np.ndarray] = None
    scaler: Optional[object] = None
    aug_cols: Optional[List[int]] = None



