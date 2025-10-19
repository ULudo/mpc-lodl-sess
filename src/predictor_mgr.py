from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Dict, Union, Tuple, List

from omegaconf import DictConfig

from src.model.prediction.recurrent_predictor import RecurrentPredictor

class PredictorType(Enum):
    RECURRENT_NET = RecurrentPredictor

class LearningMgr(ABC):

    def __init__(self, config: DictConfig, log_dir: Path):
        self.log_dir = log_dir
        self.config = config

    @abstractmethod
    def train(self) -> Any:
        pass

    @abstractmethod
    def evaluate(self, predictor: Any) -> None:
        pass

    def train_and_evaluate(self) -> None:
        predictor = self.train()
        self.evaluate(predictor)


class BasePredictorMgr(LearningMgr, ABC):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data: Optional[Any] = None
        self.two_d = self.config.model.get("two_d", False)

    @abstractmethod
    def prepare_data(self) -> None:
        pass

    
    def _get_eval_periods(
        self, data_classes: List[str]
    ) -> Dict[str, Tuple[
        Dict[str, Union[Tuple[int, int], List[Tuple[int, int]]]], 
        Dict[str, Union[Tuple[int, int], List[Tuple[int, int]]]]
    ]]:
        """
        For each data class, extract its evaluation and test periods from the config.
        
        Returns a dictionary with the data class as key, and value a dict with keys "eval" and "test".
        """
        periods_by_class = {}
        for d_class in data_classes:
            data_eval_defs = self.config.data.get(d_class, None)
            eval_periods = self._get_periods_for_set(data_eval_defs, "eval") if data_eval_defs else {}
            test_periods = self._get_periods_for_set(data_eval_defs, "test") if data_eval_defs else {}
            periods_by_class[d_class] = {"eval": eval_periods, "test": test_periods}
        return periods_by_class

    @staticmethod
    def _get_periods_for_set(
            conf: Any,
            check_set: str
    ) -> Dict[str, Union[Tuple[int, int], List[Tuple[int, int]]]]:
        get_periods = lambda item: [(i.start_date, i.end_date) for i in getattr(item, "periods")]
        if data_sets := getattr(conf, check_set, None):
            return {d.name: get_periods(d) for d in data_sets}
        return {}
    
    def _ensure_data_is_set(self):
        if self.data is None:
            raise ValueError("Data is not set. Please call prepare_data() before proceeding.")
