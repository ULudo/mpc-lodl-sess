from dataclasses import dataclass
from typing import Tuple, Any, Optional, Dict

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.model.base_controller import BaseController
from src.model.optimizer import MPCOptimizer
from src.model.prediction.base_predictor import BasePredictor
from src.util.consts_and_types import PredictionModelSets


@dataclass
class PredictorBundle:
    predictor: BasePredictor
    scaler: StandardScaler
    two_d: bool
    scaler_target_idx: Optional[int] = None


class PerfectMPController(BaseController):

    def __init__(self, optimizer: MPCOptimizer) -> None:
        self.optimizer = optimizer

    def step(self, obs: np.ndarray, info: Dict) -> int | float | np.ndarray:
        opti_params = self._get_params_from_info(info)
        p_charge_opt, p_discharge_opt = self.optimizer.optimize(*opti_params)
        return self.optimizer.get_action(p_charge_opt, p_discharge_opt)

    def _get_params_from_info(self, info:Dict) -> Any:
        obs_dict = info["clean_obs"]
        soc = obs_dict["soc"][0]
        opti_params = [soc] + [obs_dict[name] for name in ["loads", "prices", "gens"]]
        return opti_params
    

class SingleNetPredictorMPController(BaseController):
    """
    An MPC controller that uses imperfect (learned) predictions for MPC.
    """

    def __init__(
            self,
            predictor: PredictorBundle,
            optimizer: MPCOptimizer,
            history_length: int,
    ) -> None:
        self.optimizer = optimizer
        self.predictor = predictor
        self.history_length = history_length

        self._load_history = np.zeros(history_length, dtype=np.float32)
        self._pv_history = np.zeros(history_length, dtype=np.float32)
        self._price_history = np.zeros(history_length, dtype=np.float32)
        self._time_feat_history = np.zeros((history_length, 6), dtype=np.float32)

    def step(self, obs: np.ndarray, info: Dict) -> int | float | np.ndarray:
        cur_load, cur_price, cur_pv, soc, time_feats = self.extract_information_from_info(info)
        self._update_histories(cur_load, cur_price, cur_pv, time_feats)
        forecast_load, forecast_price, forecast_pv = self._make_predictions(cur_load, cur_price, cur_pv)
        p_charge_opt, p_discharge_opt = self.optimizer.optimize(
            soc, forecast_load, forecast_price, forecast_pv
        )
        return self.optimizer.get_action(p_charge_opt, p_discharge_opt)

    def _make_predictions(
            self, cur_load: float, cur_price: float, cur_pv: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        predictor_input = np.column_stack((self._load_history, self._pv_history, self._price_history, self._time_feat_history))

        n_building_features = 3
        predictor_input = self._scale_and_reshape_data(predictor_input, self.predictor.scaler, self.predictor.two_d)
        predictions = self.predictor.predictor.predict(predictor_input)
        assert predictions.shape[0] == 1, "Expected a batch of size 1"
        if predictions.ndim == 2:
            predictions = predictions.reshape(-1, n_building_features)
        if self.predictor.scaler:
            data_stack = np.zeros((predictions.shape[0], self.predictor.scaler.scale_.shape[0]))
            data_stack[:, :n_building_features] = predictions
            predictions = self.predictor.scaler.inverse_transform(data_stack)[:, :n_building_features]
        prediction_batch = np.vstack(([cur_load, cur_pv, cur_price], predictions))

        forecast_load = prediction_batch[:, 0]
        forecast_pv = prediction_batch[:, 1]
        forecast_price = prediction_batch[:, 2]

        return forecast_load, forecast_price, forecast_pv

    def _update_histories(
            self, cur_load: float, cur_price: float, cur_pv: float, time_feats: np.ndarray
    ) -> None:
        def _rolling_insert(arr, val):
            arr[:-1] = arr[1:]
            arr[-1] = val

        _rolling_insert(self._load_history, np.float32(cur_load))
        _rolling_insert(self._pv_history, np.float32(cur_pv))
        _rolling_insert(self._price_history, np.float32(cur_price))
        _rolling_insert(self._time_feat_history, time_feats.astype(np.float32))

    @staticmethod
    def extract_information_from_info(info: Dict) -> Tuple[float, float, float, float, np.ndarray]:
        obs_dict = info["clean_obs"]
        soc = obs_dict["soc"][0]
        cur_load = obs_dict["loads"][0]
        cur_pv = obs_dict["gens"][0]
        cur_price = obs_dict["prices"][0]
        time_feats = obs_dict["time_features"]
        return cur_load, cur_price, cur_pv, soc, time_feats

    @staticmethod
    def _scale_and_reshape_data(
            data_stack: np.ndarray,
            scaler: Optional[StandardScaler],
            two_d: bool
    ) -> np.ndarray:
        if scaler:
            data_stack = scaler.transform(data_stack)

        if two_d:
            # shape: (history_length, num_features) -> (1, history_length * num_features)
            data_stack = data_stack.reshape(1, -1)
        else:
            # shape: (history_length, num_features) -> (1, history_length, num_features)
            data_stack = np.expand_dims(data_stack, axis=0)

        return data_stack
    