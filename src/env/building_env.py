from copy import deepcopy
import time
from dataclasses import asdict, astuple, fields
from typing import Tuple, Optional, Union, List, Dict

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.util.consts_and_types import (
    BuildingEnvDF,
    BuildEnvDataSpecs,
    BuildingEnvObservation,
    ActionSpaceType,
    ScalingType,
    DataColumn,
    DATA_FREQUENCY,
    RNG_SOC_MIN,
    RNG_SOC_MAX
)
from src.env.battery import Battery, BatAction
from src.env.data_manager import BuildingDataManager
from src.util.functions import (
    sin_encode_hour,
    cos_encode_hour,
    sin_encode_day,
    cos_encode_day,
    sin_encode_month,
    cos_encode_month,
    through_kilo
)

wh_to_kwh = lambda x: through_kilo(x)


class BuildingEnv(gym.Env):
    dtype = np.float32

    def __init__(
            self,
            dataset_args: Optional[Union[List[str], Dict[str, str]]] = None,
            battery_efficiency: float = 0.95,
            battery_max_power: float = 8000,
            battery_capacity: float = 20000,
            use_time_features: bool = True,
            random_time_init: bool = False,
            random_soc_init: bool = False,
            upper_time_bound: Optional[int] = None,
            init_soc: float = 0.5,
            init_time: Optional[int] = None,
            episode_length: Optional[int] = None,
            tax: float = 0.0,
            apply_deadband: bool = False,
            scaling_method: str = 'none',
            load_stats: Optional[Tuple[float, float]] = None,
            price_stats: Optional[Tuple[float, float]] = None,
            pv_stats: Optional[Tuple[float, float]] = None,
            prediction_horizon: int = 0,
            action_space_type: str = 'discrete',
    ) -> None:
        super(BuildingEnv, self).__init__()

        self.df_building = None
        self.min_data_time = None
        self.max_data_time = None
        self.upper_time_bound = upper_time_bound
        self.prediction_horizon = prediction_horizon
        self.dataset_args = dataset_args
        self.random_time_init = random_time_init
        self.init_time = init_time
        self.episode_length = episode_length
        self.tax = tax
        self.apply_deadband = apply_deadband
        self.use_time_features = use_time_features
        self.scaling_method = scaling_method
        self.normalization_stats = None
        self.random_soc_init = random_soc_init
        self.init_soc = init_soc
        self.battery = None
        self.action_space_type = action_space_type
        self.action_space: Optional[spaces.Space] = None
        self.observation_space: Optional[spaces.Space] = None
        self.sim_start_time = None
        self.sim_stop_time = None
        self.env_done_time = None
        self.terminal = None
        self.building_id = None
        self.price_id = None

        self._setup_obs_scaling(load_stats, price_stats, pv_stats)
        self._init_soc_and_setup_battery(
            battery_efficiency, battery_max_power, battery_capacity)
        self._setup_spaces()

    def _setup_obs_scaling(
            self,
            load_stats: Tuple[float, float],
            price_stats: Tuple[float, float],
            pv_stats: Tuple[float, float]
    ) -> None:
        method = self.scaling_method
        try:
            self.scaling_method = ScalingType[method.upper()]
        except KeyError:
            raise ValueError(f"Invalid normalization method: {method}")
        if self.scaling_method is not ScalingType.NONE:
            self.normalization_stats = BuildEnvDataSpecs(
                load=self._get_stats(load_stats),
                price=self._get_stats(price_stats),
                pv=self._get_stats(pv_stats)
            )

    def _init_soc_and_setup_battery(
        self,
        battery_efficiency: float,
        battery_max_power: float,
        battery_capacity: float
        ) -> None:
        # SoC
        if self.random_soc_init:
            self.init_soc = self._gen_random_soc()
        if not (0 <= self.init_soc <= 1):
            raise ValueError("Initial SoC must be between 0 and 1.")
        # Battery
        self.battery = Battery(
            dt=DATA_FREQUENCY,
            efficiency=battery_efficiency,
            max_power=battery_max_power,
            capacity=battery_capacity,
            initial_soc=self.init_soc,
        )

    def _setup_spaces(self) -> None:
        # Type of action space
        if self.action_space_type:
            try:
                self.action_space_type = ActionSpaceType[self.action_space_type.upper()]
            except KeyError:
                raise ValueError(f"Invalid action space type: {self.action_space_type}")
        else:
            self.action_space_type = ActionSpaceType.DISCRETE
        # Action and observation space gym variables
        self.action_space = self._create_action_space()
        self.observation_space = self._create_observation_space()

    def _get_stats(self, stats: Union[Tuple[float, float], List[float]]) -> np.ndarray:
        if len(stats) != 2:
            raise ValueError(f"Stats must be a list or tuple of two elements, e.g. (mean, std).")
        if not all(isinstance(x, (int, float)) for x in stats):
            raise ValueError(f"All values in stats must be numbers.")
        return np.array(stats, dtype=self.dtype)

    def _determine_init_time_and_episode_length(self) -> None:
        upper_bound = lambda: self.max_data_time - self.episode_length - self.prediction_horizon
        if self.random_time_init:
            assert self.episode_length, "Episode length must be given when random time init is set."
            assert upper_bound() > self.min_data_time, "Episode length exceeds the available data range."
            self.init_time = self._get_random_time(self.min_data_time, upper_bound())
        if not self.init_time:
            self.init_time = self.min_data_time
        if not self.episode_length:
            self.episode_length = self.max_data_time - self.init_time
        assert self.episode_length > 0, "Episode length must be greater than 0."
        assert self.init_time <= upper_bound(), f"Episode length exceeds the available data range by {self.init_time - upper_bound()}s."

    def _read_current_data_value(self, column: str) -> np.ndarray:
        return np.array(
            [self.df_building.loc[self.sim_start_time, column].item()],
            dtype=self.dtype
        )

    def _read_prediction_values(self, column: str) -> np.ndarray:
        if self.prediction_horizon == 0:
            return np.array([])
        start_idx = self.sim_stop_time
        end_idx = self.sim_stop_time + self.prediction_horizon - DATA_FREQUENCY
        return self.df_building.loc[start_idx:end_idx, column].values.flatten().astype(np.float32)

    def _read_current_and_prediction_values(self, column: str) -> np.ndarray:
        return np.concatenate((
            self._read_current_data_value(column),
            self._read_prediction_values(column)
        ))

    @staticmethod
    def _determine_time_features(unix_ts: int) -> np.ndarray:
        dt = time.gmtime(unix_ts)
        return np.array([
            sin_encode_day(dt.tm_wday), cos_encode_day(dt.tm_wday),
            sin_encode_hour(dt.tm_hour), cos_encode_hour(dt.tm_hour),
            sin_encode_month(dt.tm_mon - 1), cos_encode_month(dt.tm_mon - 1)
        ], dtype=BuildingEnv.dtype)

    def _gen_random_soc(self):
        return self.np_random.uniform(RNG_SOC_MIN, RNG_SOC_MAX)

    def _get_random_time(self, min_unix_time: int, max_unix_time: int) -> int:
        num_steps = (max_unix_time - min_unix_time) // DATA_FREQUENCY
        return min_unix_time + self.np_random.integers(num_steps) * DATA_FREQUENCY

    @staticmethod
    def _generate_observation_keys(obs: BuildingEnvObservation) -> list:
        keys = []
        for field in fields(obs):
            field_name = field.name
            value = getattr(obs, field_name)
            value_size = value.size
            if value_size == 1:
                keys.append(field_name)
            else:
                for i in range(value_size):
                    keys.append(f"{field_name}_{i}")
        return keys

    def _create_info(
            self,
            action: int | float,
            obs: BuildingEnvObservation,
            proc_obs: np.ndarray,
            reward: float,
            terminal: bool
    ) -> Dict:
        state_message = {
            "soc": float(obs.soc[0]),
            "load": float(obs.loads[0]),
            "e_bat": float(obs.bat_energy[0]),
            "price": float(obs.prices[0]),
            "pv_gen": float(obs.gens[0])
        }
        obs_keys = self._generate_observation_keys(obs)
        info = {
            "time": self.sim_start_time,
            "action": action,
            "state": state_message,
            "reward": reward,
            "done": terminal,
            "observation": dict(zip(obs_keys, proc_obs)),
            "clean_obs": asdict(obs)
        }
        return info

    def _discrete_action(self, action: int) -> float:
        try:
            action = BatAction(action)
        except ValueError:
            raise ValueError(f"Invalid action: {action}")

        action_methods = {
            BatAction.IDLE: self.battery.idle,
            BatAction.CHARGE: self.battery.charge,
            BatAction.DISCHARGE: self.battery.discharge
        }
        return action_methods[action]()

    def _continuous_action(self, action: float) -> float:
        if self.apply_deadband:
            if action > 0.05:
                action = (action - 0.05) / 0.95
            elif action < -0.05:
                action = (action + 0.05) / 0.95
            else:
                action = 0.0
        return self.battery.continuous_action(action)

    def _perform_action(self, action: int | float) -> float:
        if self.action_space_type == ActionSpaceType.DISCRETE:
            return self._discrete_action(action)
        else:
            return self._continuous_action(action)

    def _update_sim_times_and_terminal(self) -> None:
        self.sim_start_time = self.sim_stop_time
        self.sim_stop_time = self.sim_stop_time + DATA_FREQUENCY
        self.terminal = self.sim_stop_time > self.env_done_time

    def _set_building_data(self, data: BuildingEnvDF) -> None:
        self.df_building = data.df
        self.min_data_time = data.min_time
        self.max_data_time = data.max_time if self.upper_time_bound is None else self.upper_time_bound
        self.building_id = data.building_id
        self.price_id = data.price_id

    def _determine_building_data(self) -> None:
        """
        Determine and set the building data for the environment.

        If `dataset_args` are provided and `df_building` is not already set, the specified building and price data
        will be retrieved from the `BuildingDataManager` and set in the environment. If `dataset_args` are not provided,
        a random building and price data will be chosen from the `BuildingDataManager` and set in the environment.

        Raises:
            ValueError: If `dataset_args` is provided but is not a list or dictionary.
        """
        if self.dataset_args and self.df_building is None:
            if isinstance(self.dataset_args, list):
                data = BuildingDataManager.get_building_data(*self.dataset_args)
            elif isinstance(self.dataset_args, dict):
                data = BuildingDataManager.get_building_data(**self.dataset_args)
            else:
                raise ValueError("Invalid dataset arguments.")
            self._set_building_data(data)
        elif not self.dataset_args:
            data = BuildingDataManager.get_random_dataset()
            self._set_building_data(data)

    def reset(self, seed=None, **kwargs) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        try:
            # Init times
            self._determine_building_data()
            self._determine_init_time_and_episode_length()
            self.sim_start_time = self.init_time
            self.sim_stop_time = self.sim_start_time + DATA_FREQUENCY
            self.env_done_time = self.init_time + self.episode_length

            # Get current environment state
            self.terminal = False
            self.init_soc = self._gen_random_soc() if self.random_soc_init else self.init_soc
            self.battery.set_soc(self.init_soc)
            obs = self._next_observation(0.0)
            proc_obs = self._postprocess_observation(obs)

            info = self._create_info(0, obs, proc_obs, 0, self.terminal)
            return proc_obs, info
        except Exception as e:
            print(f"Error in reset: {str(e)}. Sim start time: {self.sim_start_time}, "
                  f"Available timestamps: {self.min_data_time} to {self.max_data_time}, "
                  F"Data IDs: {self.building_id}, {self.price_id}")
            raise

    def step(self, action: int | float | np.ndarray) -> tuple[np.ndarray, float, bool, bool, Dict]:
        if self.terminal:
            raise RuntimeError("Environment terminated. Please reset the environment.")
        try:
            to_scalar = lambda x: np.atleast_1d(x)[0]
            action = to_scalar(action)
            self._update_sim_times_and_terminal()
            e_bat = wh_to_kwh(self._perform_action(action))
            obs = self._next_observation(e_bat)
            reward = self.calculate_reward(obs)
            proc_obs = self._postprocess_observation(obs)
            info = self._create_info(action, obs, proc_obs, reward, self.terminal)
            return proc_obs, reward, self.terminal, False, info
        except Exception as e:
            print(f"Error in step: {str(e)}. Sim start time: {self.sim_start_time}, "
                  f"Available timestamps: {self.min_data_time} to {self.max_data_time}, "
                  f"Data IDs: {self.building_id}, {self.price_id}")
            raise

    def _create_action_space(self) -> spaces.Space:
        if self.action_space_type == ActionSpaceType.DISCRETE:
            return spaces.Discrete(len(BatAction))
        elif self.action_space_type == ActionSpaceType.CONTINUOUS:
            return spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=self.dtype)
        else:
            raise ValueError("Invalid action space type.")

    def _create_observation_space(self) -> spaces.Space:
        gen_item_array = lambda n, item: np.full(n, item, dtype=self.dtype)
        bat_vars_bounds = {
            "soc": (0.0, 1.0),
            "bat_energy": (-self._e_bat_max, self._e_bat_max)
        }
        data_variable_bounds = (-np.inf, np.inf)
        time_feature_bounds = (-1.0, 1.0)

        n_prediction_steps = self.prediction_horizon // DATA_FREQUENCY
        n_data_variables = len(BuildEnvDataSpecs._fields)
        n_time_features = (
            self._determine_time_features(0).size
            if self.use_time_features else 0
        )

        bounds = {}
        if self.scaling_method is not ScalingType.NONE:
            num_items = (len(bat_vars_bounds) 
                         + n_data_variables # current time step
                         + n_prediction_steps * n_data_variables # prediction steps
                         + n_time_features)
            val = 1.0 if self.scaling_method == ScalingType.NORMALIZE else 3.0
            bounds["lower"] = gen_item_array(num_items, -val)
            bounds["upper"] = gen_item_array(num_items, val)
        else:
            for idx, key in enumerate(["lower", "upper"]):
                bat_vars = np.array([v[idx] for _, v in bat_vars_bounds.items()], dtype=self.dtype)
                data_vars = gen_item_array(
                    n_prediction_steps * n_data_variables + n_data_variables,
                    data_variable_bounds[idx]
                )
                time_vars = gen_item_array(n_time_features, time_feature_bounds[idx])
                bounds[key] = np.concatenate((bat_vars, data_vars, time_vars))

        observation_space = spaces.Box(bounds["lower"], bounds["upper"], dtype=self.dtype)
        return observation_space

    def _next_observation(self, bat_energy: float) -> BuildingEnvObservation:
        obs = BuildingEnvObservation()
        obs.soc = np.array([self.battery.soc], dtype=self.dtype)
        obs.bat_energy = np.array([bat_energy], dtype=self.dtype)
        obs.loads = self._read_current_and_prediction_values(DataColumn.LOAD)
        obs.prices = self._read_current_and_prediction_values(DataColumn.PRICE)
        obs.gens = self._read_current_and_prediction_values(DataColumn.PV)
        obs.time_features = (self._determine_time_features(self.sim_start_time)
                                            if self.use_time_features else np.array([]))
        return obs

    @property
    def _e_bat_max(self):
        return wh_to_kwh(self.battery.max_energy)

    def _normalize_obs(self, obs: BuildingEnvObservation) -> BuildingEnvObservation:
        normalize = lambda obs, bounds: 2 * (obs - bounds[0]) / (bounds[1] - bounds[0]) - 1

        nor_res = BuildingEnvObservation()
        nor_res.soc = normalize(obs.soc, (0.0, 1.0))
        nor_res.bat_energy = normalize(obs.bat_energy, (-self._e_bat_max, self._e_bat_max))
        nor_res.loads = normalize(obs.loads, self.normalization_stats.load)
        nor_res.prices = normalize(obs.prices, self.normalization_stats.price)
        nor_res.gens = normalize(obs.gens, self.normalization_stats.pv)
        nor_res.time_features = obs.time_features.copy()
        return nor_res

    def _standardize_obs(self, obs: BuildingEnvObservation) -> BuildingEnvObservation:
        standardize = lambda obs, dists: (obs - dists[0]) / dists[1]

        std_res = BuildingEnvObservation()
        std_res.soc = standardize(obs.soc, (0.5, 0.5))
        std_res.bat_energy = standardize(obs.bat_energy, (0.0, self._e_bat_max))
        std_res.loads = standardize(obs.loads, self.normalization_stats.load)
        std_res.prices = standardize(obs.prices, self.normalization_stats.price)
        std_res.gens = standardize(obs.gens, self.normalization_stats.pv)
        std_res.time_features = obs.time_features.copy()
        return std_res

    def _postprocess_observation(self, obs: BuildingEnvObservation) -> np.ndarray:
        scaled_obs = deepcopy(obs)
        if self.scaling_method == ScalingType.NORMALIZE:
            scaled_obs = self._normalize_obs(scaled_obs)
        elif self.scaling_method == ScalingType.STANDARDIZE:
            scaled_obs = self._standardize_obs(scaled_obs)
        return np.concatenate(astuple(scaled_obs), dtype=self.dtype)

    def calculate_reward(self, obs: BuildingEnvObservation) -> float:
        bat_energy, load, price, gen = obs.bat_energy[0], obs.loads[0], obs.prices[0], obs.gens[0]
        net_consumption = bat_energy + load - gen
        if net_consumption > 0:
            price = price + self.tax
        return -price * net_consumption

    def render(self, mode: str = "plot") -> None:
        pass

    def close(self) -> None:
        pass
