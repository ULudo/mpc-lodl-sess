import random
import shutil
from pathlib import Path
from time import gmtime, strftime
from typing import List, Optional, Tuple, Any

import numpy as np
from omegaconf import OmegaConf, DictConfig, ListConfig
import torch

from src.env import BuildingEnv, BuildingDataManager
from src.model.env_loop import evaluate_and_report
from src.model.mpc import PerfectMPController, PredictorBundle, SingleNetPredictorMPController
from src.model.optimizer import MPCOptimizer
from src.mse_predictor_mgr import MsePredictorMgr
from src.util.consts_and_types import ScalingType, FormatType, ActionSpaceType, ModelType, DataColumn
from src.util.functions import check_path


class ConfigManager:

    def __init__(self, config_path: str):
        self.config, self.config_path = self._read_config(config_path)

    @staticmethod
    def _read_config(config_path: str) -> Tuple[DictConfig, Path]:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        return OmegaConf.load(config_path), config_path

    def run(self):
        if seed := self.config.model.get("seed", None):
            self._set_random_seed(seed)

        log_dir = self._create_log_dir()
        self._copy_config_to_log_dir(log_dir)
        self._init_data_mgr(self.config.data)
        model_type = ModelType[self.config.model.name.upper()]
        if model_type == ModelType.MPC:
            self._run_mpc(log_dir)
        elif model_type == ModelType.PREDICTOR:
            mgr = MsePredictorMgr(config=self.config, log_dir=log_dir)
            mgr.prepare_data()
            mgr.save_scaler()
            if self.config.model.action == "train":
                mgr.train()
            elif self.config.model.action == "evaluate":
                predictor = mgr.load_predictor()
                mgr.evaluate(predictor)
            elif self.config.model.action == "train_and_evaluate":
                mgr.train_and_evaluate()
            else:
                raise ValueError(f"Action {self.config.model.action} not supported")
        else:
            raise ValueError(f"Model {self.config.model.name} not supported")

    def _set_random_seed(self, seed: int):
        # Set all random seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        
        # For CUDA operations
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _copy_config_to_log_dir(self, log_dir: Path) -> None:
        config_copy_path = log_dir / self.config_path.name
        shutil.copy(self.config_path, config_copy_path)

    def _run_mpc(self, log_dir: Path):
        self._check_scaling_for_mpc()
        self._check_action_space_type_for_mpc()

        if self.config.model.mpc_type == "perfect":
            mpc_optimizer = self._create_optimizer(MPCOptimizer)
            controller = PerfectMPController(mpc_optimizer)
        elif self.config.model.mpc_type == "single_net_predictor":
            mpc_optimizer = self._create_optimizer(MPCOptimizer)
            predictor = self._load_predictor(self.config.model.predictor)
            controller = SingleNetPredictorMPController(
                predictor=predictor,
                optimizer=mpc_optimizer,
                history_length=self.config.model.history_length,
            )
        else:
            raise ValueError(f"Unknown MPC type: {self.config.model.mpc_type}")

        evaluate_and_report(
            log_dir=log_dir,
            n_runs=self.config.get("n_eval_runs", 1),
            envs=self._create_envs(self.config.environment),
            controller=controller,
            heat_up_steps=self.config.model.get('history_length', 0),
            plot_trajectories=self.config.get("plot_trajectories", False)
        )

    def _create_optimizer(self, class_type: type) -> Any:
        return class_type(
            n_predictions=self.config.model.prediction_length,
            bat_efficiency=self.config.environment.battery.efficiency,
            bat_capacity=self.config.environment.battery.capacity,
            bat_max_power=self.config.environment.battery.max_power,
            tax=self.config.environment.get("tax", 0.0)
        )

    @staticmethod
    def _load_predictor(predictor_config: DictConfig, target:Optional[str] = None) -> PredictorBundle:
        predictor_class = MsePredictorMgr.get_predictor(predictor_config.predictor_type)
        predictor = predictor_class(**predictor_config.get("kwargs", {}))
        predictor.load(predictor_config.predictor_path)
        if scaler_target_idx := target:
            scaler_target_idx = 1 if target == DataColumn.PV else 0
        if scaler_path := getattr(predictor_config, "scaler_path", None):
            scaler = MsePredictorMgr.load_scaler(scaler_path)
        else:
            scaler = None
        return PredictorBundle(
                predictor=predictor,
                scaler=scaler,
                scaler_target_idx=scaler_target_idx,
                two_d=predictor_config.two_d,
            )

    @staticmethod
    def _init_data_mgr(conf: DictConfig):
        datasets = conf.datasets
        if not datasets or not isinstance(datasets, ListConfig):
            raise ValueError("No datasets provided")
        house_data = {
            getattr(ds, 'name', f"house_{idx}"): check_path(ds.house_data_file)
            for idx, ds in enumerate(datasets)
        }
        price_data = check_path(conf.price_data_file)
        data_format = FormatType[conf.data_format.upper()]
        BuildingDataManager.load_datasets(
            datasets=house_data,
            price_data_file=price_data,
            input_format=data_format,
        )

    def _create_log_dir(self) -> Path:
        get_time_string = lambda: strftime("%Y%m%d_%H%M%S", gmtime())
        out_dir = getattr(self.config, "out_dir", "./out")
        name = getattr(self.config, "name", "experiment")
        log_dir = Path(out_dir) / name / get_time_string()
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir

    @staticmethod
    def _create_building_env_from_config(env_config: DictConfig, log_dir: Optional[Path] = None) -> BuildingEnv:
        if dataset_args := env_config.get("dataset_args", None):
            dataset_args = OmegaConf.to_container(dataset_args, resolve=True)
        
        env = BuildingEnv(
            dataset_args=dataset_args,
            battery_efficiency=env_config.get("battery_efficiency", 0.95),
            battery_max_power=env_config.get("battery_max_power", 8000),
            battery_capacity=env_config.get("battery_capacity", 20000),
            use_time_features=env_config.get("use_time_features", True),
            random_time_init=env_config.get("random_time_init", False),
            random_soc_init=env_config.get("random_soc_init", False),
            upper_time_bound=env_config.get("upper_time_bound", None),
            init_soc=env_config.get("init_soc", 0.5),
            init_time=env_config.get("init_time", None),
            episode_length=env_config.get("episode_length", None),
            tax=env_config.get("tax", 0.0),
            apply_deadband=env_config.get("apply_deadband", False),
            scaling_method=env_config.get("scaling_method", 'none'),
            load_stats=env_config.get("load_stats", None),
            price_stats=env_config.get("price_stats", None),
            pv_stats=env_config.get("pv_stats", None),
            prediction_horizon=env_config.get("prediction_horizon", 0),
            action_space_type=env_config.get("action_space_type", 'discrete')
        )
        return env

    @staticmethod
    def get_create_env_fun(conf, log_dir: Optional[Path] = None) -> callable:
        return lambda: ConfigManager._create_building_env_from_config(conf, log_dir)

    @staticmethod
    def _create_envs(env_config: DictConfig) -> List[BuildingEnv]:
        num_envs = getattr(env_config, "num_envs", 1)
        return [ConfigManager.get_create_env_fun(env_config)() for _ in range(num_envs)]

    def _check_action_space_type_for_mpc(self):
        action_space_type = self.config.environment.action_space_type.upper()
        action_space_type = ActionSpaceType[action_space_type]
        assert action_space_type is ActionSpaceType.CONTINUOUS, \
            "Evaluation env must have continuous action space for MPC"

    def _check_scaling_for_mpc(self):
        scaling_method = self.config.environment.scaling_method.upper()
        scaling_method = ScalingType[scaling_method]
        assert scaling_method is ScalingType.NONE, \
            "Environment observations should not be scaled for MPC"

