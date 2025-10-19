import os
from typing import Union, List, Tuple, Any, Dict, Type, Optional

import joblib
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.preprocessing import StandardScaler

from src.env import BuildingDataManager
from src.model.diff_mpc.lodl_dispatcher import GlobalLODL, lodl_loss_handler
from src.model.mpc import MPCOptimizer
from src.model.prediction.base_predictor import BasePredictor
from src.model.prediction.data_preprocessor import DataBundle, DataPreprocessor, FeatureExtractor
from src.model.prediction.evaluation import evaluate_prediction_model
from src.predictor_mgr import BasePredictorMgr, PredictorType
from src.util.consts_and_types import PredictionModelSets, DataColumn


class MsePredictorMgr(BasePredictorMgr):

    def train(self) -> BasePredictor:
        self._ensure_data_is_set()
        predictor = self.create_predictor()
        res_eval_train, res_eval_val = predictor.fit(self.data)
        np.save(os.path.join(self.log_dir, f'fit_eval_train.npy'), res_eval_train)
        np.save(os.path.join(self.log_dir, f'fit_eval_val.npy'), res_eval_val)
        predictor.save(self.log_dir / 'predictor.pkl')
        return predictor

    def create_predictor(self) -> BasePredictor:
        kwargs = self._read_predictor_kwargs()
        predictor = self.get_predictor(self.config.model.predictor_type)(**kwargs)
        return predictor

    def _read_predictor_kwargs(self):
        kwargs = dict(self.config.model.get('kwargs', {}))
        if kwargs.get('early_stopping', False):
            kwargs["checkpoint_path"] = str(self.log_dir / 'model_checkpoint.pt')
        if self.config.model.loss == "lodl":
            kwargs["criterion"] = GlobalLODL(
                optimizer=MPCOptimizer(
                    n_predictions=self.config.model.prediction_steps,
                    bat_efficiency=self.config.environment.battery.efficiency,
                    bat_capacity=self.config.environment.battery.capacity,
                    bat_max_power=self.config.environment.battery.max_power,
                    tax=self.config.environment.get("tax", 0.0)
                ),
                sigma=self.config.model.get("sigma", 0.05),
                sigma_vec=self.config.model.get("sigma_vec", None),
                n_iters=self.config.model.get("n_iters", 500),
                max_surrogates=self.config.model.get("max_surrogates", 20000),
                use_clusters=self.config.model.get("use_clusters", False),
                n_clusters=self.config.model.get("n_clusters", 100),
                rank=self.config.model.get("rank", 6),
                save_targets=self.config.model.get("targets_dir", str(self.log_dir / "lodl_targets.pt")),
                device=kwargs.get("device", "cpu"),
            )
            kwargs["loss_handler"] = lodl_loss_handler
        return kwargs

    def load_predictor(self) -> BasePredictor:
        model_dir = self.config.model.get("model_dir")
        predictor = self.create_predictor()
        predictor.load(model_dir)
        return predictor

    def evaluate(self, predictor: BasePredictor) -> None:
        self._ensure_data_is_set()
        batch_size = self.config.model.get("kwargs", {}).get("batch_size", 0)
        names = ['train', 'validation', 'test']
        input_sets = [self.data.X_train, self.data.X_val, self.data.X_test]
        output_sets = [self.data.y_train, self.data.y_val, self.data.y_test]
        results = []
        for name, X, y in zip(names, input_sets, output_sets):
            if X is not None:
                if batch_size:
                    y_pred = predictor.predict_in_batches(X, batch_size)
                else:
                    y_pred = predictor.predict(X)
                eval_results = evaluate_prediction_model(y, y_pred)
                results.append({'set': name, **eval_results})
        # save results as csv
        df_results = pd.DataFrame(results)
        df_results.to_csv(self.log_dir / 'evaluation_results.csv', index=False)

    def train_and_evaluate(self) -> None:
        predictor = self.train()
        self.evaluate(predictor)

    def prepare_data(self) -> None:
        """
        Prepares the training and evaluation datasets for the prediction model based on the configuration.
        
        This method performs the following steps:
        1. Determines the predictor data including dataframes and feature extractor using _determine_predictor_data.
        2. Instantiates a DataPreprocessor with model-specific parameters (e.g., sequence length and prediction steps).
        3. Retrieves evaluation periods for each data class using _get_eval_periods and organizes the corresponding
            data bundles (with eval and test periods) into a structured format.
        4. Constructs the training dataset by calling prepare_train_data on the preprocessor, which returns the training features,
            labels, target indices, a fitted scaler, and the number of original data columns.
        5. Constructs the validation and test datasets by invoking the private method _prepare_check_data. This method selects and processes
            the appropriate check periods from each data bundle for evaluation purposes.
        6. Optionally computes augmentation columns if data augmentation is enabled.
        7. Aggregates the training, validation, and test sets, along with auxiliary information (target indices, scaler, augmentation columns),
            into a PredictionModelSets instance, which is then stored in self.data.
        
        Raises:
            ValueError: If the provided data (building or price) is insufficient or if evaluation/test periods are not defined properly.
        """
        data_classes, data_mgr_dfs, feature_extractor = self._determine_predictor_data()

        # Create preprocessor
        preprocessor = DataPreprocessor(
            seq_length=self.config.model.sequence_length,
            prediction_steps=self.config.model.prediction_steps
        )

        # Get evaluation periods
        periods_by_class = self._get_eval_periods(data_classes)
        data_bundles = {}
        for (d_class, periods), (arg_name, dfs) in zip(periods_by_class.items(), data_mgr_dfs.items()):
            assert d_class in arg_name, f"Mismatch in order: {d_class} != {arg_name}"
            data_bundles[arg_name] = DataBundle(dfs=dfs, eval_periods=periods["eval"], test_periods=periods["test"])
        
        # Prepare training data
        X_train, y_train, target_indexes, scaler, n_orig_data_columns = preprocessor.prepare_train_data(
            feature_extractor=feature_extractor,
            **data_bundles,
            two_d=self.two_d,
            scaled=self.config.model.get('scale_data', True),
            combined=False,
        )

        # Prepare validation and test data
        X_val, y_val, _ = self._prepare_check_data(data_bundles, "val", preprocessor, feature_extractor, scaler)
        X_test, y_test, _ = self._prepare_check_data(data_bundles, "test", preprocessor, feature_extractor, scaler)

        # Set augmentation columns if enabled
        aug_cols = [i for i in range(n_orig_data_columns)] \
            if self.config.model.get('data_augmentation', False) else []

        # Store the processed data
        self.data = PredictionModelSets(
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
            target_indexes=target_indexes,
            scaler=scaler,
            aug_cols=aug_cols,
        )

    def _prepare_check_data(
        self,
        bundles: Dict[str, DataBundle],
        period_type: str,
        preprocessor: DataPreprocessor,
        feature_extractor: FeatureExtractor,
        scaler: StandardScaler
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        modified_bundles = {}
        eval_periods_given = False
        for key, bundle in bundles.items():
            if period_type == "val":
                modified_bundles[key] = DataBundle(
                    dfs=bundle.dfs,
                    eval_periods=bundle.eval_periods,
                    test_periods=None
                )
                eval_periods_given = bool(bundle.eval_periods)
            elif period_type == "test":
                modified_bundles[key] = DataBundle(
                    dfs=bundle.dfs,
                    eval_periods=None,
                    test_periods=bundle.test_periods
                )
                eval_periods_given = bool(bundle.test_periods)
            else:
                raise ValueError(f"Unknown period type: {period_type}")
        if eval_periods_given:
            return preprocessor.prepare_evaluation_data(
                feature_extractor=feature_extractor,
                building_data=modified_bundles.get("building_data", None),
                price_data=modified_bundles.get("price_data", None),
                scaler=scaler,
                two_d=self.two_d,
            )
        return None, None, None

    def save_scaler(self) -> None:
        self._ensure_data_is_set()
        joblib.dump(self.data.scaler, str(self.log_dir / 'scaler.pkl'))

    @staticmethod
    def load_scaler(path:str) -> StandardScaler:
        return joblib.load(path)

    def _determine_predictor_data(self) -> Tuple[List[str], Dict[str, pd.DataFrame], FeatureExtractor]:
        labels = self._read_labels()
        data, feature_extractor = self._determine_data_and_format(labels)
        data_classes = self._determine_data_classes(data)
        return data_classes, data, feature_extractor

    def _determine_data_classes(self, data: Dict[str, pd.DataFrame]) -> List[str]:
        data_classes = []
        if "building_data" in data:
            data_classes.append("building")
        if "price_data" in data:
            data_classes.append("price")
        return data_classes

    def _determine_data_and_format(
            self, labels: List[DataColumn]
    ) -> Tuple[Dict[str, pd.DataFrame], FeatureExtractor]:
        data = {}
        default_features = []

        if DataColumn.LOAD in labels or DataColumn.PV in labels:
            data["building_data"] = BuildingDataManager.building_dfs
            default_features += BuildingDataManager.building_cols
        
        if DataColumn.PRICE in labels:
            data["price_data"] = BuildingDataManager.price_dfs
            default_features += BuildingDataManager.price_cols
        feature_extractor = FeatureExtractor(
            features=self.config.data.get('features', default_features),
            labels=labels,
            use_time_features=self.config.model.get('use_time_features', True),
            sin_cos_encoding=self.config.model.get('sin_cos_encoding', True),
        )
        return data, feature_extractor

    def _read_labels(self) -> List[DataColumn]:
        cfg_label = self.config.data.get("label", None)
        if cfg_label is None:
            labels = [DataColumn.LOAD, DataColumn.PV, DataColumn.PRICE]
        else:
            if isinstance(cfg_label, str):
                cfg_label = [cfg_label]
            elif not isinstance(cfg_label, list):
                cfg_label = OmegaConf.to_container(cfg_label, resolve=True)
            labels = []
            for x in cfg_label:
                assert x.upper() in DataColumn.__members__, f"Invalid label '{x}'. Available labels: {list(DataColumn.__members__.keys())}"
                labels.append(DataColumn[x.upper()])
        return labels

    @staticmethod
    def get_predictor(name:str) -> Type[BasePredictor]:
        return PredictorType[name.upper()].value

