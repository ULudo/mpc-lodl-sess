from dataclasses import dataclass
from typing import Tuple, List, Dict, Union, Type, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.util.consts_and_types import DATA_FREQUENCY, DataColumn
from src.util.functions import sin_encode_day, cos_encode_day, sin_encode_hour, cos_encode_hour, sin_encode_month, \
    cos_encode_month


@dataclass
class DataBundle:
    dfs: Dict[str, pd.DataFrame]
    eval_periods: Optional[Dict[str, Union[Tuple[int, int], List[Tuple[int, int]]]]] = None
    test_periods: Optional[Dict[str, Union[Tuple[int, int], List[Tuple[int, int]]]]] = None


class FeatureExtractor:
    _labels = {}
    times = {
        'hour': lambda df: df.index.hour,
        'day_of_week': lambda df: df.index.dayofweek,
        'month': lambda df: df.index.month,
    }
    encodings = {
        'hour_sin': lambda df: sin_encode_hour(df.hour),
        'hour_cos': lambda df: cos_encode_hour(df.hour),
        'day_of_week_sin': lambda df: sin_encode_day(df.day_of_week),
        'day_of_week_cos': lambda df: cos_encode_day(df.day_of_week),
        'month_sin': lambda df: sin_encode_month(df.month),
        'month_cos': lambda df: cos_encode_month(df.month)
    }

    def __init__(
            self,
            features: Optional[List[DataColumn]] = None,
            labels: Optional[List[DataColumn]] = None,
            use_time_features: bool = True,
            sin_cos_encoding: bool = True
    ) -> None:
        self._features = {f: lambda df: df[f] for f in features} if features else {}
        self._labels = {l: lambda df: df[l] for l in labels} if labels else {}
        self.use_time_features = use_time_features
        self.sin_cos_encoding = sin_cos_encoding

    @property
    def features(self) -> List[str]:
        if self.use_time_features:
            time_features = self.encodings.keys() if self.sin_cos_encoding else self.times.keys()
            return list(self._features.keys()) + list(time_features)
        return list(self._features.keys())

    @property
    def labels(self) -> List[str]:
        return list(self._labels.keys())


class DataPreprocessor:

    def __init__(
            self,
            seq_length: int,
            prediction_steps: int,
    ) -> None:
        self.seq_length = seq_length
        self.prediction_steps = prediction_steps

    def prepare_train_data(
            self,
            feature_extractor: FeatureExtractor,
            building_data: Optional[DataBundle] = None,
            price_data: Optional[DataBundle] = None,
            two_d: bool = False,
            scaled: bool = True,
            combined: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler, int]:
        """
        Prepares training data for prediction tasks by:
        
          1. Extracting valid training segments by removing any evaluation or test periods.
          2. Optionally merging building and price data if both are provided.
          3. Adding time features if enabled in the feature extractor.
          4. Optionally scaling the features using a StandardScaler.
          5. Creating labeled sequences for training.
          6. Optionally converting the data segments to a combined array (useful for multi-target models).
          7. Reshaping the output to 2D if required.

        Args:
            feature_extractor (FeatureExtractor): Defines the features and labels to be extracted from the data.
            building_data (Optional[DataBundle]): A bundle containing building-related data (e.g. LOAD, PV). 
                Expected to include evaluation and test period definitions which are removed.
            price_data (Optional[DataBundle]): A bundle containing price-related data and associated periods.
            two_d (bool): If True, reshapes the resulting feature array to 2D.
            scaled (bool): If True, scales the segmentation data.
            combined (bool): If True, converts data segments into a single combined array; 
                in this case, no labeled sequences are created and y and indices are empty arrays.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler, int]:
                - X: The features array. It is a 3D array by default or reshaped to 2D if two_d is True.
                - y: The labels array corresponding to the prediction targets. It is reshaped to 2D.
                - idxs: A 1D array with the indices in the feature array corresponding to target columns.
                - scaler: The fitted StandardScaler used to scale the data (or None if scaled is False).
                - n_orig_data_columns: The number of columns in the original, unmodified data segments.
        
        Raises:
            ValueError: If neither building_data nor price_data is provided, or if no training data remains 
                        following the removal of evaluation and test periods.
        """

        if bool(building_data is None) ^ bool(price_data is None):
            data = building_data or price_data
            dfs = self._remove_check_test_periods(data)
        elif building_data and price_data:
            cleaned_building_dfs = self._remove_check_test_periods(building_data)
            cleaned_price_dfs = self._remove_check_test_periods(price_data)

            dfs = []
            for building_df in cleaned_building_dfs:
                for price_df in cleaned_price_dfs:
                    merged_df = building_df.merge(price_df, left_index=True, right_index=True, how='inner')
                    dfs.append(merged_df)

        else:
            raise ValueError("Either building or price data must be provided")
        
        assert dfs, 'No training data remains after removing evaluation and test periods.'
        n_orig_data_columns = dfs[0].shape[1]

        # Add time features
        if feature_extractor.use_time_features:
            dfs = [self._add_time_features(df, feature_extractor.sin_cos_encoding) for df in dfs]

        # Features and labels
        features = feature_extractor.features

        if scaled:
            dfs, scaler = self._scale_data_segments(features, dfs)
        else:
            scaler = None

        if combined:
            X = self._convert_to_combined_segment_array(features, dfs)
            y, idxs = np.array([]), np.array([])
        else:
            # Scale and create labeled sequences
            labels = feature_extractor.labels
            X, y, idxs = self._create_labeled_sequences(features, labels, dfs)
            y = y.reshape(y.shape[0], -1)

        if two_d:
            X = X.reshape(X.shape[0], -1)
        return X, y, idxs, scaler, n_orig_data_columns


    def prepare_evaluation_data(
            self,
            feature_extractor: FeatureExtractor,
            building_data: Optional[DataBundle] = None,
            price_data: Optional[DataBundle] = None,
            scaler: StandardScaler = None,
            two_d: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepares evaluation (or check) data for prediction tasks by:
        
          1. Validating that each provided DataBundle (building_data or price_data) contains only evaluation periods 
             or test periods (not both).
          2. Selecting the relevant check periods from the given DataBundle(s) based on the defined evaluation or test periods.
          3. If both building_data and price_data are provided, merging their data on matching indices.
          4. Adding time features to the data segments if enabled in the feature extractor.
          5. Optionally scaling the features using the provided StandardScaler (if available).
          6. Creating labeled sequences for the evaluation data.
          7. Reshaping the features array to 2D if specified.

        Args:
            feature_extractor (FeatureExtractor): Specifies the features and labels to be extracted from the data.
            building_data (Optional[DataBundle]): A bundle of building-related data (e.g. LOAD, PV) that includes 
                definitions for evaluation or test periods.
            price_data (Optional[DataBundle]): A bundle of price-related data with associated evaluation or test periods.
            scaler (StandardScaler, optional): A pre-fitted scaler used to transform the features. Defaults to None.
            two_d (bool, optional): If True, reshapes the resulting features array to 2D. Defaults to False.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - X: A 3D numpy array of features with shape (num_samples, sequence_length, num_features). 
                     If two_d is True, X is reshaped to a 2D array.
                - y: A 3D numpy array of labels with shape (num_samples, sequence_length, num_labels), 
                     reshaped to 2D.
                - idxs: A 1D numpy array containing the indices of the target columns.
        """
        # Enforce that each data bundle only has eval_periods or test_periods set, not both.
        def check_bundle(bundle: DataBundle, bundle_name: str):
            if bundle.eval_periods and bundle.test_periods:
                raise ValueError(f"For {bundle_name}, only eval_periods or test_periods should be set, not both.")
        if building_data:
            check_bundle(building_data, "building_data")
        if price_data:
            check_bundle(price_data, "price_data")

        get_periods = lambda item: item.eval_periods or item.test_periods
        if bool(building_data is None) ^ bool(price_data is None):
            dfs = building_data or price_data
            dfs = self._select_check_data(dfs.dfs, get_periods(dfs))
        elif building_data and price_data:
            building_dfs = self._select_check_data(building_data.dfs, get_periods(building_data))
            price_dfs = self._select_check_data(price_data.dfs, get_periods(price_data))
            dfs = []
            for building_df in building_dfs:
                for price_df in price_dfs:
                    merged_df = building_df.merge(price_df, left_index=True, right_index=True, how='inner')
                    dfs.append(merged_df)
        else:
            raise ValueError("Either building or price data must be provided")

        assert dfs, 'No evaluation data remains after selecting check periods.'

        # Add time features
        if feature_extractor.use_time_features:
            dfs = [self._add_time_features(df, feature_extractor.sin_cos_encoding) for df in dfs]

        # Features and labels
        features = feature_extractor.features
        labels = feature_extractor.labels

        # Scale and create labeled sequences
        if scaler:
            for df in dfs:
                df[features] = scaler.transform(df[features])

        X, y, idxs = self._create_labeled_sequences(features, labels, dfs)
        if two_d:
            X = X.reshape(X.shape[0], -1)
        y = y.reshape(y.shape[0], -1)
        return X, y, idxs


    def _convert_to_combined_segment_array(
            self, features, dfs):
        combined_segs = []
        for df in dfs:
            data = df[features].values

            if len(data) < self.seq_length + self.prediction_steps:
                segs = np.array([])
            else:
                X = []
                end_limit = len(data) - self.seq_length - self.prediction_steps + 1
                for idx in range(end_limit):
                    sep_idx = idx + self.seq_length + self.prediction_steps
                    X.append(data[idx:sep_idx])

                segs = np.array(X)

            if segs.size > 0:
                combined_segs.append(segs)
        if not combined_segs:
            raise ValueError('No data remains after creating sequences.')
        return np.concatenate(combined_segs, axis=0) if len(combined_segs) > 1 else combined_segs[0]

    @staticmethod
    def _add_time_features(
            df: pd.DataFrame,
            sin_cos_encoding: bool = True
    ) -> pd.DataFrame:
        df = df.copy()
        df.index = pd.to_datetime(df.index, unit='s', utc=True)
        df.index = df.index.tz_convert(None)
        for name, fun in FeatureExtractor.times.items():
            df[name] = fun(df)
        if sin_cos_encoding:
            for name, fun in FeatureExtractor.encodings.items():
                df[name] = fun(df)
        return df

    @staticmethod
    def _scale_features(
            df: pd.DataFrame, features: List[str]
    ) -> Tuple[pd.DataFrame, StandardScaler]:
        scaler = StandardScaler()
        df[features] = scaler.fit_transform(df[features])
        return df, scaler

    def _create_sequences(
            self, data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if len(data) < self.seq_length + self.prediction_steps:
            return np.array([]), np.array([])

        X, y = [], []
        total_length = len(data)
        end_limit = total_length - self.seq_length - self.prediction_steps + 1
        for idx in range(end_limit):
            sep_idx = idx + self.seq_length
            pred_end_idx = sep_idx + self.prediction_steps
            X.append(data[idx:sep_idx])
            y.append(data[sep_idx:pred_end_idx])
        return np.array(X), np.array(y)

    def _remove_check_test_periods(self, data: DataBundle) -> List[pd.DataFrame]:
        dfs = []
        for name, df in data.dfs.items():
            intervals = self._collect_intervals(data.eval_periods, data.test_periods, name)
            intervals.sort(key=lambda x: x[0])
            merged_intervals = self._merge_overlapping_or_adjacent_intervals(intervals)
            segments = self._build_training_segments(df.copy(), merged_intervals)
            dfs.extend(segments)
        return dfs

    @staticmethod
    def _build_training_segments(
            df: pd.DataFrame,
            merged_intervals: List[Tuple[int, int]]
    ) -> List[pd.DataFrame]:
        segments: List[pd.DataFrame] = []
        if not merged_intervals:
            # No forbidden intervals, entire df is training data
            segments.append(df)
        else:
            # Start from the beginning of the dataset to the first interval
            df_start = df.index.min()
            df_end = df.index.max()

            current_start = df_start - DATA_FREQUENCY
            for (f_start, f_end) in merged_intervals:
                # Segment before forbidden interval
                seg = df.loc[(df.index > current_start) & (df.index < f_start)]
                if not seg.empty:
                    segments.append(seg)
                current_start = f_end

            # After the last forbidden interval
            if current_start < df_end:
                seg = df.loc[(df.index > current_start) & (df.index <= df_end)]
                if not seg.empty:
                    segments.append(seg)
        return segments

    @staticmethod
    def _merge_overlapping_or_adjacent_intervals(
            intervals: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        merged_intervals = []
        for interval in intervals:
            if not merged_intervals:
                merged_intervals.append(interval)
            else:
                last_start, last_end = merged_intervals[-1]
                cur_start, cur_end = interval
                # Check if intervals are within one data frequency step
                if cur_start - last_end <= DATA_FREQUENCY:
                    merged_intervals[-1] = (last_start, max(last_end, cur_end))
                else:
                    merged_intervals.append(interval)
        return merged_intervals

    @staticmethod
    def _collect_intervals(
            eval_periods: Dict[str, Union[Tuple[int, int], List[Tuple[int, int]]]],
            test_periods: Dict[str, Union[Tuple[int, int], List[Tuple[int, int]]]],
            name: str
    ) -> List[Tuple[int, int]]:
        intervals = []
        eval_slices = eval_periods.get(name, [])
        test_slices = test_periods.get(name, [])
        # Ensure eval_slices and test_slices are lists
        if isinstance(eval_slices, tuple):
            eval_slices = [eval_slices]
        if isinstance(test_slices, tuple):
            test_slices = [test_slices]
        intervals.extend(eval_slices)
        intervals.extend(test_slices)
        return intervals

    @staticmethod
    def _scale_data_segments(
            features: List[str],
            dfs: List[pd.DataFrame]
    ) -> Tuple[List[pd.DataFrame], StandardScaler]:
        # Concat and scale data
        combined_df = pd.concat(dfs)
        scaler = StandardScaler()
        combined_df[features] = scaler.fit_transform(combined_df[features])
        # Split back to segments after scaling
        scaled_dfs = []
        offset = 0
        for seg in dfs:
            seg_len = len(seg)
            scaled_seg = combined_df.iloc[offset:offset + seg_len]
            scaled_dfs.append(scaled_seg)
            offset += seg_len
        return scaled_dfs, scaler

    def _create_labeled_sequences(
            self,
            features: List[str],
            labels: List[str],
            dfs: List[pd.DataFrame]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        target_indices = [features.index(label) for label in labels]
        X_segments, y_segments = [], []
        for seg in dfs:
            data_array = seg[features].values
            X_seg, y_seg = self._create_sequences(data_array)
            if X_seg.size > 0:
                y_seg = y_seg[..., target_indices]
                X_segments.append(X_seg)
                y_segments.append(y_seg)
        if not X_segments:
            raise ValueError('No data remains after creating sequences.')
        X = np.concatenate(X_segments, axis=0) if len(X_segments) > 1 else X_segments[0]
        y = np.concatenate(y_segments, axis=0) if len(y_segments) > 1 else y_segments[0]
        return X, y, np.array(target_indices)

    @staticmethod
    def _select_check_data(
            data_mgr_dfs: Dict[str, pd.DataFrame],
            check_periods: Dict[str, Union[Tuple[int, int], List[Tuple[int, int]]]]
    ) -> List[pd.DataFrame]:
        dfs = []
        for name, periods in check_periods.items():
            if isinstance(periods, tuple):
                periods = [periods]
            for start, end in periods:
                df = data_mgr_dfs[name].loc[(data_mgr_dfs[name].index >= start) & (data_mgr_dfs[name].index <= end)]
                if not df.empty:
                    dfs.append(df)
        return dfs
