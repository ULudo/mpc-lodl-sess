from typing import Dict, Tuple, Union, List

import numpy as np
import pandas as pd
from tabulate import tabulate

from src.util.consts_and_types import (
    DATA_FREQUENCY,
    BuildingEnvDF,
    FormatType,
    DataColumn
)


def check_data_format_and_convert_to_feather(data_path: str, save_path: str) -> None:
    # Check if the data file contains 'unixtime' as the first column
    df = pd.read_csv(data_path)
    if df.columns[0] != DataColumn.INDEX:
        raise ValueError(f"The first column must be '{DataColumn.INDEX}'")

    # Read the data file with 'unixtime' as index
    df = pd.read_csv(data_path, index_col=DataColumn.INDEX)
    df.index = df.index.astype(int)
    df = df.sort_index()

    # Check that data has a resolution of DATA_FREQUENCY
    time_diffs = df.index.to_series().diff().dropna()
    if not time_diffs.eq(DATA_FREQUENCY).all():
        raise ValueError(f"Data does not have a consistent frequency of {DATA_FREQUENCY} seconds.")

    # Check that no data is missing between the first and last timestamp
    expected_times = np.arange(df.index.min(), df.index.max() + DATA_FREQUENCY, DATA_FREQUENCY)
    if not np.array_equal(df.index.values, expected_times):
        missing_times = set(expected_times) - set(df.index.values)
        raise ValueError(f"Missing data at timestamps: {sorted(missing_times)}")

    # Print data validity information
    total_hours = (df.index.max() - df.index.min()) / 3600
    print(f"Data valid: {total_hours} hours of data is available.")
    print("Statistics:")
    df_stats = df.describe().loc[['min', 'max', 'mean', 'std'], :].T
    df_stats.index.name = 'Variable'
    print(tabulate(df_stats, headers='keys', tablefmt='fancy_grid', floatfmt=".3f", numalign="right", stralign="right"))

    # Save data as Feather file in the save_path
    df.reset_index().to_feather(save_path)


class BuildingDataManager:
    building_dfs: dict[str, pd.DataFrame] = {}
    price_dfs: dict[str, pd.DataFrame] = {}
    building_cols: List[str] = []
    price_cols: List[str] = []
    _lookup_table: dict[(str, str): pd.DataFrame] = {}

    @staticmethod
    def _load_data(data_format: FormatType, data_path: str) -> pd.DataFrame:
        if data_format == FormatType.CSV:
            return pd.read_csv(data_path, index_col=DataColumn.INDEX)
        elif data_format == FormatType.FEATHER:
            return pd.read_feather(data_path).set_index(DataColumn.INDEX)
        else:
            raise ValueError("Invalid data format.")

    @classmethod
    def _build_lookup_table(cls) -> None:
        building_time_ranges = {name: (df.index.min(), df.index.max()) for name, df in cls.building_dfs.items()}
        price_time_ranges = {name: (df.index.min(), df.index.max()) for name, df in cls.price_dfs.items()}

        # Compute available data ranges for each combination
        cls._lookup_table = {}
        for b_name, (b_min, b_max) in building_time_ranges.items():
            for p_name, (p_min, p_max) in price_time_ranges.items():
                overlap_start, overlap_end = max(b_min, p_min), min(b_max, p_max)
                data_range = overlap_end - overlap_start if overlap_end > overlap_start else 0
                cls._lookup_table[(b_name, p_name)] = data_range

    @classmethod
    def _set_df_dicts(cls, datasets: dict, price_data_file: str, format_type: FormatType) -> None:
        cls._datasets = []
        for idx, (name, path) in enumerate(datasets.items()):
            df_house = cls._load_data(format_type, path)
            if not cls.building_cols:
                cls.building_cols = df_house.columns.to_list()
            df_house = df_house[[DataColumn.LOAD, DataColumn.PV]]
            cls.building_dfs[name] = df_house
        df_price = cls._load_data(format_type, price_data_file)
        cls.price_dfs = {col: df_price[col].rename(str(DataColumn.PRICE)).to_frame().dropna() for col in df_price.columns}
        if cls.price_dfs and not cls.price_cols:
            cls.price_cols = next(iter(cls.price_dfs.values())).columns.to_list()

    @classmethod
    def load_datasets(cls, datasets: dict, price_data_file: str, input_format: FormatType) -> None:
        if cls.building_dfs or cls.price_dfs:
            print("Datasets already loaded. Call reset_datasets to reload.")
            return
        if not datasets:
            raise ValueError("No datasets given.")
        cls._set_df_dicts(datasets, price_data_file, input_format)
        cls._build_lookup_table()

    @staticmethod
    def _get_data(data_dict: dict, key: Union[str, int], data_name: str) -> pd.DataFrame:
        if isinstance(key, int):
            if key < 0 or key >= len(data_dict):
                raise IndexError(f"{data_name.capitalize()} index out of bounds: {key}")
            key = list(data_dict.keys())[key]
        elif isinstance(key, str):
            if key not in data_dict:
                raise KeyError(f"{data_name.capitalize()} data not found: {key}")
        else:
            raise TypeError(f"{data_name.capitalize()} must be a string or an integer.")
        return data_dict[key]

    @staticmethod
    def _determine_min_max_data_times(df: pd.DataFrame) -> Tuple[int, int]:
        tmin, tmax = df.index.min(), df.index.max()
        return tmin, tmax

    @classmethod
    def _check_data_available(cls) -> None:
        if not cls.building_dfs or not cls.price_dfs:
            raise ValueError("Datasets not loaded. Call load_datasets first.")

    @classmethod
    def get_building_data(cls, building: Union[str, int], price: Union[str, int]) -> BuildingEnvDF:
        cls._check_data_available()
        df_loads = cls._get_data(cls.building_dfs, building, "building")
        df_price = cls._get_data(cls.price_dfs, price, "price")
        data = {
            "building": [building, df_loads],
            "price": [price, df_price]
        }
        return cls._concatenate_datasets(data)

    @classmethod
    def _determine_random_dataset(cls) -> Dict[str, List[Union[str, pd.DataFrame]]]:
        n_combinations = list(cls._lookup_table.keys())
        probabilities = cls._calculate_probabilities()
        selected_index = np.random.choice(len(n_combinations), p=probabilities)
        building_name, price_name = list(cls._lookup_table.keys())[selected_index]
        return {
            "building": [building_name, cls.building_dfs[building_name]],
            "price": [price_name, cls.price_dfs[price_name]]
        }

    @classmethod
    def _calculate_probabilities(cls) -> np.ndarray:
        data_ranges = np.array(list(cls._lookup_table.values()), dtype=np.float32)
        total_data_range = sum(cls._lookup_table.values())
        probabilities = data_ranges / total_data_range
        return probabilities

    @classmethod
    def _concatenate_datasets(cls, data:Dict) -> BuildingEnvDF:
        df_building = pd.concat([data["building"][1], data["price"][1]], axis=1, join='inner')
        tmin, tmax = cls._determine_min_max_data_times(df_building)
        return BuildingEnvDF(df_building, tmin, tmax, data["building"][0], data["price"][0])

    @classmethod
    def get_random_dataset(cls) -> BuildingEnvDF:
        cls._check_data_available()
        data = cls._determine_random_dataset()
        return cls._concatenate_datasets(data)

    @classmethod
    def reset_datasets(cls) -> None:
        cls.building_dfs = {}
        cls.price_dfs = {}
        cls.building_cols = []
        cls.price_cols = []
        cls._lookup_table = {}


