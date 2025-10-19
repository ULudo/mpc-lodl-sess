from typing import Union
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def mse(y_true: Union[pd.Series, np.ndarray], y_pred: np.ndarray) -> float:
    return mean_squared_error(y_true, y_pred)

def rmse(y_true: Union[pd.Series, np.ndarray], y_pred: np.ndarray) -> float:
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true: Union[pd.Series, np.ndarray], y_pred: np.ndarray) -> float:
    return mean_absolute_error(y_true, y_pred)

def mape(y_true: Union[pd.Series, np.ndarray], y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0
    return (np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))) * 100.0

def r2(y_true: Union[pd.Series, np.ndarray], y_pred: np.ndarray) -> float:
    return r2_score(y_true, y_pred)

def evaluate_prediction_model(y_true: Union[pd.Series, np.ndarray], y_pred: np.ndarray) -> dict:
    """
    Return a dictionary of common evaluation metrics.
    """
    return {
        'mse': mse(y_true, y_pred),
        'rmse': rmse(y_true, y_pred),
        'mae': mae(y_true, y_pred),
        'mape': mape(y_true, y_pred),
        'r2': r2(y_true, y_pred),
        **mean_and_std(y_true, y_pred)
    }

def mean_and_std(y_true: Union[pd.Series, np.ndarray], y_pred: np.ndarray) -> dict:
    """
    Return the mean and standard deviation of the prediction errors.
    """
    errors = y_true - y_pred
    return {
        'mean': np.mean(errors),
        'std': np.std(errors)
    }
