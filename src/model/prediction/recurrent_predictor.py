from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
from optuna import Trial

from ..nets.recurrent_net import RNNModel
from .base_predictor import BasePredictor


class RecurrentPredictor(BasePredictor):
    """
    A recurrent neural network (LSTM or GRU) predictor for multi-step time series forecasting.
    Predicts 'horizon' future steps at once (direct multi-step).
    """

    def __init__(
            self,
            rnn_type: str = 'LSTM',
            hidden_units: int = 64,
            num_layers: int = 1,
            dropout: float = 0.0,
            horizon: int = 96,
            **kwargs
    ):
        """
        Args:
            rnn_type: Type of RNN ('LSTM' or 'GRU')
            hidden_units: Number of hidden units in each RNN layer
            num_layers: Number of stacked RNN layers
            dropout: Dropout rate for RNN
            horizon: Number of future time steps to predict
            **kwargs: Additional parameters passed to BasePredictor
        """
        super().__init__(**kwargs)
        self.rnn_type = rnn_type
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.dropout = dropout
        
    def _build_model(self) -> nn.Module:
        """
        Build an RNN model.
        
        Returns:
            A recurrent neural network model
        """
        model = RNNModel(
            input_size=self.input_size,
            hidden_size=self.hidden_units,
            num_layers=self.num_layers,
            horizon=self.output_size,
            rnn_type=self.rnn_type,
            dropout=self.dropout
        )
        return model.to(self._device)

    @classmethod
    def _sample_model_params(cls, trial: Trial, given_kwargs:Dict[str, Any]) -> Dict[str, Any]:
        """
        Suggest hyperparameters for Optuna or other tuning.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of hyperparameter suggestions
        """
        rnn_type = given_kwargs.get('rnn_type', trial.suggest_categorical('rnn_type', ['LSTM', 'GRU']))
        hidden_units = given_kwargs.get('hidden_units', trial.suggest_categorical('hidden_units', [64, 128, 256, 512]))
        num_layers = given_kwargs.get('num_layers', trial.suggest_int('num_layers', 2, 8, step=2))
        dropout = given_kwargs.get('dropout', trial.suggest_float('dropout', 0.0, 0.5, step=0.1))
        return {
            'rnn_type': rnn_type,
            'hidden_units': hidden_units,
            'num_layers': num_layers,
            'dropout': dropout,
        }
