import abc
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from optuna import Trial

from src.model.diff_mpc.lodl_dispatcher import GlobalLODL
from src.util.consts_and_types import PredictionModelSets
from src.model.prediction.helpers import default_loss_handler, train_and_evaluate_model


class BasePredictor(object):
    """
    Abstract base class for prediction models.
    """
    def __init__(
            self,
            input_size: int = 1,
            output_size: int = 1,
            learning_rate: float = 1e-3,
            batch_size: int = 32,
            epochs: int = 20,
            early_stopping: bool = False,
            patience: int = 10,
            delta: float = 0.0,
            batches_per_epoch: Optional[int] = None,
            val_batches: Optional[int] = None,
            noise_std: float = 0.0,
            scale_std: float = 1.0,
            criterion: nn.Module = nn.MSELoss(),
            loss_handler: callable = default_loss_handler,
            loss_args: Dict[str, Any] = {},
            checkpoint_path: str = "checkpoint.pt",
            verbose: bool = True,
            device: str = 'cpu',
            log_dir: Optional[Path] = None,
            seed: Optional[int] = None,
    ) -> None:
        # Model parameters
        self.input_size = input_size
        self.output_size = output_size

        # Training parameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.batches_per_epoch = batches_per_epoch
        self.val_batches = val_batches
        self.early_stopping = early_stopping
        self.patience = patience
        self.delta = delta
        self.checkpoint_path = checkpoint_path
        self.seed = seed
        
        # Loss parameters
        self.criterion = criterion
        self.loss_handler = loss_handler
        self.loss_args = loss_args
        self.noise_std = noise_std
        self.scale_std = scale_std
        
        # Device and logging
        self.verbose = verbose
        self.log_dir = log_dir
        self._device = torch.device(device)
        
        # The model instance (to be set by child classes)
        self.model = None

    @abc.abstractmethod
    def _build_model(self) -> nn.Module:
        """
        Build and return a model instance.
        Implementation depends on the specific predictor type.
        """
        pass

    def fit(self, data: PredictionModelSets) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit the model to the data.
        
        Args:
            data: A data bundle containing training, validation sets and optional scaler
            
        Returns:
            A tuple containing training and validation loss histories
        """
        if self.model is None:
            self.model = self._build_model()

        if type(self.criterion) is GlobalLODL:
            self.criterion.load(data.X_train, data.y_train, data.scaler)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        train_losses, val_losses = train_and_evaluate_model(
            model=self.model,
            data=data,
            batch_size=self.batch_size,
            device=self._device,
            epochs=self.epochs,
            optimizer=optimizer,
            criterion=self.criterion,
            loss_handler=self.loss_handler,
            extra_loss_args=self.loss_args,
            early_stopping=self.early_stopping,
            patience=self.patience,
            delta=self.delta,
            checkpoint_path=self.checkpoint_path,
            noise_std=self.noise_std,
            scale_std=self.scale_std,
            batches_per_epoch=self.batches_per_epoch,
            val_batches=self.val_batches,
            log_dir=self.log_dir,
            verbose=self.verbose,
            seed=self.seed,
        )
        return train_losses, val_losses

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions for input data X.
        
        Args:
            X: Input data
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")

        self.model.eval()
        X_torch = torch.tensor(X, dtype=torch.float32).to(self._device)
        with torch.no_grad():
            preds = self.model(X_torch)
        return preds.cpu().numpy()

    def predict_in_batches(self, X: np.ndarray, batch_size:int) -> np.ndarray:
        """
        Helper method to predict in batches to avoid memory issues with large datasets.
        
        Args:
            X: Input data
            batch_size: Size of batch to use
            
        Returns:
            Predictions for all input data
        """
        y_pred = []
        for i in range(0, len(X), batch_size):
            X_batch = X[i:i + batch_size]
            y_batch_pred = self.predict(X_batch)
            if y_batch_pred.ndim == 1:
                y_batch_pred = y_batch_pred.reshape(1, -1)
            y_pred.append(y_batch_pred)
        return np.concatenate(y_pred, axis=0)

    def save(self, path: str) -> None:
        """
        Save model weights.
        
        Args:
            path: File path to save the model
        """
        if self.model is None:
            raise RuntimeError("No model to save.")
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        """
        Load model weights.
        
        Args:
            path: File path to load the model from
        """
        if self.model is None:
            self.model = self._build_model()
        self.model.load_state_dict(torch.load(path, map_location=self._device))

    @classmethod
    def _sample_model_params(cls, trial: Trial, given_kwargs:Dict[str, Any]) -> Dict[str, Any]:
        """
        Sample hyperparameters for tuning.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of hyperparameter suggestions
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    @classmethod
    def sample(cls, trial:Trial, given_kwargs:Dict[str, Any]) -> Dict[str, Any]:
        """
        Sample hyperparameters for tuning with Optuna.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of hyperparameter suggestions
        """
        params = cls._sample_model_params(trial, given_kwargs)
        params["learning_rate"] = given_kwargs.get('learning_rate', trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True))
        params["noise_std"] = given_kwargs.get('noise_std', trial.suggest_float('noise_std', 0.0, 1.0, step=0.01))
        params["scale_std"] = given_kwargs.get('scale_std', trial.suggest_float('scale_std', 0.0, 1.0, step=0.01))
        params["batch_size"] = given_kwargs.get('batch_size', trial.suggest_categorical('batch_size', [32, 64, 128, 256]))
        return params