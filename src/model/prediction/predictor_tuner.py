import os
import pickle as pkl
import time
import torch
import queue
import traceback
import multiprocessing
import numpy as np
from pathlib import Path
from typing import Type, Dict, Any, Optional

import optuna
from optuna import Trial
from optuna.pruners import SuccessiveHalvingPruner, MedianPruner, NopPruner, BasePruner
from optuna.samplers import RandomSampler, TPESampler, NSGAIISampler, BaseSampler
from .base_predictor import BasePredictor
from .evaluation import mse
from ...util.consts_and_types import PredictionModelSets


def create_gpu_queue(n_gpus: int):
    """
    Create a shared queue that holds the GPU IDs [0, 1, 2, ..., n_gpus-1].
    """
    manager = multiprocessing.Manager()
    gpu_queue = manager.Queue()
    for gpu_id in range(n_gpus):
        gpu_queue.put(gpu_id)
    return gpu_queue


class PredictorTuner:
    """
    Class for hyperparameter tuning of BasePredictor models using Optuna.
    Measures training time and uses custom evaluation metrics.
    """

    def __init__(
            self,
            log_dir: Path,
            predictor: Type[BasePredictor],
            data: PredictionModelSets,
            n_trials: int = 100,
            sampler: str = 'tpe',
            n_startup_trials: int = 5,
            pruner: str = "median",
            n_jobs: int = 1,
            direction: str = 'minimize',
            study_name: Optional[str] = None,
            fix_kwargs: Optional[dict[str, Any]] = None,
            n_gpus: int = 0,
            sampler_kwargs: Optional[dict[str, Any]] = None,
            seed: Optional[int] = None
    ):
        self.log_dir = log_dir
        self.predictor = predictor
        self.data = data
        self.n_trials = n_trials
        self.fix_kwargs = fix_kwargs

        self.direction = direction
        self.study_name = study_name
        self.n_jobs = n_jobs

        self.sampler = self._get_sampler(n_startup_trials, sampler, seed)
        self.pruner = self._get_pruner(n_startup_trials, pruner)

        self.study = None
        self.sampler_kwargs = sampler_kwargs if sampler_kwargs is not None else {}
        if n_gpus > 0:
            self.gpu_queue = create_gpu_queue(n_gpus)
        self.seed = seed

    def objective(self, trial: Trial) -> float:

        # Sample hyperparameters ---
        params = self.predictor.sample(trial, self.fix_kwargs)
        params.update(self.fix_kwargs)

        # Choose GPU if possible ---
        device, gpu_id = self._get_device()
        params['device'] = device

        # Initialize model ---
        model = self.predictor(**params, seed=self.seed)

        # Train & Evaluate ---
        try:
            start_time = time.time()
            res_eval_train, res_eval_val = model.fit(self.data)
            training_time = time.time() - start_time

            if batch_size := params.get('batch_size', 0):
                y_pred = model.predict_in_batches(self.data.X_val, batch_size)
            else:
                y_pred = model.predict(self.data.X_val)
            loss = mse(self.data.y_val, y_pred)

            # Indicate that this trial did not end due to an exception.
            trial.set_user_attr("pruned", "")

        except Exception as e:
            # Print the stack trace and exception
            print("Exception occurred:", repr(e))
            traceback.print_exc()

            # Store the exception info in trial attributes (useful for debugging).
            trial.set_user_attr("pruned", f"Exception: {repr(e)}")
            # Raise TrialPruned so Optuna marks it as a pruned/failed trial.
            raise optuna.exceptions.TrialPruned()

        finally:
            # Cleanup ---
            if device != 'cpu':
                # Clear out PyTorch’s CUDA cache to reduce fragmentation for next trial
                torch.cuda.empty_cache()
                # Release the GPU back to the queue
                if gpu_id is not None:
                    self.gpu_queue.put(gpu_id)

        # Store training time in trial attributes ---
        trial.set_user_attr('training_time', training_time)

        # Save evaluation metrics to disk (optional) ---
        eval_dir = self.log_dir / 'evaluation'
        eval_dir.mkdir(exist_ok=True)
        np.save(os.path.join(eval_dir, f'{trial.number}_eval_train.npy'), res_eval_train)
        np.save(os.path.join(eval_dir, f'{trial.number}_eval_val.npy'), res_eval_val)

        return loss

    def _get_device(self):
        device = 'cpu'
        gpu_id = None  # keep track of which GPU is allocated
        if hasattr(self, 'gpu_queue'):
            try:
                gpu_id = self.gpu_queue.get_nowait()
                device = f'cuda:{gpu_id}'
            except queue.Empty:
                pass
        return device, gpu_id

    def tune(self) -> Dict[str, Any]:
        self.study = optuna.create_study(
            sampler=self.sampler,
            pruner=self.pruner,
            study_name=self.study_name,
            direction=self.direction)

        try:
            self.study.optimize(
                self.objective,
                n_jobs=self.n_jobs,
                n_trials=self.n_trials)
        except KeyboardInterrupt:
            pass
        finally:
            # Save study
            with open(os.path.join(self.log_dir, "study.pkl"), "wb+") as f:
                pkl.dump(self.study, f)

        # Write report
        df_trials = self.study.trials_dataframe()
        df_trials.to_csv(os.path.join(self.log_dir, "trials.csv"))

        # Logging
        best_trial = self.study.best_trial
        best_params = self.study.best_params
        best_training_time = best_trial.user_attrs.get('training_time', None)
        print(f"Best trial: {best_trial.value}")
        print(f"Best params: {best_params}")
        print(f"Best training time: {best_training_time}")

        return best_params

    @staticmethod
    def _get_sampler(n_startup_trials: int, sampler_name: str, seed: int) -> BaseSampler:
        if sampler_name == 'random':
            sampler = RandomSampler()
        elif sampler_name == 'tpe':
            sampler = TPESampler(n_startup_trials=n_startup_trials, seed=seed, multivariate=True)
        elif sampler_name == 'nsga2':
            sampler = NSGAIISampler(seed=seed)
        else:
            raise ValueError(f"Sampler {sampler_name} not supported.")
        return sampler

    @staticmethod
    def _get_pruner(n_startup_trials: int, pruner_name: str) -> BasePruner:
        if pruner_name == "halving":
            pruner = SuccessiveHalvingPruner(
                min_resource=1,
                reduction_factor=4,
                min_early_stopping_rate=0)
        elif pruner_name == "median":
            pruner = MedianPruner(n_startup_trials=n_startup_trials)
        elif pruner_name == "none":
            pruner = NopPruner()
        else:
            raise ValueError(f"Unknown pruner: {pruner_name}")
        return pruner
