from typing import Tuple, List, Dict, Optional, Any, Callable
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.utils.tensorboard.writer import SummaryWriter

from src.util.consts_and_types import PredictionModelSets

MAX_LOSS_THRESHOLD = 1e5

LossHandlerCallable = Callable[
    ..., Tuple[torch.Tensor, Dict[str, Any]]
]

def gen_data_loader(X: np.ndarray, y: np.ndarray, batch_size: int, seed: Optional[int] = None) -> DataLoader:
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    X_torch = torch.tensor(X, dtype=torch.float32)
    y_torch = torch.tensor(y, dtype=torch.float32)
    dataset = IndexedTensorDataset(X_torch, y_torch)

    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g)


class IndexedTensorDataset(torch.utils.data.Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X, self.y = X, y
    def __len__(self): return self.X.size(0)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], idx


class EarlyStopping:
    def __init__(self, patience: int = 10, delta: float = 0, verbose: bool = False, path: str = 'checkpoint.pt'):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.path = path
        self.best_model_state = None

    def __call__(self, val_loss: float, model: torch.nn.Module):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}: Best loss: {self.best_loss:.4f}, current loss: {val_loss:.4f}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        self.best_model_state = model.state_dict().copy()
        if self.verbose:
            print(f"Validation loss improved. Saving model...")


def default_loss_handler(
    predictions: torch.Tensor, 
    targets: torch.Tensor, 
    criterion: torch.nn.Module,
    **_kwargs
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    loss = criterion(predictions, targets)
    return loss, {"loss": loss.item()}


def train_and_evaluate_model(
        model: torch.nn.Module,
        data: PredictionModelSets,
        batch_size: int,
        device: torch.device,
        epochs: int,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        loss_handler: LossHandlerCallable = default_loss_handler,
        extra_loss_args: Dict[str, Any] = {},
        early_stopping: bool = False,
        patience: int = 10,
        delta: float = 0.0,
        checkpoint_path: str = 'checkpoint.pt',
        noise_std: float = 0.0,
        scale_std: float = 1.0,
        batches_per_epoch: Optional[int] = None,
        val_batches: Optional[int] = None,
        log_dir: Optional[Path] = None,
        verbose: bool = False,
        seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Train and evaluate a model with progress tracking and logging."""
    val_seed = seed + 1 if seed else None
    train_loader = gen_data_loader(data.X_train, data.y_train, batch_size, seed=seed)
    val_loader = gen_data_loader(data.X_val, data.y_val, batch_size, seed=val_seed) if data.X_val is not None and data.y_val is not None else None
    augmentation_columns = data.aug_cols
    
    writer = None
    if log_dir is not None:
        writer = SummaryWriter(log_dir=log_dir)
    
    train_losses = []
    val_losses = []
    early_stopper = None
    global_step = 0

    if early_stopping:
        early_stopper = EarlyStopping(patience=patience, delta=delta, verbose=verbose, path=checkpoint_path)

    epoch_pbar = tqdm(range(epochs), desc="Training epochs", disable=not verbose)
    
    for epoch in epoch_pbar:
        epoch_loss, global_step = _train_model(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            loss_handler=loss_handler,
            extra_loss_args=extra_loss_args,
            augmentation_columns=augmentation_columns,
            noise_std=noise_std,
            scale_std=scale_std,
            batches_per_epoch=batches_per_epoch,
            writer=writer,
            global_step=global_step,
            epoch=epoch,
            epochs=epochs,
            verbose=verbose,
            seed=seed,
        )
        
        train_losses.append(epoch_loss)
        log_str = f"Train Loss: {epoch_loss:.4f}"
        
        if val_loader is not None:
            val_epoch_loss = _validate_model(
                val_loader, model, criterion, device, loss_handler, extra_loss_args, 
                writer, global_step, val_batches, verbose, seed=val_seed
            )
            val_losses.append(val_epoch_loss)
            log_str += f", Val Loss: {val_epoch_loss:.4f}"
            
            if early_stopping and early_stopper is not None:
                early_stopper(val_epoch_loss, model)
                if early_stopper.early_stop:
                    epoch_pbar.write("Early stopping triggered!")
                    if early_stopper.best_model_state is not None:
                        model.load_state_dict(early_stopper.best_model_state)
                    break
        
        epoch_pbar.set_postfix_str(log_str)
        
        if writer is not None:
            writer.flush()
    
    if writer is not None:
        writer.close()
    
    return np.array(train_losses), np.array(val_losses)


def _validate_model(
        val_loader: DataLoader,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        device: torch.device,
        loss_handler: LossHandlerCallable = default_loss_handler,
        extra_loss_args: Dict[str, Any] = {},
        writer: Optional[SummaryWriter] = None,
        global_step: int = 0,
        val_batches: Optional[int] = None,
        verbose: bool = False,
        seed: Optional[int] = None,
) -> float:
    model.eval()
    val_running_loss = 0.0
    total_samples = 0
    
    # If val_batches is specified, randomly sample batches
    if val_batches is not None:
        all_val_batches = list(val_loader)
        total_val_batches = len(all_val_batches)
        
        # Randomly select batches
        g = torch.Generator()
        if seed:
            g.manual_seed(seed)
        indices = torch.randperm(total_val_batches, generator=g)[:min(val_batches, total_val_batches)]
        val_batches_to_use = [all_val_batches[i] for i in indices]
        
        # Create progress bar for sampled validation batches
        val_pbar = tqdm(val_batches_to_use, 
                        desc=f"Validation (sampling {len(indices)}/{total_val_batches} batches)", 
                        leave=False, disable=not verbose)
    else:
        # Use all validation batches
        val_pbar = tqdm(val_loader, desc="Validation", leave=False, disable=not verbose)
    
    with torch.no_grad():
        for batch_idx, (val_inputs, val_targets, idx_batch) in enumerate(val_pbar):
            val_inputs = val_inputs.to(device)
            val_targets = val_targets.to(device)
            batch_size = val_inputs.size(0)
            total_samples += batch_size
            
            # Forward pass
            val_predictions = model(val_inputs)
            
            # Process predictions and compute loss using the handler
            loss_seed = seed + batch_idx if seed is not None else None
            extra_loss_args['batch_indices'] = idx_batch
            val_loss, val_metrics = loss_handler(val_predictions, val_targets, criterion, **extra_loss_args, seed=loss_seed)
            
            # Log metrics to tensorboard
            if writer is not None:
                for metric_name, metric_value in val_metrics.items():
                    writer.add_scalar(f"val/{metric_name}", metric_value, global_step)
            
            # Update progress bar with current loss
            val_pbar.set_postfix({'loss': f"{val_loss.item():.4f}"})
            
            val_running_loss += val_loss.item() * batch_size
    
    # Calculate average loss
    val_epoch_loss = val_running_loss / total_samples
    return val_epoch_loss


def _train_model(
        model: torch.nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        device: torch.device,
        loss_handler: LossHandlerCallable,
        extra_loss_args: Dict[str, Any],
        augmentation_columns: Optional[List[int]],
        noise_std: float,
        scale_std: float,
        batches_per_epoch: Optional[int],
        writer: Optional[SummaryWriter],
        global_step: int,
        epoch: int,
        epochs: int,
        verbose: bool,
        seed: Optional[int] = None,
) -> Tuple[float, int]:
    model.train()
    running_loss = 0.0
    samples_processed = 0
    
    # If batches_per_epoch is set, randomly sample batches
    if batches_per_epoch is not None:
        # Get the dataset from the original loader
        dataset = train_loader.dataset
        total_batches = len(train_loader)
        batch_size = train_loader.batch_size
        
        # Calculate how many samples we need for the requested batches
        samples_per_epoch = min(batches_per_epoch * (batch_size or 1), len(dataset))
        
        # Create indices for random sampling (without replacement)
        g = torch.Generator()
        if seed:
            g.manual_seed(seed + epoch)
        indices = torch.randperm(len(dataset), generator=g)[:samples_per_epoch]
        
        # Create a sampler and a new DataLoader just for this epoch
        sampler = torch.utils.data.SubsetRandomSampler(indices.tolist())
        epoch_loader = DataLoader(
            dataset, 
            batch_size=batch_size,
            sampler=sampler,
            num_workers=train_loader.num_workers,
            pin_memory=train_loader.pin_memory
        )
        
        # Create progress bar for the sampled batches
        train_pbar = tqdm(epoch_loader, 
                        desc=f"Epoch {epoch+1}/{epochs} (Training - ~{samples_per_epoch//(batch_size or 1)}/{total_batches} batches)", 
                        leave=False, disable=not verbose)
    else:
        # Use all batches as normal
        train_pbar = tqdm(train_loader, 
                        desc=f"Epoch {epoch+1}/{epochs} (Training)", 
                        leave=False, disable=not verbose)
    
    for inputs, targets, idx_batch in train_pbar:
        # Prepare model inputs and targets
        inputs = inputs.to(device)
        targets = targets.to(device)
        batch_size_actual = inputs.size(0)
        samples_processed += batch_size_actual
        
        # Apply data augmentation if configured
        if augmentation_columns and noise_std > 0:
            noise = torch.randn_like(inputs[..., augmentation_columns]) * noise_std
            inputs[..., augmentation_columns] += noise
        
        if augmentation_columns and scale_std > 0:
            scale_factor = 1.0 + torch.randn(batch_size_actual, 1, 1, device=device) * scale_std
            inputs[..., augmentation_columns] *= scale_factor
        
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(inputs)
        
        # Process predictions and compute loss using the handler
        loss_seed = seed + global_step if seed is not None else None
        extra_loss_args['batch_indices'] = idx_batch
        loss, metrics = loss_handler(predictions, targets, criterion, **extra_loss_args, seed=loss_seed)
        
        # Check for exploding loss
        if loss.item() > MAX_LOSS_THRESHOLD:
            train_pbar.write(f"Warning: Loss exploded to {loss.item()} > {MAX_LOSS_THRESHOLD}. Skipping batch.")
            continue
        
        # Backward pass and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Log metrics to tensorboard
        if writer is not None:
            for metric_name, metric_value in metrics.items():
                # If the metric value is a tensor (e.g., for histograms), handle it appropriately
                if isinstance(metric_value, torch.Tensor):
                    if len(metric_value.shape) <= 1 or metric_value.shape[0] <= 10000:  # Avoid giant histograms
                        writer.add_histogram(f"train/{metric_name}", metric_value, global_step)
                else:
                    writer.add_scalar(f"train/{metric_name}", metric_value, global_step)
            
            # Optionally log gradient norms
            if epoch == 0 or global_step % 100 == 0:  # Don't log too frequently
                grad_norm = torch.stack([p.grad.norm() for p in model.parameters() 
                                       if p.grad is not None]).mean()
                writer.add_scalar("train/grad_norm", grad_norm.item(), global_step)
        
        # Update progress bar with current loss
        train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        running_loss += loss.item() * batch_size_actual
        global_step += 1
    
    # Calculate average epoch loss
    epoch_loss = running_loss / samples_processed
    return epoch_loss, global_step