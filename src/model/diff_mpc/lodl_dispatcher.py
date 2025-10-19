import os
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from collections import defaultdict

from src.model.optimizer import MPCOptimizer

from functools import lru_cache



class DirectedQuadraticLODL(nn.Module):
    def __init__(self, dim: int, rank: int = 8):
        super().__init__()
        self.dim, self.rank = dim, rank
        self.L_pp = nn.Parameter(torch.zeros(dim, rank))
        self.L_pn = nn.Parameter(torch.zeros(dim, rank))
        self.L_np = nn.Parameter(torch.zeros(dim, rank))
        self.L_nn = nn.Parameter(torch.zeros(dim, rank))

    def _quad(self, delta: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        z = delta @ L
        return (z ** 2).sum(dim=1)

    def forward(self, delta: torch.Tensor):
        quad = self._quad(delta, self.L_pp)
        return quad


class GlobalLODL(nn.Module):
    
    def __init__(
        self,
        optimizer: MPCOptimizer, 
        sigma: float = 0.05, 
        sigma_vec: Optional[List[float]] = None,
        n_iters: int = 128, 
        max_surrogates: int = 20000, 
        use_clusters: bool = False, 
        n_clusters: int = 100,
        rank: int = 6,
        save_targets: Optional[str] = None,
        device: str = 'cpu',
    ):
        super().__init__()
        self.optimizer = optimizer
        
        self.sigma = sigma
        self.sigma_vec = sigma_vec
        self.K = n_iters
        self.max_surrogates = max_surrogates
        self.use_clusters = use_clusters
        self.n_clusters = n_clusters
        self.rank = rank
        self.save_targets = save_targets
        self.device = device
        
        self.lodls = None
        self.cluster_assignments = None
        self.cluster_centers = None
        self.idx_map = None

    @staticmethod
    @lru_cache(maxsize=None)
    def _mpc_objective(
        optimizer: MPCOptimizer,
        soc_initial: float,
        load_ts: tuple,
        pv_ts: tuple,
        price_ts: tuple,
    ) -> float:
        _, _, net_import, net_export = optimizer.optimize(
            soc_initial,
            np.asarray(load_ts),
            np.asarray(price_ts),
            np.asarray(pv_ts),
            ret_full=True,
        )
        price_import = np.asarray(price_ts) + optimizer.tax
        return float((price_import * net_import - np.asarray(price_ts) * net_export).sum())
    
    def _get_perturbation_sigmas(self, y_true: np.ndarray) -> np.ndarray:
        if self.sigma_vec is not None:
            sigma_matrix = np.zeros_like(y_true)
            for i, sigma_val in enumerate(self.sigma_vec[:3]):
                sigma_matrix[:, i] = sigma_val
            return sigma_matrix
        else:
            return np.ones_like(y_true) * self.sigma
    
    def _fit_lodl_for_instance(
        self,
        y_true: np.ndarray,
        init_soc: float = 0.5
    ) -> nn.Module:
        N = y_true.shape[0]
        dim = 3 * N
        base_cost = self._mpc_objective(
            self.optimizer,
            init_soc,
            tuple(y_true[:, 0]),
            tuple(y_true[:, 1]),
            tuple(y_true[:, 2]),
        )

        sigma_matrix = self._get_perturbation_sigmas(y_true)
        deltas, costs = [], []
        
        for i in range(dim):
            pert_vector = np.zeros(dim)
            row_idx = i // 3
            col_idx = i % 3
            pert_value = sigma_matrix[row_idx, col_idx]
            pert_vector[i] = pert_value
            
            noise = pert_vector.reshape(N, 3)
            y_pert = y_true + noise
            pert_cost = self._mpc_objective(
                self.optimizer,
                init_soc,
                tuple(y_pert[:, 0]),
                tuple(y_pert[:, 1]),
                tuple(y_pert[:, 2]),
            )
            deltas.append(pert_vector)
            costs.append(pert_cost - base_cost)
            
            if len(deltas) < self.K:
                pert_vector = np.zeros(dim)
                pert_vector[i] = -pert_value
                noise = pert_vector.reshape(N, 3)
                y_pert = y_true + noise
                pert_cost = self._mpc_objective(
                    self.optimizer,
                    init_soc,
                    tuple(y_pert[:, 0]),
                    tuple(y_pert[:, 1]),
                    tuple(y_pert[:, 2]),
                )
                deltas.append(pert_vector)
                costs.append(pert_cost - base_cost)
        
        while len(deltas) < self.K:
            noise = np.random.randn(*y_true.shape) * sigma_matrix
            noise_flat = noise.reshape(-1)
            y_pert = y_true + noise
            pert_cost = self._mpc_objective(
                self.optimizer,
                init_soc,
                tuple(y_pert[:, 0]),
                tuple(y_pert[:, 1]),
                tuple(y_pert[:, 2]),
            )
            deltas.append(noise_flat)
            costs.append(pert_cost - base_cost)

        phi = np.vstack(deltas)
        C = np.asarray(costs)
        w, *_ = np.linalg.lstsq(phi ** 2, C, rcond=None)
        w = np.clip(w, 0.0, None)

        lodl = DirectedQuadraticLODL(dim=dim, rank=self.rank)
        with torch.no_grad():
            diag = torch.from_numpy(w.clip(min=0)).sqrt()
            L0 = torch.zeros(dim, self.rank)
            L0[:, 0] = diag
            if self.rank > 1:
                L0[:, 1:] = torch.randn(dim, self.rank-1) * 0.01
                
            lodl.L_pp.copy_(L0)
            lodl.L_nn.copy_(L0)
            if self.rank > 1:
                small_init = torch.randn(dim, self.rank) * 0.01
                lodl.L_pn.copy_(small_init)
                lodl.L_np.copy_(small_init)
        
        for p in lodl.parameters():
            p.requires_grad_(False)
        return lodl
    
    def _inverse_transform(self, arr: np.ndarray, scaler: StandardScaler) -> np.ndarray:
        mean = scaler.mean_[:3]
        std = scaler.scale_[:3]
        return (arr[..., :3] * std) + mean
    
    def load(self, train_inputs:np.ndarray, train_targets: np.ndarray, scaler:StandardScaler) -> None:
        if self.save_targets is not None and os.path.exists(self.save_targets):
            saved_data = torch.load(self.save_targets)
            self.lodls = saved_data['lodls'].to(self.device)
            self.idx_map = saved_data['idx_map']
            if 'cluster_assignments' in saved_data:
                self.cluster_assignments = saved_data['cluster_assignments']
            if 'cluster_centers' in saved_data:
                self.cluster_centers = saved_data['cluster_centers']
            return
        
        assert train_inputs.ndim == 3, "train_inputs must be 3D (B, T, F)"
        assert train_inputs.shape[2] == 9, "train_inputs must have 9 features (load, pv, price, 6 time features)"
        
        # Get the last time step of inputs as current states
        current_states = self._inverse_transform(train_inputs, scaler)[:, -1, :3]
        
        # Reshape train_targets if needed
        reshaped_targets = train_targets.copy()
        if reshaped_targets.ndim == 2:
            # flatten → (B, 3N)
            assert reshaped_targets.shape[1] % 3 == 0, "labels must be multiple of 3"
            N = reshaped_targets.shape[1] // 3
            reshaped_targets = reshaped_targets.reshape(reshaped_targets.shape[0], N, 3)
        
        # Apply inverse transform to get physical values
        targets_physical = self._inverse_transform(reshaped_targets, scaler)
        
        # Record the total dataset size before sampling
        total_dataset_size = targets_physical.shape[0]
        
        # Initialize the index mapping tensor with -1 (invalid)
        self.idx_map = torch.full((total_dataset_size,), -1, dtype=torch.long)
        
        # Sample a subset of instances if the dataset is large
        if total_dataset_size > self.max_surrogates:
            sampled_indices = np.random.choice(total_dataset_size, size=self.max_surrogates, replace=False)
            targets_physical = targets_physical[sampled_indices]
            current_states = current_states[sampled_indices]
            
            for new_idx, orig_idx in enumerate(sampled_indices):
                self.idx_map[orig_idx] = new_idx
        else:
            self.idx_map = torch.arange(total_dataset_size, dtype=torch.long)
            sampled_indices = np.arange(total_dataset_size)
        
        if self.use_clusters and self.n_clusters < len(targets_physical):
            flat_targets = targets_physical.reshape(targets_physical.shape[0], -1)
            
            # Perform k-means clustering
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            self.cluster_assignments = kmeans.fit_predict(flat_targets)
            self.cluster_centers = kmeans.cluster_centers_.reshape(-1, targets_physical.shape[1], 3)
            
            # Create a mapping from cluster ID to the indices of instances in that cluster
            cluster_to_indices = defaultdict(list)
            for i, cluster_id in enumerate(self.cluster_assignments):
                cluster_to_indices[cluster_id].append(i)
            
            # Select a representative instance from each cluster (closest to center)
            selected_instances = []
            selected_states = []
            selected_indices = []  # Track which original indices were selected
            
            # Keep track of mapping from cluster ID to representative index
            cluster_to_rep_idx = {}
            
            for cluster_id, center in enumerate(self.cluster_centers):
                # Get indices of instances in this cluster
                cluster_indices = cluster_to_indices[cluster_id]
                if not cluster_indices:
                    continue
                    
                # Find the instance closest to the center
                cluster_instances = targets_physical[cluster_indices]
                center_flat = center.flatten()
                distances = np.sum((cluster_instances.reshape(len(cluster_indices), -1) - center_flat)**2, axis=1)
                closest_local_idx = np.argmin(distances)
                closest_idx = cluster_indices[closest_local_idx]
                
                # Store the representative instance
                selected_instances.append(targets_physical[closest_idx])
                selected_states.append(current_states[closest_idx])
                selected_indices.append(closest_idx)
                
                # Record which representative to use for this cluster
                cluster_to_rep_idx[cluster_id] = len(selected_indices) - 1
            
            # Create a temporary idx_map for the sampled indices
            temp_idx_map = {}
            for i, cluster_id in enumerate(self.cluster_assignments):
                rep_idx = cluster_to_rep_idx[cluster_id]
                temp_idx_map[i] = rep_idx
            
            # Map every sampled instance to its cluster representative
            rep_idxs = torch.tensor(
                [temp_idx_map[int(self.idx_map[s])] for s in sampled_indices],
                dtype=torch.long
            )
            self.idx_map[sampled_indices] = rep_idxs
            
            # Use these representatives for LODL fitting
            targets_physical = np.array(selected_instances)
            current_states = np.array(selected_states)
            print(f"Using {len(targets_physical)} cluster representatives for LODL surrogate fitting")
        
        # Fill remaining -1s in idx_map by nearest neighbour (or 0)
        unmapped = (self.idx_map < 0).nonzero(as_tuple=True)[0]
        if len(unmapped):
            fallback = self.idx_map[sampled_indices[0]].item()  # any valid surrogate
            print(f"Found {len(unmapped)} unmapped instances, assigning them to surrogate {fallback}")
            self.idx_map[unmapped] = fallback
        
        # Create augmented targets with current state at position 0
        augmented_targets = np.zeros((targets_physical.shape[0], targets_physical.shape[1] + 1, 3))
        augmented_targets[:, 0 , :] = current_states
        augmented_targets[:, 1:, :] = targets_physical

        # Fit LODL models
        modules = []
        print(f"Fitting {len(augmented_targets)} LODL surrogate models...")
        for y_i in augmented_targets:
            soc0 = np.random.uniform(0.0, 1.0)  # random initial SoC
            modules.append(self._fit_lodl_for_instance(y_i.copy(), soc0))

        self.lodls = nn.ModuleList(modules).to(self.device)
        # Ensure all parameters are frozen
        for p in self.parameters():
            p.requires_grad_(False)
        
        if self.save_targets is not None:
            save_data = {
                'lodls': self.lodls,
                'idx_map': self.idx_map
            }
            
            if hasattr(self, 'cluster_assignments') and self.cluster_assignments is not None:
                save_data['cluster_assignments'] = self.cluster_assignments
            if hasattr(self, 'cluster_centers') and self.cluster_centers is not None:
                save_data['cluster_centers'] = self.cluster_centers
            os.makedirs(os.path.dirname(os.path.abspath(self.save_targets)), exist_ok=True)
            torch.save(save_data, self.save_targets)
    
    def _map_batch_to_surrogates(self, batch_indices):
        if self.idx_map.device != batch_indices.device:
            self.idx_map = self.idx_map.to(batch_indices.device)
        return self.idx_map[batch_indices]

    def forward(self, y_hat: torch.Tensor, y_true: torch.Tensor,
                indices: torch.Tensor) -> torch.Tensor:
        # Pad with zeros for current state if needed
        if y_hat.shape[1] + 3 == self.lodls[0].dim:
            zero_pad = torch.zeros(y_hat.size(0), 3, device=y_hat.device, dtype=y_hat.dtype)
            y_hat = torch.cat([zero_pad, y_hat], dim=1)
            y_true = torch.cat([zero_pad, y_true], dim=1)
        deltas = y_hat - y_true
        out = torch.empty(deltas.size(0), device=deltas.device)
        surrogate_indices = self._map_batch_to_surrogates(indices)
        
        for j, idx in enumerate(surrogate_indices):
            idx_int = int(idx)
            if 0 <= idx_int < len(self.lodls):
                out[j] = self.lodls[idx_int](deltas[j:j+1])
            else:
                # Fallback for invalid indices
                out[j] = (deltas[j] ** 2).sum()
        
        return torch.relu(out)



def lodl_loss_handler(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        criterion: GlobalLODL,
        batch_indices: torch.Tensor,
        **_kwargs
) -> Tuple[torch.Tensor, Dict[str, Any]]:

    loss_vec = criterion(predictions, targets, batch_indices)
    loss_vec = loss_vec / (100.0 * (predictions.shape[1] // 3 + 1))
    loss = loss_vec.mean()

    metrics = {
        "loss/lodl_mean": loss.item(),
        "loss/lodl_max":  loss_vec.max().item(),
    }
    return loss, metrics

