# File: pll_nn/nn_dataset.py

import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from omegaconf import OmegaConf
from pyDOE import lhs

# Change this import to wherever your ODE modelling helper is:
from src.dataset.create_dataset_pll_functions import ODE_modellingPLLRom

class DataSamplerPLLRom:
    """
    Load and sample PLL-ROM dataset for training a PINN.

    Args:
        cfg (OmegaConf): Configuration containing dataset settings.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_flag = cfg.model.model_flag  # should be "PLL_ROM"
        self.shuffle = cfg.dataset.shuffle
        self.split_ratio = cfg.dataset.split_ratio
        self.new_coll_points_flag = cfg.dataset.new_coll_points_flag
        self.total_time = cfg.time

        # Load raw trajectory data from .pkl
        self.data, self.input_dim, self.total_trajectories = self.load_data()

        # Convert trajectories to (x,y) pairs up to time limit
        self.x, self.y = self.data_input_target_limited(self.data, self.total_time)
        self.sample_per_trajectory = self.x.shape[0] // self.total_trajectories

        # Split into train/val/test sets
        if cfg.dataset.validation_flag:
            splits = self.train_val_test_split(self.x, self.y, self.split_ratio, True)
            self.x_train, self.x_val, self.x_test, self.y_train, self.y_val, self.y_test = splits
        else:
            splits = self.train_val_test_split(self.x, self.y, self.split_ratio, False)
            self.x_train, self.x_test, self.y_train, self.y_test = splits

        # Collocation points: either reuse x_train or generate new ones
        if self.new_coll_points_flag:
            self.x_train_col = self.create_col_points().requires_grad_(True)
        else:
            self.x_train_col = self.x_train.clone().detach().to(self.device).requires_grad_(True)

        # Input/output normalization parameters if needed
        if self.cfg.dataset.transform_input != "None":
            self.minus_input, self.divide_input = self.define_minus_divide(self.x_train, self.x_train_col)
            self.minus_input = torch.nn.Parameter(self.minus_input, requires_grad=False)
            self.divide_input = torch.nn.Parameter(self.divide_input, requires_grad=False)

        if self.cfg.dataset.transform_output != "None":
            self.minus_target, self.divide_target = self.define_minus_divide(self.y_train, torch.empty(0))
            self.minus_target = torch.nn.Parameter(self.minus_target, requires_grad=False)
            self.divide_target = torch.nn.Parameter(self.divide_target, requires_grad=False)

    def load_data(self):
        """
        Load trajectory data from a pickle file.

        Returns:
            data_list (list): List of ODE solution arrays.
            input_dim (int): Dimensionality of each trajectory record.
            total_trajectories (int): Number of trajectories.
        """
        number_of_dataset = self.cfg.dataset.number
        name = f"{self.model_flag}/dataset_v{number_of_dataset}.pkl"
        dataset_path = os.path.join(self.cfg.dirs.dataset_dir, name)

        if not os.path.isfile(dataset_path):
            raise FileNotFoundError(f"Cannot find dataset file at {dataset_path}")

        with open(dataset_path, 'rb') as f:
            sol_list = pickle.load(f)

        input_dim = sol_list[0].shape[0]  # e.g., [time; delta; omega]
        total_trajectories = len(sol_list)

        if self.shuffle:
            np.random.shuffle(sol_list)

        return sol_list, input_dim, total_trajectories

    def data_input_target_limited(self, data, time_limit):
        """
        Convert raw trajectories to x (input) and y (target) up to time_limit.

        Input x format: [t, δ, ω] for each time step. Target y: [δ, ω] next-step or same-step.
        """
        x_list = []
        y_list = []

        for traj in data:
            arr = torch.tensor(traj, dtype=torch.float32)
            # arr shape: [3, num_time_points] or similar
            # Transpose to [num_time_points, 3]
            arr_t = arr.T
            arr_t = arr_t[arr_t[:,0] <= time_limit];  # filter by time
            if arr_t.shape[0] == 0:
                continue

            # Input x = [t_i, δ_i, ω_i]
            x_traj = arr_t.clone().detach().requires_grad_(True)
            # Target y = [δ_i, ω_i]
            y_traj = arr_t[:, 1:].clone().detach()

            x_list.append(x_traj)
            y_list.append(y_traj)

        x_cat = torch.cat(x_list, dim=0)
        y_cat = torch.cat(y_list, dim=0)
        # Update time in cfg in case truncated
        self.total_time = min(time_limit, x_cat[:,0].max().item())
        return x_cat.to(self.device), y_cat.to(self.device)

    def train_val_test_split(self, x_data, y_data, split_ratio, val_flag=True):
        """
        Split data into train / val / test sets based on full trajectories.

        Args:
            x_data (torch.Tensor): All inputs, shape [N_total, input_dim].
            y_data (torch.Tensor): All targets, shape [N_total, output_dim].
            split_ratio (float): Fraction of trajectories used for training.
            val_flag (bool): Whether to produce a validation set.

        Returns:
            If val_flag=True: x_train, x_val, x_test, y_train, y_val, y_test
            else: x_train, x_test, y_train, y_test
        """
        total_points_per_traj = self.sample_per_trajectory
        total_traj = self.total_trajectories

        # Number of training trajectories
        num_train_traj = int(total_traj * split_ratio)
        train_end = num_train_traj * total_points_per_traj

        if val_flag:
            # Split remaining for val and test (half each)
            num_val_traj = (total_traj - num_train_traj) // 2
            val_end = train_end + num_val_traj * total_points_per_traj

            x_train = x_data[:train_end]
            x_val = x_data[train_end:val_end]
            x_test = x_data[val_end:]
            y_train = y_data[:train_end]
            y_val = y_data[train_end:val_end]
            y_test = y_data[val_end:]

            return x_train, x_val, x_test, y_train, y_val, y_test
        else:
            x_train = x_data[:train_end]
            x_test = x_data[train_end:]
            y_train = y_data[:train_end]
            y_test = y_data[train_end:]
            return x_train, x_test, y_train, y_test

    def create_col_points(self):
        """
        Create collocation points via Latin Hypercube sampling of initial δ/ω,
        then expand in time from 0 to self.total_time.
        """
        # Use your ODE_modellingPLLRom helper to sample random ICs
        ode_helper = ODE_modellingPLLRom(self.cfg)
        init_conditions_set = ode_helper.create_init_conditions_set3()  # shape [N_ic, dim_states]
        init_conditions_set = torch.tensor(init_conditions_set, dtype=torch.float32)

        # Create a time grid from 0 to total_time
        num_pts = self.cfg.num_of_points
        t_lin = torch.linspace(0.0, self.total_time, num_pts).unsqueeze(1)  # Shape [num_pts, 1]

        colloc_list = []
        for i in range(init_conditions_set.shape[0]):
            ic = init_conditions_set[i].unsqueeze(0)     # shape [1, dim_states]
            ic_repeated = ic.repeat(num_pts, 1)           # [num_pts, dim_states]
            t_ic = torch.cat([t_lin, ic_repeated], dim=1) # [num_pts, 1 + dim_states]
            colloc_list.append(t_ic)

        colloc_tensor = torch.cat(colloc_list, dim=0).to(self.device)
        return colloc_tensor

    # 你还可以把 transform_input / detransform_input / ...
    # 及 find_norm_values, find_std_values, define_minus_divide 等方法也搬进来并调整 import