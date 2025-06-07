# File: pll_nn/nn_actions.py

import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt

import wandb
from omegaconf import OmegaConf

# Removed: from src.ode.sm_models_d import SynchronousMachineModels
# Add instead the PLL-ROM model import:
from src.ode.pll_rom import pll_rom
from src.nn.gradient_based_weighting import PINNWeighting
# If you also copied over early_stopping, gradient_based_weighting, nn_inference, etc.,
# make sure to import them from the pll_nn folder:
from src.pll_nn.early_stopping import EarlyStopping
from src.pll_nn.gradient_based_weighting import PINNWeighting as PLLPINNWeighting
from src.pll_nn.nn_inference import predict, define_nn_model
from torch.utils.data import DataLoader, TensorDataset

class NeuralNetworkActions:
    """
    A class used to define the actions of the neural network model for PLL-ROM.

    Attributes:
        cfg (dict): Configuration dictionary loaded from YAML.
        input_dim (int): Number of input features for the network.
        hidden_dim (int): Number of hidden neurons per layer.
        output_dim (int): Number of output features.
        device (torch.device): Device on which tensors are allocated.
        model (nn.Module): The neural network model instance.
        optimizer (torch.optim.Optimizer): The optimizer for network parameters.
        criterion (callable): The loss function.
        weighting (PINNWeighting): Object to manage dynamic PINN weights.
        early_stopper (EarlyStopping): Early stopping utility.
    """

    def __init__(self, cfg):
        """
        Initialize NeuralNetworkActions with the provided configuration.

        Args:
            cfg (OmegaConf): The configuration containing all training parameters.
        """
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get input/output dimensions from config
        self.input_dim = cfg.nn.input_dim
        self.hidden_dim = cfg.nn.hidden_dim
        self.output_dim = cfg.nn.output_dim

        # Instantiate the neural network for PLL-ROM
        self.model = pll_rom(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            output_size=self.output_dim,
            num_layers=cfg.nn.hidden_layers,
            activation=cfg.nn.activation
        ).to(self.device)

        # Set up optimizer
        if cfg.nn.optimizer == "Adam":
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=cfg.nn.learning_rate
            )
        elif cfg.nn.optimizer == "LBFGS":
            self.optimizer = optim.LBFGS(
                self.model.parameters(), lr=cfg.nn.learning_rate
            )
        else:
            raise NotImplementedError(f"Optimizer {cfg.nn.optimizer} not supported.")

        # Define loss criterion
        if cfg.nn.loss_criterion == "MSELoss":
            self.criterion = nn.MSELoss()
        elif cfg.nn.loss_criterion == "L1Loss":
            self.criterion = nn.L1Loss()
        else:
            raise NotImplementedError(f"Loss {cfg.nn.loss_criterion} not supported.")

        # Initialize dynamic PINN weighting (optional)
        self.weighting = PLLPINNWeighting(
            cfg.nn.weight_data,
            cfg.nn.weight_pde,
            cfg.nn.weight_ic,
            cfg.nn.weighting.update_weights_freq,
            device=self.device
        )

        # Early stopping utility
        self.early_stopper = EarlyStopping(
            patience=cfg.nn.early_stopping_patience,
            min_delta=cfg.nn.early_stopping_min_delta,
            verbose=True
        )

    def pinn_train(self, dataloader, collocation_points, validation_loader=None):
        """
        Train the PINN for PLL-ROM.

        Args:
            dataloader (DataLoader): DataLoader for ground-truth data.
            collocation_points (torch.Tensor): PDE collocation points.
            validation_loader (DataLoader, optional): DataLoader for validation.
        """
        for epoch in range(self.cfg.nn.num_epochs):
            self.model.train()
            total_loss = 0.0

            for (x_data, y_data) in dataloader:
                x_data = x_data.to(self.device)
                y_data = y_data.to(self.device)

                # Zero out gradients
                self.optimizer.zero_grad()

                # Prediction for data points
                y_pred_data = self.model(x_data)

                # Compute data loss
                loss_data = self.criterion(y_pred_data, y_data)

                # Compute PDE residual loss at collocation points
                x_col = collocation_points.clone().detach().requires_grad_(True).to(self.device)
                y_col_pred = self.model(x_col)
                # Assume PLL-ROM ODE residual function is provided in PLL_ROM_Model
                pde_residual = self.model.compute_pde_residual(x_col, y_col_pred)
                loss_pde = torch.mean(pde_residual**2)

                # Compute initial-condition loss if needed
                # Suppose first row of collocation_points is initial condition at t=0
                x_ic = collocation_points[collocation_points[:,0] == 0]
                y_ic_pred = self.model(x_ic)
                y_ic_true = self.model.get_initial_condition_values(x_ic)
                loss_ic = self.criterion(y_ic_pred, y_ic_true)

                # Combine losses with dynamic weights
                weight_data, weight_pde, weight_ic = self.weighting.get_weights()
                loss_total = weight_data * loss_data + weight_pde * loss_pde + weight_ic * loss_ic

                # Backward pass and optimizer step
                loss_total.backward()
                self.optimizer.step()

                total_loss += loss_total.item()

                # Update weights if needed
                self.weighting.update(loss_data.item(), loss_pde.item(), loss_ic.item())

            # Validation (if provided)
            if validation_loader is not None:
                val_loss = self.validate(validation_loader)
                wandb.log({"epoch": epoch, "train_loss": total_loss / len(dataloader),
                           "val_loss": val_loss})
                if self.early_stopper.step(val_loss):
                    print(f"Early stopping at epoch {epoch}")
                    break
            else:
                wandb.log({"epoch": epoch, "train_loss": total_loss / len(dataloader)})

    def validate(self, validation_loader):
        """
        Validate the model on the validation set.

        Args:
            validation_loader (DataLoader): DataLoader for validation data.

        Returns:
            float: Average validation loss.
        """
        self.model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for (x_val, y_val) in validation_loader:
                x_val = x_val.to(self.device)
                y_val = y_val.to(self.device)
                y_pred = self.model(x_val)
                val_loss_total += self.criterion(y_pred, y_val).item()

        return val_loss_total / len(validation_loader)

    def save_model(self, save_path=None):
        """
        Save the trained model to disk.

        Args:
            save_path (str): Path to save the model. If None, use cfg.dirs.model_dir.
        """
        if save_path is None:
            save_path = os.path.join(self.cfg.dirs.model_dir, "pll_rom_model.pth")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.state_dict(), save_path)

    def load_model(self, load_path):
        """
        Load a trained model from disk.

        Args:
            load_path (str): Path to the saved .pth file.
        """
        self.model.load_state_dict(torch.load(load_path, map_location=self.device))
        self.model.to(self.device)