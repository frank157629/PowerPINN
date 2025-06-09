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
from src.pll_nn.pll_model import Network
# from src.nn.gradient_based_weighting import PINNWeighting

# If you also copied over early_stopping, gradient_based_weighting, nn_inference, etc.,
# make sure to import them from the pll_nn folder:

from src.pll_nn.early_stopping import EarlyStopping
# from src.pll_nn.gradient_based_weighting import PINNWeighting as PLLPINNWeighting
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
        self.lam_d = cfg.nn.weight_data
        self.lam_pd = cfg.nn.weight_pde_data
        self.lam_pc = cfg.nn.weight_pde_col
        self.lam_ic = cfg.nn.weight_ic

        # Instantiate the neural network for PLL-ROM
        self.model = Network(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            output_size=self.output_dim,
            num_layers=cfg.nn.hidden_layers,
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

        # # Initialize dynamic PINN weighting (optional)
        # self.weighting = PLLPINNWeighting(
        #     cfg.nn.weight_data,
        #     cfg.nn.weight_pde_data,
        #     cfg.nn.weight_pde_col,
        #     cfg.nn.weight_ic,
        #     update_freq=None,
        #     device=self.device
        # )

        # # Early stopping utility
        # self.early_stopper = EarlyStopping(
        #     patience=cfg.nn.early_stopping_patience,
        #     min_delta=cfg.nn.early_stopping_min_delta,
        #     verbose=True
        # )

    def pinn_train(self, dataloader, collocation_points, validation_loader=None):
        """
        Train the PINN for PLL-ROM.

        Args:
            dataloader (DataLoader): DataLoader for ground-truth data.
            collocation_points (torch.Tensor): PDE collocation points.
            validation_loader (DataLoader, optional): DataLoader for validation.
        """
        # 放在 for epoch 外，或者 for batch 前都行
        params = {
            'k_p': torch.tensor(self.cfg.physics.k_p, device=self.device),
            'k_i': torch.tensor(self.cfg.physics.k_i, device=self.device),
            'L_g': torch.tensor(self.cfg.physics.L_g, device=self.device),
            'r_Lg': torch.tensor(self.cfg.physics.r_Lg, device=self.device),
            'i_d_c': torch.tensor(self.cfg.physics.i_d_c, device=self.device),
            'i_q_c': torch.tensor(self.cfg.physics.i_q_c, device=self.device),
            'v_g': torch.tensor(self.cfg.physics.v_g, device=self.device),
            'omega_g': torch.tensor(self.cfg.physics.omega_g, device=self.device),
        }

        for epoch in range(self.cfg.nn.num_epochs):
            self.model.train()
            total_loss = 0.0

            # for (x_data, y_data) in dataloader:
            for batch_idx, (x_data, y_data) in enumerate(dataloader):
                x_data = x_data.to(self.device)
                y_data = y_data.to(self.device)

                # Zero out gradients
                self.optimizer.zero_grad()

                # ─── 2. physics-on-data（新增）────────────
                x_data_req_grad = x_data.detach().clone().requires_grad_(True)
                y_pred_data = self.model(x_data_req_grad)  # ← 重算一次
                diff = y_pred_data - y_data  # [batch, 2]
                loss_data = diff.norm(p=2, dim=1).mean()
                res_pd = self.model.compute_pde_residual(
                    x_data_req_grad,  # x with grad
                    y_pred_data,  # y_pred attached to that graph
                    params  # 物理参数 dict
                )
                # —— PDE‐on‐data loss：L_pde_data = (1/N) Σ_i ‖res_pd_i‖₂²
                loss_pde_data = res_pd.norm(p=2, dim=1).pow(2).mean()
                # Compute PDE residual loss at collocation points
                x_col = collocation_points.clone().detach().requires_grad_(True).to(self.device)
                y_col_pred = self.model(x_col)
                # Assume PLL-ROM ODE residual function is provided in PLL_ROM_Model
                pde_residual = self.model.compute_pde_residual(x_col, y_col_pred, params)
                # —— PDE‐collocation loss：L_pde_col = (1/Nc) Σ_j ‖res_col_j‖₂²
                loss_pde_col = pde_residual.norm(p=2, dim=1).pow(2).mean()
                # Compute initial-condition loss if needed
                # Suppose first row of collocation_points is initial condition at t=0
                x_ic = collocation_points[collocation_points[:,0] == 0]
                y_ic_pred = self.model(x_ic)
                y_ic_true = self.model.get_initial_condition_values(x_ic)
                diff_ic = y_ic_pred - y_ic_true
                loss_ic = diff_ic.norm(p=2, dim=1).mean()
                if epoch % 100 == 0 and batch_idx == 0:  # 只在每100个epoch的第一个batch打印
                    print("==== IC DEBUG INFO ====")
                    print("IC points: ", x_ic[:5])
                    print("IC output (pred_ic): ", y_ic_pred[:5])
                    print("IC target (real_ic): ", y_ic_true[:5])
                    print("IC loss: ", loss_ic.item())
                    print("nan in pred_ic:", torch.isnan(y_ic_pred).sum().item())
                    print("inf in pred_ic:", torch.isinf(y_ic_pred).sum().item())
                    print("nan in real_ic:", torch.isnan(y_ic_true).sum().item())
                    print("inf in real_ic:", torch.isinf(y_ic_true).sum().item())
                # Combine losses with dynamic weights
                # weight_data, weight_pde, weight_ic = self.weighting.get_weights()

                loss_total = (self.lam_d * loss_data +
                              self.lam_pd * loss_pde_data +
                              self.lam_pc * loss_pde_col +
                              self.lam_ic * loss_ic)
                               # Backward pass and optimizer step
                loss_total.backward()
                self.optimizer.step()

                total_loss += loss_total.item()

                # Update weights if needed
                # self.weighting.update(loss_data.item(), loss_pde_data, loss_pde_col.item(), loss_ic.item())

            # Validation (if provided)
            if validation_loader is not None:
                val_loss = self.validate(validation_loader)
                wandb.log({
                    "epoch": epoch,
                    "loss/train_data": loss_data.item(),
                    "loss/train_pde_data": loss_pde_data.item(),
                    "loss/train_pde_col": loss_pde_col.item(),
                    "loss/train_ic": loss_ic.item(),
                    "loss/total": loss_total.item(),
                    "loss/val_total": val_loss
                })
                # if self.early_stopper.step(val_loss):
                #     print(f"Early stopping at epoch {epoch}")
                #     break
            else:
                wandb.log({"epoch": epoch, "train_loss": total_loss / len(dataloader)})
        print("All unique t in collocation_points:", torch.unique(collocation_points[:, 0]))
        print("Any t==0?", (collocation_points[:, 0] == 0).sum().item())

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
                diff = (y_pred - y_val).norm(p=2, dim=1)  # ‖·‖₂
                val_loss_total += diff.mean().item()
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