# File: pll_nn/nn_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    A simple multi-layer perceptron for the PLL-ROM problem.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, activation="tanh"):
        super(MLP, self).__init__()
        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if activation.lower() == "tanh":
                layers.append(nn.Tanh())
            elif activation.lower() == "relu":
                layers.append(nn.ReLU())
            else:
                raise ValueError(f"Unsupported activation: {activation}")
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def compute_pde_residual(self, x, y_pred):
        """
        Compute PDE residual for PLL-ROM: dδ/dt = ω, dω/dt = f(δ, ω).

        Args:
            x (torch.Tensor): shape [N, input_dim], where x[:,0]=t, x[:,1]=δ.
            y_pred (torch.Tensor): shape [N, output_dim], preds for [δ, ω].

        Returns:
            torch.Tensor: shape [N, 2], each row is [residual1, residual2].
        """
        # Extract t, δ from input. y_pred has [δ_pred, ω_pred] columns.
        t = x[:, 0].unsqueeze(1)
        delta_pred = y_pred[:, 0].unsqueeze(1)
        omega_pred = y_pred[:, 1].unsqueeze(1)

        # Compute ∂ω/∂t via autograd
        domega_dt = torch.autograd.grad(
            omega_pred, x, torch.ones_like(omega_pred), create_graph=True, retain_graph=True
        )[0][:, 0].unsqueeze(1)

        # PDE: dδ/dt − ω = 0  → residual1
        ddelta_dt = omega_pred
        residual1 = ddelta_dt - omega_pred

        # PDE: dω/dt − f(δ, ω) = 0  → residual2
        # For example: f(δ, ω) = -K*sin(δ) - (D/M) * ω. 这里仅举例，参数可从 cfg 或 params_pll.yaml 读取
        K = 10.0
        D = 0.5
        M = 1.0
        residual2 = domega_dt - ( - (K / M) * torch.sin(delta_pred) - (D / M) * omega_pred )

        return torch.cat([residual1, residual2], dim=1)

    def get_initial_condition_values(self, x_ic):
        """
        Return true initial condition values for t=0.

        Args:
            x_ic (torch.Tensor): shape [N_ic, input_dim], where x_ic[:,0]==0, x_ic[:,1]=δ0

        Returns:
            torch.Tensor: shape [N_ic, output_dim], i.e. [δ0, ω0].
        """
        delta0 = x_ic[:, 1].unsqueeze(1)
        # Suppose initial ω0 = 0 for all runs:
        omega0 = torch.zeros_like(delta0)
        return torch.cat([delta0, omega0], dim=1)