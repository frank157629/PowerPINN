# File: pll_nn/nn_inference.py

import torch
import numpy as np
import matplotlib.pyplot as plt

from pll_nn.pll_dataset import DataSamplerPLLRom  # 或者 delete if unused
from pll_nn.pll_model import PLL_ROM_Model
from pll_nn.functions import some_physics_function   # 只有在你用了它时才保留

class NNInference:
    """
    A utility class for performing inference once the PLL-ROM model has been trained.
    """
    def __init__(self, model_path, cfg):
        """
        Args:
            model_path (str): Path to the saved .pth file of the trained model.
            cfg (OmegaConf): Configuration dict used to build the model architecture.
        """
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build the same PLL-ROM NN architecture as used in training
        self.model = PLL_ROM_Model(
            input_size=cfg.nn.input_dim,
            hidden_size=cfg.nn.hidden_dim,
            output_size=cfg.nn.output_dim,
            num_layers=cfg.nn.hidden_layers,
            activation=cfg.nn.activation
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, x_input):
        """
        Run a forward pass on new input points to get predicted outputs.

        Args:
            x_input (torch.Tensor): Shape [N, input_dim].

        Returns:
            np.ndarray: Shape [N, output_dim], network predictions.
        """
        with torch.no_grad():
            y_pred = self.model(x_input.to(self.device))
        return y_pred.cpu().numpy()

    def plot_trajectory(self, initial_conditions, t_max, dt=0.001):
        """
        Plot PLL-ROM trajectory over time given initial conditions.

        Args:
            initial_conditions (torch.Tensor): Shape [1, input_dim-1], e.g. [1,1] if input = [t, delta].
            t_max (float): Maximum time to simulate.
            dt (float): Time step for simulation.
        """
        t_values = torch.arange(0.0, t_max, dt).unsqueeze(1)  # Shape [N_time, 1]
        ic = initial_conditions.repeat(len(t_values), 1)      # Broadcast to all time steps
        inputs = torch.cat([t_values.to(self.device), ic.to(self.device)], dim=1)

        with torch.no_grad():
            outputs = self.model(inputs)  # Shape [N_time, output_dim]

        outputs = outputs.cpu().numpy()
        times = t_values.cpu().numpy().flatten()

        # Suppose output_dim = 2 → [delta_pred, omega_pred]
        plt.figure(figsize=(8, 4))
        plt.plot(times, outputs[:, 0], label="δ(t)")
        plt.plot(times, outputs[:, 1], label="ω(t)")
        plt.xlabel("Time (s)")
        plt.ylabel("Value")
        plt.title("PLL-ROM Trajectory")
        plt.legend()
        plt.grid(True)
        plt.show()