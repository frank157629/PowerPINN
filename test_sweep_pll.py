"""
test_sweep_pll.py

Performs a W&B hyperparameter sweep for training a Physics-Informed Neural Network (PINN)
on the PLL-ROM. This script:

 1) Loads dataset configuration and instantiates a DataSampler.
 2) Loads PINN configuration (network architecture, optimizer, epochs, etc.).
 3) Overrides PINN hyperparameters from the sweep.
 4) Prepares PyTorch data loaders for data and collocation points.
 5) Runs the PINN training loop.
"""

import torch
import wandb
from omegaconf import OmegaConf
from src.pll_nn.pll_dataset import DataSampler
from src.pll_nn.nn_actions import NeuralNetworkActions

def train(config=None):
    """
    Main training entrypoint for a W&B sweep job.

    Args:
      config (dict): the run.config from wandb.init()

    Workflow:
      1. Initialize W&B run & load sweep parameters.
      2. Load dataset-only config, set seed, create DataSampler.
      3. Load PINN-only config, set seed, override hyperparameters.
      4. Instantiate the PINN trainer.
      5. Build PyTorch DataLoader for training (and validation if available).
      6. Call the PINN training routine.
      7. Finish W&B run.
    """
    # --- 1) Initialize W&B run and extract sweep config ---
    run = wandb.init(config=config)
    sweep_cfg = run.config

    # --- 2) Dataset config & DataSampler ---
    # Load only the dataset-related settings
    ds_cfg = OmegaConf.load("src/conf/setup_pll_dataset.yaml")
    ds_cfg.seed = sweep_cfg.seed
    # Create the object that loads/splits/normalizes your PLL data
    dataset = DataSampler(ds_cfg)

    # --- 3) PINN config & trainer setup ---
    # Load only the PINN-related settings (network, optimizer, epochs…)
    pinn_cfg = OmegaConf.load("src/conf/setup_pll_pinn.yaml")
    pinn_cfg.seed = sweep_cfg.seed
    # Override hidden dimensions, layers, learning rate and loss weights from the sweep
    pinn_cfg.nn.hidden_dim    = sweep_cfg.nn_hidden_dim
    pinn_cfg.nn.hidden_layers = sweep_cfg.nn_hidden_layers
    pinn_cfg.nn.learning_rate = sweep_cfg.nn_learning_rate
    pinn_cfg.nn.weight_data   = sweep_cfg.nn_weight_data
    pinn_cfg.nn.weight_pde    = sweep_cfg.nn_weight_pde
    pinn_cfg.nn.weight_ic     = sweep_cfg.nn_weight_ic
    # Instantiate the PINN training logic
    nn_actions = NeuralNetworkActions(pinn_cfg)

    # --- 4) Prepare data loaders ---
    # Standard data points
    train_ds    = torch.utils.data.TensorDataset(dataset.x_train, dataset.y_train)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True)
    # Collocation points for the PDE residual
    collocation_points = dataset.x_train_col

    # Optional validation loader if your dataset split created one
    val_loader = None
    if hasattr(dataset, "x_val"):
        val_ds    = torch.utils.data.TensorDataset(dataset.x_val, dataset.y_val)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=256, shuffle=False)

    # --- 5) Run the actual PINN training ---
    nn_actions.pinn_train(
        train_loader,
        collocation_points,
        validation_loader=val_loader
    )

    # --- 6) Finish the W&B run ---
    run.finish()

if __name__ == "__main__":
    # Define your W&B hyperparameter sweep
    sweep_definition = {
        "method": "random",
        "metric": {"name": "val/total_loss", "goal": "minimize"},
        "parameters": {
            "seed":             {"values": [42, 7, 123]},
            "nn_hidden_dim":    {"values": [32, 64, 128]},
            "nn_hidden_layers": {"values": [3, 4, 5]},
            "nn_learning_rate": {"values": [1e-3, 5e-4, 1e-4]},
            "nn_weight_data":   {"values": [1.0]},
            "nn_weight_pde":    {"values": [0.01]},
            "nn_weight_ic":     {"values": [0.01]},
        }
    }
    sweep_id = wandb.sweep(sweep_definition, project="PLL-ROM")
    wandb.agent(sweep_id, function=train)