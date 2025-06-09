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
    # 1) 加载并实例化 DataSampler
    ds_cfg = OmegaConf.load("src/conf/setup_pll_dataset.yaml")
    ds_cfg.seed = sweep_cfg.seed
    dataset = DataSampler(ds_cfg)

    # test_sweep_pll.py 里的 train()
    dataset = DataSampler(ds_cfg)

    col = dataset.x_train_col
    print("⏺ collocation points count:", col.shape)
    print("   time min/max:", col[:, 0].min().item(), "/", col[:, 0].max().item())
    print("   δ,ω ranges:", col[:, 1:].min(0).values.tolist(), "/", col[:, 1:].max(0).values.tolist())

    ic = col[col[:, 0] == 0]
    print("→ 初始条件点 (t==0) 数量:", ic.shape[0])

    # 2) 加载 PINN config
    pinn_cfg = OmegaConf.load("src/conf/setup_pll_pinn.yaml")
    params = OmegaConf.load("src/conf/params_pll.yaml")
    OmegaConf.set_struct(pinn_cfg, False)  # 允许增字段
    pinn_cfg.physics = params  # ← 手动挂进去

    pinn_cfg.seed = sweep_cfg.seed

    # 3) **动态灌入** 网络维度：
    pinn_cfg.nn.input_dim = dataset.input_dim
    pinn_cfg.nn.output_dim = dataset.y_train.shape[1]  # 或者 DataSampler 提供的 dataset.output_dim

    # 4) 再把 sweep 超参覆盖进去
    pinn_cfg.nn.hidden_dim = sweep_cfg.nn_hidden_dim
    pinn_cfg.nn.hidden_layers = sweep_cfg.nn_hidden_layers
    pinn_cfg.nn.learning_rate = sweep_cfg.nn_learning_rate
    pinn_cfg.nn.weight_data = sweep_cfg.nn_weight_data
    pinn_cfg.nn.weight_pde_data = sweep_cfg.nn_weight_pde_data
    pinn_cfg.nn.weight_pde_col = sweep_cfg.nn_weight_pde_col
    pinn_cfg.nn.weight_ic = sweep_cfg.nn_weight_ic

    # 5) 用这份完整的 pinn_cfg 实例化
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
    # sweep_definition = {
    #     "method": "random",
    #     "metric": {"name": "loss/val_total", "goal": "minimize"},
    #     "parameters": {
    #         "seed":             {"values": [42, 7, 123]},
    #         "nn_hidden_dim":    {"values": [32, 64, 128]},
    #         "nn_hidden_layers": {"values": [3, 4, 5]},
    #         "nn_learning_rate": {"values": [1e-3, 5e-4, 1e-4]},
    #         "nn_weight_data":   {"values": [1.0]},
    #         "nn_weight_pde":    {"values": [0.01]},
    #         "nn_weight_pde_col": {"values": [0.03]},
    #         "nn_weight_ic":     {"values": [0.01]},
    #     }
    # }
    # sweep_id = wandb.sweep(sweep_definition, project="PLL-ROM")
    # wandb.agent(sweep_id, function=train)
    single_cfg = dict(
        seed=42,
        nn_hidden_dim=64,
        nn_hidden_layers=4,
        nn_learning_rate=1e-3,
        nn_weight_data=1.0,
        nn_weight_pde_data=0.01,
        nn_weight_ic=0.01,
        nn_weight_pde_col=0.03,
    )
    train(config=single_cfg)