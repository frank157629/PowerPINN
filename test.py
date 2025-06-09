# test.py
import torch
from torch.utils.data import TensorDataset, DataLoader
from omegaconf import OmegaConf
import wandb

# —— 1) 初始化 W&B，不禁用 ——
run = wandb.init(
    project="PowerPINN",
    entity="liuhaitian888-rwth-aachen-university",
    config={
        "hidden_dim": 64,
        "hidden_layers": 4,
        "learning_rate": 1e-3,
        "weight_data": 1.0,
        "weight_pde_data": 0.01,
        "weight_pde_col": 0.03,
        "weight_ic": 0.03,
        "num_epochs": 1000,
        "seed": 42,
    }
)

from src.pll_nn.pll_dataset import DataSampler
from src.pll_nn.nn_actions  import NeuralNetworkActions

def main():
    # 1) DataSampler
    ds_cfg = OmegaConf.load("src/conf/setup_pll_dataset.yaml")
    ds_cfg.seed = wandb.config.seed
    dataset = DataSampler(ds_cfg)

    # 2) PINN 配置
    pinn_cfg = OmegaConf.load("src/conf/setup_pll_pinn.yaml")
    params    = OmegaConf.load("src/conf/params_pll.yaml")
    OmegaConf.set_struct(pinn_cfg, False)
    pinn_cfg.physics = params
    pinn_cfg.seed    = wandb.config.seed

    pinn_cfg.nn.input_dim     = dataset.input_dim
    pinn_cfg.nn.output_dim    = dataset.y_train.shape[1]
    # 从 wandb.config 取超参
    pinn_cfg.nn.hidden_dim      = wandb.config.hidden_dim
    pinn_cfg.nn.hidden_layers   = wandb.config.hidden_layers
    pinn_cfg.nn.learning_rate   = wandb.config.learning_rate
    pinn_cfg.nn.weight_data     = wandb.config.weight_data
    pinn_cfg.nn.weight_pde_data = wandb.config.weight_pde_data
    pinn_cfg.nn.weight_pde_col  = wandb.config.weight_pde_col
    pinn_cfg.nn.weight_ic       = wandb.config.weight_ic
    pinn_cfg.nn.num_epochs      = wandb.config.num_epochs

    # 3) 实例化 trainer
    nn_actions = NeuralNetworkActions(pinn_cfg)

    # 4) DataLoader
    train_ds        = TensorDataset(dataset.x_train, dataset.y_train)
    train_loader    = DataLoader(train_ds, batch_size=256, shuffle=True)
    collocation_pts = dataset.x_train_col

    val_loader = None
    if hasattr(dataset, "x_val"):
        val_ds     = TensorDataset(dataset.x_val, dataset.y_val)
        val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    # 5) 训练
    nn_actions.pinn_train(
        train_loader,
        collocation_pts,
        validation_loader=val_loader
    )

if __name__ == "__main__":
    main()
    run.finish()