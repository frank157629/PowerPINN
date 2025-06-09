# debug.py
import pickle
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from omegaconf import OmegaConf
from src.pll_nn.pll_model import Network
from src.ode.pll_rom import pll_rom

import wandb
wandb.init(mode="disabled")

# ---- 1) 加载你的 pkl 数据 ----
pkl_path = Path("dataset/PLL_ROM/dataset_v2.pkl")
with pkl_path.open("rb") as f:
    all_runs = pickle.load(f)

# ---- 2) 拼接成 x_train, y_train ----
# 把每条轨迹的 (t, δ(t), ω(t)) 按列堆起来
arrays = []
for t, delta, omega in all_runs:
    # t, delta, omega 都是 numpy 一维数组，拼成 (n_points, 3)
    arr = np.vstack([t, delta, omega]).T
    arrays.append(arr)
data = np.concatenate(arrays, axis=0)          # shape = (n_samples * n_points, 3)
x_all = torch.tensor(data, dtype=torch.float32)     # [N, 3]
y_all = x_all[:, 1:].clone()                        # [N, 2] 目标 δ, ω

# ---- 3) 随机抽一小批 collocation points ----
num_col = 20
perm = torch.randperm(x_all.size(0))
x_col = x_all[perm[:num_col]].clone().detach().requires_grad_(True)

# ---- 4) 构造 DataLoader ----
train_ds     = TensorDataset(x_all, y_all)
train_loader = DataLoader(train_ds, batch_size=x_all.size(0), shuffle=True)

# ---- 5) 配置并实例化超简版 PINN ----
# 我们直接 hard-code 超参数，也可以从 YAML 里读
input_dim, output_dim = 3, 2
hidden_dim, hidden_layers = 64, 4
lr, epochs = 1e-3, 100

model = Network(input_size=input_dim,
                hidden_size=hidden_dim,
                output_size=output_dim,
                num_layers=hidden_layers)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.MSELoss()

# 把物理参数也 hard-code 进来（也可以从 YAML + OmegaConf 里读）
# 这里只示范：先 load 一次 params_pll.yaml
params_conf = OmegaConf.load("src/conf/params_pll.yaml")
params = {k: torch.tensor(float(v)) for k, v in params_conf.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
x_all = x_all.to(device)
y_all = y_all.to(device)
x_col = x_col.to(device)

# ---- 6) 训练打印每轮 loss ----f
for epoch in range(epochs):
    model.train()
    # 只一个 batch
    x_data, y_data = next(iter(train_loader))
    x_data = x_data.to(device)
    y_data = y_data.to(device)

    optimizer.zero_grad()
    # —— data loss ——
    x_req = x_data.clone().detach().requires_grad_(True)
    y_pred = model(x_req)
    diff = y_pred - y_data  # [batch, 2]
    loss_data = diff.norm(dim=1).mean()  # 1/N Σ_i ||diff_i||₂
    # —— PDE‐on‐data loss ——
    res_pd = model.compute_pde_residual(x_req, y_pred, params)
    loss_pde_data = res_pd.norm(p=2, dim=1).pow(2).mean()

    # —— PDE‐collocation loss ——
    y_col_pred = model(x_col)
    res_col   = model.compute_pde_residual(x_col, y_col_pred, params)
    loss_pde_col = res_col.norm(p=2, dim=1).pow(2).mean()

    # 不做 IC loss，这里设为 0
    loss_ic = torch.tensor(0.0, device=device)

    # —— 总 loss ——
    α = 1.0  # data
    β = 0.01  # pde
    loss = α * loss_data + β * (loss_pde_data + loss_pde_col)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch:>2d} | "
          f"data={loss_data.item():.3e}  "
          f"pde_data={loss_pde_data.item():.3e}  "
          f"pde_col={loss_pde_col.item():.3e}  "
          f"total={loss.item():.3e}")

