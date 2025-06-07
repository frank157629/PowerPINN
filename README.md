# PowerPINN 

Physics-Informed Neural Networks (PINNs) for Power System Components

## Overview
This repository provides a framework for generating and training **Physics-Informed Neural Networks (PINNs)** for power system components. It allows users to define Ordinary Differential Equations (ODEs), generate datasets, and train PINNs to approximate system dynamics efficiently. 

## Features
- Define and integrate new sets of **ODEs** for different power system components.
- Configure initial conditions and variable ranges.
- Automatically generate datasets using numerical solvers.
- Train and test PINNs using **WandB** for tracking.
- Fully parameterized using YAML configuration files.
- Modular design for easy extension.

## Installation
### Prerequisites
Ensure you have Python installed (>=3.8). Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage
### 1. Define ODEs
ODEs are stored in `src/ode/sm_models_d.py`. You can add any new ODE model in this directory. 

### 2. Configure Variables
The independent variables should be defined in `modellings_guide.yaml`, ensuring they are in the same order as in the ODEs.

### 3. Set Initial Conditions
Initial condition values and ranges should be specified in respective YAML files, located in the `src/conf/initial_conditions/` folder under the corresponding ODE name (e.g., `SM_AVR_GOV/init_cond.yaml`).

### 4. Define Machine Parameters
Different machine parameters can be configured in the `src/conf/params/` folder.

### 5. Generate Dataset
To generate the dataset for PINN training, use:
```bash
python create_dataset_d.py
```
**Configuration file:** `setup_dataset.yaml`
- `time`: Total simulation time.
- `num_of_points`: Number of data points per trajectory.
- `modelling_method`: Defines how state variables evolve.
- `model`: Specifies which ODE model to use (e.g., `SM_AVR_GOV`).
- `sampling`: Type of sampling for initial conditions (`Lhs`, `Linear`, `Random`).
- `dirs`: Paths for storing parameters, initial conditions, and dataset.

### 6. Train & Test a PINN
Train the PINN model with:
```bash
python test_sweep.py
```
**Configuration file:** `setup_dataset_nn.yaml`
- `time`, `num_of_points`, `modelling_method`: Same as dataset setup.
- `seed`: Ensures reproducibility.
- `dataset`: Defines data usage (`shuffle`, `split_ratio`, `transform_input/output`).
- `nn` (Neural Network Configuration):
  - `type`: Network architecture (`DynamicNN`, `StaticNN`, etc.).
  - `hidden_layers`, `hidden_dim`: Network depth and width.
  - `optimizer`: (`Adam`, `LBFGS`, etc.).
  - `weighting`: Adjusts loss function balance.
- `dirs`: Paths for dataset, model, and training parameters.

## Configuration Files
| File | Purpose |
|---------------------|--------------------------------------------------|
| `setup_dataset.yaml` | Defines parameters for dataset generation |
| `setup_dataset_nn.yaml` | Defines parameters for neural network training |
| `modellings_guide.yaml` | Lists different ODE models and their variables |
| `initial_conditions/` | Folder containing initial conditions for each ODE |
| `params/` | Folder containing different synchronous machine parameters |

## Citation
If you use this repository in your research, please cite the following paper:

**Ioannis Karampinis, Petros Ellinas, Ignasi Ventura Nadal, Rahul Nellikkath, Spyros Chatzivasileiadis**, *Toolbox for Developing Physics-Informed Neural Networks for Power System Components*, DTU.

## License
This project is licensed under the MIT License.

# PowerPINN
