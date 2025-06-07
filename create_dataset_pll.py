# File: PowerPINN/create_dataset_pll.py

import os
import torch
import hydra
from omegaconf import OmegaConf

# Import our PLLDatasetGenerator which handles ODE solving and .pkl saving
from src.dataset.create_dataset_pll_functions import PLLDatasetGenerator


@hydra.main(config_path="src/conf", config_name="setup_pll_dataset.yaml", version_base=None)
def main(config):
    """
    Entry point for generating a dataset for the reduced-order PLL model.
    Steps:
      1. Read PLL parameters from params_pll.yaml
      2. Read initial-condition bounds from init_cond.yaml
      3. Solve the 2-state PLL ODE for `n_samples` random initial conditions
      4. Save all trajectories into a .pkl file under dataset/PLL_ROM/

    Paths and sampling settings are defined in setup_pll_dataset.yaml.
    """

    # 1) Print whether CUDA is available (not required for CPU-based ODE solving)
    print("CUDA available?", torch.cuda.is_available())

    # 2) Hydra changes cwd to a temporary directory under outputs/.
    #    To build correct paths, use os.getcwd() as base.
    runtime_dir = os.getcwd()

    # 3) Build absolute paths to the two YAML files for PLL:
    #    - params_pll.yaml    (contains k_p, k_i, L_g, etc.)
    #    - init_cond.yaml     (contains delta0.min, delta0.max, omega0.min, omega0.max)
    params_path = os.path.join(
        runtime_dir,
        config.dirs.params_dir,
        "params_pll.yaml"
    )
    init_cond_path = os.path.join(
        runtime_dir,
        config.dirs.init_conditions_dir,
        "init_cond.yaml"
    )

    # 4) The base dataset directory; PLLDatasetGenerator will create a "PLL_ROM" subfolder
    dataset_base_dir = os.path.join(runtime_dir, config.dirs.dataset_dir)

    # 5) Instantiate the PLLDatasetGenerator
    pll_generator = PLLDatasetGenerator(
        params_path=params_path,
        init_cond_path=init_cond_path,
        dataset_dir=dataset_base_dir
    )

    # 6) Read simulation settings from Hydra config
    total_time = float(config.time)            # e.g., 1.0 second
    num_points = int(config.num_of_points)     # e.g., 1001 points
    n_samples  = int(config.sampling.n_samples)  # e.g., 10 random trajectories

    # 7) Generate and save the PLL-ROM dataset
    pll_generator.generate(
        time=total_time,
        num_points=num_points,
        n_samples=n_samples
    )

    return None


if __name__ == "__main__":
    main()