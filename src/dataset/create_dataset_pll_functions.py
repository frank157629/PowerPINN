# File: PowerPINN/src/dataset/create_dataset_pll_functions.py

import os
import pickle
import numpy as np
from scipy.integrate import solve_ivp
import torch
from omegaconf import OmegaConf
# Import your PLL-ROM ODE function
from src.ode.pll_rom import pll_rom


class PLLDatasetGenerator:
    """
    A class that generates and saves a dataset for the reduced-order PLL model,
    following the same pickle-based logic used by the original synchronous-machine code.
    """

    def __init__(self, params_path: str, init_cond_path: str, dataset_dir: str):
        """
        Initialize the PLLDatasetGenerator.

        Args:
            params_path (str): Path to 'params_pll.yaml' (contains k_p, k_i, etc.).
            init_cond_path (str): Path to 'init_cond.yaml' (contains δ₀ range, ω₀ range).
            dataset_dir (str): Base directory where 'PLL_ROM' folder lives.
        """
        # Load PLL parameters (all entries in YAML are cast to float)
        pll_conf = OmegaConf.load(params_path)
        self.params_pll = {k: float(pll_conf[k]) for k in pll_conf}

        # Load initial-condition bounds from YAML
        init_conf = OmegaConf.load(init_cond_path)
        self.delta_min = float(init_conf.delta0.min)
        self.delta_max = float(init_conf.delta0.max)
        self.omega_min = float(init_conf.omega0.min)
        self.omega_max = float(init_conf.omega0.max)

        # Directory under which we will create / load PLL data
        # e.g., dataset_dir = "/path/to/PowerPINN/dataset"
        self.dataset_dir = dataset_dir

        # Ensure the base folder exists
        os.makedirs(self.dataset_dir, exist_ok=True)

    def generate(self, time: float, num_points: int, n_samples: int):
        """
        Generate 'n_samples' trajectories of the PLL-ROM ODE, then save them to a .pkl file.

        Each trajectory has:
          - t: 1D array of length 'num_points'
          - delta(t): 1D array of length 'num_points'
          - omega(t): 1D array of length 'num_points'

        The saved dataset is a Python list of length 'n_samples', where each element is itself
        a list: [t_array, delta_array, omega_array].

        The file is written to:
          <self.dataset_dir>/dataset_vX.pkl
        where X is one more than the number of existing files in that folder.

        Args:
            time (float): End time (seconds) for each simulation.
            num_points (int): Number of time points (including t=0, t=time).
            n_samples (int): How many random (δ₀, ω₀) initial pairs to generate.
        """
        # 1) Sample initial conditions uniformly
        np.random.seed(42)
        delta0_arr = np.random.uniform(self.delta_min, self.delta_max, size=n_samples)
        omega0_arr = np.random.uniform(self.omega_min, self.omega_max, size=n_samples)

        # 2) Create the time grid for all runs
        t_eval = np.linspace(0.0, time, num_points)

        # 3) Container for all trajectories
        all_runs = []

        for idx, (d0, w0) in enumerate(zip(delta0_arr, omega0_arr), start=1):
            # Wrap the PLL-ROM ODE to work with solve_ivp (numpy→torch→numpy)
            def rhs_numpy(t, x_np):
                # x_np: ndarray shape (2,), convert to torch.Tensor shape (1,2)
                x_tensor = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0)  # (1,2)
                dxdt_tensor = pll_rom(t, x_tensor, self.params_pll)         # (1,2)
                return dxdt_tensor.detach().cpu().numpy().flatten()            # (2,)

            # Solve ODE
            sol = solve_ivp(
                fun=rhs_numpy,
                t_span=(0.0, time),
                y0=[d0, w0],
                t_eval=t_eval,
                method="RK45",
                atol=1e-8,
                rtol=1e-8
            )

            # sol.t: shape (num_points,)
            # sol.y: shape (2, num_points); sol.y[0] is delta array, sol.y[1] is omega array

            # Build a Python list: [ t_array, delta_array, omega_array ]
            run_data = [
                sol.t.copy(),
                sol.y[0].copy(),
                sol.y[1].copy()
            ]
            all_runs.append(run_data)

            print(f"Run {idx} completed: δ₀={d0:.4f}, ω₀={w0:.4f}")

        # 4) Serialize 'all_runs' as a pickle file under dataset_dir
        #    Naming: dataset_v{N+1}.pkl, where N = number of existing files
        existing_files = [
            f for f in os.listdir(self.dataset_dir)
            if os.path.isfile(os.path.join(self.dataset_dir, f)) and f.endswith(".pkl")
        ]
        num_existing = len(existing_files)
        next_version = num_existing + 1
        filename = f"dataset_v{next_version}.pkl"
        full_path = os.path.join(self.dataset_dir, filename)

        with open(full_path, "wb") as fp:
            # Dump the entire Python list (all_runs)
            pickle.dump(all_runs, fp)

        print(f"Saved PLL-ROM dataset to: {full_path}")
        return all_runs

    def load(self, filename: str):
        """
        Load a previously saved PLL-ROM dataset from a .pkl file.

        Args:
            filename (str): Name of the pickle file under '<dataset_dir>'.
                            For example: 'dataset_v1.pkl'.

        Returns:
            list: The same Python list structure that was saved:
                  [ [t_array1, delta1, omega1], [t_array2, delta2, omega2], … ]
        """
        full_path = os.path.join(self.dataset_dir, filename)
        with open(full_path, "rb") as fp:
            dataset = pickle.load(fp)
        return dataset