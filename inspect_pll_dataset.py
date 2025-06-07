# File: inspect_pll_dataset.py

import os
import pickle
import numpy as np

# Enforce a known Matplotlib backend that your system can support.
# Here we choose "TkAgg". Make sure your environment has tkinter installed.
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# STEP 1: Define path to the saved dataset .pkl file
# We assume the current working directory is PowerPINN/.
dataset_folder = os.path.join(os.getcwd(), "dataset", "PLL_ROM")
pkl_filename = "dataset_v1.pkl"
pkl_path = os.path.join(dataset_folder, pkl_filename)

# STEP 2: Load the pickle file
if not os.path.isfile(pkl_path):
    raise FileNotFoundError(f"Cannot find dataset file: {pkl_path}")

with open(pkl_path, "rb") as f:
    data_list = pickle.load(f)

print(f"Loaded {len(data_list)} trajectories from {pkl_path}")

# STEP 3: Inspect the first trajectory structure
# In our case, each trajectory is stored as a list [t_array, delta_array, omega_array].
first_traj = data_list[0]
t_array = np.array(first_traj[0])
delta_array = np.array(first_traj[1])
omega_array = np.array(first_traj[2])

print("Example of first trajectory:")
print("  Type of t array:    ", type(first_traj[0]), "Shape:", t_array.shape)
print("  Type of δ array:    ", type(first_traj[1]), "Shape:", delta_array.shape)
print("  Type of ω array:    ", type(first_traj[2]), "Shape:", omega_array.shape)
print("  First few time points:", t_array[:5])
print("  First few δ values:   ", delta_array[:5])
print("  First few ω values:   ", omega_array[:5])

# STEP 4: Plot all δ(t) trajectories. This call will block until you close the window.
plt.figure(figsize=(8, 5))
for idx, traj in enumerate(data_list):
    t = np.array(traj[0])
    delta = np.array(traj[1])
    plt.plot(t, delta, label=f"Run {idx+1}")
plt.xlabel("Time (s)")
plt.ylabel("δ (rad)")
plt.title("δ(t) trajectories for all runs")
plt.legend()
plt.grid(True)

# Show δ(t) figure and block until closed.
plt.show()


# STEP 6: Plot all ω(t) trajectories. This too will block until you close the window.
plt.figure(figsize=(8, 5))
for idx, traj in enumerate(data_list):
    t = np.array(traj[0])
    omega = np.array(traj[2])
    plt.plot(t, omega, label=f"Run {idx+1}")
plt.xlabel("Time (s)")
plt.ylabel("ω (pu)")
plt.title("ω(t) trajectories for all runs")
plt.legend()
plt.grid(True)

# Show ω(t) figure
plt.show()