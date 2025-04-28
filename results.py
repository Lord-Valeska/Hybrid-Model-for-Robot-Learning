import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from test_model_single import test_saved_model_single
from test_pcc import baseline_pcc_from_angle_increments

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1) Hybrid model (single-step) rollout
hybrid_states = test_saved_model_single(
    model_path    = 'saved_models/single_step_dynamics_model.pt',
    received_path = 'data/received_data_test.csv',
    marker_path   = 'data/cleaned_xyz_test.csv',
    command_path  = 'data/command_test.csv',
    device        = device
)
hybrid_xyz = hybrid_states[:, 3:6]

# 1.5) Multi-step model rollout
# multi_states = test_saved_model_multi(
#     model_path    = 'saved_models/multiple_step_dynamics_model.pt',
#     received_path = 'data/received_data_test.csv',
#     marker_path   = 'data/cleaned_xyz_test.csv',
#     command_path  = 'data/command_test.csv',
#     device        = device
# )
# multi_xyz = multi_states[:, 3:6]

# 2) PCC baseline rollout
_, pcc_xyz = baseline_pcc_from_angle_increments(
    received_path = 'data/received_data_test.csv',
    command_path  = 'data/command_test.csv',
    d             = 28.0,
    device        = device
)

# 3) Read ground truth
df = pd.read_csv('data/cleaned_xyz_test.csv')
cols = df.columns.str.strip()
if set(['X', 'Y', 'Z']).issubset(cols):
    actual_xyz = df[['X', 'Y', 'Z']].to_numpy(dtype=np.float32)
else:
    df_vals = pd.read_csv('data/cleaned_xyz_test.csv', header=None)
    actual_xyz = df_vals.iloc[:, :3].to_numpy(dtype=np.float32)

# 4) Trim sequences
n = min(actual_xyz.shape[0], hybrid_xyz.shape[0], pcc_xyz.shape[0])
actual_xyz = actual_xyz[:n]
hybrid_xyz = hybrid_xyz[:n]
pcc_xyz    = pcc_xyz[:n]
# multi_xyz  = multi_xyz[:n]

# 4.5) Align PCC to ground truth origin
offset = actual_xyz[0] - pcc_xyz[0]
print(f"Offset to apply: {offset}")
pcc_xyz = pcc_xyz + offset

# 4.6) Compute MSE
mse_pcc    = np.mean((pcc_xyz - actual_xyz)**2)
mse_hybrid = np.mean((hybrid_xyz - actual_xyz)**2)
# mse_multi  = np.mean((multi_xyz  - actual_xyz)**2)
print(f"PCC XYZ MSE:        {mse_pcc:.6f}")
print(f"Single-step Hybrid XYZ MSE:     {mse_hybrid:.6f}")
# print(f"Multi-step Hybrid XYZ MSE: {mse_multi:.6f}")

# 5) Plot trajectories
fig = plt.figure()
ax  = fig.add_subplot(111, projection='3d')

ax.plot(actual_xyz[:,0], actual_xyz[:,1], actual_xyz[:,2],
        label='Cleaned XYZ (GT)', marker='o', linestyle='-')
ax.plot(pcc_xyz[:,0],    pcc_xyz[:,1],    pcc_xyz[:,2],
        label='PCC Baseline (aligned)', marker='^', linestyle='--')
ax.plot(hybrid_xyz[:,0], hybrid_xyz[:,1], hybrid_xyz[:,2],
        label='Hybrid Model (single-step)', marker='s', linestyle='-.')
# ax.plot(multi_xyz[:,0],  multi_xyz[:,1],  multi_xyz[:,2],
#         label='Hybrid Model (multi-step)', marker='x', linestyle=':')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.title('Trajectory Comparison: Ground Truth vs PCC vs Hybrid Single-step')
plt.tight_layout()
plt.show()
