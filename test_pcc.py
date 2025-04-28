import numpy as np
import pandas as pd
import torch
from models import PCCModel, angle_to_length

def baseline_pcc_from_angle_increments(received_path,
                                       command_path,
                                       d,
                                       device=torch.device('cpu')):
    """
    Reads initial encoder angles from received_path,
    then reads angle increments from command_path,
    accumulates them, converts to lengths, runs PCCModel,
    and prints each (x,y,z) step.
    """
    # 0) Prepare model
    model = PCCModel().to(device)
    model.eval()

    # 1) Load initial angles (first row: cols [traj, enc1, enc2, enc3])
    rec_df    = pd.read_csv(received_path, header=None)
    init_ang  = rec_df.iloc[0, 1:4].to_numpy(dtype=np.float32)  # shape (3,)

    # 2) Load angle increments (cols [traj, Δa1, Δa2, Δa3])
    cmd_df    = pd.read_csv(command_path, header=None)
    increments = cmd_df.iloc[:, 1:4].to_numpy(dtype=np.float32)
    # If you want to skip the very first increment, uncomment:
    increments = increments[1:]

    # 3) Build cumulative angles sequence
    angles = [init_ang]
    for inc in increments:
        angles.append(angles[-1] + inc)
    angles = np.stack(angles, axis=0)  # shape (T+1, 3)

    # 4) Convert angles → lengths for each step
    #    angle_to_length accepts a NumPy array of shape (...,3)
    lengths = angle_to_length(angles)  # shape (T+1, 3)

    # 5) Run through PCCModel and print
    q = torch.tensor(lengths, dtype=torch.float32, device=device)
    with torch.no_grad():
        xyz_pred = model(q).cpu().numpy()  # (T+1, 3)

    print("Step |    angle1    |    angle2    |    angle3    ||    x     y     z")
    print("-----+--------------+--------------+--------------++-----------------")
    for i, (ang, pos) in enumerate(zip(angles, xyz_pred)):
        a1,a2,a3 = ang
        x,y,z     = pos
        print(f"{i:4d} | {a1:10.3f} | {a2:10.3f} | {a3:10.3f} || {x:7.3f} {y:7.3f} {z:7.3f}")
    
    return angles, xyz_pred


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    angles, xyz = baseline_pcc_from_angle_increments(
        received_path = 'data/received_data_test.csv',
        command_path  = 'data/command_test.csv',
        d             = 28.0,
        device        = device
    )
