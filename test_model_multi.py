import numpy as np
import pandas as pd
import torch
from models_xyz import HybridDynamicsModel  # adjust import as needed

def test_saved_model_multi(model_path,
                           received_path,
                           marker_path,
                           command_path,
                           device=torch.device('cpu')):
    """
    Loads a saved HybridDynamicsModel trained with multi-step loss,
    reads initial encoder angles and xyz, then rollouts by taking the first
    prediction at each step.
    """
    # 1) load the model
    state_dim, control_dim = 6, 3
    model = HybridDynamicsModel(control_dim).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()

    # 2) read initial encoder angles
    rec_df = pd.read_csv(received_path, header=None)
    init_enc = rec_df.iloc[0, 1:4].to_numpy(dtype=np.float32)

    # 3) read initial xyz
    cleaned_df = pd.read_csv(marker_path)
    init_xyz = cleaned_df[['X', 'Y', 'Z']].iloc[0].to_numpy(dtype=np.float32)

    # 4) read all actions (skip the very first)
    cmd_df = pd.read_csv(command_path, header=None)
    all_actions = cmd_df.iloc[:, 1:4].to_numpy(dtype=np.float32)
    actions = all_actions[1:]

    # 5) rollout
    states = []
    current = torch.tensor(
        np.concatenate([init_enc, init_xyz]),
        dtype=torch.float32,
        device=device
    )
    states.append(current.cpu().numpy())

    with torch.no_grad():
        for a in actions:
            u = torch.tensor(a, dtype=torch.float32, device=device)
            # take the first prediction from the multi-step model
            nxt = model(current.unsqueeze(0), u.unsqueeze(0)).squeeze(0)
            states.append(nxt.cpu().numpy())
            current = nxt

    # 6) stack into array
    states_pred = np.stack(states)

    # 7) print in baseline format:
    print("Step |    enc1    |    enc2    |    enc3    ||    x     y     z")
    print("-----+--------------+--------------+--------------++-----------------")
    for i, st in enumerate(states_pred):
        e1, e2, e3, x, y, z = st
        print(f"{i:4d} | {e1:10.3f} | {e2:10.3f} | {e3:10.3f} || {x:7.3f} {y:7.3f} {z:7.3f}")

    return states_pred

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    states_pred = test_saved_model_multi(
        model_path    = './saved_models/multiple_step_dynamics_model.pt',
        received_path = 'data/received_data_test.csv',
        marker_path   = 'data/cleaned_xyz_test.csv',
        command_path  = 'data/command_test.csv',
        device        = device
    )
    print(f"Predicted multi-step rollout shape: {states_pred.shape}")
