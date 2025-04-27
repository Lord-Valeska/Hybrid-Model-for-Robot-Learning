import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

class SingleStepDynamicsDataset(Dataset):
    """
    Each data sample is a dictionary containing (x_t, u_t, x_{t+1}) in the form:
    {'state': x_t,
     'action': u_t,
     'next_state': x_{t+1},
    }
    where:
     x_t: torch.float32 tensor of shape (state_size,)
     u_t: torch.float32 tensor of shape (action_size,)
     x_{t+1}: torch.float32 tensor of shape (state_size,)
    """

    def __init__(self, collected_data):
        self.data = collected_data
        self.trajectory_length = self.data[0]['actions'].shape[0]

    def __len__(self):
        return len(self.data) * self.trajectory_length

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def __getitem__(self, item):
        """
        Return the data sample corresponding to the index <item>.
        :param item: <int> index of the data sample to produce.
            It can take any value in range 0 to self.__len__().
        :return: data sample corresponding to encoded as a dictionary with keys (state, action, next_state).
        The class description has more details about the format of this data sample.
        """
        sample = {
            'state': None,
            'action': None,
            'next_state': None,
        }
        trajectory_idx = item // self.trajectory_length
        time_step = item % self.trajectory_length
        sample = {
            'state': torch.from_numpy(self.data[trajectory_idx]['states'][time_step]),
            'action': torch.from_numpy(self.data[trajectory_idx]['actions'][time_step]),
            'next_state': torch.from_numpy(self.data[trajectory_idx]['states'][time_step + 1])
        }
        return sample

class MultiStepDynamicsDataset(Dataset):
    """
    Dataset containing multi-step dynamics data.

    Each data sample is a dictionary containing (state, action, next_state) in the form:
    {'state': x_t, -- initial state of the multipstep torch.float32 tensor of shape (state_size,)
     'action': [u_t,..., u_{t+num_steps-1}] -- actions applied in the muli-step.
                torch.float32 tensor of shape (num_steps, action_size)
     'next_state': [x_{t+1},..., x_{t+num_steps} ] -- next multiple steps for the num_steps next steps.
                torch.float32 tensor of shape (num_steps, state_size)
    }
    """

    def __init__(self, collected_data, num_steps=3):
        self.data = collected_data
        self.trajectory_length = self.data[0]['actions'].shape[0] - num_steps + 1
        self.num_steps = num_steps

    def __len__(self):
        return len(self.data) * (self.trajectory_length)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def __getitem__(self, item):
        """
        Return the data sample corresponding to the index <item>.
        :param item: <int> index of the data sample to produce.
            It can take any value in range 0 to self.__len__().
        :return: data sample corresponding to encoded as a dictionary with keys (state, action, next_state).
        The class description has more details about the format of this data sample.
        """
        sample = {
            'state': None,
            'action': None,
            'next_state': None
        }
        trajectory_idx = item // self.trajectory_length
        time_step = item % self.trajectory_length
        sample = {
            'state': torch.from_numpy(self.data[trajectory_idx]['states'][time_step]),
            'action': torch.from_numpy(self.data[trajectory_idx]['actions'][time_step:time_step + self.num_steps]),
            'next_state': torch.from_numpy(self.data[trajectory_idx]['states'][time_step + 1: time_step + 1 + self.num_steps])
        }
        return sample
    
def process_data_single_step(collected_data, batch_size=500):
    """
    Process the collected data and returns a DataLoader for train and one for validation.
    The data provided is a list of trajectories (like collect_data_random output).
    Each DataLoader must load dictionary as {'state': x_t,
     'action': u_t,
     'next_state': x_{t+1},
    }
    where:
     x_t: torch.float32 tensor of shape (batch_size, state_size)
     u_t: torch.float32 tensor of shape (batch_size, action_size)
     x_{t+1}: torch.float32 tensor of shape (batch_size, state_size)

    The data should be split in a 80-20 training-validation split.
    :param collected_data:
    :param batch_size: <int> size of the loaded batch.
    :return:

    Hints:
     - Pytorch provides data tools for you such as Dataset and DataLoader and random_split
     - You should implement SingleStepDynamicsDataset below.
        This class extends pytorch Dataset class to have a custom data format.
    """
    train_loader = None
    val_loader = None
    dataset = SingleStepDynamicsDataset(collected_data)
    train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def process_data_multiple_step(collected_data, batch_size=500, num_steps=4):
    """
    Process the collected data and returns a DataLoader for train and one for validation.
    The data provided is a list of trajectories (like collect_data_random output).
    Each DataLoader must load dictionary as
    {'state': x_t,
     'action': u_t, ..., u_{t+num_steps-1},
     'next_state': x_{t+1}, ... , x_{t+num_steps}
    }
    where:
     state: torch.float32 tensor of shape (batch_size, state_size)
     next_state: torch.float32 tensor of shape (batch_size, num_steps, action_size)
     action: torch.float32 tensor of shape (batch_size, num_steps, state_size)

    Each DataLoader must load dictionary dat
    The data should be split in a 80-20 training-validation split.
    :param collected_data:
    :param batch_size: <int> size of the loaded batch.
    :param num_steps: <int> number of steps to load the multistep data.
    :return:

    Hints:
     - Pytorch provides data tools for you such as Dataset and DataLoader and random_split
     - You should implement MultiStepDynamicsDataset below.
        This class extends pytorch Dataset class to have a custom data format.
    """
    train_loader = None
    val_loader = None
    dataset = MultiStepDynamicsDataset(collected_data)
    train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def read_data(file_paths, seq_len=51):
    """
    :param file_paths: str or List[str], paths to one or more CSV files
                       each file must have columns: traj, encoder1-3, X, Y, Z, angle1-3
    :return: List[dict], each with
             - "states": ndarray(N, 6)
             - "actions": ndarray(N-1, 3)  # first action ignored
    """
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    all_trajectories = []

    for file_path in file_paths:
        # Read the merged data (with header)
        df = pd.read_csv(file_path)
        # Drop rows with missing actions
        df = df.dropna(subset=['angle1', 'angle2', 'angle3'])

        # Group by trajectory ID
        for traj_id, group in df.groupby('traj'):
            # Ensure original ordering
            group = group.sort_index()

            states  = group[['encoder1', 'encoder2', 'encoder3', 'X', 'Y', 'Z']].values.astype(np.float64)
            actions_all = group[['angle1', 'angle2', 'angle3']].values.astype(np.float64)

            # Ignore the first action so that actions has one fewer row than states
            actions = actions_all[1:]

            all_trajectories.append({
                'states': states,   # (N,6)
                'actions': actions, # (N-1,3)
            })

    total_traj = len(all_trajectories)
    print(f"Stored and returning {total_traj} trajectories from {len(file_paths)} file(s).")
    return all_trajectories
