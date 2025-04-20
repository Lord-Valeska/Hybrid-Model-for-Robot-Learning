import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

from dataset import process_data_single_step  
from models import HybridDynamicsModel   

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class PoseLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pose_pred, pose_target):
        pose_loss = None
        loss_x = F.mse_loss(pose_pred[:, 0], pose_target[:, 0])
        loss_y = F.mse_loss(pose_pred[:, 1], pose_target[:, 1])
        pose_loss = loss_x + loss_y
        return pose_loss

class SingleStepLoss(nn.Module):

    def __init__(self, loss_fn):
        super().__init__()
        self.loss = loss_fn

    def forward(self, model, state, action, target_state):
        """
        Compute the single step loss resultant of querying model with (state, action) and comparing the predictions with target_state.
        """
        single_step_loss = None
        pred_state = model(state, action)
        single_step_loss = self.loss(pred_state, target_state)
        return single_step_loss
    
def train_step(model, train_loader, optimizer) -> float:
    """
    Performs an epoch train step.
    :param model: Pytorch nn.Module
    :param train_loader: Pytorch DataLoader
    :param optimizer: Pytorch optimizer
    :return: train_loss <float> representing the average loss among the different mini-batches.
        Loss needs to be MSE loss.
    """
    train_loss = 0. 
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        state = batch['state'].to(device)
        action = batch['action'].to(device)
        next_state_gth = batch['next_state'].to(device)
        optimizer.zero_grad()
        loss = pose_loss(model, state, action, next_state_gth)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss/len(train_loader)

def val_step(model, val_loader) -> float:
    """
    Perfoms an epoch of model performance validation
    :param model: Pytorch nn.Module
    :param train_loader: Pytorch DataLoader
    :param optimizer: Pytorch optimizer
    :return: val_loss <float> representing the average loss among the different mini-batches
    """
    val_loss = 0. # TODO: Modify the value
    # Initialize the validation loop
    model.eval()
    for batch_idx, batch in enumerate(val_loader):
        loss = None
        state = batch['state'].to(device)
        action = batch['action'].to(device)
        next_state_gth = batch['next_state'].to(device)
        loss = pose_loss(model, state, action, next_state_gth)
        val_loss += loss.item()
    return val_loss/len(val_loader)

def train_model(model, train_dataloader, val_dataloader, num_epochs=100, lr=1e-3):
    """
    Trains the given model for `num_epochs` epochs. Use SGD as an optimizer.
    You may need to use `train_step` and `val_step`.
    :param model: Pytorch nn.Module.
    :param train_dataloader: Pytorch DataLoader with the training data.
    :param val_dataloader: Pytorch DataLoader with the validation data.
    :param num_epochs: <int> number of epochs to train the model.
    :param lr: <float> learning rate for the weight update.
    :return:
    """
    optimizer = None
    optimizer = optim.Adam(model.parameters(), lr=lr)
    pbar = tqdm(range(num_epochs))
    train_losses = []
    val_losses = []
    for epoch_i in pbar:
        train_loss_i = None
        val_loss_i = None
        train_loss_i = train_step(model, train_dataloader, optimizer)
        val_loss_i = val_step(model, val_dataloader)
        pbar.set_description(f'Train Loss: {train_loss_i:.4f} | Validation Loss: {val_loss_i:.4f}')
        train_losses.append(train_loss_i)
        val_losses.append(val_loss_i)
    return train_losses, val_losses

if __name__ == "__main__":
    # Load your collected_data (list of dicts with 'states' and 'actions')
    collected = np.load('collected_data.npy', allow_pickle=True)
    collected_data = collected.tolist()

    train_loader, val_loader = process_data_single_step(collected_data, batch_size=64)

    dt = 0.01
    state_dim  = collected_data[0]['states'].shape[1]
    action_dim = collected_data[0]['actions'].shape[1]
    model = HybridDynamicsModel(state_dim, action_dim, dt=dt).to(device)

    pose_loss = PoseLoss()
    pose_loss = SingleStepLoss(pose_loss).to(device)

    LR = 0.001
    NUM_EPOCHS = 1000
    train_losses, val_losses = train_model(model,
                                           train_loader, val_loader, num_epochs=NUM_EPOCHS, lr=LR)
    
    # plot train loss and test loss:
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 3))
    axes[0].plot(train_losses)
    axes[0].grid()
    axes[0].set_title('Train Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Train Loss')
    axes[0].set_yscale('log')
    axes[1].plot(val_losses)
    axes[1].grid()
    axes[1].set_title('Validation Loss')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Validation Loss')
    axes[1].set_yscale('log')
    plt.show()

    # save model:
    os.makedirs('./saved_models', exist_ok=True)
    save_path = './saved_models/single_step_dynamics_model.pt'
    torch.save(model.state_dict(), save_path)