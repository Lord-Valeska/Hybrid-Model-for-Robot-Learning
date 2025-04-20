import torch
import torch.nn as nn

class PCCModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, u):
        # TODO: Implement the forward pass
        pass

class ResidualNet(nn.Module):
    def __init__(self, state_dim, control_dim, hidden_sizes=(64,64)):
        super().__init__()
        layers = []
        inp_dim = state_dim + control_dim
        for h in hidden_sizes:
            layers += [nn.Linear(inp_dim, h), nn.ReLU()]
            inp_dim = h
        layers += [nn.Linear(inp_dim, state_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, u):
        xu = torch.cat([x, u], dim=-1)
        return self.net(xu)

class HybridDynamicsModel(nn.Module):
    def __init__(self, state_dim, control_dim, dt=0.01):
        super().__init__()
        self.physics = PCCModel()
        self.residual = ResidualNet(state_dim, control_dim)
        self.dt = dt

    def forward(self, x, u):
        phys_pred = self.physics(x, u)
        res_pred  = self.residual(x, u)
        return phys_pred + res_pred
