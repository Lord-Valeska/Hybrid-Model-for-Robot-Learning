import math
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import fsolve

def angle_to_length(angleIN):
    """
    Polynomial mapping from actuator angle to cable length.
    """
    return (1.327e-10 * angleIN**5
          - 7.894e-08 * angleIN**4
          + 1.314e-05 * angleIN**3
          - 0.001259  * angleIN**2
          - 0.1502    * angleIN
          + 80.92)

def mpcr_forward(inputs):
    # Inputs = [L1, L2, L3] in mm
    L1 = inputs[0] / 1000.0  # convert to meters
    L2 = inputs[1] / 1000.0
    L3 = inputs[2] / 1000.0
    # Constants
    r = 28e-3  # meters
    c_n = 6.0
    bottom_length = 0.030
    top_length = 0.021
    # Base angles
    a1 = math.pi / 2.0
    a2 = a1 + 2.0 * math.pi / 3.0
    a3 = a1 + 4.0 * math.pi / 3.0
    MIN = 1e-4
    if abs(L1 - L2) < MIN and abs(L1 - L3) < MIN:
        return [0.0, 0.0, L1 * 1000.0]
    # Step 1: φ
    K = (L1 - L2) / (L1 - L3)
    # check for NaN or division by zero
    if abs(L1 - L3) < MIN:
        if L2 - L1 < 0:
            phi = math.pi
        else:
            phi = 0.0
    else:
        if abs(K - 1.0) < MIN:
            if K >= 1.0:
                phi = math.pi / 2.0
            else:
                phi = -math.pi / 2.0
        else:
            tan_phi = (K + 1.0) / (math.sqrt(3) * (K - 1.0))
            phi = math.atan(tan_phi)
    # Step 2: ρ
    try:
        rho = r * (L1 * math.cos(a2 - phi) - L2 * math.cos(a1 - phi)) / (L1 - L2)
        if math.isnan(rho):
            raise ValueError("NaN fallback")
    except (ValueError, ZeroDivisionError) as _:
        rho = r * (L1 * math.cos(a3 - phi) - L3 * math.cos(a1 - phi)) / (L1 - L3)
    # Step 3: θ
    denom = 2.0 * c_n * (rho - r * math.cos(a1 - phi))
    sin_theta_over_cn = L1 / denom
    sin_theta_over_cn = max(min(sin_theta_over_cn, 1.0), -1.0)
    theta = 2.0 * c_n * math.asin(sin_theta_over_cn)
    # Final pose computation
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    x = rho * cos_phi * (1.0 - cos_theta) + top_length * sin_theta * cos_phi
    y = rho * sin_phi * (1.0 - cos_theta) + top_length * sin_theta * sin_phi
    z = rho * sin_theta + bottom_length + top_length * cos_theta
    # Convert to millimeters
    return [x * 1000.0, y * 1000.0, z * 1000.0]


class PCCModel(nn.Module):
    """
    Pure physics mapping from cable lengths to end-effector pose.
    Uses the analytic mpcr_forward solver.
    """
    def __init__(self):
        super().__init__()

    def forward(self, q: torch.Tensor):
        """
        q: Tensor of shape (..., 3), each entry is [L1, L2, L3] in mm
        Returns:
            Tensor of shape (..., 3) with end-effector [x, y, z] in mm
        """
        single = (q.ndim == 1)
        # Move to numpy for analytic solver
        q_np = q.detach().cpu().numpy()
        if single:
            q_np = q_np.reshape(1, 3)

        poses = []
        for qi in q_np:
            # qi is a length-3 array in mm
            xyz = mpcr_forward(qi.tolist())
            poses.append(xyz)

        # Back to torch
        pos = torch.tensor(poses, dtype=q.dtype, device=q.device)
        # Collapse batch dim if needed
        return pos.squeeze(0) if single else pos

class ResidualNet(nn.Module):
    """
    Learns only Δxyz (3-vector) from current xyz + control u.
    """
    def __init__(self, control_dim, hidden_sizes=(128,128,128,128)):
        super().__init__()
        layers = []
        inp_dim = 3 + control_dim
        for h in hidden_sizes:
            layers += [nn.Linear(inp_dim, h), nn.ReLU()]
            inp_dim = h
        layers += [nn.Linear(inp_dim, 3)]
        self.net = nn.Sequential(*layers)

    def forward(self, xyz, u):
        xu = torch.cat([xyz.float(), u.float()], dim=-1)
        return self.net(xu)


class HybridDynamicsModel(nn.Module):
    """
    Combines a physics-based model (PCCModel) with a learned residual.
    """
    def __init__(self, control_dim):
        super().__init__()
        self.physics  = PCCModel()
        self.residual = ResidualNet(control_dim)

    def forward(self, x, u):
        enc = x[..., :3].float()
        xyz = x[..., 3:].float()
        next_enc = enc + u.float()
        next_lengths = angle_to_length(next_enc)

        # physics returns (poses, updated_guesses)
        phys_xyz = self.physics(next_lengths)

        delta_xyz = self.residual(xyz, u)
        # print(delta_xyz)
        next_xyz = phys_xyz + delta_xyz
        out = torch.cat([next_enc, next_xyz], dim=-1)
        return out
