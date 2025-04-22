import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import fsolve

# Geometry parameters (from the MATLAB code)
d = 28    # Distance from the chamber center to the base platform center
aa = 43.5   # Offset from base to the chamber cross-section (set to zero as in MATLAB)

def end_effector_equations(vars, q_meas, d):
    """
    Solve for phi, k, and theta such that:
      q1 = theta*(1/k - d*sin(phi))
      q2 = theta*(1/k + d*sin(pi/3 + phi))
      q3 = theta*(1/k - d*cos(pi/6 + phi))
    """
    phi, k, theta = vars
    q1, q2, q3 = q_meas

    eq1 = theta*(1.0/k - d*np.sin(phi)) - q1
    eq2 = theta*(1.0/k + d*np.sin(np.pi/3 + phi)) - q2
    eq3 = theta*(1.0/k - d*np.cos(np.pi/6 + phi)) - q3

    return [eq1, eq2, eq3]

def angle_to_length (angleIN):

    return 1.327e-10*angleIN **5 - 7.894e-08*angleIN **4 + 1.314e-05*angleIN **3 - 0.001259*angleIN**2 - 0.1502*angleIN + 80.92

def solve_end_effector(q_meas, d, init_guess=(0.0, 0.05, 0.3)):
    """
    Given actuator lengths q_meas = [q1, q2, q3], solve for phi, k, theta.
    Then compute the end–effector position (x, y, z) and its normal vector.
    Enforces: k > 0, theta < 0, and z > 0.
    """
    # Special case: all three lengths are nearly equal
    if abs(q_meas[0]-q_meas[1]) < 1e-6 and abs(q_meas[1]-q_meas[2]) < 1e-6:
        base_normal = np.array([0, 0, 1])
        pos = q_meas[0] * base_normal  # along z
        ee_normal = base_normal.copy()
        return (pos[0], pos[1], pos[2], 0.0, 0.0, 0.0, ee_normal)

    sol, info, ier, mesg = fsolve(
        end_effector_equations, init_guess,
        args=(q_meas, d),
        full_output=True,
        xtol=1e-12,
        maxfev=int(1e8)
    )
    if ier != 1:
        raise RuntimeError(f"physics solver failed to converge: {mesg}")

    phi, k, theta = sol

    # Enforce sign conventions:
    k_eff = abs(k)         # curvature > 0
    theta_eff = -theta     # ensure theta is negative

    # Constant curvature backbone:
    # In a plane, if the curvature is k (R = 1/k) and the bending angle is |theta_eff|,
    # a common parameterization is:
    #   x_local(s) = (1 - cos(s)) / k_eff,   z_local(s) = sin(s) / k_eff,  with s in [0, |theta_eff|]
    # Our computed tip is:
    #   x_tip = x_local(|theta_eff|)*cos(phi), y_tip = x_local(|theta_eff|)*sin(phi), z_tip = z_local(|theta_eff|)
    r0 = (1 - np.cos(abs(theta_eff))) / k_eff
    z0 = np.sin(abs(theta_eff)) / k_eff

    x = r0 * np.cos(phi)
    y = r0 * np.sin(phi)
    z = abs(z0)

    return (x, y, z, phi, k_eff, theta_eff)

class PCCModel(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d

    def forward(self, q: torch.Tensor):
        single = (q.ndim == 1)
        q_np = q.detach().cpu().numpy()
        if single:
            q_np = q_np.reshape(1,3)

        poses = []
        for qi in q_np:
            x, y, z, *_ = solve_end_effector(qi, self.d)
            poses.append([x, y, z])

        pos = torch.tensor(poses, dtype=q.dtype, device=q.device)
        return pos.squeeze(0) if single else pos

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
        x = x.float()
        u = u.float()
        xu = torch.cat([x, u], dim=-1)
        return self.net(xu)

class HybridDynamicsModel(nn.Module):
    def __init__(self, state_dim, control_dim):
        super().__init__()
        self.physics = PCCModel(d)
        self.residual = ResidualNet(state_dim, control_dim)

    def forward(self, x, u):
        x = x.float()
        u = u.float()

        enc = x[..., :3]    # (...,3)
        xyz = x[..., 3:]    # (...,3)
        next_enc = enc + u  # (...,3)

        try:
            phys_xyz = self.physics(next_enc)
        except RuntimeError:
            # fallback: skip physics update, keep current xyz
            print("Physics model failed, using current xyz")
            phys_xyz = xyz

        delta = self.residual(x, u)       # (...,6)
        delta_enc = delta[..., :3]        # (...,3)
        delta_xyz = delta[..., 3:]        # (...,3)

        next_enc = next_enc + delta_enc   # (...,3)
        next_xyz = phys_xyz + delta_xyz   # (...,3)

        return torch.cat([next_enc, next_xyz], dim=-1)  # (...,6)
