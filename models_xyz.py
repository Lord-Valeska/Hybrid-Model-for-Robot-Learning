import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import fsolve

# Geometry parameters (from the MATLAB code)
d = 28    # Distance from the chamber center to the base platform center
aa = 50   # Offset from base to the chamber cross-section (set to zero as in MATLAB)

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

def solve_end_effector(q_meas, init_guess=(0.0, 0.0127, 1.0)):
    # Straight posture:
    if abs(q_meas[0]-q_meas[1]) < 1e-6 and abs(q_meas[1]-q_meas[2]) < 1e-6:
        # equal leg lengths → pure z-offset
        z = q_meas[0] + aa
        return (0.0, 0.0, z, 0.0, 0.0, 0.0)

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
    k_eff = abs(k)
    theta_eff = -theta
    θ = abs(theta_eff)                       # bending angle magnitude

    # radius and height of the arc from the PCC model
    r0 = (1 - np.cos(θ)) / k_eff
    z0 = np.sin(θ) / k_eff

    # now *add* the cross-section offset aa:
    x = r0 * np.cos(phi) + aa * np.sin(θ) * np.cos(phi)
    y = r0 * np.sin(phi) + aa * np.sin(θ) * np.sin(phi)
    z = z0            + aa * np.cos(θ)

    return (x, y, z, phi, k_eff, theta_eff)

class PCCModel(nn.Module):
    """
    Pure physics mapping from cable lengths to end-effector pose.
    Uses the previous xyz to initialize the solver for continuity.
    """
    def __init__(self):
        super().__init__()

    def forward(self, q: torch.Tensor, prev_xyz: torch.Tensor=None):
        single = (q.ndim == 1)
        # make a numpy copy for solver
        q_np = q.detach().cpu().numpy()
        if single:
            q_np = q_np.reshape(1,3)

        # prepare prev_xyz numpy if provided
        if prev_xyz is not None:
            prev_np = prev_xyz.detach().cpu().numpy()
            if single:
                prev_np = prev_np.reshape(1,3)

        poses = []
        for i, qi in enumerate(q_np):
            if prev_xyz is not None:
                # derive init_guess from previous xyz
                p = prev_np[i]
                phi0 = np.arctan2(p[1], p[0])
                r_xy = np.linalg.norm(p[:2])
                k0 = 1.0 / (r_xy + 1e-6)
                theta0 = r_xy * k0
                init_guess = (phi0, k0, theta0)
                x, y, z, *_ = solve_end_effector(qi, init_guess=init_guess)
            else:
                x, y, z, *_ = solve_end_effector(qi)
            poses.append([x, y, z])

        pos = torch.tensor(poses, dtype=q.dtype, device=q.device)
        return pos.squeeze(0) if single else pos

class ResidualNet(nn.Module):
    """
    Learns only Δxyz (3-vector) from current xyz + control u.
    """
    def __init__(self, control_dim, hidden_sizes=(64,64,64)):
        super().__init__()
        layers = []
        inp_dim = 3 + control_dim    # only xyz plus u
        for h in hidden_sizes:
            layers += [nn.Linear(inp_dim, h), nn.ReLU()]
            inp_dim = h
        layers += [nn.Linear(inp_dim, 3)]  # output: Δx,Δy,Δz
        self.net = nn.Sequential(*layers)

    def forward(self, xyz, u):
        # xyz: (...,3), u: (...,control_dim)
        xu = torch.cat([xyz.float(), u.float()], dim=-1)
        return self.net(xu)  # (...,3)

class HybridDynamicsModel(nn.Module):
    """
    Combines a physics-based model (PCCModel) with a learned residual.
    """
    def __init__(self, control_dim):
        super().__init__()
        self.physics  = PCCModel()
        self.residual = ResidualNet(control_dim)
        # self.res_scale = nn.Parameter(torch.tensor(10.0))

    def forward(self, x, u):
        enc = x[..., :3].float()
        xyz = x[..., 3:].float()
        next_enc = enc + u.float()
        next_lengths = angle_to_length(next_enc)

        # pass prev xyz into physics solver
        try:
            phys_xyz = self.physics(next_lengths)
        except RuntimeError:
            print("Physics solver failed, using current xyz")
            phys_xyz = xyz

        delta_xyz = self.residual(xyz, u)
        # print(f"delta_enc: {delta_xyz}")

        next_xyz = phys_xyz + delta_xyz
        return torch.cat([next_enc, next_xyz], dim=-1)  # (...,6)
