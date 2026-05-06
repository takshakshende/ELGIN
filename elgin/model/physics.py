"""physics.py — Analytical physics kernels for the CFD-GNN.

All functions are pure PyTorch operations (differentiable, GPU-compatible).

Functions
---------
cunningham_correction        Cunningham slip correction for sub-micron particles
stokes_drag_acc              Stokes drag acceleration with Cunningham correction
evaporation_diameter         Wells' law droplet evaporation  (S2)
saffman_lift_acc             Saffman shear-lift force        (S6)
brownian_sigma               Brownian diffusion coefficient  (S11)
turbulent_dispersion_kick    Stochastic TKE-based velocity kick  (S3/S11)
gravity_vector               Gravity vector for dim-D domain

References
----------
Cunningham (1910), Stokes (1851), Wells (1934), Saffman (1965),
Einstein (1905), McLaughlin (1991), Pope (2000).
"""

from __future__ import annotations
import math
from typing import Optional

import torch


# ── Physical constants ────────────────────────────────────────────────────────
K_B    = 1.380649e-23   # J/K  — Boltzmann constant
LAMBDA = 68e-9          # m    — mean free path of air at 20 °C, 101 325 Pa
MU_AIR = 1.81e-5        # Pa·s — dynamic viscosity of air at 20 °C


# ── Cunningham correction ─────────────────────────────────────────────────────

def cunningham_correction(d_p: torch.Tensor,
                          lam: float = LAMBDA) -> torch.Tensor:
    """Cunningham slip correction factor.

        Cc = 1 + (2λ/d_p)[1.257 + 0.400·exp(-1.10·d_p/(2λ))]

    Significant for d_p < 2 µm; reaches ~1.4 at d_p = 1 µm.

    Args:
        d_p : (N,) particle diameter [m]
    Returns:
        (N,) Cunningham factor >= 1
    """
    Kn = 2.0 * lam / d_p.clamp(min=1e-12)
    return 1.0 + Kn * (1.257 + 0.400 * torch.exp(-1.10 / (Kn + 1e-30)))


# ── Stokes drag ───────────────────────────────────────────────────────────────

def stokes_drag_acc(
    v_p:   torch.Tensor,   # (N, dim) particle velocity
    u_f:   torch.Tensor,   # (N, dim) fluid velocity at particle location
    d_p:   torch.Tensor,   # (N,)     particle diameter  [m]
    rho_p: torch.Tensor,   # (N,)     particle density   [kg/m^3]
    mu:    float = MU_AIR,
) -> torch.Tensor:
    """Cunningham-corrected Stokes drag acceleration.

        a_drag = (u_fluid - v_p) / (tau_p / Cc)
        tau_p  = rho_p * d_p^2 / (18 * mu)

    Returns: (N, dim)
    """
    tau_p = rho_p * d_p ** 2 / (18.0 * mu)          # (N,)
    Cc    = cunningham_correction(d_p)                # (N,)
    tau_c = (tau_p / Cc).clamp(min=1e-9)
    return (u_f - v_p) / tau_c.unsqueeze(-1)


# ── S2: Droplet evaporation (Wells' law) ──────────────────────────────────────

def evaporation_diameter(
    d_p0:  torch.Tensor,   # (N,) initial diameter  [m]
    age:   torch.Tensor,   # (N,) particle age       [s]
    rho_p: torch.Tensor,   # (N,) particle density  [kg/m^3]
    D_v:   float = 2.6e-5, # m^2/s  vapour diffusivity in air at 20 °C
    B_M:   float = 0.0263, # Spalding mass-transfer number  (pure water at 50 %RH)
    eps:   float = 1e-9,
) -> torch.Tensor:
    """Wells' d^2-law evaporation model for droplet nuclei.

    The D^2-law gives:
        d_p(t)^2 = d_p0^2 - K * t
        K = 8 * rho_air * D_v * ln(1 + B_M) / rho_p   [m^2/s]

    Returns d_p(t) clamped to nucleus diameter d_nuc = d_p0/sqrt(2)
    (once all free water has evaporated, the residue diameter ~70 % of d_p0).

    Args:
        d_p0  : (N,) initial diameter        [m]
        age   : (N,) time since injection     [s]
        rho_p : (N,) droplet density         [kg/m^3]
        D_v   : vapour diffusivity            [m^2/s]
        B_M   : Spalding mass-transfer number [-]
    Returns:
        (N,) current diameter, >= 0.5 * d_p0  (nucleus floor)
    """
    rho_air = 1.225   # kg/m^3
    K = 8.0 * rho_air * D_v * math.log(1.0 + B_M) / rho_p.clamp(min=1.0)
    d2_new  = (d_p0 ** 2 - K * age).clamp(min=(0.5 * d_p0) ** 2)
    return d2_new.sqrt()


# ── S6: Saffman shear-lift force ──────────────────────────────────────────────

def saffman_lift_acc(
    v_p:     torch.Tensor,            # (N, dim) particle velocity
    u_f:     torch.Tensor,            # (N, dim) fluid velocity at particle
    d_p:     torch.Tensor,            # (N,)     particle diameter  [m]
    rho_p:   torch.Tensor,            # (N,)     particle density   [kg/m^3]
    du_dy:   Optional[torch.Tensor],  # (N,)     wall-normal shear |dU/dy| [1/s]
    nu:      float = 1.5e-5,          # m^2/s kinematic viscosity
    rho_f:   float = 1.225,           # kg/m^3 fluid density
) -> torch.Tensor:
    """Saffman shear-lift acceleration (McLaughlin 1991 correction).

    The Saffman lift force acts perpendicular to the slip velocity in a
    shear flow:
        F_lift = 1.615 * mu * d_p * sqrt(Re_G) * (u_fluid - v_p) x omega_fluid
        Re_G   = d_p^2 * |du/dy| / nu

    Simplified 2-D form (lift in y-direction for shear dU_x/dy):
        a_lift_y = C_L * |u_slip| * sqrt(|du/dy|) / (rho_p/rho_f * d_p)

    where C_L = 1.615 * sqrt(nu) / (pi / 6).

    If du_dy is None (no shear information), returns zero.

    Returns: (N, dim)  lift acceleration
    """
    if du_dy is None:
        return torch.zeros_like(v_p)

    mu   = nu * rho_f                             # dynamic viscosity
    Vp   = math.pi / 6.0                          # volume of unit-diameter sphere

    # Shear Reynolds number
    Re_G = (d_p ** 2 * du_dy.abs() / nu).clamp(min=0.0)

    # Slip velocity magnitude
    slip     = u_f - v_p                          # (N, dim)
    slip_mag = slip.norm(dim=-1)                   # (N,)

    # Lift coefficient magnitude  (per unit mass)
    #   F_lift / m_p = 1.615 * mu * d_p * sqrt(Re_G) * slip / (m_p)
    #   m_p          = rho_p * Vp * d_p^3
    m_p      = (rho_p * Vp * d_p ** 3).clamp(min=1e-30)
    F_mag    = 1.615 * mu * d_p * Re_G.sqrt() * slip_mag   # (N,)
    a_mag    = F_mag / m_p                                  # (N,)

    # Direction: perpendicular to slip, in the plane of the flow
    # For 2-D flow, rotate slip by 90°: (vx, vy) -> (-vy, vx)
    if v_p.shape[-1] == 2:
        slip_norm = slip / (slip_mag.unsqueeze(-1) + 1e-8)
        lift_dir  = torch.stack([-slip_norm[:, 1], slip_norm[:, 0]], dim=-1)
    else:
        # 3-D: use (slip x z_hat) as lift direction
        z_hat     = torch.zeros_like(v_p)
        z_hat[:, 2] = 1.0
        cross     = torch.cross(slip, z_hat, dim=-1)
        cross_mag = cross.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        lift_dir  = cross / cross_mag

    # Correct sign: lift pushes away from wall (toward high velocity side)
    # Use sign of shear to orient
    sign     = du_dy.sign().unsqueeze(-1)
    lift_acc = sign * a_mag.unsqueeze(-1) * lift_dir    # (N, dim)

    return lift_acc


# ── S11: Brownian motion ──────────────────────────────────────────────────────

def brownian_sigma(
    d_p:   torch.Tensor,   # (N,)  diameter [m]
    T:     float = 293.15, # K    air temperature
    dt:    float = 0.01,   # s    timestep
    rho_p: Optional[torch.Tensor] = None,  # unused but kept for API consistency
    mu:    float = MU_AIR,
    lam:   float = LAMBDA,
) -> torch.Tensor:
    """Brownian diffusion displacement standard deviation per timestep.

    From the Einstein-Smoluchowski relation:
        D_B  = k_B * T * Cc / (3 * pi * mu * d_p)
        sigma = sqrt(2 * D_B * dt)            [m per sqrt(dt)]

    Significant only for d_p < 0.5 µm.

    Returns: (N,) sigma [m] — the std-dev of the Gaussian Brownian kick
    """
    Cc  = cunningham_correction(d_p, lam=lam)
    D_B = (K_B * T * Cc) / (math.pi * 3.0 * mu * d_p.clamp(min=1e-12))
    return (2.0 * D_B * dt).clamp(min=0.0).sqrt()


# ── S3: Stochastic turbulent dispersion ───────────────────────────────────────

def turbulent_dispersion_kick(
    k_fluid: torch.Tensor,   # (N,)  TKE at particle location [m^2/s^2]
    dt:      float = 0.01,   # s     timestep
    nu:      float = 1.5e-5, # m^2/s kinematic viscosity
    C_mu:    float = 0.09,
    dim:     int   = 2,
    device:  Optional[torch.device] = None,
) -> torch.Tensor:
    """Stochastic turbulent dispersion velocity perturbation.

    Based on the Discrete Random Walk (DRW) model used by OpenFOAM's
    StochasticDispersionRAS submodel:

        v' ~ N(0, 2k/3 * I)   for a Lagrangian time scale tau_t

    The perturbation is applied once per LagrangianGNN step:
        v_dispersion = sqrt(2k/3) * epsilon,  epsilon ~ N(0, I)

    Returns: (N, dim) stochastic velocity perturbation [m/s]
    """
    # Turbulent velocity fluctuation magnitude: sigma_v = sqrt(2k/3)
    sigma_v = (2.0 * k_fluid.clamp(min=0.0) / 3.0).sqrt()  # (N,)

    eps = torch.randn(k_fluid.shape[0], dim,
                      dtype=k_fluid.dtype,
                      device=k_fluid.device if device is None else device)
    return sigma_v.unsqueeze(-1) * eps                       # (N, dim)


# ── Gravity vector ────────────────────────────────────────────────────────────

def gravity_vector(g: float = 9.81, dim: int = 2,
                   dtype=torch.float32,
                   device: Optional[torch.device] = None) -> torch.Tensor:
    """Return gravity vector pointing in the -y direction."""
    vec = torch.zeros(dim, dtype=dtype, device=device)
    vec[1] = -g
    return vec
