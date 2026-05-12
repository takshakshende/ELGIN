"""TurbulenceClosure — learned and analytic k-ω SST surrogate.
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn

from .config import CfdGNNConfig


# ---------------------------------------------------------------------------
#  SST constants (Menter 1994)
# ---------------------------------------------------------------------------

_a1    = 0.31
_beta_star = 0.09
_sigma_omega2 = 0.856
_kappa = 0.41


def _mlp(in_d, hid, out_d, layers, ln=True):
    mods = [nn.Linear(in_d, hid), nn.ReLU()]
    for _ in range(layers - 1):
        mods += [nn.Linear(hid, hid), nn.ReLU()]
    mods.append(nn.Linear(hid, out_d))
    if ln:
        mods.append(nn.LayerNorm(out_d))
    return nn.Sequential(*mods)


def analytic_nu_t_sst(
    k:       torch.Tensor,   # (N,)
    omega:   torch.Tensor,   # (N,)
    d_wall:  torch.Tensor,   # (N,)  distance to nearest wall (m)
    Omega_mag: torch.Tensor, # (N,)  vorticity magnitude |∇×U|
    nu:      float = 1.5e-5,
) -> torch.Tensor:
    """Compute turbulent viscosity from k-ω SST model.

    Implements Eq. (13)-(17) in mathematical_formulation.tex.

    Args:
        k:         Turbulent kinetic energy at each cell.
        omega:     Specific dissipation rate at each cell.
        d_wall:    Distance from each cell to the nearest no-slip wall.
        Omega_mag: Vorticity magnitude at each cell.
        nu:        Kinematic viscosity of air.

    Returns:
        nu_t: (N,) turbulent viscosity.
    """
    k     = k.clamp(min=1e-8)
    omega = omega.clamp(min=1e-8)
    d     = d_wall.clamp(min=1e-6)

    sqrt_k = k.sqrt()

    # SST F2 blending function
    arg2 = torch.max(
        2.0 * sqrt_k / (_beta_star * omega * d),
        500.0 * nu / (omega * d**2)
    )
    F2 = torch.tanh(arg2**2)

    # Turbulent viscosity: limited by vorticity (Bradshaw hypothesis)
    nu_t_vortex = _a1 * k / torch.max(_a1 * omega, Omega_mag * F2)
    nu_t_simple = k / omega

    # Full SST formula blends between the two
    nu_t = torch.min(nu_t_vortex, nu_t_simple)
    return nu_t.clamp(min=0.0, max=1.0)   # physical bounds: ν_t ∈ [0, 1 m²/s]


def effective_viscosity(
    nu_t:  torch.Tensor,   # (N,)
    nu:    float = 1.5e-5,
) -> torch.Tensor:
    """Effective viscosity ν_eff = ν + ν_t."""
    return nu + nu_t


# ---------------------------------------------------------------------------
#  Turbulence closure module
# ---------------------------------------------------------------------------

class TurbulenceClosure(nn.Module):
    """Computes turbulent viscosity ν_t for use in momentum equation.

    Supports two modes:
      analytic_closure = True:  Use SST formula directly.
      analytic_closure = False: SST formula + learned residual correction.

    The learned correction is particularly important near the spray nozzle
    where standard SST assumptions break down due to high-curvature streamlines.
    """

    def __init__(self, cfg: CfdGNNConfig):
        super().__init__()
        self.cfg     = cfg
        self.analytic = cfg.analytic_closure

        if not cfg.analytic_closure:
            # Predict residual correction δ(ν_t) from [k, ω, d_wall, |S|]
            self.correction_net = _mlp(
                cfg.turb_in, cfg.turb_hidden, 1,
                cfg.turb_layers, ln=False
            )

    def forward(
        self,
        k:         torch.Tensor,   # (N,)
        omega:     torch.Tensor,   # (N,)
        d_wall:    torch.Tensor,   # (N,)
        S_mag:     Optional[torch.Tensor] = None,   # (N,) strain-rate mag
        Omega_mag: Optional[torch.Tensor] = None,   # (N,) vorticity mag
    ) -> torch.Tensor:
        """Compute ν_t (N,) at each mesh cell.

        Args:
            k:         Turbulent kinetic energy.
            omega:     Specific dissipation rate.
            d_wall:    Distance to nearest wall.
            S_mag:     Strain-rate magnitude (for learned correction).
            Omega_mag: Vorticity magnitude (for SST F2 function).

        Returns:
            nu_t: (N,) turbulent viscosity.
        """
        if Omega_mag is None:
            Omega_mag = torch.zeros_like(k)
        if S_mag is None:
            S_mag = torch.zeros_like(k)

        # Analytic SST
        nu_t = analytic_nu_t_sst(k, omega, d_wall, Omega_mag,
                                  nu=self.cfg.nu)

        if not self.analytic:
            # Learned residual correction
            L_ref = self.cfg.L_ref
            feats = torch.stack([
                k.log1p(),                     # log(1 + k)
                omega.log1p(),                 # log(1 + ω)
                (d_wall / L_ref).clamp(0, 1),  # normalised wall distance
                S_mag.clamp(0, 100) / 100.0,   # normalised strain rate
            ], dim=-1)                         # (N, 4)
            delta = self.correction_net(feats).squeeze(-1)  # (N,)
            nu_t  = (nu_t + delta).clamp(min=0.0, max=1.0)

        return nu_t

    def production_term(
        self,
        k:     torch.Tensor,
        omega: torch.Tensor,
        nu_t:  torch.Tensor,
        S_mag: torch.Tensor,
    ) -> torch.Tensor:
        """Turbulent production P_k = min(ν_t S², 10 β* k ω).

        Used to update k and ω in the Eulerian GNN output de-normalisation.
        """
        P_k = nu_t * S_mag**2
        return P_k.clamp(max=10.0 * _beta_star * k * omega)
