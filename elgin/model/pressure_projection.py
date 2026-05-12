"""PressureProjection — enforce div(U) = 0 after Eulerian GNN prediction.
"""

from __future__ import annotations
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .config import CfdGNNConfig


def _mlp(in_d, hid, out_d, layers, ln=True):
    mods = [nn.Linear(in_d, hid), nn.ReLU()]
    for _ in range(layers - 1):
        mods += [nn.Linear(hid, hid), nn.ReLU()]
    mods.append(nn.Linear(hid, out_d))
    if ln:
        mods.append(nn.LayerNorm(out_d))
    return nn.Sequential(*mods)


# ---------------------------------------------------------------------------
#  Graph-based divergence and Laplacian operators
# ---------------------------------------------------------------------------

def compute_divergence(
    U:            torch.Tensor,    # (N, dim)  velocity at each cell
    edge_index:   torch.Tensor,    # (2, E)
    face_normals: torch.Tensor,    # (E, dim)  outward unit normal per face
    face_areas:   torch.Tensor,    # (E,)      face area / length
    cell_volumes: torch.Tensor,    # (N,)      cell volume / area
) -> torch.Tensor:
    """Compute cell-centred divergence using face-centred fluxes.

    D_i = (1/A_i) sum_{faces f of cell i} U_f . n_f . L_f
    where U_f = (U_i + U_j)/2  (linear face interpolation).

    Returns: (N,) divergence at each cell.
    """
    src, dst = edge_index
    U_face = 0.5 * (U[src] + U[dst])                  # (E, dim)
    flux   = (U_face * face_normals).sum(dim=-1)       # (E,)
    flux   = flux * face_areas                         # (E,)

    N   = U.shape[0]
    div = flux.new_zeros(N).scatter_add(0, dst, flux)
    return div / cell_volumes.clamp(min=1e-12)


def build_laplacian_rhs(
    div:          torch.Tensor,    # (N,)
    edge_index:   torch.Tensor,    # (2, E)
    face_areas:   torch.Tensor,    # (E,)
    face_dists:   torch.Tensor,    # (E,)  |x_i - x_j|
    cell_volumes: torch.Tensor,    # (N,)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the weighted Laplacian matrix coefficients.

    Returns:
        diag: (N,)   diagonal entries of L
        off:  (E,)   off-diagonal entries (one per edge)
        rhs:  (N,)   right-hand side = divergence
    """
    w = face_areas / face_dists.clamp(min=1e-8)   # (E,)
    N = div.shape[0]
    src, dst = edge_index

    diag = w.new_zeros(N).scatter_add(0, dst, w)
    return diag, -w, div


def cg_solve(
    diag:       torch.Tensor,   # (N,)
    off:        torch.Tensor,   # (E,)
    ei:         torch.Tensor,   # (2, E)
    b:          torch.Tensor,   # (N,)  RHS
    max_iter:   int   = 20,
    tol:        float = 1e-4,
    use_jacobi: bool  = True,
) -> Tuple[torch.Tensor, List[float]]:
    """Preconditioned Conjugate Gradient solver for L phi = b.

    Improvement #11: Jacobi (diagonal) preconditioner.
    When use_jacobi=True, applies M^{-1} = diag(L)^{-1} to accelerate
    convergence for ill-conditioned graph Laplacians.

    The system matrix is:
        (L phi)_i = diag_i * phi_i + sum_{j in N(i)} off_{ij} * phi_j

    Returns:
        phi        : (N,)   pressure correction field
        residuals  : list of ||r_k||_2 per iteration (for logging)
    """
    src, dst = ei
    N = b.shape[0]

    def matvec(phi: torch.Tensor) -> torch.Tensor:
        """Apply graph Laplacian: y = L @ phi  (non-inplace for autograd)."""
        y   = diag * phi
        msg = off * phi[src]
        agg = msg.new_zeros(N).scatter_add(0, dst, msg)
        return y + agg

    # Jacobi preconditioner: M_inv = 1 / diag (clamped for stability)
    if use_jacobi:
        M_inv = 1.0 / diag.clamp(min=1e-8)
    else:
        M_inv = torch.ones_like(diag)

    phi  = torch.zeros_like(b)
    r    = b - matvec(phi)            # initial residual
    z    = M_inv * r                  # preconditioned residual
    p    = z.clone()
    rz   = (r * z).sum()

    residuals: List[float] = []

    for _ in range(max_iter):
        r_norm = r.norm().item()
        residuals.append(r_norm)
        if r_norm < tol:
            break

        Ap    = matvec(p)
        alpha = rz / ((p * Ap).sum() + 1e-12)
        phi   = phi + alpha * p
        r     = r   - alpha * Ap
        z     = M_inv * r
        rz_new = (r * z).sum()
        beta  = rz_new / (rz + 1e-12)
        p     = z + beta * p
        rz    = rz_new

    return phi, residuals


def apply_pressure_correction(
    U_hat:        torch.Tensor,    # (N, dim)  raw GNN velocity
    phi:          torch.Tensor,    # (N,)      pressure correction
    edge_index:   torch.Tensor,    # (2, E)
    face_normals: torch.Tensor,    # (E, dim)
    face_areas:   torch.Tensor,    # (E,)
    face_dists:   torch.Tensor,    # (E,)
    cell_volumes: torch.Tensor,    # (N,)
) -> torch.Tensor:
    """Apply grad(phi) correction: U = U_hat - (1/V_i) sum phi_j n_ij A_ij / |Dx|.

    Returns: (N, dim) divergence-free velocity.
    """
    src, dst = edge_index
    w = face_areas / face_dists.clamp(min=1e-8)          # (E,)
    grad_phi_term = w.unsqueeze(-1) * phi[src].unsqueeze(-1) * face_normals
    N   = U_hat.shape[0]
    idx = dst.unsqueeze(-1).expand(-1, U_hat.shape[1])
    grad_phi = (grad_phi_term.new_zeros(U_hat.shape)
                .scatter_add(0, idx, grad_phi_term))
    grad_phi = grad_phi / cell_volumes.unsqueeze(-1).clamp(min=1e-12)
    return U_hat - grad_phi


# ---------------------------------------------------------------------------
#  Master pressure projection module
# ---------------------------------------------------------------------------

class PressureProjection(nn.Module):
    """Enforce incompressibility (div(U) = 0) after GNN velocity prediction.

    Improvement #11:
        - Jacobi preconditioned CG when cfg.use_jacobi_precond = True
        - cg_residuals stored as self.last_cg_residuals for logging
        - Separate CG iteration budgets per training stage

    Usage:
        proj = PressureProjection(cfg)
        U_div_free, phi = proj(U_hat, edge_index, ...)
        print(proj.last_cg_residuals)   # list of residual norms
    """

    def __init__(self, cfg: CfdGNNConfig):
        super().__init__()
        self.cfg = cfg
        self.last_cg_residuals: List[float] = []
        self._cg_iters = cfg.pressure_cg_iters   # overridden by trainer per stage

        if cfg.learned_pressure:
            self.phi_head = _mlp(
                cfg.fluid_hidden, cfg.pressure_hidden, 1,
                cfg.fluid_mlp_layers, ln=False
            )

    def set_cg_iters(self, n: int) -> None:
        """Called by the trainer to adjust iteration budget per stage."""
        self._cg_iters = n

    def forward(
        self,
        U_hat:        torch.Tensor,   # (N, dim)
        edge_index:   torch.Tensor,   # (2, E)
        face_normals: torch.Tensor,   # (E, dim)
        face_areas:   torch.Tensor,   # (E,)
        face_dists:   torch.Tensor,   # (E,)
        cell_volumes: torch.Tensor,   # (N,)
        latent:       Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project velocity to divergence-free subspace.

        Returns:
            U:    (N, dim)  corrected velocity field
            phi:  (N,)      pressure correction field
        """
        if self.cfg.learned_pressure and latent is not None:
            phi = self.phi_head(latent).squeeze(-1)
            self.last_cg_residuals = []
        else:
            div  = compute_divergence(U_hat, edge_index, face_normals,
                                       face_areas, cell_volumes)
            diag, off, rhs = build_laplacian_rhs(
                div, edge_index, face_areas, face_dists, cell_volumes
            )
            # Improvement #11: preconditioned CG with convergence logging
            phi, residuals = cg_solve(
                diag, off, edge_index, rhs,
                max_iter=self._cg_iters,
                tol=self.cfg.pressure_cg_tol,
                use_jacobi=self.cfg.use_jacobi_precond,
            )
            self.last_cg_residuals = residuals

        U_corrected = apply_pressure_correction(
            U_hat, phi, edge_index, face_normals, face_areas,
            face_dists, cell_volumes
        )
        return U_corrected, phi

    def compute_div_residual(
        self,
        U:            torch.Tensor,
        edge_index:   torch.Tensor,
        face_normals: torch.Tensor,
        face_areas:   torch.Tensor,
        cell_volumes: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ||div(U)||_2 for use as continuity residual loss."""
        div = compute_divergence(U, edge_index, face_normals,
                                  face_areas, cell_volumes)
        return (div ** 2).mean()
