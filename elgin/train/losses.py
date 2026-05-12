"""PDE residual loss functions for the ELGIN framework.

"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
#  Field prediction MSE (supervised, normalised)
# ---------------------------------------------------------------------------

def fluid_mse_loss(
    pred:   torch.Tensor,   # (N_cells, 5)
    target: torch.Tensor,   # (N_cells, 5)
    weights: Optional[torch.Tensor] = None,  # (5,) per-variable weights
    fluid_mean: Optional[torch.Tensor] = None,  # (5,)
    fluid_std:  Optional[torch.Tensor] = None,  # (5,)
) -> torch.Tensor:
    """MSE on the (optionally) normalised fluid state.

    When ``fluid_mean`` / ``fluid_std`` are supplied the error is computed
    on (q - mean) / std so each channel contributes on a common scale.
    Without normalisation the absolute pressure scale (~10^5 Pa) would
    dominate the loss and effectively switch off learning of velocity / k /
    omega.
    """
    if fluid_std is not None:
        std = fluid_std.view(1, -1).clamp(min=1e-8)
        if fluid_mean is not None:
            mean = fluid_mean.view(1, -1)
            pred   = (pred   - mean) / std
            target = (target - mean) / std
        else:
            pred   = pred   / std
            target = target / std
    err = (pred - target) ** 2
    if weights is not None:
        err = err * weights.view(1, -1)
    return err.mean()


def particle_mse_loss(
    pred_pos:   torch.Tensor,   # (N_part, dim)
    target_pos: torch.Tensor,   # (N_part, dim)
    L_ref:      float = 4.0,
) -> torch.Tensor:
    """Normalised MSE on particle positions."""
    return ((pred_pos - target_pos) ** 2).mean() / (L_ref ** 2)


# ---------------------------------------------------------------------------
#  Improvement #10 — Angular momentum conservation loss
# ---------------------------------------------------------------------------

def angular_momentum_loss(
    pred_pos:   torch.Tensor,   # (N_part, dim)  predicted positions
    tgt_pos:    torch.Tensor,   # (N_part, dim)  target positions
    pred_vel:   torch.Tensor,   # (N_part, dim)  predicted velocities
    tgt_vel:    torch.Tensor,   # (N_part, dim)  target velocities
    d_p:        torch.Tensor,   # (N_part,)      particle diameter [m]
    rho_p:      torch.Tensor,   # (N_part,)      particle density [kg/m^3]
    centroid:   Optional[torch.Tensor] = None,  # (dim,) domain centre
    L_ref:      float = 4.0,
    U_ref:      float = 20.0,
) -> torch.Tensor:
    """Angular momentum conservation penalty.

    Computes the squared difference in bulk angular momentum between
    predicted and target particle clouds.

    For 2-D flow the z-component of angular momentum is:
        L_z = sum_i m_i * (r_i x v_i)_z
             = sum_i m_i * (dx_i * vy_i - dy_i * vx_i)

    where r_i = x_i - centroid, and m_i = rho_p * pi/6 * d_p^3.

    The loss is:
        L_ang = (L_z_pred - L_z_tgt)^2 / (N * (m_mean * L_ref * U_ref)^2)

    Normalisation makes the loss scale-independent and comparable across
    different particle sizes and densities.

    Args:
        pred_pos, tgt_pos : (N_part, 2)  particle positions
        pred_vel, tgt_vel : (N_part, 2)  particle velocities (or accelerations)
        d_p   : (N_part,)  particle diameter
        rho_p : (N_part,)  particle density
        centroid : (2,)    domain centroid for computing moment arm

    Returns:
        scalar loss
    """
    import math
    N = pred_pos.shape[0]
    if N == 0:
        return pred_pos.new_zeros(())

    # Particle mass (spherical)
    m = rho_p * (math.pi / 6.0) * d_p ** 3           # (N_part,)

    if centroid is None:
        centroid = torch.tensor([L_ref / 2.0, L_ref / 2.0],
                                dtype=pred_pos.dtype, device=pred_pos.device)

    # Moment arms relative to domain centroid
    r_pred = pred_pos - centroid.unsqueeze(0)         # (N, 2)
    r_tgt  = tgt_pos  - centroid.unsqueeze(0)         # (N, 2)

    # z-component of (r x v):  r_x*v_y - r_y*v_x
    Lz_pred = (m * (r_pred[:, 0] * pred_vel[:, 1]
                    - r_pred[:, 1] * pred_vel[:, 0])).sum()
    Lz_tgt  = (m * (r_tgt[:, 0]  * tgt_vel[:, 1]
                    - r_tgt[:, 1]  * tgt_vel[:, 0])).sum()

    # Normalisation: typical angular momentum scale
    m_mean  = m.mean().detach().clamp(min=1e-20)
    scale   = (N * (m_mean * L_ref * U_ref) ** 2).clamp(min=1e-30)

    return (Lz_pred - Lz_tgt) ** 2 / scale


# ---------------------------------------------------------------------------
#  Continuity residual  ||div(U)||^2
# ---------------------------------------------------------------------------

def continuity_loss(
    U:            torch.Tensor,   # (N_cells, dim)
    edge_index:   torch.Tensor,   # (2, E)
    face_normals: torch.Tensor,   # (E, dim)
    face_areas:   torch.Tensor,   # (E,)
    cell_volumes: torch.Tensor,   # (N_cells,)
) -> torch.Tensor:
    """Compute ||div(U)||^2 — should be zero for incompressible flow."""
    src, dst = edge_index
    U_face = 0.5 * (U[src] + U[dst])
    flux   = (U_face * face_normals).sum(dim=-1) * face_areas

    N   = U.shape[0]
    div = flux.new_zeros(N).scatter_add(0, dst, flux)
    div = div / cell_volumes.clamp(min=1e-12)

    return (div ** 2).mean()


# ---------------------------------------------------------------------------
#  Momentum equation residual
# ---------------------------------------------------------------------------

def _graph_gradient(
    scalar:       torch.Tensor,
    edge_index:   torch.Tensor,
    face_normals: torch.Tensor,
    face_areas:   torch.Tensor,
    face_dists:   torch.Tensor,
    cell_volumes: torch.Tensor,
) -> torch.Tensor:
    """Gauss-theorem graph gradient of a scalar field. Returns (N, dim)."""
    src, dst = edge_index
    phi_face = 0.5 * (scalar[src] + scalar[dst])
    flux_vec = phi_face.unsqueeze(-1) * face_normals * face_areas.unsqueeze(-1)

    N   = scalar.shape[0]
    dim = face_normals.shape[-1]
    idx  = dst.unsqueeze(-1).expand(-1, dim)
    grad = flux_vec.new_zeros(N, dim).scatter_add(0, idx, flux_vec)
    return grad / cell_volumes.unsqueeze(-1).clamp(min=1e-12)


def _graph_laplacian_scalar(
    scalar:       torch.Tensor,
    edge_index:   torch.Tensor,
    face_areas:   torch.Tensor,
    face_dists:   torch.Tensor,
    cell_volumes: torch.Tensor,
) -> torch.Tensor:
    """FV diffusion Laplacian of a scalar field. Returns (N,)."""
    src, dst = edge_index
    w    = face_areas / face_dists.clamp(min=1e-8)
    diff = scalar[src] - scalar[dst]
    flux = w * diff

    N   = scalar.shape[0]
    lap = flux.new_zeros(N).scatter_add(0, dst, flux)
    return lap / cell_volumes.clamp(min=1e-12)


def momentum_residual_loss(
    U_pred:       torch.Tensor,
    U_prev:       torch.Tensor,
    p_pred:       torch.Tensor,
    nu_t:         torch.Tensor,
    edge_index:   torch.Tensor,
    face_normals: torch.Tensor,
    face_areas:   torch.Tensor,
    face_dists:   torch.Tensor,
    cell_volumes: torch.Tensor,
    dt:    float = 0.1,
    nu:    float = 1.5e-5,
    U_ref: float = 20.0,
    L_ref: float = 4.0,
) -> torch.Tensor:
    """RANS momentum residual: R = dU/dt + div(UU) + grad(p) - div[(nu+nu_t)grad U].

    Normalised by (U_ref/L_ref)^2 so the loss is O(1) for a well-trained model,
    making lambda_momentum directly comparable to the normalised MSE terms.
    """
    dim = U_pred.shape[-1]
    src, dst = edge_index

    dU_dt  = (U_pred - U_prev) / dt
    grad_p = _graph_gradient(p_pred, edge_index, face_normals,
                              face_areas, face_dists, cell_volumes)

    nu_eff    = nu + nu_t
    diff_cols = [
        _graph_laplacian_scalar(nu_eff * U_pred[:, d],
                                edge_index, face_areas, face_dists,
                                cell_volumes).unsqueeze(-1)
        for d in range(dim)
    ]
    diff_term = torch.cat(diff_cols, dim=-1)

    # Symmetric face-average convection: div(U⊗U) accumulated to owner cell
    U_face    = 0.5 * (U_pred[src] + U_pred[dst])
    flux_U    = (U_face * face_normals).sum(-1)          # normal flux at face
    contrib   = (flux_U * face_areas).unsqueeze(-1) * U_face  # use face vel, not src
    idx_nd    = dst.unsqueeze(-1).expand_as(contrib)
    conv_term = contrib.new_zeros(U_pred.shape).scatter_add(0, idx_nd, contrib)
    conv_term = conv_term / cell_volumes.unsqueeze(-1).clamp(1e-12)

    R          = dU_dt + conv_term + grad_p - diff_term
    scale_sq   = (U_ref / L_ref) ** 2          # (m/s / m)^2 = s^-2  reference shear rate²
    return (R ** 2).mean() / scale_sq


# ---------------------------------------------------------------------------
#  Turbulence equation residuals (k and omega)
# ---------------------------------------------------------------------------

def turbulence_residual_loss(
    k_pred:       torch.Tensor,
    k_prev:       torch.Tensor,
    omega_pred:   torch.Tensor,
    omega_prev:   torch.Tensor,
    U_pred:       torch.Tensor,
    nu_t:         torch.Tensor,
    S_mag:        Optional[torch.Tensor],
    edge_index:   torch.Tensor,
    face_normals: torch.Tensor,
    face_areas:   torch.Tensor,
    face_dists:   torch.Tensor,
    cell_volumes: torch.Tensor,
    dt:        float = 0.01,
    nu:        float = 1.5e-5,
    beta_star: float = 0.09,
    beta1:     float = 0.075,
    sigma_k:   float = 0.85,
    sigma_om:  float = 0.5,
    alpha1:    float = 5.0 / 9.0,
) -> torch.Tensor:
    """Residuals of k and omega transport equations (k-omega SST)."""
    src, dst = edge_index

    if S_mag is None:
        S_mag = torch.zeros_like(k_pred)

    P_k = (nu_t * S_mag**2).clamp(max=10 * beta_star * k_pred * omega_pred)

    def _transport_res(phi_new, phi_old, source_plus, source_minus, D_coeff):
        dphidt   = (phi_new - phi_old) / dt
        phi_face = 0.5 * (phi_new[src] + phi_new[dst])
        U_face   = 0.5 * (U_pred[src]  + U_pred[dst])
        flux_n   = (U_face * face_normals).sum(-1)
        src_flux = flux_n * phi_face * face_areas
        conv     = src_flux.new_zeros(phi_new.shape[0]).scatter_add(0, dst, src_flux)
        conv     = conv / cell_volumes.clamp(1e-12)
        diff_phi = _graph_laplacian_scalar(
            D_coeff * phi_new, edge_index, face_areas, face_dists, cell_volumes
        )
        return dphidt + conv + source_minus * phi_new - source_plus - diff_phi

    R_k = _transport_res(
        k_pred, k_prev,
        source_plus  = P_k,
        source_minus = beta_star * omega_pred,
        D_coeff      = nu + sigma_k * nu_t,
    )

    k_safe = k_pred.clamp(min=1e-8)
    R_om   = _transport_res(
        omega_pred, omega_prev,
        source_plus  = alpha1 * omega_pred / k_safe * P_k,
        source_minus = beta1 * omega_pred,
        D_coeff      = nu + sigma_om * nu_t,
    )

    return (R_k**2).mean() + (R_om**2).mean()


# ---------------------------------------------------------------------------
#  Combined loss
# ---------------------------------------------------------------------------

def total_loss(
    fluid_pred:   torch.Tensor,
    fluid_tgt:    torch.Tensor,
    part_pred:    torch.Tensor,
    part_tgt:     torch.Tensor,
    nu_t:         torch.Tensor,
    edge_index:   torch.Tensor,
    face_normals: torch.Tensor,
    face_areas:   torch.Tensor,
    face_dists:   torch.Tensor,
    cell_volumes: torch.Tensor,
    cfg,
    compute_pde_losses: bool = False,
    d_p:          Optional[torch.Tensor] = None,   # for angular loss
    rho_p:        Optional[torch.Tensor] = None,   # for angular loss
    kl_loss:      Optional[torch.Tensor] = None,   # S3: from stochastic decoder
    fluid_norm_mean: Optional[torch.Tensor] = None,  # (5,)
    fluid_norm_std:  Optional[torch.Tensor] = None,  # (5,)
) -> dict:
    """Compute all loss terms and return dict with individual values.

    Improvement #10: angular momentum conservation loss is included when
    compute_pde_losses=True and d_p/rho_p are provided.

    Args:
        compute_pde_losses: If False, only compute MSE (stages 1-2).

    Returns:
        dict with keys: total, mse_fluid, mse_particle,
                        continuity, momentum, turbulence, angular.
    """
    losses = {}

    fluid_mean_t = fluid_norm_mean
    fluid_std_t  = fluid_norm_std
    if fluid_mean_t is None and hasattr(cfg, "fluid_mean"):
        fluid_mean_t = torch.tensor(cfg.fluid_mean, dtype=fluid_pred.dtype,
                                    device=fluid_pred.device)
        fluid_std_t  = torch.tensor(cfg.fluid_std,  dtype=fluid_pred.dtype,
                                    device=fluid_pred.device)

    losses["mse_fluid"]    = fluid_mse_loss(
        fluid_pred, fluid_tgt,
        fluid_mean=fluid_mean_t, fluid_std=fluid_std_t,
    )
    losses["mse_particle"] = particle_mse_loss(part_pred, part_tgt, cfg.L_ref)

    losses["continuity"] = torch.tensor(0.0, device=fluid_pred.device)
    losses["momentum"]   = torch.tensor(0.0, device=fluid_pred.device)
    losses["turbulence"] = torch.tensor(0.0, device=fluid_pred.device)
    losses["angular"]    = torch.tensor(0.0, device=fluid_pred.device)
    losses["kl"]         = torch.tensor(0.0, device=fluid_pred.device)

    if compute_pde_losses:
        U_pred  = fluid_pred[:, :cfg.dim]
        p_pred  = fluid_pred[:, 2]
        k_pred  = fluid_pred[:, 3]
        om_pred = fluid_pred[:, 4]
        U_prev  = fluid_tgt[:, :cfg.dim]

        losses["continuity"] = continuity_loss(
            U_pred, edge_index, face_normals, face_areas, cell_volumes
        ) * cfg.lambda_continuity

        losses["momentum"] = momentum_residual_loss(
            U_pred, U_prev, p_pred, nu_t,
            edge_index, face_normals, face_areas, face_dists, cell_volumes,
            dt=cfg.dt, nu=cfg.nu,
            U_ref=cfg.U_ref, L_ref=cfg.L_ref,
        ) * cfg.lambda_momentum

        losses["turbulence"] = turbulence_residual_loss(
            k_pred, fluid_tgt[:, 3], om_pred, fluid_tgt[:, 4],
            U_pred, nu_t, None,
            edge_index, face_normals, face_areas, face_dists, cell_volumes,
            dt=cfg.dt, nu=cfg.nu,
        ) * cfg.lambda_turbulence

        # Improvement #10: angular momentum conservation
        if (d_p is not None and rho_p is not None
                and part_pred.shape[0] > 0
                and hasattr(cfg, "lambda_angular")
                and cfg.lambda_angular > 0.0):
            # Estimate velocity as displacement over one step
            pred_vel = part_pred - part_tgt    # approximate dv
            tgt_vel  = torch.zeros_like(part_tgt)
            losses["angular"] = angular_momentum_loss(
                part_pred, part_tgt,
                pred_vel, tgt_vel,
                d_p, rho_p,
                L_ref=cfg.L_ref, U_ref=cfg.U_ref,
            ) * cfg.lambda_angular

    # S3: KL divergence from stochastic decoder
    if kl_loss is not None and hasattr(cfg, "lambda_kl") and cfg.lambda_kl > 0.0:
        losses["kl"] = kl_loss * cfg.lambda_kl

    losses["total"] = (
        losses["mse_fluid"]
        + cfg.lambda_particle * losses["mse_particle"]
        + losses["continuity"]
        + losses["momentum"]
        + losses["turbulence"]
        + losses["angular"]
        + losses["kl"]
    )
    return losses
