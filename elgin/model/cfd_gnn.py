"""CfdGNN — master model: Eulerian + Lagrangian + cross-graph.
"""

from __future__ import annotations
import dataclasses
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .config              import CfdGNNConfig
from .eulerian_graph      import EulerianGNN
from .lagrangian_graph    import LagrangianGNN
from .physics             import (evaporation_diameter, stokes_drag_acc,
                                   gravity_vector)
from .pressure_projection import PressureProjection
from .turbulence_closure  import TurbulenceClosure


# ---------------------------------------------------------------------------
#  Cross-graph interpolation (fluid -> particle)
# ---------------------------------------------------------------------------

def interpolate_fluid_to_particles(
    fluid_field:  torch.Tensor,   # (N_cells, C)
    cell_pos:     torch.Tensor,   # (N_cells, dim)
    particle_pos: torch.Tensor,   # (N_part, dim)
    k_nearest:    int = 4,
    chunk_size:   int = 2048,
) -> torch.Tensor:
    """Inverse-distance-weighted interpolation of fluid fields to particles.

    Particles are processed in chunks to avoid building the full
    (N_part × N_cells) distance matrix in one shot, which can exceed GPU
    memory when N_part > ~5 000 and N_cells > ~7 000.
    """
    P, C = particle_pos.shape[0], fluid_field.shape[-1]
    k    = min(k_nearest, cell_pos.shape[0])
    out  = torch.empty(P, C, device=fluid_field.device, dtype=fluid_field.dtype)

    for start in range(0, P, chunk_size):
        end   = min(start + chunk_size, P)
        p_chunk = particle_pos[start:end]          # (chunk, dim)

        # (chunk, N_cells, dim)  — only 2048 × 7704 × 2 floats at most
        diff   = p_chunk.unsqueeze(1) - cell_pos.unsqueeze(0)
        dist2  = (diff ** 2).sum(dim=-1)           # (chunk, N_cells)

        _, idx      = torch.topk(dist2, k=k, dim=-1, largest=False)
        near_dist2  = dist2.gather(1, idx)         # (chunk, k)
        near_vals   = fluid_field[idx.reshape(-1)].reshape(end - start, k, C)

        w = 1.0 / (near_dist2.sqrt() + 1e-8)
        w = w / w.sum(dim=-1, keepdim=True)        # (chunk, k)
        out[start:end] = (w.unsqueeze(-1) * near_vals).sum(dim=1)

    return out


# ---------------------------------------------------------------------------
#  S7 — Two-way coupling: particle-to-fluid momentum source term
# ---------------------------------------------------------------------------

def compute_particle_source(
    particle_pos:  torch.Tensor,   # (N_part, dim)
    d_p:           torch.Tensor,   # (N_part,)
    rho_p:         torch.Tensor,   # (N_part,)
    v_p:           torch.Tensor,   # (N_part, dim)  particle velocity
    u_fluid_at_p:  torch.Tensor,   # (N_part, dim)  fluid vel at particle
    cell_pos:      torch.Tensor,   # (N_cells, dim)
    cell_volumes:  torch.Tensor,   # (N_cells,)
    k_nearest:     int = 1,
) -> torch.Tensor:
    """Compute Eulerian momentum source term from Lagrangian parcels.

    For each cell i:
        S_i = -(1/V_i) * sum_{p: nearest(p)=i} F_drag^(p)

    where F_drag^(p) = m_p * a_drag^(p) is the drag force on parcel p.
    The negative sign comes from Newton's 3rd law: the force the fluid
    exerts on the particle is equal and opposite to the force the particle
    exerts on the fluid.

    Returns: (N_cells, dim)  momentum source [m/s^2]
    """
    import math
    m_p    = rho_p * (math.pi / 6.0) * d_p ** 3        # (N_part,)
    a_drag = stokes_drag_acc(v_p, u_fluid_at_p, d_p, rho_p)
    F_drag = m_p.unsqueeze(-1) * a_drag                 # (N_part, dim)

    # Map each particle to its nearest cell (chunked to avoid OOM)
    _chunk = 2048
    nearest_cell = torch.empty(particle_pos.shape[0], dtype=torch.long,
                                device=particle_pos.device)
    for _s in range(0, particle_pos.shape[0], _chunk):
        _e  = min(_s + _chunk, particle_pos.shape[0])
        _d2 = ((particle_pos[_s:_e].unsqueeze(1) - cell_pos.unsqueeze(0)) ** 2).sum(-1)
        nearest_cell[_s:_e] = _d2.argmin(dim=-1)

    N_cells = cell_pos.shape[0]
    dim     = F_drag.shape[-1]
    source  = torch.zeros(N_cells, dim, dtype=F_drag.dtype,
                          device=F_drag.device)
    idx     = nearest_cell.unsqueeze(-1).expand_as(F_drag)
    source  = source.scatter_add(0, idx, -F_drag)       # negative sign
    source  = source / cell_volumes.unsqueeze(-1).clamp(min=1e-12)
    return source


# ---------------------------------------------------------------------------
#  CfdGNN
# ---------------------------------------------------------------------------

class CfdGNN(nn.Module):
    """Full CFD surrogate: fluid field + particle transport on graphs.
    """

    def __init__(self, cfg: CfdGNNConfig):
        super().__init__()
        self.cfg            = cfg
        self.turb_closure   = TurbulenceClosure(cfg)
        self.eulerian_gnn   = EulerianGNN(cfg)
        self.pressure_proj  = PressureProjection(cfg)
        self.lagrangian_gnn = LagrangianGNN(cfg)

    def step(
        self,
        fluid_field:    torch.Tensor,    # (N_cells, 5)
        cell_pos:       torch.Tensor,
        bc_type:        torch.Tensor,
        bc_values:      torch.Tensor,
        edge_index:     torch.Tensor,
        face_normals:   torch.Tensor,
        face_areas:     torch.Tensor,
        face_dists:     torch.Tensor,
        cell_volumes:   torch.Tensor,
        d_wall:         torch.Tensor,
        particle_hist:  torch.Tensor,    # (N_part, H+1, dim)
        particle_type:  torch.Tensor,
        d_p:            torch.Tensor,    # (N_part,) current diameter
        d_p0:           torch.Tensor,    # (N_part,) initial diameter (for S2)
        rho_p:          torch.Tensor,
        S_mag:          Optional[torch.Tensor] = None,
        Omega_mag:      Optional[torch.Tensor] = None,
        face_type:      Optional[torch.Tensor] = None,
        du_dy:          Optional[torch.Tensor] = None,  # (N_part,) for S6
        particle_age:   Optional[torch.Tensor] = None,  # (N_part,) seconds for S2
        clip_particles: bool = True,
        inlet_cond:     Optional[torch.Tensor] = None,  # (dim,) per-case U_in
        wall_normal:    Optional[torch.Tensor] = None,  # (N_cells, dim)
        domain_bounds:  Optional[torch.Tensor] = None,  # (dim, 2) override cfg
        bypass_eulerian: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Execute one simulation timestep.

        Returns dict with keys:
            fluid_field   (N_cells, 5)
            particle_pos  (N_part, dim)
            d_p_new       (N_part,)   updated diameters after evaporation
            nu_t          (N_cells,)
            phi           (N_cells,)
            div_residual  scalar
        """
        # ── Step 1: Turbulence closure ────────────────────────────────────────
        k_field = fluid_field[:, 3]
        omega   = fluid_field[:, 4]
        nu_t    = self.turb_closure(k_field, omega, d_wall, S_mag, Omega_mag)

        if bypass_eulerian:
           new_fluid  = fluid_field
            U_div_free = fluid_field[:, :self.cfg.dim]
            phi        = fluid_field.new_zeros(fluid_field.shape[0])
            div_res    = fluid_field.new_zeros(())
        else:
            # ── Step 2: Eulerian GNN ──────────────────────────────────────────
            new_fluid = self.eulerian_gnn.next_field(
                fluid_field, cell_pos, bc_type, edge_index,
                face_normals, face_areas, bc_values, nu_t,
                face_type=face_type,
                inlet_cond=inlet_cond,
            )

            # ── Step 3: Pressure projection ───────────────────────────────────
            U_raw = new_fluid[:, :self.cfg.dim]
            U_div_free, phi = self.pressure_proj(
                U_raw, edge_index, face_normals, face_areas,
                face_dists, cell_volumes
            )
            new_fluid = torch.cat([U_div_free, new_fluid[:, self.cfg.dim:]], dim=-1)

            div_res = self.pressure_proj.compute_div_residual(
                U_div_free, edge_index, face_normals, face_areas, cell_volumes
            )

        # ── Step 4: Fluid-to-particle cross-graph interpolation ───────────────
        particle_cur = particle_hist[:, -1]

        u_fluid_at_p = interpolate_fluid_to_particles(
            U_div_free, cell_pos, particle_cur,
            k_nearest=self.cfg.cross_k_nearest,
        )

        k_at_p: Optional[torch.Tensor] = None
        if self.cfg.use_tke_dispersion:
            k_at_p = interpolate_fluid_to_particles(
                new_fluid[:, 3:4], cell_pos, particle_cur,
                k_nearest=self.cfg.cross_k_nearest,
            ).squeeze(-1)

        # ── Step 5 (S2): Update particle diameter via evaporation ────────────
        d_p_new = d_p
        if self.cfg.use_evaporation and particle_age is not None:
            d_p_new = evaporation_diameter(
                d_p0=d_p0,
                age=particle_age,
                rho_p=rho_p,
                D_v=self.cfg.evap_D_v,
                B_M=self.cfg.evap_B_M,
            )

        # ── Step 6 (S7): Two-way coupling momentum source ─────────────────────
        if self.cfg.use_two_way_coupling:
            v_p = particle_hist[:, -1] - particle_hist[:, -2]
            src = compute_particle_source(
                particle_cur, d_p_new, rho_p, v_p,
                u_fluid_at_p, cell_pos, cell_volumes,
                k_nearest=self.cfg.cross_k_nearest,
            )
            # Add source term to velocity channels only
            src_field = torch.cat([
                src * self.cfg.dt,
                torch.zeros(new_fluid.shape[0],
                            new_fluid.shape[1] - self.cfg.dim,
                            dtype=src.dtype, device=src.device),
            ], dim=-1)
            new_fluid = new_fluid + src_field

        # ── Step 6b: Wall features at particle positions (S-wall) ─────────────
        wall_feat_at_p: Optional[torch.Tensor] = None
        if getattr(self.cfg, "use_wall_features_lag", False) and wall_normal is not None:
            d_at_p = interpolate_fluid_to_particles(
                d_wall.unsqueeze(-1), cell_pos, particle_cur,
                k_nearest=self.cfg.cross_k_nearest,
            )
            n_at_p = interpolate_fluid_to_particles(
                wall_normal, cell_pos, particle_cur,
                k_nearest=self.cfg.cross_k_nearest,
            )
            wall_feat_at_p = torch.cat([d_at_p, n_at_p], dim=-1)

        # ── Step 7: Particle transport ────────────────────────────────────────
        g_vec = gravity_vector(g=self.cfg.g, dim=self.cfg.dim,
                                dtype=particle_hist.dtype,
                                device=particle_hist.device)
        particle_next = self.lagrangian_gnn.next_position(
            particle_hist, particle_type,
            u_fluid_at_p, d_p_new, d_p0, rho_p,
            g_vec=g_vec, k_fluid=k_at_p, du_dy=du_dy,
            wall_feat_p=wall_feat_at_p,
        )

       if clip_particles:
            if domain_bounds is not None:
                bounds = domain_bounds
            else:
                bounds = torch.tensor(self.cfg.domain_bounds,
                                      dtype=particle_next.dtype,
                                      device=particle_next.device)
            lo = bounds[:, 0].unsqueeze(0)  # (1, dim)
            hi = bounds[:, 1].unsqueeze(0)  # (1, dim)
            # Reflect off lower walls
            mask_lo = particle_next < lo
            particle_next = torch.where(mask_lo, 2.0 * lo - particle_next, particle_next)
            # Reflect off upper walls
            mask_hi = particle_next > hi
            particle_next = torch.where(mask_hi, 2.0 * hi - particle_next, particle_next)
            # Failsafe hard clamp for extreme overshoots (> domain width)
            particle_next = particle_next.clamp(min=bounds[:, 0], max=bounds[:, 1])

        return {
            "fluid_field":  new_fluid,
            "particle_pos": particle_next,
            "d_p_new":      d_p_new,
            "nu_t":         nu_t,
            "phi":          phi,
            "div_residual": div_res,
        }

    # ── Autoregressive rollout ────────────────────────────────────────────────

    @torch.no_grad()
    def rollout(
        self,
        fluid_field0:   torch.Tensor,
        cell_pos:       torch.Tensor,
        bc_type:        torch.Tensor,
        bc_values:      torch.Tensor,
        edge_index:     torch.Tensor,
        face_normals:   torch.Tensor,
        face_areas:     torch.Tensor,
        face_dists:     torch.Tensor,
        cell_volumes:   torch.Tensor,
        d_wall:         torch.Tensor,
        particle_hist0: torch.Tensor,
        particle_type:  torch.Tensor,
        d_p:            torch.Tensor,
        rho_p:          torch.Tensor,
        n_steps:        int = 100,
        face_type:      Optional[torch.Tensor] = None,
        inlet_cond:     Optional[torch.Tensor] = None,
        wall_normal:    Optional[torch.Tensor] = None,
        domain_bounds:  Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Autoregressive rollout with deposition tracking and evaporation.

        Returns:
            fluid_traj      (T, N_cells, 5)
            particle_traj   (T, N_part, dim)
            deposition_step (N_part,) — step at which deposited (-1=airborne)
            deposition_pos  (N_part, dim)
            dp_traj         (T, N_part) — diameter evolution (S2)
        """
        self.eval()
        fluid   = fluid_field0.clone()
        p_hist  = particle_hist0.clone()
        d_p0    = d_p.clone()          # initial diameters for evaporation
        d_p_cur = d_p.clone()
        device  = fluid.device
        N_part  = p_hist.shape[0]
        dim     = p_hist.shape[-1]

        # Deposition tracking
        deposited = torch.zeros(N_part, dtype=torch.bool,   device=device)
        depo_step = torch.full((N_part,), -1, dtype=torch.long, device=device)
        depo_pos  = torch.zeros(N_part, dim, dtype=p_hist.dtype, device=device)

        if domain_bounds is not None:
            bounds = domain_bounds.to(device=device, dtype=p_hist.dtype)
        else:
            bounds = torch.tensor(self.cfg.domain_bounds,
                                  dtype=p_hist.dtype, device=device)
        tol    = self.cfg.deposition_wall_tol

        fluid_traj    = []
        particle_traj = []
        dp_traj       = []

        for step_idx in range(n_steps):
            age = torch.full((N_part,), step_idx * self.cfg.dt,
                             dtype=d_p.dtype, device=device)

            out = self.step(
                fluid, cell_pos, bc_type, bc_values,
                edge_index, face_normals, face_areas, face_dists,
                cell_volumes, d_wall,
                p_hist, particle_type, d_p_cur, d_p0, rho_p,
                face_type=face_type, particle_age=age,
                clip_particles=True,
                inlet_cond=inlet_cond,
                wall_normal=wall_normal,
                domain_bounds=domain_bounds,
            )
            fluid   = out["fluid_field"]
            p_next  = out["particle_pos"]
            d_p_cur = out["d_p_new"]

            # Deposition: distance to nearest wall *face* (incl. obstacles).
            d_at_p = interpolate_fluid_to_particles(
                d_wall.unsqueeze(-1), cell_pos, p_next,
                k_nearest=self.cfg.cross_k_nearest,
            ).squeeze(-1)
            box_dist = torch.stack([
                (p_next - bounds[:, 0]).min(dim=-1).values,
                (bounds[:, 1] - p_next).min(dim=-1).values,
            ], dim=-1).min(dim=-1).values
            dist_to_walls = torch.minimum(d_at_p, box_dist)

            newly_dep                = (~deposited) & (dist_to_walls < tol)
            depo_step[newly_dep]     = step_idx
            depo_pos[newly_dep]      = p_next[newly_dep]
            deposited                = deposited | newly_dep
            p_next[deposited]        = depo_pos[deposited]

            p_hist  = torch.cat([p_hist[:, 1:], p_next.unsqueeze(1)], dim=1)

            fluid_traj.append(fluid.cpu())
            particle_traj.append(p_next.cpu())
            dp_traj.append(d_p_cur.cpu())

        return {
            "fluid_traj":      torch.stack(fluid_traj,    dim=0),
            "particle_traj":   torch.stack(particle_traj, dim=0),
            "dp_traj":         torch.stack(dp_traj,       dim=0),
            "deposition_step": depo_step.cpu(),
            "deposition_pos":  depo_pos.cpu(),
        }


# ---------------------------------------------------------------------------
#  Checkpoint helpers
# ---------------------------------------------------------------------------

def save_cfd_gnn_checkpoint(model, path, epoch, val_loss):
    torch.save({
        "config":   dataclasses.asdict(model.cfg),
        "model":    model.state_dict(),
        "epoch":    epoch,
        "val_loss": val_loss,
    }, path)


def load_cfd_gnn_checkpoint(path: str) -> CfdGNN:
    ckpt  = torch.load(path, map_location="cpu", weights_only=False)
    cfg   = CfdGNNConfig(**ckpt["config"])
    model = CfdGNN(cfg)
    model.load_state_dict(ckpt["model"])
    return model
