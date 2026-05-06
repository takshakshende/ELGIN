"""LagrangianGNN — particle transport GNN.

Improvements (Round 1 + Round 2)
----------------------------------
#1  Bug fix   : velocity from pos-diff not raw pos
#2  LSTM      : recurrent temporal encoder
#3  Stormer-V : symplectic integrator
#7  log(d_p)  : explicit size feature

S1  SE(2) equivariance  — edge features expressed in edge-local frames
S2  Evaporation         — Wells' d^2-law: d_p(t) shrinks as droplet evaporates
S3  Stochastic decoder  — probabilistic acceleration head with KL loss
S6  Saffman lift        — shear-induced lift force node feature
S9  Heterogeneous graph — fine/coarse particle embeddings
S11 Brownian motion     — Einstein-Smoluchowski kick for sub-micron particles

Physics basis
-------------
Maxey-Riley simplified:
    dv/dt = F_drag/m + F_lift/m + F_grav/m + F_Brownian/m + F_turb/m
    F_drag     = m*(u_fluid - v) / (tau_p/Cc)          [Stokes + Cunningham]
    F_lift     = C_L * |du/dy|^0.5 * |u-v| * lift_dir  [Saffman 1965]
    F_Brownian ~ N(0, 2*D_B/dt)                         [Einstein 1905]
    F_turb     ~ N(0, 2k/3)                             [DRW / Pope 2000]
"""

from __future__ import annotations
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .config  import CfdGNNConfig
from .physics import (
    cunningham_correction,
    stokes_drag_acc,
    evaporation_diameter,
    saffman_lift_acc,
    brownian_sigma,
    turbulent_dispersion_kick,
)


# ---------------------------------------------------------------------------
#  Shared MLP builder
# ---------------------------------------------------------------------------

def _mlp(in_d: int, hid: int, out_d: int, layers: int,
         ln: bool = True) -> nn.Sequential:
    mods: list = [nn.Linear(in_d, hid), nn.ReLU()]
    for _ in range(layers - 1):
        mods += [nn.Linear(hid, hid), nn.ReLU()]
    mods.append(nn.Linear(hid, out_d))
    if ln:
        mods.append(nn.LayerNorm(out_d))
    return nn.Sequential(*mods)


# ---------------------------------------------------------------------------
#  Pure-PyTorch radius graph
# ---------------------------------------------------------------------------

def _radius_graph(pos: torch.Tensor, r: float,
                  max_neighbors: int = 32,
                  chunk_size: int = 512) -> torch.Tensor:
    """Build radius graph without materialising the full N×N distance matrix.

    Processes rows in chunks of `chunk_size` to keep peak memory at
    O(chunk_size × N) rather than O(N²). With chunk_size=512 and N=3000
    the peak extra allocation is 512 × 3000 × 4 ≈ 6 MB instead of 36 MB,
    and the benefit is larger for N ≥ 10 000.
    """
    N = pos.shape[0]
    k = min(max_neighbors, N - 1)
    rows, cols = [], []
    pos_f = pos.float()

    for start in range(0, N, chunk_size):
        end   = min(start + chunk_size, N)
        chunk = pos_f[start:end]                           # (C, dim)
        dist  = torch.cdist(chunk, pos_f)                  # (C, N)
        dist[:, start:end].fill_diagonal_(float("inf"))    # no self-loops

        # k nearest within radius r
        top_d, top_i = torch.topk(dist, k=k, dim=1, largest=False)
        valid = top_d < r                                  # (C, k)

        src = (torch.arange(start, end, device=pos.device)
               .unsqueeze(1).expand_as(top_i)[valid])
        dst = top_i[valid]
        rows.append(src)
        cols.append(dst)

    if rows:
        return torch.stack([torch.cat(rows), torch.cat(cols)], dim=0)
    return torch.zeros(2, 0, dtype=torch.long, device=pos.device)


# ---------------------------------------------------------------------------
#  S1 — SE(2) edge-local reference frame helper
# ---------------------------------------------------------------------------

def _edge_local_frame(pos_i: torch.Tensor,
                      pos_j: torch.Tensor,
                      r:     float
                      ) -> torch.Tensor:
    """Project relative position into an edge-aligned local frame.

    For edge (i -> j) we define:
        t_hat = (x_j - x_i) / ||x_j - x_i||          (tangent)
        n_hat = rotate(t_hat, +90°) = (-t_y, t_x)     (normal, 2-D)

    The local-frame features:
        [d_t/r, d_n/r, ||delta||/r]

    are invariant to rigid translations and equivariant to 2-D rotations:
    rotating the room merely rotates the frame along with the coordinates,
    so the feature values are unchanged.

    References
    ----------
    - Sharma & Fink (2025) Dynami-CAL GraphNet: edge-local reference frames
    - Villar et al. (2021) Scalable and equivariant spherical CNNs
    """
    delta = pos_j - pos_i                              # (E, 2)
    dist  = delta.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    t_hat = delta / dist                               # (E, 2) unit tangent
    # 2-D normal: rotate t_hat by +90 degrees
    n_hat = torch.stack([-t_hat[:, 1], t_hat[:, 0]], dim=-1)  # (E, 2)

    d_t   = (delta * t_hat).sum(-1, keepdim=True)     # (E, 1) tangential distance
    d_n   = (delta * n_hat).sum(-1, keepdim=True)     # (E, 1) normal distance

    return torch.cat([d_t / r, d_n / r, dist / r], dim=-1)  # (E, 3)


# ---------------------------------------------------------------------------
#  LSTM temporal encoder (Round-1 #2)
# ---------------------------------------------------------------------------

class LSTMTemporalEncoder(nn.Module):
    def __init__(self, input_dim: int, lstm_hidden: int, out_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, lstm_hidden, batch_first=True)
        self.proj = nn.Linear(lstm_hidden, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, vel_history: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(vel_history)
        return self.norm(self.proj(h_n.squeeze(0)))


# ---------------------------------------------------------------------------
#  S3 — Stochastic (VAE-style) decoder for turbulent dispersion
# ---------------------------------------------------------------------------

class StochasticDecoder(nn.Module):
    """Probabilistic acceleration head.

    Outputs a mean mu and log-variance log_sigma^2 over acceleration.
    During training, samples a = mu + sigma * eps, eps ~ N(0,I).
    During eval, returns the mean (deterministic).

    The reparameterisation trick keeps the gradient flowing through mu and
    sigma, and the KL loss KL(N(mu,sigma)||N(0,1)) is returned alongside
    the sample for use in the training loss.

    References
    ----------
    - Kingma & Welling (2014) VAE
    - Lino et al. (2025) Diffusion Graph Networks
    - Pope (2000) Turbulent Flows — stochastic Lagrangian dispersion
    """

    def __init__(self, hidden: int, dim: int, layers: int):
        super().__init__()
        self.mu_head    = _mlp(hidden, hidden, dim, layers, ln=False)
        self.logv_head  = _mlp(hidden, hidden, dim, layers, ln=False)

    def forward(self, h: torch.Tensor,
                sample: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            acc_sample : (N, dim) — sampled (or mean) acceleration
            kl_loss    : scalar   — KL divergence per sample
        """
        mu    = self.mu_head(h)                        # (N, dim)
        logv  = self.logv_head(h).clamp(-10, 4)        # (N, dim) log-variance
        sigma = (0.5 * logv).exp()                     # (N, dim) std-dev

        if sample and self.training:
            eps    = torch.randn_like(sigma)
            a_samp = mu + sigma * eps
        else:
            a_samp = mu

        # KL divergence:  0.5 * sum(mu^2 + sigma^2 - log(sigma^2) - 1)
        kl = 0.5 * (mu ** 2 + sigma ** 2 - logv - 1.0).sum(dim=-1).mean()

        return a_samp, kl


# ---------------------------------------------------------------------------
#  Hamiltonian decoder (Round-1 #3 symplectic)
# ---------------------------------------------------------------------------

class HamiltonianDecoder(nn.Module):
    def __init__(self, hidden: int, dim: int, layers: int):
        super().__init__()
        self.head_q = _mlp(hidden, hidden, dim, layers, ln=False)
        self.head_p = _mlp(hidden, hidden, dim, layers, ln=False)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.head_q(h), self.head_p(h)


# ---------------------------------------------------------------------------
#  Particle interaction block
# ---------------------------------------------------------------------------

class ParticleInteractionBlock(nn.Module):
    def __init__(self, hidden: int, mlp_layers: int):
        super().__init__()
        self.edge_mlp = _mlp(3 * hidden, hidden, hidden, mlp_layers)
        self.node_mlp = _mlp(2 * hidden, hidden, hidden, mlp_layers)

    def forward(self, x, edge_index, e):
        src, dst = edge_index
        e_new = self.edge_mlp(torch.cat([x[dst], x[src], e], dim=-1))
        N   = x.shape[0]
        idx = dst.unsqueeze(-1).expand_as(e_new)
        agg = torch.zeros(N, e_new.shape[-1], dtype=x.dtype,
                          device=x.device).scatter_add(0, idx, e_new)
        x_new = x + self.node_mlp(torch.cat([x, agg], dim=-1))
        return x_new, e_new


# ---------------------------------------------------------------------------
#  LagrangianGNN
# ---------------------------------------------------------------------------

class LagrangianGNN(nn.Module):
    """Particle transport GNN with all Round-1 and Round-2 improvements."""

    def __init__(self, cfg: CfdGNNConfig):
        super().__init__()
        self.cfg = cfg
        D   = cfg.dim
        H   = cfg.particle_hidden

        # Feature flags
        self.use_lstm       = cfg.use_lstm_encoder
        self.use_symplectic = cfg.use_symplectic
        self.use_log_dp      = cfg.use_log_dp
        self.use_tke         = cfg.use_tke_dispersion
        self.use_equivar     = cfg.use_equivariant_edges
        self.use_stoch       = cfg.use_stochastic_decoder
        self.use_saffman     = cfg.use_saffman_lift
        self.use_hetero      = cfg.use_heterogeneous_graph
        self.use_brownian    = cfg.use_brownian_motion
        self.use_evap        = cfg.use_evaporation
        self.use_drag_feat   = getattr(cfg, "use_drag_features", True)
        self.use_gravity     = getattr(cfg, "use_gravity", True)
        self.use_wall_feat   = getattr(cfg, "use_wall_features_lag", False)

        # ── Node input dimension ──────────────────────────────────────────────
        vel_feat_dim = H if self.use_lstm else cfg.history_length * D

        node_in = (vel_feat_dim
                   + 2 * D                               # SDF wall dists (bbox-relative)
                   + cfg.particle_embed_dim              # type embedding
                   + (D if self.use_drag_feat  else 0)   # Stokes drag (optional)
                   + (1 if self.use_log_dp     else 0)
                   + (1 if self.use_tke        else 0)
                   + (D if self.use_saffman    else 0)
                   + (1 if self.use_evap       else 0)   # S2: d_p/d_p0
                   + (D if self.use_hetero     else 0)   # S9: coarse/fine signal
                   + ((1 + D) if self.use_wall_feat else 0))  # d_wall + wall_normal

        # ── Embeddings ────────────────────────────────────────────────────────
        if self.use_hetero:
            # S9: two embeddings — fine (<= threshold) and coarse
            self.fine_embed   = nn.Embedding(cfg.particle_type_count,
                                             cfg.particle_embed_dim)
            self.coarse_embed = nn.Embedding(cfg.particle_type_count,
                                             cfg.particle_embed_dim)
            # Fine/coarse indicator projected to D dimensions
            self.size_class_proj = _mlp(1, D, D, 1, ln=False)
        else:
            self.type_embed = nn.Embedding(cfg.particle_type_count,
                                           cfg.particle_embed_dim)

        if self.use_lstm:
            self.lstm_encoder = LSTMTemporalEncoder(D, cfg.lstm_hidden, H)

        self.node_encoder = _mlp(node_in, H, H, cfg.particle_mlp_layers)

        # ── Edge input dimension ──────────────────────────────────────────────
        # S1 equivariant: 3 (d_t/r, d_n/r, dist/r) [+ D drag if enabled]
        # Standard:       D + 1 (rel_pos + dist)    [+ D drag if enabled]
        edge_in = (3 if self.use_equivar else D + 1) + (D if self.use_drag_feat else 0)

        self.edge_encoder = _mlp(edge_in, H, H, cfg.particle_mlp_layers)

        # ── Processor ─────────────────────────────────────────────────────────
        self.processor = nn.ModuleList(
            ParticleInteractionBlock(H, cfg.particle_mlp_layers)
            for _ in range(cfg.particle_mp_steps)
        )

        # ── Decoder ───────────────────────────────────────────────────────────
        if self.use_stoch:
            # S3: probabilistic decoder — also has latent dim
            self.decoder = StochasticDecoder(H, D, cfg.particle_mlp_layers)
        elif self.use_symplectic:
            self.decoder = HamiltonianDecoder(H, D, cfg.particle_mlp_layers)
        else:
            self.decoder = _mlp(H, H, D, cfg.particle_mlp_layers, ln=False)

        # ── Normalisation buffers ─────────────────────────────────────────────
        self.register_buffer("vel_mean",
            torch.tensor(cfg.vel_mean, dtype=torch.float32))
        self.register_buffer("vel_std",
            torch.tensor(cfg.vel_std,  dtype=torch.float32))
        self.register_buffer("acc_mean",
            torch.tensor(cfg.acc_mean, dtype=torch.float32))
        self.register_buffer("acc_std",
            torch.tensor(cfg.acc_std,  dtype=torch.float32))

        # Cache for KL loss
        self._last_kl: torch.Tensor = torch.tensor(0.0)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _norm_vel(self, v):
        return (v - self.vel_mean) / (self.vel_std + 1e-8)

    def _denorm_acc(self, a):
        return a * (self.acc_std + 1e-8) + self.acc_mean

    def _type_embed(self, ptype: torch.Tensor,
                    d_p:   torch.Tensor) -> torch.Tensor:
        """Return type embedding (heterogeneous or standard)."""
        if self.use_hetero:
            is_fine   = (d_p < self.cfg.fine_particle_threshold).long()
            fine_feat  = self.fine_embed(ptype)   * is_fine.unsqueeze(-1)
            coarse_feat= self.coarse_embed(ptype) * (1 - is_fine).unsqueeze(-1)
            return fine_feat + coarse_feat
        return self.type_embed(ptype)

    # ── Node feature builder ──────────────────────────────────────────────────

    def _node_feat(
        self,
        pos_hist: torch.Tensor,             # (N, H+1, D)
        ptype:    torch.Tensor,             # (N,)
        u_fluid:  torch.Tensor,             # (N, D)
        d_p:      torch.Tensor,             # (N,)  current diameter
        d_p0:     torch.Tensor,             # (N,)  initial diameter (for S2)
        rho_p:    torch.Tensor,             # (N,)
        k_fluid:  Optional[torch.Tensor],   # (N,)  TKE
        du_dy:    Optional[torch.Tensor],   # (N,)  wall-normal shear (for S6)
        wall_feat_p: Optional[torch.Tensor] = None,  # (N, 1+D) [d_wall, wn_x, wn_y]
    ) -> torch.Tensor:
        cfg    = self.cfg
        D      = cfg.dim
        bounds = torch.tensor(cfg.domain_bounds,
                              dtype=pos_hist.dtype, device=pos_hist.device)
        room_scale = torch.stack([bounds[0, 1] - bounds[0, 0],
                                  bounds[1, 1] - bounds[1, 0]])

        # Velocity history
        vels     = pos_hist[:, 1:] - pos_hist[:, :-1]     # (N, H, D)
        norm_vel = self._norm_vel(vels)

        vel_feat = (self.lstm_encoder(norm_vel)
                    if self.use_lstm
                    else norm_vel.reshape(norm_vel.size(0), -1))

        # SDF wall distances
        cur      = pos_hist[:, -1]
        dist_lo  = (cur - bounds[:, 0]) / room_scale
        dist_hi  = (bounds[:, 1] - cur) / room_scale
        wall_feat = torch.cat([dist_lo.clamp(-1, 1),
                                dist_hi.clamp(-1, 1)], dim=-1)

        type_feat = self._type_embed(ptype, d_p)
        parts = [vel_feat, wall_feat, type_feat]

        # Stokes drag node feature (disabled in simplified / baseline-equivalent mode)
        v_p = pos_hist[:, -1] - pos_hist[:, -2]   # position difference (m/step)
        if self.use_drag_feat:
            # Convert position-difference to physical velocity before passing to drag
            v_p_ms = v_p / (self.cfg.dt + 1e-12)  # m/s
            a_drag  = stokes_drag_acc(v_p_ms, u_fluid, d_p, rho_p)
            # Normalise by acc_std scaled to physical units (acc_std is in m/step²
            # = physical_acc * dt²), so divide by dt² to match m/s² scale
            norm_drag = a_drag * (self.cfg.dt ** 2) / self.acc_std.clamp(min=1e-8)
            parts.append(norm_drag)

        # S2: evaporation ratio d_p/d_p0  (1 = un-evaporated, 0.5 = nucleus)
        if self.use_evap:
            evap_ratio = (d_p / d_p0.clamp(min=1e-12)).clamp(0.0, 1.0)
            parts.append(evap_ratio.unsqueeze(-1))

        # #7: log(d_p) scaled
        if self.use_log_dp:
            log_dp_norm = (d_p.clamp(min=1e-9).log() + 13.5) / 4.5
            parts.append(log_dp_norm.unsqueeze(-1))

        # #4: TKE turbulent dispersion intensity
        if self.use_tke and k_fluid is not None:
            tke_feat = (k_fluid.clamp(min=0.0).sqrt() / cfg.U_ref).unsqueeze(-1)
            parts.append(tke_feat)

        # S6: Saffman lift vector — also needs physical velocity
        if self.use_saffman:
            v_p_ms = v_p / (self.cfg.dt + 1e-12)
            a_lift  = saffman_lift_acc(v_p_ms, u_fluid, d_p, rho_p, du_dy,
                                       nu=cfg.nu, rho_f=cfg.rho)
            norm_lift = a_lift * (self.cfg.dt ** 2) / self.acc_std.clamp(min=1e-8)
            parts.append(norm_lift)

        # S9: size-class signal (continuous fine-ness indicator)
        if self.use_hetero:
            fine_frac = (1.0 - (d_p / cfg.fine_particle_threshold)
                         .clamp(0.0, 1.0)).unsqueeze(-1)  # 1=fine, 0=coarse
            parts.append(self.size_class_proj(fine_frac))

        # Wall awareness: distance + outward normal at the parcel position.
        # d_wall is normalised by L_ref so it is O(1).
        if self.use_wall_feat:
            if wall_feat_p is None:
                wall_feat_p = pos_hist.new_zeros(pos_hist.shape[0], 1 + D)
            else:
                wf = wall_feat_p.clone()
                wf[..., 0] = (wf[..., 0] / cfg.L_ref).clamp(0.0, 2.0)
                wall_feat_p = wf
            parts.append(wall_feat_p)

        return torch.cat(parts, dim=-1)

    # ── Edge feature builder ──────────────────────────────────────────────────

    def _edge_feat(
        self,
        pos:        torch.Tensor,
        pos_hist:   torch.Tensor,
        edge_index: torch.Tensor,
        u_fluid:    torch.Tensor,
        d_p:        torch.Tensor,
        rho_p:      torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index
        r        = self.cfg.particle_radius

        # S1: SE(2) equivariant local-frame features
        if self.use_equivar:
            geom = _edge_local_frame(pos[src], pos[dst], r)   # (E, 3)
        else:
            rel  = pos[dst] - pos[src]
            dist = rel.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            geom = torch.cat([rel / r, dist / r], dim=-1)     # (E, D+1)

        if self.use_drag_feat:
            v_p    = pos_hist[:, -1] - pos_hist[:, -2]
            v_p_ms = v_p / (self.cfg.dt + 1e-12)            # convert to m/s
            a_drag_src = stokes_drag_acc(v_p_ms[src], u_fluid[src],
                                         d_p[src], rho_p[src])
            norm_drag_src = (a_drag_src * (self.cfg.dt ** 2)
                             / self.acc_std.clamp(min=1e-8))
            return torch.cat([geom, norm_drag_src], dim=-1)

        return geom

    # ── Forward (predict acceleration) ───────────────────────────────────────

    def predict_acceleration(
        self,
        pos_hist: torch.Tensor,
        ptype:    torch.Tensor,
        u_fluid:  torch.Tensor,
        d_p:      torch.Tensor,
        d_p0:     torch.Tensor,
        rho_p:    torch.Tensor,
        k_fluid:  Optional[torch.Tensor] = None,
        du_dy:    Optional[torch.Tensor] = None,
        wall_feat_p: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return normalised acceleration; cache KL loss for stochastic decoder."""
        cur        = pos_hist[:, -1]
        edge_index = _radius_graph(cur, self.cfg.particle_radius)

        node_feat = self._node_feat(pos_hist, ptype, u_fluid, d_p, d_p0,
                                    rho_p, k_fluid, du_dy,
                                    wall_feat_p=wall_feat_p)

        if edge_index.shape[1] > 0:
            edge_feat = self._edge_feat(cur, pos_hist, edge_index,
                                        u_fluid, d_p, rho_p)
        else:
            E_dim = (3 if self.use_equivar else self.cfg.dim + 1) + self.cfg.dim
            edge_feat = cur.new_zeros(0, E_dim)

        x = self.node_encoder(node_feat)
        e = (self.edge_encoder(edge_feat)
             if edge_index.shape[1] > 0
             else cur.new_zeros(0, self.cfg.particle_hidden))

        for block in self.processor:
            if edge_index.shape[1] > 0:
                x, e = block(x, edge_index, e)

        # Decode
        self._last_kl = torch.tensor(0.0, device=x.device)
        if self.use_stoch:
            # S3: stochastic decoder — also stores KL for loss computation
            acc, kl = self.decoder(x)
            self._last_kl = kl
            return acc
        elif self.use_symplectic:
            dH_dq, dH_dp = self.decoder(x)
            self._last_dH_dp = dH_dp
            return -dH_dq
        else:
            return self.decoder(x)

    # ── Integration step ──────────────────────────────────────────────────────

    def next_position(
        self,
        pos_hist: torch.Tensor,
        ptype:    torch.Tensor,
        u_fluid:  torch.Tensor,
        d_p:      torch.Tensor,
        d_p0:     torch.Tensor,
        rho_p:    torch.Tensor,
        g_vec:    Optional[torch.Tensor] = None,
        k_fluid:  Optional[torch.Tensor] = None,
        du_dy:    Optional[torch.Tensor] = None,
        wall_feat_p: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict next position with full physics stack.

        Pipeline:
            1. GNN predicts residual acceleration
            2. Add gravity
            3. S3: add stochastic TKE dispersion kick
            4. S11: add Brownian motion for sub-micron particles
            5. Integrate via Stormer-Verlet (Round-1 #3) or Euler
        """
        norm_acc  = self.predict_acceleration(
            pos_hist, ptype, u_fluid, d_p, d_p0, rho_p, k_fluid, du_dy,
            wall_feat_p=wall_feat_p,
        )
        acc       = self._denorm_acc(norm_acc)

        # Gravity: g_vec is in m/s²; acc is in position units (m = a·dt²)
        # Multiply by dt² to convert to the same position-difference space.
        if g_vec is not None and self.use_gravity:
            acc = acc + g_vec.unsqueeze(0) * (self.cfg.dt ** 2)

        # S3: stochastic TKE dispersion kick — v_turb is in m/s.
        # Convert to position units: displacement = v * dt (metres per step).
        if self.use_tke and k_fluid is not None:
            v_turb = turbulent_dispersion_kick(k_fluid, dt=self.cfg.dt,
                                               dim=self.cfg.dim,
                                               device=acc.device)
            acc = acc + v_turb * self.cfg.dt   # m/s × s = m (position units)

        last_vel = pos_hist[:, -1] - pos_hist[:, -2]
        cur_pos  = pos_hist[:, -1]

        if self.use_symplectic and hasattr(self, "_last_dH_dp"):
            # Stormer-Verlet
            vel_ham  = (self._last_dH_dp * (self.acc_std + 1e-8) + self.vel_mean)
            p_half   = last_vel + 0.5 * acc
            next_vel = 0.5 * p_half + 0.5 * vel_ham
            next_pos = cur_pos + next_vel
        else:
            # Forward Euler
            next_vel = last_vel + acc
            next_pos = cur_pos + next_vel

        # S11: Brownian motion kick for sub-micron particles
        if self.use_brownian:
            sigma_br = brownian_sigma(d_p, T=self.cfg.T_air,
                                      dt=self.cfg.dt)          # (N,)
            eps      = torch.randn(next_pos.shape,
                                   dtype=next_pos.dtype,
                                   device=next_pos.device)
            brownian = sigma_br.unsqueeze(-1) * eps
            # Only apply to particles small enough (d_p < 1 µm)
            sub_micron = (d_p < 1e-6).float().unsqueeze(-1)
            next_pos   = next_pos + sub_micron * brownian

        # Wall reflection — prevents particle pile-up at domain boundaries.
        # Reflecting the overshoot (rather than clamping) reverses the implied
        # velocity so the model's LSTM history encodes inward motion next step,
        # pulling the particle back into the domain naturally.
        bounds = torch.tensor(self.cfg.domain_bounds,
                              dtype=next_pos.dtype,
                              device=next_pos.device)   # (dim, 2)
        lo = bounds[:, 0].unsqueeze(0)                  # (1, dim)
        hi = bounds[:, 1].unsqueeze(0)                  # (1, dim)
        mask_lo = next_pos < lo
        next_pos = torch.where(mask_lo, 2.0 * lo - next_pos, next_pos)
        mask_hi = next_pos > hi
        next_pos = torch.where(mask_hi, 2.0 * hi - next_pos, next_pos)
        # Failsafe hard clamp for extreme overshoots
        next_pos = torch.max(torch.min(next_pos, hi), lo)

        return next_pos
