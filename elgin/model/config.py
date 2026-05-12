"""CfdGNNConfig — master hyperparameter dataclass for ELGIN (CfdGNN).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple


@dataclass
class CfdGNNConfig:
    """Master configuration for the ELGIN framework."""

    # ── Geometry / domain ────────────────────────────────────────────────────
    dim: int = 2
    domain_bounds: Tuple[Tuple[float, float], Tuple[float, float]] = (
        (0.0, 4.0), (0.0, 3.0)
    )
    L_ref: float = 4.0    # m   — room width (reference length)
    U_ref: float = 20.0   # m/s — injection reference speed
    dt:    float = 0.01   # s   — CFD time step

    # ── Eulerian graph (fluid field) ─────────────────────────────────────────
    fluid_node_in:    int   = 5     # [U_x, U_y, p, k, omega]
    fluid_edge_in:    int   = 6     # geometric features (face_type appended at runtime)
    fluid_hidden:     int   = 64    # d_h (paper Table III)
    fluid_mp_steps:   int   = 4     # K_E (paper Table III)
    fluid_mlp_layers: int   = 2
    fluid_out:        int   = 5     # [dU_x, dU_y, dp, dk, domega]

    use_fluid_attention: bool = True

    use_graph_transformer: bool = True   # replaces single-head gate when True
    fluid_attn_heads:      int  = 4      # number of attention heads

    face_type_count:     int = 5    # interior/inlet/outlet/wall/symmetry
    face_type_embed_dim: int = 4

    history_length:      int   = 5
    particle_type_count: int   = 9
    particle_embed_dim:  int   = 16
    particle_hidden:     int   = 64    # d_h (paper Table III)
    particle_mp_steps:   int   = 4     # K_L (paper Table III)
    particle_mlp_layers: int   = 2
    particle_radius:     float = 0.10   # m — radius graph connectivity (ELGIN).
                                        # M0 baseline: 0.30 m (paper Sec. V.A).


    use_lstm_encoder: bool = True
    lstm_hidden:      int  = 32


    use_symplectic: bool = True


    use_log_dp: bool = True


    use_equivariant_edges: bool = True


    use_stochastic_decoder: bool = False
    stochastic_latent_dim:  int  = 16   # dimension of the stochastic latent


    use_saffman_lift: bool = False


    use_heterogeneous_graph:   bool  = False
    fine_particle_threshold:   float = 5e-6   # m — particles < threshold are 'fine'


    use_brownian_motion: bool  = False
    T_air:               float = 293.15   # K — air temperature


    use_drag_features: bool = True   # Stokes drag as node/edge input feature
    use_gravity:       bool = True   # gravity vector added to predicted acceleration


    cross_k_nearest:  int = 4
    cross_hidden:     int = 64
    cross_mlp_layers: int = 2


    use_tke_dispersion: bool = True


    use_evaporation: bool  = False
    evap_D_v:        float = 2.6e-5   # m^2/s  vapour diffusivity
    evap_B_M:        float = 0.0263   # Spalding mass-transfer number (50 %RH)

    # ── Turbulence closure ────────────────────────────────────────────────────
    turb_in:          int  = 4
    turb_hidden:      int  = 32
    turb_layers:      int  = 3
    analytic_closure: bool = False

    # ── Pressure projection ───────────────────────────────────────────────────
    pressure_cg_iters:        int   = 20
    pressure_cg_iters_stage3: int   = 50
    pressure_cg_iters_stage4: int   = 50
    pressure_cg_tol:          float = 1e-4
    use_jacobi_precond:       bool  = True
    learned_pressure:         bool  = False
    pressure_hidden:          int   = 64

    # ── Boundary conditions ───────────────────────────────────────────────────

    bc_type_count: int = 16
    bc_embed_dim:  int = 8


    use_inlet_conditioning: bool = True


    use_wall_features_lag: bool = True


    deposition_wall_tol: float = 0.01   # m


    use_two_way_coupling:     bool  = False   # off by default (dense spray only)
    two_way_vol_frac_thr:     float = 0.01    # volume fraction threshold

    # ── Normalisation statistics ──────────────────────────────────────────────
    fluid_mean: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.01, 10.0)
    fluid_std:  Tuple[float, ...] = (1.0, 1.0, 1.0, 0.01,  5.0)
    vel_mean:   Tuple[float, float] = (0.0, 0.0)
    vel_std:    Tuple[float, float] = (1.0, 1.0)
    acc_mean:   Tuple[float, float] = (0.0, 0.0)
    acc_std:    Tuple[float, float] = (1.0, 1.0)

    # ── Training curriculum ───────────────────────────────────────────────────

    stage1_epochs:       int   = 60
    stage1_lr:           float = 5e-4   # reduced from 1e-3 — prevents escape from good minimum
    stage2_epochs:       int   = 60
    stage2_lr:           float = 5e-4
    stage2_freeze_fluid: bool  = True
    stage3_epochs:       int   = 120
    stage3_lr:           float = 1e-4
    stage4_epochs:       int   = 60
    stage4_lr:           float = 5e-5
    stage4_max_steps:    int   = 20


    use_bptt_loss:        bool  = True
    bptt_rollout_steps:   int   = 5      # paper Table III: BPTT unroll = 5
    bptt_loss_weight:     float = 0.5    # weight of rollout loss vs one-step
    bptt_rollout_noise:   float = 0.01  # noise [m] injected between BPTT steps
                                         # to simulate long-horizon covariate shift

    # PDE loss weights
    lambda_continuity: float = 0.1
    lambda_momentum:   float = 0.05
    lambda_turbulence: float = 0.02
    lambda_particle:   float = 1.0
    lambda_energy:     float = 0.05
    lambda_angular:    float = 0.01

    lambda_kl:         float = 0.001


    noise_std: float = 3e-4


    ensemble_size: int = 5


    pretrained_gns_checkpoint: str = ""

    # ── Physical constants ────────────────────────────────────────────────────
    nu:  float = 1.5e-5   # m^2/s  kinematic viscosity of air
    rho: float = 1.225    # kg/m^3 air density
    g:   float = 9.81     # m/s^2  gravitational acceleration
