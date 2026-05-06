"""CfdGNNConfig — master hyperparameter dataclass for the CFD-GNN.

Improvement log
---------------
Round 1 (implemented in model files)
  #1  Bug fix: velocity from pos-diff, not raw pos
  #2  LSTM temporal encoder
  #3  Stormer-Verlet symplectic integrator
  #4  TKE cross-graph interpolation
  #5  Single-head attention gate in FluidInteractionBlock
  #6  Face-type embedding in Eulerian edges
  #7  log(d_p) node feature
  #8  Particle deposition mask in rollout
  #9  Noise augmentation in training
  #10 Angular momentum conservation loss
  #11 Jacobi preconditioned CG

Round 2 (this file + corresponding model changes)
  S1  SE(2)-equivariant edge-local reference frames (lagrangian)
  S2  Wells' droplet evaporation model
  S3  Probabilistic stochastic decoder for turbulent dispersion
  S4  Differentiable multi-step rollout loss (BPTT) in train.py
  S5  Multi-head Graph Transformer attention (Eulerian)
  S6  Saffman shear-lift force
  S7  Two-way Eulerian-Lagrangian coupling
  S8  Ensemble UQ — size / config only (model instantiation in ensemble.py)
  S9  Heterogeneous size-class graph (fine vs coarse particles)
  S11 Brownian motion for sub-micron particles
  S13 Transfer-learning from pre-trained GNS checkpoint
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple


@dataclass
class CfdGNNConfig:
    """Master configuration for the CFD-GNN framework."""

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
    fluid_hidden:     int   = 128
    fluid_mp_steps:   int   = 8
    fluid_mlp_layers: int   = 2
    fluid_out:        int   = 5     # [dU_x, dU_y, dp, dk, domega]

    # Round-1 #5: single-head attention gate (kept for backward compat)
    use_fluid_attention: bool = True

    # S5: multi-head Graph Transformer attention in Eulerian processor
    use_graph_transformer: bool = True   # replaces single-head gate when True
    fluid_attn_heads:      int  = 4      # number of attention heads

    # Round-1 #6: face-type embedding
    face_type_count:     int = 5    # interior/inlet/outlet/wall/symmetry
    face_type_embed_dim: int = 4

    # ── Lagrangian graph (particle transport) ────────────────────────────────
    history_length:      int   = 5
    particle_type_count: int   = 9
    particle_embed_dim:  int   = 16
    particle_hidden:     int   = 128
    particle_mp_steps:   int   = 5
    particle_mlp_layers: int   = 2
    particle_radius:     float = 0.10   # m — radius graph connectivity. Reduced
                                        # from 0.30 m so the graph topology is
                                        # closer to stationary between the
                                        # injection cone and the dispersed
                                        # cloud later in the rollout.

    # Round-1 #2: LSTM temporal encoder
    use_lstm_encoder: bool = True
    lstm_hidden:      int  = 32

    # Round-1 #3: Stormer-Verlet symplectic integrator
    use_symplectic: bool = True

    # Round-1 #7: log(d_p) node feature
    use_log_dp: bool = True

    # S1: SE(2)-equivariant edge-local reference frames
    use_equivariant_edges: bool = True

    # S3: Probabilistic / stochastic decoder for turbulent dispersion
    use_stochastic_decoder: bool = True
    stochastic_latent_dim:  int  = 16   # dimension of the stochastic latent

    # S6: Saffman shear-lift force
    use_saffman_lift: bool = True

    # S9: Heterogeneous graph — separate embeddings for fine/coarse particles
    use_heterogeneous_graph:   bool  = True
    fine_particle_threshold:   float = 5e-6   # m — particles < threshold are 'fine'

    # S11: Brownian motion for sub-micron particles
    use_brownian_motion: bool  = True
    T_air:               float = 293.15   # K — air temperature

    # Simplification flags — disable physics components to match baseline GNS
    # Set both to False for a minimal model equivalent to GNS + fluid coupling.
    use_drag_features: bool = True   # Stokes drag as node/edge input feature
    use_gravity:       bool = True   # gravity vector added to predicted acceleration

    # ── Cross-graph (fluid -> particle coupling) ─────────────────────────────
    cross_k_nearest:  int = 4
    cross_hidden:     int = 64
    cross_mlp_layers: int = 2

    # Round-1 #4: TKE interpolation for turbulent dispersion
    use_tke_dispersion: bool = True

    # S2: Droplet evaporation (Wells' d^2-law)
    use_evaporation: bool  = True
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
    # bc_type ids 0..8: interior, inlet, outlet, wall, floor, ceiling,
    # dentist, patient, symmetry — with headroom up to 16.
    bc_type_count: int = 16
    bc_embed_dim:  int = 8

    # Per-case inlet conditioning: prescribed airInlet velocity (Ux, Uy)
    # is broadcast as a constant 2-channel feature on every cell.  Lets the
    # network distinguish cases that differ only in the inlet condition.
    use_inlet_conditioning: bool = True

    # Lagrangian wall-distance / wall-normal awareness.  When on, each
    # particle node feature is augmented with [d_wall, wall_n_x, wall_n_y]
    # interpolated from the cell-level fields stored in mesh_graph.npz.
    use_wall_features_lag: bool = True

    # ── Deposition (Round-1 #8) ───────────────────────────────────────────────
    deposition_wall_tol: float = 0.01   # m

    # S7: Two-way Eulerian-Lagrangian coupling
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
    stage1_epochs:       int   = 50
    stage1_lr:           float = 5e-4   # reduced from 1e-3 — prevents escape from good minimum
    stage2_epochs:       int   = 50
    stage2_lr:           float = 5e-4
    stage2_freeze_fluid: bool  = True
    stage3_epochs:       int   = 100
    stage3_lr:           float = 1e-4
    stage4_epochs:       int   = 50
    stage4_lr:           float = 5e-5
    stage4_max_steps:    int   = 20

    # S4: BPTT differentiable rollout loss
    use_bptt_loss:        bool  = True
    bptt_rollout_steps:   int   = 10     # steps to unroll in Stage 4
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
    # S3: KL divergence weight for stochastic decoder
    lambda_kl:         float = 0.001

    # Noise augmentation (Round-1 #9)
    noise_std: float = 3e-4

    # S8: Ensemble UQ
    ensemble_size: int = 5

    # S13: Transfer learning — path to pre-trained GNS checkpoint ('' = disabled)
    pretrained_gns_checkpoint: str = ""

    # ── Physical constants ────────────────────────────────────────────────────
    nu:  float = 1.5e-5   # m^2/s  kinematic viscosity of air
    rho: float = 1.225    # kg/m^3 air density
    g:   float = 9.81     # m/s^2  gravitational acceleration
