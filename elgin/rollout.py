"""rollout.py — Run a full ELGIN simulation (no OpenFOAM required).


Usage examples
--------------
  # Use a real CFD initial condition
  python elgin/rollout.py \\
      --model_dir experiments/elgin_case03/models \\
      --ic_file   experiments/elgin_case03/datasets/case_single.npz \\
      --mesh      experiments/elgin_case03/datasets/mesh_graph.npz \\
      --n_steps   255 \\
      --output    experiments/elgin_case03/results/rollouts

  # Use a synthetic initial condition (test without data)
  python elgin/rollout.py \\
      --model_dir experiments/elgin_case03/models \\
      --mesh      experiments/elgin_case03/datasets/mesh_graph.npz \\
      --synthetic \\
      --n_particles 300 \\
      --u_inlet 0.10 \\
      --output  predictions/synthetic_run
"""

from __future__ import annotations
import argparse
import json
import pathlib
import sys
import time
from typing import Optional

# Force UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

import numpy as np
import torch

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from elgin.model.cfd_gnn import load_cfd_gnn_checkpoint


# ---------------------------------------------------------------------------
#  Initial condition builders
# ---------------------------------------------------------------------------

def _ic_from_npz(npz_path: pathlib.Path, history_len: int = 5,
                 n_particles: int = 0, device=None, seed: int = 42,
                 t0_override: Optional[int] = None):
    """Load initial conditions from an extract_fields.py .npz file.

    Returns
    -------
    fluid0   : (N_cells, 5) tensor
    p_hist   : (N_part, H+1, dim) tensor — only parcels alive throughout
               ``[t0-H, t0]`` are included (so ``next_position`` sees real
               velocities, not zero-padded ghosts).
    d_p      : (N_part,) tensor
    rho_p    : (N_part,) tensor
    info     : dict carrying ``orig_ids``, ``selected_idx``, ``t0``,
               ``alive_mask`` and ``n_total`` for downstream GT matching.
    """
    d  = np.load(npz_path)
    T  = d["fluid_U"].shape[0]
    H  = history_len
    t0 = int(t0_override) if t0_override is not None else min(H, T - 2)
    t0 = max(H, min(t0, T - 2))

    def _fld(tt):
        return np.concatenate([
            d["fluid_U"][tt],
            d["fluid_p"][tt, :, None],
            d["fluid_k"][tt, :, None],
            d["fluid_omega"][tt, :, None],
        ], axis=-1)

    fluid0 = torch.from_numpy(_fld(t0)).float()

    p_pos = np.asarray(d["particle_pos"], dtype=np.float32)        # (T, N, 2)
    N_total = p_pos.shape[1]

    if "particle_alive_mask" in d.files:
        alive = np.asarray(d["particle_alive_mask"], dtype=bool)
    else:
        if np.isnan(p_pos).any():
            alive = ~np.isnan(p_pos).any(axis=-1)
        else:
            alive = np.ones(p_pos.shape[:2], dtype=bool)
            print("  [warn] no particle_alive_mask in IC file — assuming all "
                  "parcels alive (legacy zero-padded extractor).")

    if "orig_ids" in d.files:
        orig_ids = np.asarray(d["orig_ids"], dtype=np.int64)
    else:
        orig_ids = np.arange(N_total, dtype=np.int64)

    diam_raw = np.asarray(d["particle_diam"], dtype=np.float32)
    if diam_raw.ndim == 2:
        with np.errstate(invalid="ignore"):
            diam_const = np.where(np.isnan(diam_raw), 5e-6, diam_raw).mean(axis=0)
    else:
        diam_const = np.where(np.isnan(diam_raw), 5e-6, diam_raw)
    diam_const = diam_const.astype(np.float32)

    dens_raw = np.asarray(d["particle_dens"], dtype=np.float32)
    if dens_raw.ndim == 2:
        with np.errstate(invalid="ignore"):
            dens_const = np.where(np.isnan(dens_raw), 1000.0, dens_raw).mean(axis=0)
    else:
        dens_const = np.where(np.isnan(dens_raw), 1000.0, dens_raw)
    dens_const = np.where(dens_const <= 0.0, 1000.0, dens_const).astype(np.float32)

    # ── Pick parcels alive across the IC history window [t0-H, t0] ─────────
    window = alive[t0 - H: t0 + 1].all(axis=0)
    alive_idx = np.flatnonzero(window)
    if alive_idx.size == 0:
        raise RuntimeError(
            f"No parcels are alive across [t0-H={t0-H}, t0={t0}] in "
            f"{npz_path}. Choose a later t0 or extend the simulation window.")

    if n_particles > 0 and n_particles < alive_idx.size:
        rng = np.random.default_rng(seed)
        sel_local = rng.choice(alive_idx.size, size=n_particles, replace=False)
        sel = np.sort(alive_idx[sel_local])
        print(f"  Sub-sampled parcels: {alive_idx.size} alive → {n_particles}")
    else:
        sel = alive_idx
        if n_particles > 0:
            print(f"  Requested {n_particles} parcels but only "
                  f"{alive_idx.size} alive — using all of them.")

    # Replace NaN with zeros only after we've decided which parcels to keep
    p_pos_clean = np.where(np.isnan(p_pos), 0.0, p_pos)
    p_hist_np = np.transpose(p_pos_clean[t0 - H: t0 + 1, sel, :],
                             (1, 0, 2)).astype(np.float32)
    p_hist = torch.from_numpy(p_hist_np)
    d_p    = torch.from_numpy(diam_const[sel])
    rho_p  = torch.from_numpy(dens_const[sel])

    if device is not None:
        fluid0 = fluid0.to(device)
        p_hist = p_hist.to(device)
        d_p    = d_p.to(device)
        rho_p  = rho_p.to(device)

    info = {
        "orig_ids":     orig_ids[sel],
        "selected_idx": sel,
        "t0":           t0,
        "alive_mask":   alive,
        "n_total":      N_total,
    }
    return fluid0, p_hist, d_p, rho_p, info


def _synthetic_ic(
    n_particles: int,
    history_len: int,
    cfg,
    u_inlet: float = 0.3,
    device=None,
):
    """Build synthetic initial conditions from config parameters."""
    bounds = cfg.domain_bounds
    Lx = bounds[0][1] - bounds[0][0]
    Ly = bounds[1][1] - bounds[1][0]

    # ── Fluid field: uniform inlet flow with random turbulence ────────────────
    # (In practice, load from a CFD snapshot; this is for testing only)
    # Placeholder: all cells get the inlet velocity
    N_cells = 800   # matches synthetic mesh
    fluid0  = torch.zeros(N_cells, 5)
    fluid0[:, 0] = u_inlet        # U_x
    fluid0[:, 3] = 0.01           # k
    fluid0[:, 4] = 10.0           # ω

    # ── Particles injected near nozzle (top-left, x=0.1 m, y=Ly-0.1 m) ────
    rng = np.random.default_rng(0)
    nozzle = np.array([0.1, Ly - 0.1])
    pos_init = nozzle + rng.normal(0, 0.02, (n_particles, 2))
    pos_init = pos_init.clip([0, 0], [Lx, Ly])

    # Particle history: small random walk backwards
    p_hist_np = np.stack([
        pos_init + rng.normal(0, 0.001, pos_init.shape) * (history_len - i)
        for i in range(history_len + 1)
    ], axis=1)                                # (N_p, H+1, 2)

    p_hist = torch.from_numpy(p_hist_np.astype(np.float32))
    d_p    = torch.from_numpy(
        np.abs(rng.normal(5e-6, 2e-6, n_particles)).clip(1e-6, 50e-6).astype(np.float32)
    )
    rho_p  = torch.full((n_particles,), 1000.0)

    if device is not None:
        fluid0 = fluid0.to(device)
        p_hist  = p_hist.to(device)
        d_p     = d_p.to(device)
        rho_p   = rho_p.to(device)

    return fluid0, p_hist, d_p, rho_p


# ---------------------------------------------------------------------------
#  Matched GT writer (same parcels and same time window as the rollout)
# ---------------------------------------------------------------------------

def _write_matched_gt(
    ic_path:     pathlib.Path,
    gt_path:     pathlib.Path,
    ic_info:     dict,
    n_steps:     int,
    history_len: int,
) -> None:
    """Write a ``gt.npz`` whose ``positions`` array matches the rollout
    one-for-one in parcel order and time-step.

    The rollout's frame index ``s`` corresponds to physical time ``t0 + s + 1``
    (the model integrates one step *forward* from the IC each iteration), so
    GT frame ``s`` is taken from ``particle_pos[t0 + s + 1, sel, :]``.

    Frames where a parcel is not alive are filled with NaN so animations and
    metric scripts can ignore them.
    """
    d = np.load(ic_path)
    pos = np.asarray(d["particle_pos"], dtype=np.float32)
    T   = pos.shape[0]
    sel = ic_info["selected_idx"]
    t0  = int(ic_info["t0"])

    end = min(t0 + 1 + n_steps, T)
    gt  = pos[t0 + 1: end, sel, :].copy()        # (T_gt, N_sel, 2)
    if "particle_alive_mask" in d.files:
        am = np.asarray(d["particle_alive_mask"], dtype=bool)
        am_sel = am[t0 + 1: end, sel]            # (T_gt, N_sel)
        gt[~am_sel] = np.nan

    times = (np.asarray(d["times"], dtype=np.float32)[t0 + 1: end]
             if "times" in d.files else None)

    extras = {"positions": gt.astype(np.float32),
              "orig_ids":  ic_info["orig_ids"].astype(np.int64)}
    if times is not None and len(times) == gt.shape[0]:
        extras["times"] = times
    np.savez_compressed(gt_path, **extras)


# ---------------------------------------------------------------------------
#  Clinical metrics
# ---------------------------------------------------------------------------

def compute_clinical_metrics(
    particle_traj: torch.Tensor,   # (T, N_part, dim)
    domain_bounds: tuple,
    breathing_zone: tuple = ((0.5, 3.5), (1.3, 1.8)),   # x and y ranges [m]
    dt: float = 0.01,
) -> dict:
    """Compute clinical exposure metrics from particle trajectory.

    Returns:
        peak_bze_fraction:  max fraction of particles in breathing zone
        integrated_bze:     time-integrated BZE (particle·seconds)
        floor_deposition:   fraction of particles that reached the floor
        wall_deposition:    fraction of particles that reached a wall
    """
    T, N, dim = particle_traj.shape
    traj = particle_traj.cpu().numpy()

    bx_lo, bx_hi = breathing_zone[0]
    by_lo, by_hi = breathing_zone[1]

    # Breathing zone fraction per timestep
    in_bze = (
        (traj[:, :, 0] >= bx_lo) & (traj[:, :, 0] <= bx_hi) &
        (traj[:, :, 1] >= by_lo) & (traj[:, :, 1] <= by_hi)
    )  # (T, N)
    bze_frac = in_bze.mean(axis=1)           # (T,)
    peak_bze = float(bze_frac.max())
    ibze     = float(bze_frac.sum()) * dt    # integral over time

    # Floor deposition: y ≤ 0.01 m
    y_min = domain_bounds[1][0]
    on_floor = (traj[-1, :, 1] <= y_min + 0.01)
    floor_dep = float(on_floor.mean())

    # Wall deposition: within 0.01 m of any boundary
    x_min, x_max = domain_bounds[0]
    y_max = domain_bounds[1][1]
    on_wall = (
        (traj[-1, :, 0] <= x_min + 0.01) |
        (traj[-1, :, 0] >= x_max - 0.01) |
        (traj[-1, :, 1] >= y_max - 0.01)
    )
    wall_dep = float(on_wall.mean())

    return {
        "peak_bze_fraction": peak_bze,
        "integrated_bze_particle_s": ibze,
        "floor_deposition_fraction": floor_dep,
        "wall_deposition_fraction":  wall_dep,
        "breathing_zone": breathing_zone,
    }


# ---------------------------------------------------------------------------
#  Visualisation
# ---------------------------------------------------------------------------

def plot_fluid_field(fluid_field, cell_pos, title="ELGIN Fluid Field",
                     savepath=None):
    try:
        import matplotlib.pyplot as plt
        from matplotlib import cm
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ff = fluid_field.cpu().numpy()
        cp = cell_pos.cpu().numpy() if hasattr(cell_pos, "cpu") else cell_pos
        U_mag = np.sqrt(ff[:, 0]**2 + ff[:, 1]**2)
        sc0 = axes[0].scatter(cp[:, 0], cp[:, 1], c=U_mag, cmap="viridis",
                              s=4, vmin=0)
        plt.colorbar(sc0, ax=axes[0], label="|U| [m/s]")
        axes[0].set_title("Velocity magnitude")
        sc1 = axes[1].scatter(cp[:, 0], cp[:, 1], c=ff[:, 2], cmap="RdBu_r",
                              s=4)
        plt.colorbar(sc1, ax=axes[1], label="p [m²/s²]")
        axes[1].set_title("Pressure")
        fig.suptitle(title)
        plt.tight_layout()
        if savepath:
            plt.savefig(savepath, dpi=150)
        plt.close()
    except ImportError:
        pass


def plot_particle_traj(traj, domain_bounds, title="Particle Trajectories",
                       savepath=None):
    try:
        import matplotlib.pyplot as plt
        traj_np = traj.cpu().numpy() if hasattr(traj, "cpu") else traj
        fig, ax = plt.subplots(figsize=(10, 6))
        bounds = domain_bounds
        ax.set_xlim(bounds[0]); ax.set_ylim(bounds[1])
        # Plot last N_plot particles as faint lines
        N_plot = min(50, traj_np.shape[1])
        for i in range(N_plot):
            ax.plot(traj_np[:, i, 0], traj_np[:, i, 1],
                    alpha=0.3, lw=0.5, color="steelblue")
        # Final positions
        ax.scatter(traj_np[-1, :, 0], traj_np[-1, :, 1],
                   s=4, color="red", zorder=5, label="t=final")
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        ax.set_title(title)
        ax.legend(fontsize=8)
        plt.tight_layout()
        if savepath:
            plt.savefig(savepath, dpi=150)
        plt.close()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
#  Main rollout function
# ---------------------------------------------------------------------------

def run_rollout(args: argparse.Namespace) -> None:
    device = torch.device(
        "cuda" if args.device in ("auto", "cuda") and torch.cuda.is_available()
        else "cpu"
    )
    print(f"[ELGIN rollout]  device={device}")

    # ── Load model ──────────────────────────────────────────────────────────
    model_dir = pathlib.Path(args.model_dir)
    ckpt_path = model_dir / "best.pt"
    if not ckpt_path.exists():
        print(f"  [ERROR] No checkpoint at {ckpt_path}")
        sys.exit(1)
    model = load_cfd_gnn_checkpoint(str(ckpt_path)).to(device)
    model.eval()
    cfg   = model.cfg
    print(f"  Model loaded from {ckpt_path}")

    # ── Load mesh ────────────────────────────────────────────────────────────
    mesh_path = pathlib.Path(args.mesh)
    if not mesh_path.exists():
        print(f"  [ERROR] Mesh not found: {mesh_path}")
        sys.exit(1)
    from elgin.train.train import load_mesh
    mesh = load_mesh(mesh_path, device)
    N_cells = int(mesh["cell_pos"].shape[0])
    print(f"  Mesh: {N_cells} cells")

    # ── Initial conditions ───────────────────────────────────────────────────
    d_wall_np = None   # will be loaded from IC file if available
    ic_info: Optional[dict] = None
    if args.synthetic or args.ic_file is None:
        print(f"  Using synthetic ICs (n_particles={args.n_particles})")
        fluid0, p_hist, d_p, rho_p = _synthetic_ic(
            args.n_particles, cfg.history_length, cfg,
            u_inlet=args.u_inlet, device=device
        )
    else:
        ic_path = pathlib.Path(args.ic_file)
        print(f"  Loading ICs from {ic_path.name}")
        fluid0, p_hist, d_p, rho_p, ic_info = _ic_from_npz(
            ic_path, cfg.history_length,
            n_particles=args.n_particles, device=device,
            t0_override=args.t0,
        )
        # Load the per-cell wall-distance from the preprocessed dataset so it
        # matches exactly what the model saw during training.
        _raw = np.load(ic_path)
        if "d_wall" in _raw:
            d_wall_np = _raw["d_wall"].astype(np.float32)
            print(f"  d_wall loaded from IC file  "
                  f"(range {d_wall_np.min():.3f}–{d_wall_np.max():.3f} m)")
        del _raw

    N_part = p_hist.shape[0]
    print(f"  n_particles={N_part}, n_steps={args.n_steps}")

    # ── Wall-distance feature for the fluid GNN ──────────────────────────────
    if "d_wall" in mesh:
        d_wall = mesh["d_wall"]
    elif d_wall_np is not None:
        d_wall = torch.from_numpy(d_wall_np).to(device)
    else:
        d_wall = mesh["cell_pos"][:, 1].clamp(min=1e-3)

    # Per-case airInlet velocity (constant across the rollout)
    inlet_cond: Optional[torch.Tensor] = None
    if args.ic_file is not None:
        _raw3 = np.load(args.ic_file)
        if "inlet_velocity" in _raw3.files:
            inlet_cond = torch.from_numpy(
                _raw3["inlet_velocity"].astype(np.float32)
            ).to(device)
            print(f"  Inlet conditioning: U_in = "
                  f"({inlet_cond[0].item():.4f}, {inlet_cond[1].item():.4f}) m/s")
        del _raw3

    # Real domain bounds and wall_normal from the mesh graph (preferred)
    domain_bounds_t: Optional[torch.Tensor] = mesh.get("domain_bounds")
    wall_normal_t:   Optional[torch.Tensor] = mesh.get("wall_normal")

    # ── Boundary conditions (held fixed throughout rollout) ──────────────────
    bc_values = fluid0.clone()

    # ── Optional: pre-load ground-truth fluid trajectory for frozen-RANS mode ─
    fluid_gt_traj: Optional[np.ndarray] = None
    if args.freeze_fluid and args.ic_file is not None:
        _raw2 = np.load(args.ic_file)
        _fU   = _raw2["fluid_U"].astype(np.float32)    # (T, N_cells, 2)
        _fp   = _raw2["fluid_p"].astype(np.float32)    # (T, N_cells)
        _fk   = _raw2["fluid_k"].astype(np.float32)    # (T, N_cells)
        _fw   = _raw2["fluid_omega"].astype(np.float32) # (T, N_cells)
        fluid_gt_traj = np.concatenate([
            _fU,
            _fp[:, :, None],
            _fk[:, :, None],
            _fw[:, :, None],
        ], axis=-1)  # (T, N_cells, 5)
        del _raw2, _fU, _fp, _fk, _fw
        print(f"  Frozen-RANS mode: using GT fluid field from {pathlib.Path(args.ic_file).name}")

    # ── Rollout ──────────────────────────────────────────────────────────────
    print(f"\n  Running {args.n_steps} steps ...")
    t0 = time.time()

    if args.freeze_fluid:
        # ── Frozen-RANS rollout ───────────────────────────────────────────────
        # Keep the fluid field fixed (or stepped through the GT trajectory)
        # and only run the Lagrangian GNN for particle tracking.
        # This bypasses the unstable autoregressive Eulerian update and isolates
        # the particle-prediction quality.
        H       = model.cfg.history_length
        _H      = p_hist.shape[1] - 1
        p_cur   = p_hist.clone()
        d_p_cur = d_p.clone()
        d_p0    = d_p.clone()
        deposited = torch.zeros(N_part, dtype=torch.bool,   device=device)
        depo_step = torch.full((N_part,), -1, dtype=torch.long, device=device)
        depo_pos  = torch.zeros(N_part, p_hist.shape[-1],
                                dtype=p_hist.dtype, device=device)
        if domain_bounds_t is not None:
            bounds = domain_bounds_t.to(device=device, dtype=p_hist.dtype)
        else:
            bounds = torch.tensor(model.cfg.domain_bounds,
                                  dtype=p_hist.dtype, device=device)
        tol    = model.cfg.deposition_wall_tol

        fluid_traj_out    = []
        particle_traj_out = []
        dp_traj_out       = []

        # IC frame index in the GT fluid trajectory (same t0 used in _ic_from_npz)
        T_gt = fluid_gt_traj.shape[0] if fluid_gt_traj is not None else 1
        if ic_info is not None:
            ic_t0 = max(H, min(int(ic_info["t0"]), T_gt - 2))
        else:
            ic_t0 = min(H, T_gt - 2)

        with torch.no_grad():
            for step_idx in range(args.n_steps):
                # Choose fluid field: GT frame if available, else freeze at IC
                if fluid_gt_traj is not None:
                    gt_idx = min(ic_t0 + step_idx, T_gt - 1)
                    fluid = torch.from_numpy(fluid_gt_traj[gt_idx]).to(device)
                else:
                    fluid = fluid0

                # Interpolate fluid velocity to particle positions
                from elgin.model.cfd_gnn import interpolate_fluid_to_particles
                U_at_p = interpolate_fluid_to_particles(
                    fluid[:, :model.cfg.dim],
                    mesh["cell_pos"],
                    p_cur[:, -1],
                    k_nearest=model.cfg.cross_k_nearest,
                )
                k_at_p = interpolate_fluid_to_particles(
                    fluid[:, 3:4],
                    mesh["cell_pos"],
                    p_cur[:, -1],
                    k_nearest=model.cfg.cross_k_nearest,
                ).squeeze(-1) if model.cfg.use_tke_dispersion else None

                from elgin.model.cfd_gnn import gravity_vector
                g_vec = gravity_vector(model.cfg.g, model.cfg.dim,
                                       p_cur.dtype, device)
                age = torch.full((N_part,), step_idx * model.cfg.dt,
                                 dtype=d_p.dtype, device=device)

                from elgin.model.physics import evaporation_diameter
                if model.cfg.use_evaporation:
                    d_p_cur = evaporation_diameter(
                        d_p0=d_p0, age=age, rho_p=rho_p,
                        D_v=model.cfg.evap_D_v, B_M=model.cfg.evap_B_M,
                    )

                # Lagrangian wall awareness: interpolate d_wall + wall_normal
                # from the cell graph to each parcel position.
                wall_feat_p = None
                if (getattr(model.cfg, "use_wall_features_lag", False)
                        and wall_normal_t is not None):
                    d_at_p = interpolate_fluid_to_particles(
                        d_wall.unsqueeze(-1), mesh["cell_pos"], p_cur[:, -1],
                        k_nearest=model.cfg.cross_k_nearest,
                    )
                    n_at_p = interpolate_fluid_to_particles(
                        wall_normal_t, mesh["cell_pos"], p_cur[:, -1],
                        k_nearest=model.cfg.cross_k_nearest,
                    )
                    wall_feat_p = torch.cat([d_at_p, n_at_p], dim=-1)

                p_next = model.lagrangian_gnn.next_position(
                    p_cur,
                    torch.zeros(N_part, dtype=torch.long, device=device),
                    U_at_p, d_p_cur, d_p0, rho_p,
                    g_vec=g_vec, k_fluid=k_at_p, du_dy=None,
                    wall_feat_p=wall_feat_p,
                )

                # Clip to actual mesh bounds
                p_next = p_next.clamp(min=bounds[:, 0], max=bounds[:, 1])

                # Deposition: combine bounding-box distance with the true
                # cell d_wall (which already includes the dentist / patient
                # obstacles), so parcels that drift towards the dentist body
                # actually deposit instead of passing through.
                # Skip entirely when --no_deposition is requested (diagnostic).
                if not getattr(args, "no_deposition", False):
                    d_at_p_dep = interpolate_fluid_to_particles(
                        d_wall.unsqueeze(-1), mesh["cell_pos"], p_next,
                        k_nearest=model.cfg.cross_k_nearest,
                    ).squeeze(-1)
                    box_dist = torch.stack([
                        (p_next - bounds[:, 0]).min(dim=-1).values,
                        (bounds[:, 1] - p_next).min(dim=-1).values,
                    ], dim=-1).min(dim=-1).values
                    dist_to_walls = torch.minimum(d_at_p_dep, box_dist)
                    newly_dep            = (~deposited) & (dist_to_walls < tol)
                    depo_step[newly_dep] = step_idx
                    depo_pos[newly_dep]  = p_next[newly_dep]
                    deposited            = deposited | newly_dep
                    p_next[deposited]    = depo_pos[deposited]

                p_cur = torch.cat([p_cur[:, 1:], p_next.unsqueeze(1)], dim=1)
                fluid_traj_out.append(fluid.cpu())
                particle_traj_out.append(p_next.cpu())
                dp_traj_out.append(d_p_cur.cpu())

        result = {
            "fluid_traj":      torch.stack(fluid_traj_out,    dim=0),
            "particle_traj":   torch.stack(particle_traj_out, dim=0),
            "dp_traj":         torch.stack(dp_traj_out,       dim=0),
            "deposition_step": depo_step,
            "deposition_pos":  depo_pos,
        }
    else:
        result = model.rollout(
            fluid_field0   = fluid0,
            cell_pos       = mesh["cell_pos"],
            bc_type        = mesh["bc_type"],
            bc_values      = bc_values,
            edge_index     = mesh["edge_index"],
            face_normals   = mesh["face_normals"],
            face_areas     = mesh["face_areas"],
            face_dists     = mesh["face_dists"],
            cell_volumes   = mesh["cell_volumes"],
            d_wall         = d_wall,
            particle_hist0 = p_hist,
            particle_type  = torch.zeros(N_part, dtype=torch.long, device=device),
            d_p            = d_p,
            rho_p          = rho_p,
            n_steps        = args.n_steps,
            face_type      = mesh.get("face_type"),
            inlet_cond     = inlet_cond,
            wall_normal    = wall_normal_t,
            domain_bounds  = domain_bounds_t,
        )
    elapsed = time.time() - t0
    print(f"  Rollout complete in {elapsed:.1f}s  ({args.n_steps/elapsed:.1f} steps/s)")

    # ── Clinical metrics ─────────────────────────────────────────────────────
    metrics = compute_clinical_metrics(
        result["particle_traj"], cfg.domain_bounds, dt=cfg.dt
    )
    print("\n  Clinical Metrics:")
    print(f"    Peak BZE fraction:  {metrics['peak_bze_fraction']:.3f}")
    print(f"    Integrated BZE:     {metrics['integrated_bze_particle_s']:.3f} particle.s")
    print(f"    Floor deposition:   {metrics['floor_deposition_fraction']:.3f}")
    print(f"    Wall deposition:    {metrics['wall_deposition_fraction']:.3f}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    out_dir = pathlib.Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    rollout_extras: dict = {}
    if ic_info is not None:
        rollout_extras["orig_ids"]    = ic_info["orig_ids"].astype(np.int64)
        rollout_extras["selected_idx"] = ic_info["selected_idx"].astype(np.int64)
        rollout_extras["ic_t0"]       = np.int64(ic_info["t0"])

    # Build a physical-time vector for the predicted trajectory.  Prefer the
    # IC file's "times" array (preserves the case's actual sampling); fall
    # back to t = step_idx * cfg.dt if not available.  This makes the
    # animation script display real seconds instead of frame indices.
    n_pred_frames = int(result["particle_traj"].shape[0])
    times_pred: Optional[np.ndarray] = None
    if ic_info is not None:
        try:
            ic_times_full = np.load(args.ic_file, allow_pickle=False)
            if "times" in ic_times_full.files:
                t_full = np.asarray(ic_times_full["times"], dtype=np.float32)
                t0     = int(ic_info["t0"])
                end    = min(t0 + 1 + n_pred_frames, t_full.shape[0])
                times_pred = t_full[t0 + 1: end].astype(np.float32)
                if times_pred.shape[0] != n_pred_frames:
                    times_pred = None    # length mismatch -> fall back below
        except Exception:
            times_pred = None
    if times_pred is None:
        # Fall back to model dt
        times_pred = (np.arange(n_pred_frames, dtype=np.float32)
                      * float(cfg.dt))
    rollout_extras["times"] = times_pred

    np.savez_compressed(
        out_dir / "rollout.npz",
        fluid_traj    = result["fluid_traj"].numpy(),
        particle_traj = result["particle_traj"].numpy(),
        cell_pos      = mesh["cell_pos"].cpu().numpy(),
        **rollout_extras,
    )

    # ── Matched GT NPZ (same parcels, same time-window) ─────────────────────
    if ic_info is not None and not args.no_gt:
        gt_path = out_dir / "gt.npz"
        _write_matched_gt(
            ic_path     = pathlib.Path(args.ic_file),
            gt_path     = gt_path,
            ic_info     = ic_info,
            n_steps     = args.n_steps,
            history_len = cfg.history_length,
        )
        print(f"  GT written to {gt_path}")

    (out_dir / "clinical_metrics.json").write_text(
        json.dumps({"clinical_metrics": metrics, "n_steps": args.n_steps,
                    "elapsed_s": elapsed}, indent=2)
    )

    # ── Plots ─────────────────────────────────────────────────────────────────
    if not args.no_plots:
        plot_fluid_field(
            result["fluid_traj"][-1], mesh["cell_pos"].cpu(),
            title=f"ELGIN Fluid Field at t={args.n_steps*cfg.dt:.1f}s",
            savepath=str(out_dir / "fluid_field_final.png")
        )
        plot_particle_traj(
            result["particle_traj"], cfg.domain_bounds,
            title=f"Particle Trajectories ({N_part} particles, {args.n_steps} steps)",
            savepath=str(out_dir / "particle_trajectories.png")
        )
        print(f"\n  Plots saved to {out_dir}/")

    print(f"\n  Outputs saved to {out_dir}/")
    print(f"    rollout.npz       — full trajectories")
    print(f"    clinical_metrics.json")
    if not args.no_plots:
        print(f"    fluid_field_final.png")
        print(f"    particle_trajectories.png")


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_dir",   type=str, required=True)
    parser.add_argument("--mesh",        type=str, required=True)
    parser.add_argument("--ic_file",     type=str, default=None,
                        help="Path to case_XX.npz for real initial conditions.")
    parser.add_argument("--synthetic",   action="store_true",
                        help="Use synthetic initial conditions.")
    parser.add_argument("--n_particles", type=int, default=1000,
                        help="Number of particles for rollout. If the IC file "
                             "has more particles, they are randomly sub-sampled "
                             "to this count (avoids GPU OOM). Use 0 to keep all.")
    parser.add_argument("--n_steps",     type=int, default=200)
    parser.add_argument("--u_inlet",     type=float, default=0.3,
                        help="Inlet velocity magnitude [m/s].")
    parser.add_argument("--output",      type=str, default="predictions/elgin_run")
    parser.add_argument("--device",      type=str, default="auto")
    parser.add_argument("--freeze_fluid", action="store_true",
                        help="Fix the RANS fluid field at IC values (step through GT "
                             "fluid trajectory if --ic_file is given). Bypasses the "
                             "unstable autoregressive Eulerian GNN update and isolates "
                             "the Lagrangian particle-tracking quality.")
    parser.add_argument("--t0", type=int, default=None,
                        help="Frame index used as the rollout initial condition. "
                             "Default: history_length (the earliest valid frame).")
    parser.add_argument("--no_gt", action="store_true",
                        help="Disable the automatic ID-matched gt.npz writer.")
    parser.add_argument("--no_deposition", action="store_true",
                        help="Disable the geometry-aware deposition lock-in "
                             "(parcels never get pinned at walls).  Useful "
                             "for diagnosing whether a wall-clinging "
                             "rollout pattern is caused by deposition or "
                             "by biased Lagrangian predictions.")
    parser.add_argument("--no_plots",    action="store_true")
    args = parser.parse_args()
    run_rollout(args)

if __name__ == "__main__":
    main()
