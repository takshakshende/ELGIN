"""train.py — Four-stage training pipeline for the CFD-GNN.

Round-2 additions
------------------
S2  d_p0 passed through to model.step() for evaporation tracking.

S3  KL loss from stochastic decoder is retrieved from
    model.lagrangian_gnn._last_kl and added to total_loss().

S4  Differentiable multi-step rollout loss (BPTT).
    Stage 4 now unrolls cfg.bptt_rollout_steps forward passes with
    torch.enable_grad() and backpropagates through all steps.  This is
    the training scheme used by Sanchez-Gonzalez et al. (2020) which is
    essential for long-horizon rollout stability.

    The Stage-4 loss is a weighted average:
        L = (1 - w) * L_one_step  +  w * L_bptt_rollout
    where w = cfg.bptt_loss_weight (default 0.5).

    Gradient checkpointing (torch.utils.checkpoint) is used every 5 steps
    to keep GPU memory bounded during long unrolls.
"""

from __future__ import annotations
import argparse
import json
import pathlib
import random
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Force UTF-8 on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from elgin.model.config    import CfdGNNConfig
from elgin.model.cfd_gnn   import CfdGNN, save_cfd_gnn_checkpoint
from elgin.data.dataset    import CfdGNNDataset, compute_normalisation_stats
from elgin.train.losses    import total_loss


# ---------------------------------------------------------------------------
#  Noise augmentation (Round-1 #9)
# ---------------------------------------------------------------------------

def _add_trajectory_noise(p_hist: torch.Tensor,
                           noise_std: float) -> torch.Tensor:
    """Proportional Gaussian noise on velocity history."""
    if noise_std <= 0.0:
        return p_hist
    vels    = p_hist[:, 1:] - p_hist[:, :-1]
    sigma   = noise_std * vels.norm(dim=-1, keepdim=True) + 1e-8
    vels_n  = vels + torch.randn_like(vels) * sigma
    pos_n   = torch.zeros_like(p_hist)
    pos_n[:, 0] = p_hist[:, 0]
    for t in range(vels_n.shape[1]):
        pos_n[:, t + 1] = pos_n[:, t] + vels_n[:, t]
    return pos_n


# ---------------------------------------------------------------------------
#  Mesh loader
# ---------------------------------------------------------------------------

def load_mesh(mesh_path: pathlib.Path,
              device: torch.device) -> Dict[str, torch.Tensor]:
    d = np.load(mesh_path)
    mesh = {
        "cell_pos":     torch.from_numpy(d["cell_pos"]).to(device),
        "edge_index":   torch.from_numpy(d["edge_index"]).long().to(device),
        "face_normals": torch.from_numpy(d["face_normals"]).to(device),
        "face_areas":   torch.from_numpy(d["face_areas"]).to(device),
        "face_dists":   torch.from_numpy(d["face_dists"]).to(device),
        "cell_volumes": torch.from_numpy(d["cell_volumes"]).to(device),
        "bc_type":      torch.from_numpy(d["bc_type"]).long().to(device),
    }
    E = mesh["edge_index"].shape[1]
    mesh["face_type"] = (
        torch.from_numpy(d["face_type"]).long().to(device)
        if "face_type" in d.files
        else torch.zeros(E, dtype=torch.long, device=device)
    )
    if "d_wall" in d.files:
        mesh["d_wall"] = torch.from_numpy(d["d_wall"]).to(device)
    if "wall_normal" in d.files:
        mesh["wall_normal"] = torch.from_numpy(d["wall_normal"]).to(device)
    if "domain_bounds" in d.files:
        mesh["domain_bounds"] = torch.from_numpy(d["domain_bounds"]).to(device)
    return mesh


# ---------------------------------------------------------------------------
#  S4 — BPTT differentiable rollout loss
# ---------------------------------------------------------------------------

def _bptt_rollout_loss(
    model:     CfdGNN,
    fluid_in:  torch.Tensor,        # (N_cells, 5)
    p_hist:    torch.Tensor,        # (N_part, H+1, dim)
    p_traj:    torch.Tensor,        # (N_part, T, dim) ground-truth future positions
    d_p:       torch.Tensor,        # (N_part,)
    rho_p:     torch.Tensor,
    mesh:      Dict[str, torch.Tensor],
    cfg:       CfdGNNConfig,
    n_steps:   int,
    inlet_cond: Optional[torch.Tensor] = None,
    gt_fluid_traj: Optional[torch.Tensor] = None,   # (T, N_cells, 5) GT fluid
    rollout_noise_std: float = 0.0,                  # noise added between steps
) -> torch.Tensor:
    """Unroll the model n_steps and backpropagate through all steps.

    This implements the BPTT training scheme from Sanchez-Gonzalez et al.
    (2020): each step uses the previous step's predicted state, so errors
    accumulate over the rollout horizon.  The loss is the mean position
    MSE over all steps.

    Gradient checkpointing (every 5 steps) keeps memory usage linear in
    the checkpoint interval rather than in n_steps.

    Args:
        p_traj  : (N_part, T, dim) — ground-truth positions for steps 1..T
    Returns:
        scalar BPTT loss
    """
    import torch.utils.checkpoint as ckpt_util

    fluid      = fluid_in.clone()
    p_hist_cur = p_hist.clone()
    d_p0       = d_p.clone()
    d_p_cur    = d_p.clone()

    bc_values  = fluid_in.clone()
    d_wall     = (mesh["d_wall"] if "d_wall" in mesh
                  else mesh["cell_pos"][:, 1].clamp(min=1e-3))
    p_type     = torch.zeros(p_hist.shape[0], dtype=torch.long,
                             device=fluid_in.device)

    total = torch.tensor(0.0, device=fluid_in.device, requires_grad=False)
    T_avail = min(n_steps, p_traj.shape[1])

    for t in range(T_avail):
        age = torch.full((p_hist_cur.shape[0],),
                         t * cfg.dt, dtype=d_p.dtype, device=d_p.device)

        # When GT fluid is provided, bypass the EulerianGNN so the Lagrangian
        # sees the same raw-GT-interpolated fluid as rollout.py --freeze_fluid.
        _bypass = (gt_fluid_traj is not None)

        def _step_fn(fluid_f, p_hist_f, d_p_f, d_p0_f):
            return model.step(
                fluid_field      = fluid_f,
                cell_pos         = mesh["cell_pos"],
                bc_type          = mesh["bc_type"],
                bc_values        = bc_values,
                edge_index       = mesh["edge_index"],
                face_normals     = mesh["face_normals"],
                face_areas       = mesh["face_areas"],
                face_dists       = mesh["face_dists"],
                cell_volumes     = mesh["cell_volumes"],
                d_wall           = d_wall,
                particle_hist    = p_hist_f,
                particle_type    = p_type,
                d_p              = d_p_f,
                d_p0             = d_p0_f,
                rho_p            = rho_p,
                face_type        = mesh.get("face_type"),
                particle_age     = age,
                inlet_cond       = inlet_cond,
                wall_normal      = mesh.get("wall_normal"),
                domain_bounds    = mesh.get("domain_bounds"),
                bypass_eulerian  = _bypass,
            )

        # Use gradient checkpointing every 5 steps to save memory
        if (t % 5) == 0 and t > 0:
            out = ckpt_util.checkpoint(
                _step_fn, fluid, p_hist_cur, d_p_cur, d_p0,
                use_reentrant=False
            )
        else:
            out = _step_fn(fluid, p_hist_cur, d_p_cur, d_p0)

        p_pred  = out["particle_pos"]
        d_p_cur = out["d_p_new"]

        # Accumulate position MSE at this step
        step_loss = ((p_pred - p_traj[:, t]) ** 2).mean() / (cfg.L_ref ** 2)
        total     = total + step_loss / T_avail

        # Fluid update for the NEXT iteration.
        # When GT fluid trajectory is provided (training matches inference
        # under --freeze_fluid), feed the next GT fluid frame so the
        # Lagrangian always sees a physically consistent input — exactly
        # like rollout.py --freeze_fluid.  Otherwise fall back to the
        # autoregressive Eulerian prediction.
        if gt_fluid_traj is not None and (t + 1) < T_avail:
            fluid = gt_fluid_traj[t]            # GT at time t_0 + t + 1
        else:
            fluid = out["fluid_field"]

        # Inject noise into the predicted position before feeding it back
        # as history.  This simulates accumulated rollout error (covariate
        # shift) so the model learns to be robust to its own prediction
        # errors over long horizons — the core technique from
        # Sanchez-Gonzalez et al. 2020 (GNS paper, Section 3.2).
        if rollout_noise_std > 0.0 and model.training and (t + 1) < T_avail:
            p_pred = p_pred + torch.randn_like(p_pred) * rollout_noise_std

        # Slide particle history (detach old history to limit graph depth)
        p_hist_cur = torch.cat(
            [p_hist_cur[:, 1:].detach(), p_pred.unsqueeze(1)], dim=1
        )

    return total


# ---------------------------------------------------------------------------
#  Single forward pass + loss
# ---------------------------------------------------------------------------

def _process_single(
    model:       CfdGNN,
    fluid_in:    torch.Tensor,
    fluid_tgt:   torch.Tensor,
    p_hist:      torch.Tensor,
    p_tgt:       torch.Tensor,
    d_p:         torch.Tensor,
    rho_p:       torch.Tensor,
    mesh:        Dict[str, torch.Tensor],
    cfg:         CfdGNNConfig,
    compute_pde: bool = False,
    noise_std:   float = 0.0,
    fluid_norm_mean: Optional[torch.Tensor] = None,
    fluid_norm_std:  Optional[torch.Tensor] = None,
    inlet_cond:  Optional[torch.Tensor] = None,
    bypass_eulerian: bool = False,
) -> Dict[str, torch.Tensor]:
    if noise_std > 0.0 and model.training:
        p_hist = _add_trajectory_noise(p_hist, noise_std)

    p_type    = torch.zeros(p_hist.shape[0], dtype=torch.long,
                            device=fluid_in.device)
    bc_values = fluid_in.clone()
    d_wall    = (mesh["d_wall"] if "d_wall" in mesh
                 else mesh["cell_pos"][:, 1].clamp(min=1e-3))
    d_p0      = d_p.clone()   # S2: use current d_p as initial (no prior age info)

    out = model.step(
        fluid_field      = fluid_in,
        cell_pos         = mesh["cell_pos"],
        bc_type          = mesh["bc_type"],
        bc_values        = bc_values,
        edge_index       = mesh["edge_index"],
        face_normals     = mesh["face_normals"],
        face_areas       = mesh["face_areas"],
        face_dists       = mesh["face_dists"],
        cell_volumes     = mesh["cell_volumes"],
        d_wall           = d_wall,
        particle_hist    = p_hist,
        particle_type    = p_type,
        d_p              = d_p,
        d_p0             = d_p0,
        rho_p            = rho_p,
        face_type        = mesh.get("face_type"),
        inlet_cond       = inlet_cond,
        wall_normal      = mesh.get("wall_normal"),
        domain_bounds    = mesh.get("domain_bounds"),
        bypass_eulerian  = bypass_eulerian,
    )

    # S3: retrieve KL loss from stochastic decoder
    kl = getattr(model.lagrangian_gnn, "_last_kl", None)

    return total_loss(
        fluid_pred   = out["fluid_field"],
        fluid_tgt    = fluid_tgt,
        part_pred    = out["particle_pos"],
        part_tgt     = p_tgt,
        nu_t         = out["nu_t"],
        edge_index   = mesh["edge_index"],
        face_normals = mesh["face_normals"],
        face_areas   = mesh["face_areas"],
        face_dists   = mesh["face_dists"],
        cell_volumes = mesh["cell_volumes"],
        cfg          = cfg,
        compute_pde_losses = compute_pde,
        d_p          = d_p,
        rho_p        = rho_p,
        kl_loss      = kl,
        fluid_norm_mean = fluid_norm_mean,
        fluid_norm_std  = fluid_norm_std,
    )


def process_batch(
    model:       CfdGNN,
    batch:       Dict[str, torch.Tensor],
    mesh:        Dict[str, torch.Tensor],
    cfg:         CfdGNNConfig,
    device:      torch.device,
    compute_pde: bool = False,
    noise_std:   float = 0.0,
    fluid_norm_mean: Optional[torch.Tensor] = None,
    fluid_norm_std:  Optional[torch.Tensor] = None,
    bypass_eulerian: bool = False,
) -> Dict[str, torch.Tensor]:
    B         = batch["fluid_in"].shape[0]
    fluid_in  = batch["fluid_in"].to(device)
    fluid_tgt = batch["fluid_tgt"].to(device)
    p_hist    = batch["particle_hist"].to(device)
    p_tgt     = batch["particle_tgt"].to(device)
    d_p       = batch["d_p"].to(device)
    rho_p     = batch["rho_p"].to(device)
    inlet_vel = (batch["inlet_velocity"].to(device)
                 if "inlet_velocity" in batch else None)

    agg: Dict[str, torch.Tensor] = {}
    for b in range(B):
        losses = _process_single(
            model, fluid_in[b], fluid_tgt[b],
            p_hist[b], p_tgt[b], d_p[b], rho_p[b],
            mesh, cfg, compute_pde, noise_std,
            fluid_norm_mean=fluid_norm_mean,
            fluid_norm_std=fluid_norm_std,
            inlet_cond=(inlet_vel[b] if inlet_vel is not None else None),
            bypass_eulerian=bypass_eulerian,
        )
        for k, v in losses.items():
            agg[k] = agg.get(k, v * 0.0) + v / B
    return agg


# ---------------------------------------------------------------------------
#  Stage runners
# ---------------------------------------------------------------------------

def _freeze_fluid(model: CfdGNN, freeze: bool):
    for p in model.eulerian_gnn.parameters():
        p.requires_grad = not freeze
    for p in model.turb_closure.parameters():
        p.requires_grad = not freeze
    for p in model.pressure_proj.parameters():
        p.requires_grad = not freeze


def _run_epoch(
    model, loader, mesh, cfg, device, optimizer,
    train: bool, compute_pde: bool = False, noise_std: float = 0.0,
    use_bptt: bool = False,
    use_gt_fluid_in_bptt: bool = False,
    fluid_norm_mean: Optional[torch.Tensor] = None,
    fluid_norm_std:  Optional[torch.Tensor] = None,
    bypass_eulerian: bool = False,
) -> Dict[str, float]:
    model.train(train)
    ctx = torch.enable_grad() if train else torch.no_grad()
    agg: Dict[str, float] = {}
    n   = 0

    with ctx:
        for batch in loader:
            if train and optimizer:
                optimizer.zero_grad(set_to_none=True)

            losses = process_batch(
                model, batch, mesh, cfg, device,
                compute_pde, noise_std=noise_std if train else 0.0,
                fluid_norm_mean=fluid_norm_mean,
                fluid_norm_std=fluid_norm_std,
                bypass_eulerian=bypass_eulerian,
            )

            # S4: BPTT rollout loss in Stage 4 (computed in train AND val so we
            # have a faithful early-stopping signal for rollout drift; in val
            # mode we are inside torch.no_grad() so no autograd graph is built).
            if use_bptt and cfg.use_bptt_loss:
                B         = batch["fluid_in"].shape[0]
                fluid_in  = batch["fluid_in"].to(device)
                p_hist    = batch["particle_hist"].to(device)
                d_p       = batch["d_p"].to(device)
                rho_p     = batch["rho_p"].to(device)

                # Genuine K-step trajectory target: shape (B, N_part, K, dim).
                # Falls back to the unsqueezed one-step target if the dataset
                # is an older version that only provides ``particle_tgt``.
                if "particle_traj_future" in batch:
                    p_traj_K = batch["particle_traj_future"].to(device)
                else:
                    p_tgt    = batch["particle_tgt"].to(device)
                    p_traj_K = p_tgt.unsqueeze(2)             # (B, N, 1, dim)

                # GT fluid trajectory for "training matches --freeze_fluid
                # inference" mode.  When use_gt_fluid_in_bptt is on we feed
                # this to _bptt_rollout_loss so the Lagrangian sees a
                # physically consistent fluid every step instead of the
                # autoregressive (and rapidly drifting) Eulerian output.
                if use_gt_fluid_in_bptt and "fluid_traj_future" in batch:
                    fluid_traj_K = batch["fluid_traj_future"].to(device)
                else:
                    fluid_traj_K = None

                bptt_acc = torch.tensor(0.0, device=device)
                for b in range(B):
                    bptt_acc = bptt_acc + _bptt_rollout_loss(
                        model, fluid_in[b], p_hist[b],
                        p_traj_K[b],                          # (N_part, K, dim)
                        d_p[b], rho_p[b], mesh, cfg,
                        n_steps=cfg.bptt_rollout_steps,
                        gt_fluid_traj=(fluid_traj_K[b]
                                       if fluid_traj_K is not None else None),
                        # Noise injected between BPTT steps to simulate
                        # long-horizon covariate shift (Sanchez-Gonzalez 2020).
                        # cfg.bptt_rollout_noise (~0.01 m) is ~17× larger than
                        # cfg.noise_std (3e-4 m) to reflect realistic drift.
                        rollout_noise_std=(cfg.bptt_rollout_noise if train else 0.0),
                    ) / B

                w = cfg.bptt_loss_weight
                losses["total"] = (1.0 - w) * losses["total"] + w * bptt_acc
                losses["bptt"]  = bptt_acc

            if train:
                # NaN guard:
                #   1) Skip backward entirely if the forward loss is non-finite
                #      (e.g. BPTT rollout drove a few particles off-mesh, IDW
                #      interpolation produced inf, Lagrangian forward output
                #      a NaN).  Skipping forward prevents corruption.
                #   2) Even if loss is finite, gradients can still be non-finite
                #      after backward; check explicitly because clip_grad_norm_
                #      does NOT sanitise NaNs (only rescales finite gradients).
                if not torch.isfinite(losses["total"]):
                    optimizer.zero_grad(set_to_none=True)
                    print("    [warn] non-finite loss in train batch; "
                          "skipping backward + step")
                else:
                    losses["total"].backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    grads_finite = all(
                        (p.grad is None) or torch.isfinite(p.grad).all()
                        for p in model.parameters() if p.requires_grad
                    )
                    if grads_finite:
                        optimizer.step()
                    else:
                        optimizer.zero_grad(set_to_none=True)
                        print("    [warn] non-finite gradients after backward; "
                              "skipping optimizer.step() to protect weights")

            for k, v in losses.items():
                agg[k] = agg.get(k, 0.0) + float(v)
            n += 1

    return {k: v / max(n, 1) for k, v in agg.items()}


def _print_metrics(metrics: Dict[str, float], prefix: str = ""):
    parts = [f"{k}={v:.4f}" for k, v in metrics.items() if k != "total"]
    print(f"  {prefix}  total={metrics['total']:.4f}  |  " + "  ".join(parts))


# ---------------------------------------------------------------------------
#  Main training function
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        "cuda" if args.device in ("auto", "cuda")
                  and torch.cuda.is_available()
        else "cpu"
    )
    print(f"[CFD-GNN train]  device={device}")

    data_dir  = pathlib.Path(args.data_dir)
    npz_files = sorted(data_dir.glob("case_*.npz"))
    if not npz_files:
        print(f"  [ERROR] No case_*.npz found in {data_dir}"); sys.exit(1)
    print(f"  Found {len(npz_files)} case files.")

    stats = compute_normalisation_stats(npz_files, history_len=args.history_len)

    cfg_kwargs = dict(
        history_length          = args.history_len,
        fluid_hidden            = args.hidden_size,
        particle_hidden         = args.hidden_size,
        fluid_mp_steps          = args.mp_steps,
        particle_mp_steps       = args.mp_steps,
        fluid_mean              = tuple(stats["fluid_mean"].tolist()),
        fluid_std               = tuple(stats["fluid_std"].tolist()),
        vel_mean                = tuple(stats["vel_mean"]),
        vel_std                 = tuple(stats["vel_std"]),
        acc_mean                = tuple(stats["acc_mean"]),
        acc_std                 = tuple(stats["acc_std"]),
        dt                      = args.dt,
        lambda_continuity       = args.lambda_cont,
        lambda_momentum         = args.lambda_mom,
        lambda_turbulence       = args.lambda_turb,
        lambda_particle         = args.lambda_part,
        lambda_angular          = args.lambda_ang,
        lambda_kl               = args.lambda_kl,
        noise_std               = args.noise_std,
        use_lstm_encoder        = args.use_lstm,
        use_symplectic          = args.use_sv,
        use_fluid_attention     = args.use_attn,
        use_graph_transformer   = args.use_gt,
        fluid_attn_heads        = args.attn_heads,
        use_jacobi_precond      = args.use_jacobi,
        use_equivariant_edges   = args.use_equivar,
        use_stochastic_decoder  = args.use_stoch,
        use_saffman_lift        = args.use_saffman,
        use_heterogeneous_graph = args.use_hetero,
        use_brownian_motion     = args.use_brownian,
        use_evaporation         = args.use_evap,
        use_drag_features       = args.use_drag_feat,
        use_gravity             = args.use_gravity,
        use_bptt_loss           = args.use_bptt,
        bptt_rollout_steps      = args.bptt_steps,
        bptt_loss_weight        = args.bptt_weight,
        bptt_rollout_noise      = args.bptt_rollout_noise,
    )
    if args.particle_radius is not None:
        cfg_kwargs["particle_radius"] = float(args.particle_radius)
    cfg = CfdGNNConfig(**cfg_kwargs)

    model_dir = pathlib.Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(
        json.dumps(cfg.__dict__, indent=2, default=str)
    )

    # Genuine K-step BPTT requires K future GT frames per sample.  We size
    # future_len = max(1, bptt_steps) when BPTT is active so the trainer
    # can supervise every unrolled step against ground truth.  With
    # use_bptt=False we keep future_len=1 (legacy 1-step training) to
    # avoid losing samples near the end of each case.
    bptt_future = max(1, int(args.bptt_steps)) if args.use_bptt else 1
    dataset  = CfdGNNDataset(npz_files, history_len=args.history_len,
                              noise_std=0.0,
                              n_particles=args.n_particles if args.n_particles > 0 else None,
                              future_len=bptt_future)
    n_val    = max(1, int(0.15 * len(dataset)))
    n_train  = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size,
                          shuffle=False, num_workers=0)

    mesh_path = pathlib.Path(args.mesh)
    if not mesh_path.exists():
        print(f"  [ERROR] Mesh not found: {mesh_path}"); sys.exit(1)
    mesh = load_mesh(mesh_path, device)
    print(f"  Mesh: {mesh['cell_pos'].shape[0]} cells, "
          f"{mesh['edge_index'].shape[1]} edges.")

    model = CfdGNN(cfg).to(device)

    # S13: optionally load pre-trained GNS weights
    if args.pretrained_gns:
        from elgin.utils.transfer import load_gns_into_lagrangian
        load_gns_into_lagrangian(model.lagrangian_gnn, args.pretrained_gns,
                                  device=device)
        print(f"  Loaded pre-trained GNS weights from {args.pretrained_gns}")

    # Resume from a previous best.pt (typical use: continue into Stage 4
    # after Stages 1-3 have already converged).
    if getattr(args, "resume", ""):
        resume_path = pathlib.Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"--resume checkpoint not found: {resume_path}")
        ckpt = torch.load(str(resume_path), map_location=device)
        state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  Resumed weights from {resume_path}")
        if missing:
            print(f"    [resume] {len(missing)} missing keys (e.g. {missing[:3]})")
        if unexpected:
            print(f"    [resume] {len(unexpected)} unexpected keys "
                  f"(e.g. {unexpected[:3]})")

    total_p = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_p:,}")

    # Pre-build fluid normalisation tensors once
    fluid_mean_t = torch.tensor(stats["fluid_mean"], dtype=torch.float32,
                                device=device)
    fluid_std_t  = torch.tensor(stats["fluid_std"],  dtype=torch.float32,
                                device=device)

    best_val = float("inf")
    history  = []

    def _run_stage(name, n_epochs, lr, freeze_fluid, compute_pde,
                   cg_iters, noise, use_bptt=False,
                   use_gt_fluid_in_bptt=False):
        nonlocal best_val
        # Bypass the Eulerian GNN whenever this stage uses GT fluid as input to
        # the Lagrangian:
        #   • Stage 2 (freeze_fluid=True): Eulerian is frozen and its output
        #     is garbage (mse_fluid ~ 17M for dental-room k/omega).  The
        #     Lagrangian should see GT fluid directly — exactly what
        #     rollout.py --freeze_fluid does at inference.
        #   • Stage 4 BPTT with GT fluid (use_gt_fluid_in_bptt=True): same
        #     reasoning; bypass ensures BPTT gradient matches inference.
        #   • Stage 1 / Stage 3 (freeze_fluid=False, use_gt_fluid_in_bptt=False):
        #     Eulerian runs normally (PDE losses need its output).
        bypass_eulerian = freeze_fluid or use_gt_fluid_in_bptt
        # PDE losses are only meaningful when the Eulerian GNN generates the
        # fluid prediction.  When bypass_eulerian=True the fluid output is the
        # GT input (a constant w.r.t. model parameters), so PDE gradients are
        # exactly zero yet dominate the total loss value by ~10^9×, which
        # miscalibrates AdamW's second-moment estimates and makes the BPTT
        # signal (the only real gradient) effectively invisible.
        effective_pde = compute_pde and not bypass_eulerian
        print(f"\n{'='*60}\n  {name}  (epochs={n_epochs}, lr={lr:.2e}, "
              f"pde={effective_pde}, bptt={use_bptt}, "
              f"bypass_euler={bypass_eulerian})\n{'='*60}")
        _freeze_fluid(model, freeze_fluid)
        model.pressure_proj.set_cg_iters(cg_iters)

        # When the fluid network is frozen the total-loss is dominated by the
        # (constant) mse_fluid term and does not reflect particle learning.
        # Track mse_particle directly so the best checkpoint actually captures
        # the best-trained particle model.  In Stage 4, however, mse_particle
        # is typically saturated (~0) because the one-step Lagrangian was
        # already fit in Stage 2/3, so prefer the rollout BPTT loss as the
        # tracking metric whenever it is active.
        if use_bptt:
            best_metric_key = "bptt"
        elif freeze_fluid:
            best_metric_key = "mse_particle"
        else:
            best_metric_key = "total"
        stage_best = float("inf")

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=lr * 0.01
        )
        for ep in range(1, n_epochs + 1):
            t0 = time.time()
            tr = _run_epoch(
                model, train_dl, mesh, cfg, device, optimizer,
                train=True,  compute_pde=effective_pde,
                noise_std=noise, use_bptt=use_bptt,
                use_gt_fluid_in_bptt=use_gt_fluid_in_bptt,
                fluid_norm_mean=fluid_mean_t, fluid_norm_std=fluid_std_t,
                bypass_eulerian=bypass_eulerian,
            )
            vl = _run_epoch(
                model, val_dl,   mesh, cfg, device, None,
                train=False, compute_pde=effective_pde,
                use_bptt=use_bptt,
                use_gt_fluid_in_bptt=use_gt_fluid_in_bptt,
                fluid_norm_mean=fluid_mean_t, fluid_norm_std=fluid_std_t,
                bypass_eulerian=bypass_eulerian,
            )
            scheduler.step()

            elapsed = time.time() - t0
            print(f"  ep {ep:03d}/{n_epochs}  [{elapsed:.1f}s]", end="  ")
            _print_metrics(tr, "train")
            _print_metrics(vl, "val  ")

            cg_res = model.pressure_proj.last_cg_residuals
            if cg_res:
                print(f"    CG  iters={len(cg_res)}  "
                      f"r0={cg_res[0]:.3e}  rfin={cg_res[-1]:.3e}")

            history.append({"stage": name, "epoch": ep,
                            "train": tr, "val": vl})

            tracked = vl.get(best_metric_key, vl["total"])
            if tracked < stage_best:
                stage_best = tracked
                # Also update the global best when the total loss improves
                if vl["total"] < best_val:
                    best_val = vl["total"]
                save_cfd_gnn_checkpoint(
                    model, str(model_dir / "best.pt"), ep, tracked
                )
                print(f"    *** NEW BEST  val_{best_metric_key}={tracked:.5f} ***")

        save_cfd_gnn_checkpoint(
            model,
            str(model_dir / f"stage_{name.split()[1]}_last.pt"),
            n_epochs, vl["total"]
        )

    best_ckpt = model_dir / "best.pt"

    def _load_best():
        """Reload the best checkpoint saved so far before starting the next stage."""
        if best_ckpt.exists():
            ckpt = torch.load(str(best_ckpt), map_location=device)
            state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
            model.load_state_dict(state)
            print(f"  [inter-stage] Loaded best checkpoint "
                  f"(val={ckpt.get('val_loss', '?'):.5g}) from {best_ckpt.name}")

    if args.stage1_epochs > 0:
        _run_stage("Stage 1 (fluid)", args.stage1_epochs, cfg.stage1_lr,
                   False, False, cfg.pressure_cg_iters, 0.0, False)
        _load_best()   # <-- restore best before Stage 2

    if args.stage2_epochs > 0:
        _run_stage("Stage 2 (particle)", args.stage2_epochs, cfg.stage2_lr,
                   cfg.stage2_freeze_fluid, False,
                   cfg.pressure_cg_iters, cfg.noise_std, False)
        _load_best()   # <-- restore best before Stage 3

    if args.stage3_epochs > 0:
        _run_stage("Stage 3 (PDE)", args.stage3_epochs, cfg.stage3_lr,
                   False, True, cfg.pressure_cg_iters_stage3,
                   cfg.noise_std, False)
        _load_best()   # <-- restore best before Stage 4

    if args.stage4_epochs > 0:
        # S4: BPTT rollout loss active in Stage 4.
        # When --freeze_fluid_stage4 is passed, the Eulerian sub-network is
        # frozen AND the GT fluid trajectory is fed to the Lagrangian at
        # every BPTT step (the same regime as `rollout.py --freeze_fluid`
        # at inference).  This focuses the optimiser entirely on the
        # Lagrangian rollout, and ensures training and inference see the
        # SAME fluid distribution at every step of the unroll.
        _run_stage("Stage 4 (rollout)", args.stage4_epochs, cfg.stage4_lr,
                   bool(args.freeze_fluid_stage4), True,
                   cfg.pressure_cg_iters_stage4,
                   cfg.noise_std * 2.0,
                   use_bptt=cfg.use_bptt_loss,
                   use_gt_fluid_in_bptt=bool(args.freeze_fluid_stage4))

    import json as _json
    (model_dir / "training_history.json").write_text(
        _json.dumps(history, indent=2, default=str)
    )
    print(f"\n[CFD-GNN train] Done.  Best val_total={best_val:.5f}")
    print(f"  Checkpoints: {model_dir}")


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir",    required=True)
    p.add_argument("--mesh",        required=True)
    p.add_argument("--model_dir",   default="models/cfd_gnn")
    p.add_argument("--device",      default="auto")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--batch_size",  type=int,   default=4)
    p.add_argument("--history_len", type=int,   default=5)
    p.add_argument("--hidden_size", type=int,   default=128)
    p.add_argument("--mp_steps",    type=int,   default=8)
    p.add_argument("--dt",          type=float, default=0.01)
    p.add_argument("--noise_std",   type=float, default=3e-4)
    p.add_argument("--particle_radius", type=float, default=None,
                   help="Override particle-graph connectivity radius [m]. "
                        "Default uses the value from CfdGNNConfig (0.10 m).")
    p.add_argument("--stage1_epochs", type=int, default=50)
    p.add_argument("--stage2_epochs", type=int, default=50)
    p.add_argument("--stage3_epochs", type=int, default=100)
    p.add_argument("--stage4_epochs", type=int, default=50)
    p.add_argument("--lambda_cont", type=float, default=0.1)
    p.add_argument("--lambda_mom",  type=float, default=0.05)
    p.add_argument("--lambda_turb", type=float, default=0.02)
    p.add_argument("--lambda_part", type=float, default=1.0)
    p.add_argument("--lambda_ang",  type=float, default=0.01)
    p.add_argument("--lambda_kl",   type=float, default=0.001,
                   help="KL loss weight for stochastic decoder (S3)")
    # Architecture flags
    p.add_argument("--use_lstm",     action="store_true", default=True)
    p.add_argument("--no_lstm",      action="store_false", dest="use_lstm")
    p.add_argument("--use_sv",       action="store_true", default=True)
    p.add_argument("--no_sv",        action="store_false", dest="use_sv")
    p.add_argument("--use_attn",     action="store_true", default=True)
    p.add_argument("--no_attn",      action="store_false", dest="use_attn")
    p.add_argument("--use_gt",       action="store_true", default=True,
                   help="Use Graph Transformer (S5) instead of single-head gate")
    p.add_argument("--no_gt",        action="store_false", dest="use_gt")
    p.add_argument("--attn_heads",   type=int,   default=4)
    p.add_argument("--use_jacobi",   action="store_true", default=True)
    p.add_argument("--no_jacobi",    action="store_false", dest="use_jacobi")
    p.add_argument("--use_equivar",  action="store_true", default=True,
                   help="SE(2) equivariant edge-local frames (S1)")
    p.add_argument("--no_equivar",   action="store_false", dest="use_equivar")
    p.add_argument("--use_stoch",    action="store_true", default=True,
                   help="Stochastic decoder (S3)")
    p.add_argument("--no_stoch",     action="store_false", dest="use_stoch")
    p.add_argument("--use_saffman", action="store_true", default=True,
                   help="Saffman lift force (S6)")
    p.add_argument("--no_saffman",  action="store_false", dest="use_saffman")
    p.add_argument("--use_hetero",   action="store_true", default=True,
                   help="Heterogeneous size-class graph (S9)")
    p.add_argument("--no_hetero",    action="store_false", dest="use_hetero")
    p.add_argument("--use_brownian", action="store_true", default=True,
                   help="Brownian motion for sub-micron particles (S11)")
    p.add_argument("--no_brownian",  action="store_false", dest="use_brownian")
    p.add_argument("--use_evap",     action="store_true", default=True,
                   help="Droplet evaporation model (S2)")
    p.add_argument("--no_evap",      action="store_false", dest="use_evap")
    p.add_argument("--use_drag_feat", action="store_true", default=True,
                   help="Stokes drag as Lagrangian node/edge feature")
    p.add_argument("--no_drag_feat",  action="store_false", dest="use_drag_feat")
    p.add_argument("--use_gravity",  action="store_true", default=True,
                   help="Add gravity vector to predicted particle acceleration")
    p.add_argument("--no_gravity",   action="store_false", dest="use_gravity")
    p.add_argument("--use_bptt",     action="store_true", default=True,
                   help="BPTT rollout loss in Stage 4 (S4)")
    p.add_argument("--no_bptt",      action="store_false", dest="use_bptt")
    p.add_argument("--freeze_fluid_stage4", action="store_true",
                   help="Freeze the Eulerian sub-network during Stage 4 and "
                        "feed the GT fluid field at every BPTT step.  This "
                        "matches the rollout.py --freeze_fluid regime and "
                        "is recommended when (a) inference uses "
                        "--freeze_fluid, or (b) the Lagrangian one-step "
                        "loss has saturated (mse_particle ~ 0) but rollout "
                        "drift is still large.")
    p.add_argument("--bptt_steps",   type=int,   default=10,
                   help="Number of steps to unroll for BPTT (S4)")
    p.add_argument("--bptt_weight",  type=float, default=0.5,
                   help="Weight of BPTT loss vs one-step loss (S4)")
    p.add_argument("--bptt_rollout_noise", type=float, default=0.01,
                   help="Noise std [m] injected into predicted positions between "
                        "BPTT steps to simulate accumulated rollout error "
                        "(covariate shift).  Default 0.01 m ≈ per-step noise "
                        "consistent with ~0.3 m drift over 30 steps.  Set 0 to "
                        "disable.  Much larger than cfg.noise_std (3e-4 m) "
                        "which only perturbs the history encoder input.")
    p.add_argument("--n_particles", type=int, default=0,
                   help="Fixed particle count per sample (0 = auto: use "
                        "minimum across all cases). All cases are sub-sampled "
                        "to this count so batches can be stacked.")
    p.add_argument("--resume", type=str, default="",
                   help="Path to a previous best.pt to load model weights "
                        "from before training starts.  Use this to resume "
                        "training (e.g. continue into Stage 4 with "
                        "--stage1_epochs 0 --stage2_epochs 0 --stage3_epochs 0).")
    p.add_argument("--pretrained_gns", type=str, default="",
                   help="Path to pre-trained GNS checkpoint (S13)")

    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
