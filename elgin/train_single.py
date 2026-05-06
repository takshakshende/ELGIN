"""train_single.py — Train CFD-GNN on a single OpenFOAM case.

Runs the full extract → mesh-build → train pipeline for ONE case only.
Useful for quick experiments, debugging, or verifying the framework before
committing to a 20-case training run.

With default settings (100 total epochs split across 4 stages) training
finishes in approximately 30–60 minutes on an NVIDIA GPU, or 2–4 hours on CPU.

Usage examples
--------------
    # Minimal — point at one OpenFOAM case directory
    python cfd_gnn/train_single.py \\
        --case_dir openfoam/Sweep_Case_03 \\
        --device   cuda

    # Fast smoke-test with synthetic data (no OpenFOAM needed)
    python cfd_gnn/train_single.py --synthetic

    # Custom epoch budget and output location
    python cfd_gnn/train_single.py \\
        --case_dir openfoam/Sweep_Case_03 \\
        --epochs   100 \\
        --model_dir models/cfd_gnn_single \\
        --device   cuda

    # Skip extraction if case_single.npz already exists
    python cfd_gnn/train_single.py \\
        --case_dir openfoam/Sweep_Case_03 \\
        --skip_extract \\
        --device cuda

Stage epoch budget (sums to --epochs):
    Stage 1  fluid pre-training          10 %  (short — rollout uses
                                                 --freeze_fluid by default)
    Stage 2  particle supervised         50 %  (one-step parcel MSE)
    Stage 3  PDE-informed joint          10 %  (continuity / momentum / k-omega)
    Stage 4  BPTT rollout fine-tuning    30 %  (Sanchez-Gonzalez 2020 fix)

By default the model uses LSTM history encoding, the symplectic integrator,
SE(2)-equivariant edges and the BPTT rollout loss.  The stochastic decoder,
Saffman lift, Brownian motion, evaporation and heterogeneous-graph features
are off (pass --full_model to also turn those on).
"""

from __future__ import annotations
import argparse
import pathlib
import subprocess
import sys
import time

# Force UTF-8 on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

_PYTHON     = sys.executable
_ROOT       = pathlib.Path(__file__).resolve().parent.parent
_CFD_GNN    = pathlib.Path(__file__).resolve().parent

# Default paths (overridable via CLI)
_DEFAULT_OUT     = _ROOT / "datasets" / "cfd_gnn_single"
_DEFAULT_MODEL   = _ROOT / "models"   / "cfd_gnn_single"
_DEFAULT_RESULTS = _ROOT / "results"  / "cfd_gnn_single"


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _hline(title: str = "") -> None:
    w = 60
    if title:
        pad = (w - len(title) - 2) // 2
        print("=" * pad + f" {title} " + "=" * pad)
    else:
        print("=" * w)


def _run(cmd: list[str], log_path: pathlib.Path | None = None,
         check: bool = True) -> int:
    """Run a subprocess, tee-ing output to terminal and log file live."""
    import os as _os
    env = _os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, env=env
    )
    if log_path:
        with open(log_path, "w", encoding="utf-8", errors="replace") as lf:
            for line in proc.stdout:
                print(line, end="", flush=True)
                lf.write(line)
                lf.flush()
    else:
        for line in proc.stdout:
            print(line, end="", flush=True)
    proc.wait()
    if check and proc.returncode != 0:
        print(f"\n[FAILED] exit code {proc.returncode}")
        if log_path:
            print(f"  Log: {log_path}")
        sys.exit(proc.returncode)
    return proc.returncode


# ---------------------------------------------------------------------------
#  Stage 1 — Extract field data
# ---------------------------------------------------------------------------

def stage_extract(args: argparse.Namespace, out_dir: pathlib.Path) -> pathlib.Path:
    npz_path = out_dir / "case_single.npz"

    if args.skip_extract and npz_path.exists():
        print(f"  [skip] {npz_path.name} already exists.")
        return npz_path

    if args.synthetic:
        # Generate synthetic data without OpenFOAM
        cmd = [
            _PYTHON, str(_CFD_GNN / "data" / "extract_fields.py"),
            "--case_dir", str(args.case_dir or _ROOT / "openfoam" / "Sweep_Case_01"),
            "--output",   str(npz_path),
            "--t_start",  str(args.t_start),
            "--t_end",    str(args.t_end),
            "--synthetic",
        ]
    else:
        if args.case_dir is None:
            print("[ERROR] --case_dir is required (or pass --synthetic)")
            sys.exit(1)
        cmd = [
            _PYTHON, str(_CFD_GNN / "data" / "extract_fields.py"),
            "--case_dir", str(args.case_dir),
            "--output",   str(npz_path),
            "--t_start",  str(args.t_start),
            "--t_end",    str(args.t_end),
            "--dt_keep",  str(args.dt_keep),
        ]
        if args.n_particles:
            cmd += ["--n_particles", str(args.n_particles)]

    log = _DEFAULT_RESULTS / "logs" / "extract_single.log"
    _run(cmd, log)
    return npz_path


# ---------------------------------------------------------------------------
#  Stage 2 — Build mesh graph
# ---------------------------------------------------------------------------

def stage_mesh(args: argparse.Namespace, out_dir: pathlib.Path,
               case_dir: pathlib.Path) -> pathlib.Path:
    mesh_path = out_dir / "mesh_graph.npz"

    if args.skip_mesh and mesh_path.exists():
        print(f"  [skip] {mesh_path.name} already exists.")
        return mesh_path

    cmd = [
        _PYTHON, str(_CFD_GNN / "data" / "mesh_to_graph.py"),
        "--case_dir", str(case_dir),
        "--output",   str(mesh_path),
    ]
    log = _DEFAULT_RESULTS / "logs" / "mesh_single.log"
    _run(cmd, log)
    return mesh_path


# ---------------------------------------------------------------------------
#  Stage 3 — Train
# ---------------------------------------------------------------------------

def stage_train(
    args:      argparse.Namespace,
    data_dir:  pathlib.Path,
    mesh_path: pathlib.Path,
    model_dir: pathlib.Path,
) -> None:
    # Distribute total epoch budget across the 4 training stages.
    # New split: 10 / 50 / 10 / 30  (sums to 100 %).
    #   Stage 1 (fluid pre-training) is kept short because the rollout uses
    #   --freeze_fluid (GT fluid) — Stage 1 is mostly a warm-start.
    #   Stage 2 (one-step particle MSE) is the workhorse.
    #   Stage 3 (PDE-informed) is only useful with non-zero λ weights.
    #   Stage 4 (BPTT rollout fine-tuning) is the standard fix for
    #   autoregressive drift (Sanchez-Gonzalez et al. 2020).
    n = args.epochs
    start_stage = getattr(args, "start_stage", 1)

    if start_stage >= 4:
        # Resume directly into Stage 4 using the existing best.pt checkpoint.
        # Stages 1-3 are skipped; all epochs go to the BPTT rollout stage.
        s1, s2, s3 = 0, 0, 0
        s4 = n
        print(f"  [start_stage=4] Skipping Stages 1-3.  "
              f"Stage4={s4} epochs (resuming from best.pt)")
    else:
        # Stage split optimised for single-case training:
        #   Stage 1 (fluid warm-start) — short; rollout uses --freeze_fluid so
        #           Eulerian quality matters less than Lagrangian accuracy.
        #   Stage 2 (one-step particle MSE) — moderate; enough to initialise the
        #           Lagrangian before BPTT.  With a single case it overfits quickly.
        #   Stage 3 (PDE-informed) — short pass to activate physics constraints.
        #   Stage 4 (BPTT rollout) — majority of the budget.  With
        #           --freeze_fluid_stage4 this aligns training with inference
        #           (--freeze_fluid at rollout) and directly minimises long-horizon
        #           rollout error.
        s1 = max(1, round(n * 0.10))
        s2 = max(1, round(n * 0.25))
        s3 = max(0, round(n * 0.10))
        s4 = max(1, round(n * 0.55))
        print(f"  Epoch budget: Stage1={s1}  Stage2={s2}  Stage3={s3}  Stage4={s4}  "
              f"(total={s1+s2+s3+s4})")

    import os
    env = os.environ.copy()
    # Reduce CUDA memory fragmentation on Windows/4 GB GPUs
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # Force Python unbuffered I/O so train.py flushes each print() immediately
    # instead of accumulating 8 KB blocks before the parent process sees output.
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [
        _PYTHON, "-u", str(_CFD_GNN / "train" / "train.py"),
        "--data_dir",       str(data_dir),
        "--mesh",           str(mesh_path),
        "--model_dir",      str(model_dir),
        "--device",         args.device,
        "--batch_size",     str(args.batch_size),
        "--hidden_size",    str(args.hidden_size),
        "--mp_steps",       str(args.mp_steps),
        "--noise_std",      str(args.noise_std),
        "--n_particles",    str(args.n_particles),
        "--stage1_epochs",  str(s1),
        "--stage2_epochs",  str(s2),
        "--stage3_epochs",  str(s3),
        "--stage4_epochs",  str(s4),
        # Pass the frame-interval as the model dt so position integration and
        # physics residuals use the correct time step. dt_keep (default 0.1 s)
        # is the interval between consecutive dataset frames.
        "--dt",             str(args.dt_keep),
        # PDE-residual weights for Stage 3.  mse_fluid is now computed in
        # normalised space (in losses.py / dataset.py) so these are
        # comparable to the supervised term.  Set to 0 to disable.
        "--lambda_mom",     str(args.lambda_mom),
        "--lambda_cont",    str(args.lambda_cont),
        "--lambda_turb",    str(args.lambda_turb),
        # BPTT rollout fine-tuning (Stage 4)
        "--bptt_steps",         str(args.bptt_steps),
        "--bptt_weight",        str(args.bptt_weight),
        "--bptt_rollout_noise", str(args.bptt_rollout_noise),
        # Lagrangian connectivity radius (default 0.10 m — see config.py)
        "--particle_radius", str(args.particle_radius),
    ]

    # When starting directly from Stage 4, load the existing best.pt so the
    # Lagrangian begins from the Stage-3-trained weights, not random init.
    if start_stage >= 4:
        best_ckpt = model_dir / "best.pt"
        if best_ckpt.exists():
            cmd += ["--resume", str(best_ckpt)]
            print(f"  [start_stage=4] Resuming from {best_ckpt}")
        else:
            print(f"  [WARNING] --start_stage 4 requested but no best.pt found in "
                  f"{model_dir}. Starting from random init.")

    # Align Stage 4 BPTT training with --freeze_fluid inference:
    # When rollout.py uses --freeze_fluid the Lagrangian sees GT fluid at every
    # step.  Stage 4 BPTT must match this so the gradient signal is meaningful.
    # Without this flag, BPTT trains the Lagrangian against a drifting
    # autoregressive Eulerian that it never encounters at inference — wasting
    # all Stage 4 epochs on the wrong distribution.
    if getattr(args, "freeze_fluid_stage4", False):
        cmd += ["--freeze_fluid_stage4"]

    # Default model now keeps:
    #   * LSTM history encoder    (--use_lstm,    on by default)
    #   * Symplectic integrator   (--use_sv,      on by default)
    #   * SE(2)-equivariant edges (--use_equivar, on by default)
    #   * BPTT loss (Stage 4)     (--use_bptt,    on by default)
    # Disable the still-experimental physics that need more data:
    #   * Stochastic decoder, Saffman lift, Brownian, evaporation,
    #     heterogeneous graph.
    # Pass --full_model to also enable the experimental physics stack.
    if not args.full_model:
        cmd += [
            "--no_stoch",       # deterministic decoder
            "--no_saffman",     # secondary effect
            "--no_brownian",    # sub-micron only
            "--no_evap",        # simplifies architecture
            "--no_hetero",      # single particle-type embedding
        ]

    log = _DEFAULT_RESULTS / "logs" / "train_single.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, env=env
    )
    # Write log incrementally so both the terminal and the log file update
    # after every line — not just when training finishes.
    with open(log, "w", encoding="utf-8", errors="replace") as log_f:
        for line in proc.stdout:
            print(line, end="", flush=True)
            log_f.write(line)
            log_f.flush()
    proc.wait()
    if proc.returncode != 0:
        print(f"\n[FAILED] exit code {proc.returncode}")
        print(f"  Log: {log}")
        sys.exit(proc.returncode)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Data ────────────────────────────────────────────────────────────────
    ap.add_argument("--case_dir", type=pathlib.Path, default=None,
                    help="Path to a single OpenFOAM case directory "
                         "(e.g. openfoam/Sweep_Case_03).")
    ap.add_argument("--synthetic", action="store_true",
                    help="Use synthetic data — no OpenFOAM installation required.")
    ap.add_argument("--t_start",  type=float, default=2.0,
                    help="Start time for extraction window (s). Default 2.0.")
    ap.add_argument("--t_end",    type=float, default=28.0,
                    help="End time for extraction window (s). Default 28.0.")
    ap.add_argument("--dt_keep",  type=float, default=0.1,
                    help="Sampling interval for extraction (s). Default 0.1.")
    ap.add_argument("--n_particles", type=int, default=3000,
                    help="Subsample to this many particles per timestep. "
                         "Default 3000 balances accuracy and GPU memory. "
                         "Reduce to 1000 for CPU or 4 GB GPU.")

    # ── Output directories ──────────────────────────────────────────────────
    ap.add_argument("--out_dir",   type=pathlib.Path, default=_DEFAULT_OUT,
                    help="Directory for extracted .npz data.")
    ap.add_argument("--model_dir", type=pathlib.Path, default=_DEFAULT_MODEL,
                    help="Directory where model checkpoints are saved.")

    # ── Training ────────────────────────────────────────────────────────────
    ap.add_argument("--epochs",      type=int,   default=100,
                    help="Total epoch budget split across 4 training stages. "
                         "Default 100 (≈30–60 min on GPU).")
    ap.add_argument("--batch_size",  type=int,   default=2)
    ap.add_argument("--hidden_size", type=int,   default=64,
                    help="Latent dimension for GNN layers. Default 64 for "
                         "4 GB GPU. Use 128 on ≥8 GB GPU (--hidden_size 128).")
    ap.add_argument("--mp_steps",    type=int,   default=4,
                    help="Message-passing steps per forward pass. Default 4 "
                         "for 4 GB GPU. Use 8 on ≥8 GB GPU (--mp_steps 8).")
    ap.add_argument("--noise_std",   type=float, default=3e-4)
    ap.add_argument("--device",      default="auto",
                    choices=["auto", "cuda", "cpu"])
    ap.add_argument("--full_model",  action="store_true",
                    help="Also enable the still-experimental physics "
                         "(stochastic decoder, Saffman lift, Brownian, "
                         "evaporation, heterogeneous graph). LSTM, "
                         "symplectic integration, equivariant edges and "
                         "BPTT are on by default.")
    # PDE residual weights for Stage 3
    ap.add_argument("--lambda_mom",  type=float, default=0.05,
                    help="Stage 3 momentum-residual weight. Default 0.05.")
    ap.add_argument("--lambda_cont", type=float, default=0.10,
                    help="Stage 3 continuity-residual weight. Default 0.10.")
    ap.add_argument("--lambda_turb", type=float, default=0.02,
                    help="Stage 3 turbulence-residual weight. Default 0.02.")
    # BPTT rollout fine-tuning (Stage 4)
    ap.add_argument("--bptt_steps",  type=int,   default=10,
                    help="Number of unrolled steps in the Stage 4 BPTT loss.")
    ap.add_argument("--bptt_weight", type=float, default=0.5,
                    help="Weight of the BPTT rollout loss vs. the one-step "
                         "supervised loss in Stage 4.")
    ap.add_argument("--bptt_rollout_noise", type=float, default=0.01,
                    help="Noise std [m] injected between BPTT steps to simulate "
                         "long-horizon drift (default 0.01 m).  Set 0 to disable.")
    # Lagrangian graph connectivity radius
    ap.add_argument("--particle_radius", type=float, default=0.10,
                    help="Lagrangian graph connectivity radius [m]. "
                         "Default 0.10 m.")

    # ── Skip flags ──────────────────────────────────────────────────────────
    ap.add_argument("--freeze_fluid_stage4", action="store_true",
                    help="Freeze the Eulerian sub-network during Stage 4 BPTT "
                         "and feed the GT fluid field at every unrolled step. "
                         "REQUIRED when rollout.py is invoked with --freeze_fluid "
                         "(the default in run_case03.ps1). Aligns training with "
                         "inference so Stage 4 BPTT directly minimises rollout "
                         "error under the same fluid regime seen at test time.")
    ap.add_argument("--start_stage", type=int, default=1, choices=[1, 2, 3, 4],
                    help="Start training directly from this stage (1-4). "
                         "Stage 4 skips Stages 1-3 entirely and resumes from "
                         "the existing best.pt checkpoint in --model_dir. "
                         "Use after a full Stages 1-3 run to re-run Stage 4 "
                         "with different bptt_steps without re-running earlier stages.")
    ap.add_argument("--skip_extract", action="store_true",
                    help="Skip extraction if case_single.npz already exists.")
    ap.add_argument("--skip_mesh",    action="store_true",
                    help="Skip mesh-graph build if mesh_graph.npz already exists.")
    ap.add_argument("--skip_train",   action="store_true",
                    help="Skip training (extract + mesh only).")
    ap.add_argument("--clean",        action="store_true",
                    help="Delete any existing checkpoints in --model_dir before "
                         "training. Always use this when changing model flags or "
                         "after a failed/aborted run, to avoid loading stale weights.")

    args = ap.parse_args()

    # Resolve device
    if args.device == "auto":
        try:
            import torch
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            args.device = "cpu"

    # Determine the case directory for mesh building
    case_dir = args.case_dir
    if case_dir is None and not args.synthetic:
        # Auto-discover first available Sweep_Case
        for d in sorted((_ROOT / "openfoam").glob("Sweep_Case_*")):
            if d.is_dir():
                case_dir = d
                print(f"  [auto] Using first discovered case: {case_dir.name}")
                break
        if case_dir is None:
            print("[ERROR] No OpenFOAM case found. Pass --case_dir or --synthetic.")
            sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    (_DEFAULT_RESULTS / "logs").mkdir(parents=True, exist_ok=True)

    # Remove stale checkpoints so rollout.py never loads an old broken model
    if args.clean:
        import shutil
        for ckpt in args.model_dir.glob("*.pt"):
            ckpt.unlink()
            print(f"  [clean] Removed {ckpt.name}")
        cfg_file = args.model_dir / "config.json"
        if cfg_file.exists():
            cfg_file.unlink()
            print(f"  [clean] Removed config.json")
    elif args.model_dir.exists() and any(args.model_dir.glob("*.pt")):
        print(
            f"\n  [WARNING] Old checkpoints found in {args.model_dir}.\n"
            f"  If you changed model flags or had a failed run, add --clean\n"
            f"  to delete them before retraining.\n"
        )

    t0 = time.time()

    # ------------------------------------------------------------------
    _hline("STEP 1  Extract field data")
    npz_path = stage_extract(args, args.out_dir)

    # ------------------------------------------------------------------
    _hline("STEP 2  Build mesh graph")
    mesh_path = stage_mesh(args, args.out_dir, case_dir or npz_path.parent)

    # ------------------------------------------------------------------
    if not args.skip_train:
        _hline("STEP 3  Train CFD-GNN")
        stage_train(args, args.out_dir, mesh_path, args.model_dir)
    else:
        print("  [skip_train] Skipping training.")

    # ------------------------------------------------------------------
    elapsed = time.time() - t0
    _hline()
    print(f"  Done in {elapsed/60:.1f} min")
    print(f"  Model : {args.model_dir / 'best.pt'}")
    print(f"  Data  : {npz_path}")
    print()
    print("  To run rollout with the trained model:")
    print(f"    python cfd_gnn/rollout.py \\")
    print(f"        --model_dir {args.model_dir} \\")
    print(f"        --n_steps 260 \\")
    print(f"        --output  results/cfd_gnn_single/rollout")
    _hline()


if __name__ == "__main__":
    main()
