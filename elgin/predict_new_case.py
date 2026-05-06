"""predict_new_case.py — Run CFD-GNN inference on an unknown situation.

This script is the single entry-point for predicting aerosol transport in a
room geometry that was NOT part of training.  It can accept input in two ways:

  1. A pre-extracted .npz file (produced earlier by extract_fields.py)
  2. A raw OpenFOAM case directory  (extraction happens automatically)

It then:
  • Runs the CFD-GNN rollout (freeze_fluid mode — uses the RANS fluid field
    embedded in the NPZ as a frozen background; only the particle GNN runs
    autoregressively).
  • Saves rollout.npz with fluid + particle trajectories.
  • Writes clinical_metrics.json: breathing-zone exposure, deposition
    fractions per boundary surface (wall / floor / ceiling / dentist / patient).
  • Produces two animations:
      fluid_particles.mp4   — steady air-speed colourmap + predicted aerosol
      (if GT available)  compare.mp4  — side-by-side with ground truth

Usage
-----
  # From an already-extracted NPZ
  python cfd_gnn/predict_new_case.py \\
      --input      experiments/cfd_gnn_20cases/datasets/case_21.npz \\
      --mesh       experiments/cfd_gnn_20cases/datasets/mesh_graph.npz \\
      --model_dir  experiments/cfd_gnn_20cases/models \\
      --output_dir predictions/case_21

  # From a raw OpenFOAM directory (extraction runs automatically)
  python cfd_gnn/predict_new_case.py \\
      --input      D:/openfoam/Sweep_Case_21 \\
      --mesh       experiments/cfd_gnn_20cases/datasets/mesh_graph.npz \\
      --model_dir  experiments/cfd_gnn_20cases/models \\
      --output_dir predictions/case_21

  # Custom inlet velocity (overrides what is in the NPZ)
  python cfd_gnn/predict_new_case.py \\
      --input      D:/openfoam/Sweep_Case_21 \\
      --model_dir  experiments/cfd_gnn_20cases/models \\
      --u_inlet    0.8 \\
      --output_dir predictions/case_21

  # Compare against known GT (shows both GNN and GT in animation)
  python cfd_gnn/predict_new_case.py \\
      --input      experiments/cfd_gnn_20cases/datasets/case_05.npz \\
      --gt_npz     experiments/cfd_gnn_20cases/datasets/case_05.npz \\
      --model_dir  experiments/cfd_gnn_20cases/models \\
      --output_dir predictions/case_05_check
"""

from __future__ import annotations

import argparse
import json
import os
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

_PYTHON  = sys.executable
_ROOT    = pathlib.Path(__file__).resolve().parent.parent
_CFD_GNN = pathlib.Path(__file__).resolve().parent

# Default model location (train_20cases.py output)
_DEFAULT_MODEL = _ROOT / "experiments" / "cfd_gnn_20cases" / "models"
_DEFAULT_MESH  = _ROOT / "experiments" / "cfd_gnn_20cases" / "datasets" / "mesh_graph.npz"


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _hline(title: str = "") -> None:
    w = 68
    if title:
        pad = max(1, (w - len(title) - 2) // 2)
        print("=" * pad + f" {title} " + "=" * pad)
    else:
        print("=" * w)


def _run(cmd: list[str], log_path: pathlib.Path | None = None,
         check: bool = True) -> int:
    """Stream subprocess output to console and optionally log to file."""
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    proc.wait()

    if log_path:
        log_path.write_text("".join(lines), encoding="utf-8")

    if check and proc.returncode != 0:
        print(f"\n[FAILED]  exit code {proc.returncode}")
        if log_path:
            print(f"  Log: {log_path}")
        sys.exit(proc.returncode)
    return proc.returncode


# ---------------------------------------------------------------------------
#  Step 1 — Extract from OpenFOAM (only if input is a directory)
# ---------------------------------------------------------------------------

def _extract_from_openfoam(
    case_dir:    pathlib.Path,
    out_npz:     pathlib.Path,
    n_particles: int,
    t_start:     float,
    t_end:       float,
    dt_keep:     float,
    log_dir:     pathlib.Path,
) -> pathlib.Path:
    """Extract fluid + particle data from a raw OpenFOAM case."""
    _hline(f"Extracting from OpenFOAM: {case_dir.name}")
    cmd = [
        _PYTHON, str(_CFD_GNN / "data" / "extract_fields.py"),
        "--case_dir",    str(case_dir),
        "--output",      str(out_npz),
        "--t_start",     str(t_start),
        "--t_end",       str(t_end),
        "--dt_keep",     str(dt_keep),
        "--n_particles", str(n_particles),
    ]
    _run(cmd, log_dir / "extract.log")
    print(f"  Extracted: {out_npz}")
    return out_npz


# ---------------------------------------------------------------------------
#  Step 2 — Run rollout (inference)
# ---------------------------------------------------------------------------

def _run_rollout(
    ic_file:     pathlib.Path,
    mesh_path:   pathlib.Path,
    model_dir:   pathlib.Path,
    out_dir:     pathlib.Path,
    n_particles: int,
    n_steps:     int,
    device:      str,
    log_dir:     pathlib.Path,
) -> pathlib.Path:
    """Run CFD-GNN forward simulation."""
    _hline("Running CFD-GNN rollout")
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        _PYTHON, str(_CFD_GNN / "rollout.py"),
        "--model_dir",   str(model_dir),
        "--mesh",        str(mesh_path),
        "--ic_file",     str(ic_file),
        "--n_particles", str(n_particles),
        "--n_steps",     str(n_steps),
        "--output",      str(out_dir),
        "--device",      device,
        "--freeze_fluid",
    ]
    _run(cmd, log_dir / "rollout.log")

    rollout_npz = out_dir / "rollout.npz"
    if not rollout_npz.exists():
        print(f"  [ERROR] rollout.npz not found at {rollout_npz}")
        sys.exit(1)
    print(f"  Rollout saved: {rollout_npz}")
    return rollout_npz


# ---------------------------------------------------------------------------
#  Step 3 — Print clinical metrics
# ---------------------------------------------------------------------------

def _print_metrics(out_dir: pathlib.Path) -> None:
    """Pretty-print clinical_metrics.json if it exists."""
    met_file = out_dir / "clinical_metrics.json"
    if not met_file.exists():
        return
    _hline("Clinical Metrics")
    try:
        met = json.loads(met_file.read_text())
        for k, v in met.items():
            if isinstance(v, float):
                print(f"  {k:<40} {v:.4f}")
            else:
                print(f"  {k:<40} {v}")
    except Exception as exc:
        print(f"  [WARNING] Could not parse metrics: {exc}")


# ---------------------------------------------------------------------------
#  Step 4 — Generate animations
# ---------------------------------------------------------------------------

def _animate(
    rollout_npz: pathlib.Path,
    gt_npz:      pathlib.Path | None,
    out_dir:     pathlib.Path,
    fps:         int,
    log_dir:     pathlib.Path,
) -> None:
    """Produce fluid+particle animation (and optional compare animation)."""
    _hline("Generating animations")

    # 4a — fluid speed colourmap + particles  (always produced)
    fluid_out = out_dir / "fluid_particles.mp4"
    fluid_args = [
        _PYTHON, str(_CFD_GNN / "animate_fluid_particles.py"),
        "--rollout",       str(rollout_npz),
        "--output",        str(fluid_out),
        "--mode",          "speed",
        "--fps",           str(fps),
        "--particle_size", "8",
    ]
    if gt_npz is not None and gt_npz.exists():
        fluid_args += ["--gt", str(gt_npz)]
    _run(fluid_args, log_dir / "animate_fluid.log", check=False)
    if fluid_out.exists():
        print(f"  Fluid animation : {fluid_out}")

    # 4b — particle comparison (only if GT provided)
    if gt_npz is not None and gt_npz.exists():
        compare_out = out_dir / "compare.mp4"
        _run([
            _PYTHON, str(_CFD_GNN / "render_compare.py"),
            "--rollout", str(rollout_npz),
            "--truth",   str(gt_npz),
            "--output",  str(compare_out),
            "--fps",     str(fps),
        ], log_dir / "animate_compare.log", check=False)
        if compare_out.exists():
            print(f"  Compare animation: {compare_out}")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Input ─────────────────────────────────────────────────────────────
    ap.add_argument("--input", type=pathlib.Path, required=True,
                    help="Either: (a) a .npz file from extract_fields.py,\n"
                         "or (b) a raw OpenFOAM case directory (auto-extracted).")
    ap.add_argument("--gt_npz", type=pathlib.Path, default=None,
                    help="Optional ground-truth .npz for comparison animation.")

    # ── Model ─────────────────────────────────────────────────────────────
    ap.add_argument("--model_dir", type=pathlib.Path, default=_DEFAULT_MODEL,
                    help=f"Trained model directory (default: {_DEFAULT_MODEL}).")
    ap.add_argument("--mesh", type=pathlib.Path, default=_DEFAULT_MESH,
                    help=f"mesh_graph.npz path (default: {_DEFAULT_MESH}).")

    # ── Output ────────────────────────────────────────────────────────────
    ap.add_argument("--output_dir", type=pathlib.Path,
                    default=pathlib.Path("predictions") / "new_case",
                    help="Directory for rollout.npz, animations, metrics.")

    # ── Simulation settings ───────────────────────────────────────────────
    ap.add_argument("--n_particles", type=int, default=1000,
                    help="Particles to track (0 = use all in NPZ).")
    ap.add_argument("--n_steps",     type=int, default=255,
                    help="Rollout timesteps (255 ≈ 25.5 s at dt=0.1 s).")
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cuda", "cpu"])

    # ── Extraction (only used if --input is an OpenFOAM directory) ────────
    ap.add_argument("--t_start",  type=float, default=2.0,
                    help="Simulation start time for extraction [s].")
    ap.add_argument("--t_end",    type=float, default=28.0,
                    help="Simulation end time for extraction [s].")
    ap.add_argument("--dt_keep",  type=float, default=0.1,
                    help="Timestep interval to keep [s].")

    # ── Animation settings ────────────────────────────────────────────────
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--skip_animate", action="store_true")

    args = ap.parse_args()

    # ── Resolve device ────────────────────────────────────────────────────
    if args.device == "auto":
        try:
            import torch
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            args.device = "cpu"

    # ── Set up output directories ─────────────────────────────────────────
    out_dir  = args.output_dir
    log_dir  = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    t_start_wall = time.time()

    _hline("CFD-GNN  Prediction for Unknown Case")
    print(f"  Input        : {args.input}")
    print(f"  Model dir    : {args.model_dir}")
    print(f"  Mesh         : {args.mesh}")
    print(f"  Output dir   : {out_dir}")
    print(f"  Device       : {args.device}")
    print(f"  N particles  : {args.n_particles}")
    print(f"  N steps      : {args.n_steps}")
    _hline()

    # ── Validate model ────────────────────────────────────────────────────
    best_pt = args.model_dir / "best.pt"
    if not best_pt.exists():
        print(f"[ERROR] Trained model not found: {best_pt}")
        print("  Train the model first with run_20cases.ps1 (or run_case03.ps1).")
        sys.exit(1)

    if not args.mesh.exists():
        print(f"[ERROR] Mesh file not found: {args.mesh}")
        sys.exit(1)

    # ── Step 1: Resolve IC file (extract if OpenFOAM dir given) ──────────
    ic_file: pathlib.Path
    if args.input.is_dir():
        # Raw OpenFOAM case — extract data first
        extracted_npz = out_dir / f"{args.input.name}_extracted.npz"
        ic_file = _extract_from_openfoam(
            case_dir    = args.input,
            out_npz     = extracted_npz,
            n_particles = args.n_particles,
            t_start     = args.t_start,
            t_end       = args.t_end,
            dt_keep     = args.dt_keep,
            log_dir     = log_dir,
        )
    elif args.input.suffix == ".npz" and args.input.exists():
        ic_file = args.input
        print(f"  Using pre-extracted NPZ: {ic_file}")
    else:
        print(f"[ERROR] --input must be an existing .npz file or OpenFOAM directory.")
        print(f"  Got: {args.input}")
        sys.exit(1)

    # ── Step 2: Run rollout ───────────────────────────────────────────────
    rollout_npz = _run_rollout(
        ic_file     = ic_file,
        mesh_path   = args.mesh,
        model_dir   = args.model_dir,
        out_dir     = out_dir,
        n_particles = args.n_particles,
        n_steps     = args.n_steps,
        device      = args.device,
        log_dir     = log_dir,
    )

    # ── Step 3: Print clinical metrics ───────────────────────────────────
    _print_metrics(out_dir)

    # ── Step 4: Animate ───────────────────────────────────────────────────
    if not args.skip_animate:
        _animate(
            rollout_npz = rollout_npz,
            gt_npz      = args.gt_npz,
            out_dir     = out_dir,
            fps         = args.fps,
            log_dir     = log_dir,
        )

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - t_start_wall
    _hline("Prediction Complete")
    print(f"  Total time     : {elapsed:.0f} s  ({elapsed/60:.1f} min)")
    print(f"  Rollout        : {rollout_npz}")
    print(f"  Metrics        : {out_dir / 'clinical_metrics.json'}")
    if not args.skip_animate:
        print(f"  Animation      : {out_dir / 'fluid_particles.mp4'}")
    print()
    print("  To re-run animation only:")
    print(f"    python cfd_gnn/animate_fluid_particles.py \\")
    print(f"        --rollout {rollout_npz} \\")
    print(f"        --output  {out_dir / 'fluid_particles.mp4'}")
    _hline()


if __name__ == "__main__":
    main()
