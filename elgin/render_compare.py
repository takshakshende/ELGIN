"""render_compare.py -- Side-by-side animation of OpenFOAM ground truth vs
ELGIN predicted particle trajectories.

Example
-------
    python elgin/render_compare.py \\
        --rollout experiments/elgin_case03/results/rollouts/rollout.npz \\
        --truth   experiments/elgin_case03/results/rollouts/gt.npz \\
        --output  experiments/elgin_case03/results/animations/compare.mp4

If ffmpeg is not on PATH the script falls back automatically to .gif via
matplotlib's PillowWriter.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
except ImportError as exc:                                      # pragma: no cover
    raise SystemExit("matplotlib is required for rendering.") from exc


def _load_truth(path: pathlib.Path) -> np.ndarray:
    """Load ground-truth particle positions of shape (T, N, 2)."""
    data = np.load(path, allow_pickle=True)
    if "positions" in data.files:
        return np.asarray(data["positions"])
    if "particle_pos" in data.files:
        return np.asarray(data["particle_pos"])
    raise KeyError(
        f"{path} has no 'positions' or 'particle_pos' array; "
        f"available keys: {list(data.files)}"
    )


def _load_pred(path: pathlib.Path) -> np.ndarray:
    """Load ELGIN predicted particle trajectory of shape (T, N, 2)."""
    data = np.load(path, allow_pickle=True)
    if "particle_traj" in data.files:
        traj = np.asarray(data["particle_traj"])
    elif "predictions" in data.files:
        preds = np.asarray(data["predictions"])
        traj = preds[0] if preds.dtype == object else preds[0]
    else:
        raise KeyError(
            f"{path} has no 'particle_traj' or 'predictions' array; "
            f"available keys: {list(data.files)}"
        )
    return traj


def _domain_from_data(*arrays: np.ndarray, pad: float = 0.05) -> tuple[float, float, float, float]:
    xs = np.concatenate([a[..., 0].ravel() for a in arrays])
    ys = np.concatenate([a[..., 1].ravel() for a in arrays])
    xs = xs[np.isfinite(xs)]
    ys = ys[np.isfinite(ys)]
    if xs.size == 0 or ys.size == 0:
        return 0.0, 1.0, 0.0, 1.0
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    dx = max(x_max - x_min, 1e-6) * pad
    dy = max(y_max - y_min, 1e-6) * pad
    return x_min - dx, x_max + dx, y_min - dy, y_max + dy


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--rollout", type=pathlib.Path, required=True,
                        help="ELGIN rollout .npz (from elgin/rollout.py).")
    parser.add_argument("--truth",   type=pathlib.Path, required=True,
                        help="OpenFOAM ground-truth trajectory .npz "
                             "(e.g. experiments/elgin_case03/results/rollouts/gt.npz).")
    parser.add_argument("--output",  type=pathlib.Path, required=True,
                        help="Output animation file (.mp4 or .gif).")
    parser.add_argument("--fps",       type=int,   default=20)
    parser.add_argument("--dot_size",  type=float, default=4.0)
    parser.add_argument("--max_frames", type=int, default=0,
                        help="Cap number of frames (0 = use min(T_pred, T_gt)).")
    parser.add_argument("--domain", nargs=4, type=float, default=None,
                        help="Override [xmin xmax ymin ymax].  Default: data bounds.")
    args = parser.parse_args()

    pred  = _load_pred(args.rollout)
    truth = _load_truth(args.truth)

    if pred.ndim != 3 or pred.shape[-1] < 2:
        sys.exit(f"[render_compare] unexpected pred shape: {pred.shape}")
    if truth.ndim != 3 or truth.shape[-1] < 2:
        sys.exit(f"[render_compare] unexpected truth shape: {truth.shape}")

    # Truncate to the shorter trajectory and to args.max_frames if set.
    T = min(pred.shape[0], truth.shape[0])
    if args.max_frames > 0:
        T = min(T, args.max_frames)
    pred  = pred[:T, :, :2]
    truth = truth[:T, :, :2]

    print(f"[render_compare] truth shape (T,N,2) = {truth.shape}")
    print(f"[render_compare] pred  shape (T,N,2) = {pred.shape}")

    if args.domain is None:
        xmin, xmax, ymin, ymax = _domain_from_data(pred, truth)
    else:
        xmin, xmax, ymin, ymax = args.domain

    fig, (ax_truth, ax_pred) = plt.subplots(
        1, 2, figsize=(11, 4.5), sharex=True, sharey=True
    )
    for ax, title in [
        (ax_truth, "OpenFOAM ground truth (reactingParcelFoam)"),
        (ax_pred,  "ELGIN prediction"),
    ]:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x  [m]")
        ax.set_ylabel("y  [m]")
        ax.grid(linestyle="--", alpha=0.3)

    scat_truth = ax_truth.scatter([], [], s=args.dot_size, c="steelblue", alpha=0.75)
    scat_pred  = ax_pred.scatter([], [], s=args.dot_size, c="indianred",  alpha=0.75)
    title_txt  = fig.suptitle("", fontsize=11, fontweight="bold")

    def _update(i: int):
        scat_truth.set_offsets(truth[i])
        scat_pred.set_offsets(pred[i])
        title_txt.set_text(f"frame {i + 1} / {T}")
        return scat_truth, scat_pred, title_txt

    anim = FuncAnimation(
        fig, _update, frames=T, interval=1000.0 / args.fps, blit=False,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".gif":
        anim.save(args.output, writer=PillowWriter(fps=args.fps))
    else:
        try:
            anim.save(args.output, fps=args.fps, dpi=120)
        except Exception:
            gif_path = args.output.with_suffix(".gif")
            print(f"[render_compare] ffmpeg unavailable; falling back to {gif_path}")
            anim.save(gif_path, writer=PillowWriter(fps=args.fps))
            print(f"[render_compare] wrote {gif_path}")
            return
    print(f"[render_compare] wrote {args.output}")


if __name__ == "__main__":
    main()
