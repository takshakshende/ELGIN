"""animate_fluid_particles.py — Animated fluid velocity field + aerosol particles.

Produces an MP4 (or GIF) showing the steady-state air velocity as a
background colour-map with velocity arrows, and the predicted (and optionally
ground-truth) aerosol particle cloud animated on top.

Data source
-----------
All data comes from a single ``rollout.npz`` produced by ``rollout.py``:
    fluid_traj    (T, N_cells, 5)   — [Ux, Uy, p, k, omega] in physical units
    particle_traj (T, N_part, 2)    — predicted particle positions [m]
    cell_pos      (N_cells, 2)      — mesh cell centroids [m]
    times         (T,)              — simulation times [s]

Optionally also loads ``gt.npz`` (same structure) for a side-by-side panel.

Usage
-----
    # Single panel (GNN only, fluid + particles)
    python cfd_gnn/animate_fluid_particles.py \\
        --rollout experiments/cfd_gnn_case03/results/checks/<run>/rollout.npz \\
        --output  fluid_particles.mp4

    # Two-panel: GNN | GT  (fluid background on both)
    python cfd_gnn/animate_fluid_particles.py \\
        --rollout experiments/cfd_gnn_case03/results/checks/<run>/rollout.npz \\
        --gt      experiments/cfd_gnn_case03/results/checks/<run>/gt.npz \\
        --output  fluid_particles_compare.mp4

    # Show velocity vectors instead of stream lines
    python cfd_gnn/animate_fluid_particles.py --rollout ... --mode quiver

Fluid display options (--mode):
    speed      — colour-map of |U| = sqrt(Ux²+Uy²)             [default]
    ux         — x-component of velocity
    uy         — y-component of velocity
    quiver     — velocity vectors on a regular sub-grid
    streamline — streamlines (requires scipy)
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
    import matplotlib.patches as mpatches
    import matplotlib.tri as mtri
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm
except ImportError as exc:
    raise SystemExit("matplotlib is required.  pip install matplotlib") from exc


# ---------------------------------------------------------------------------
#  bc_type integer → semantic label (matches config.py / mesh_to_graph.py)
# ---------------------------------------------------------------------------
_BC_ID = {
    "interior": 0, "inlet": 1, "outlet": 2,
    "wall": 3, "floor": 4, "ceiling": 5,
    "dentist": 6, "patient": 7, "symmetry": 8,
}


# ---------------------------------------------------------------------------
#  Geometry overlay — draws obstacles, inlet, and outlet on an axes object
# ---------------------------------------------------------------------------

def _draw_geometry_overlay(ax, mesh: dict) -> None:
    """Annotate the axis with domain geometry extracted from mesh_graph.npz.

    Draws:
      • Dentist obstacle   — grey filled rectangle + label
      • Patient obstacle   — tan  filled rectangle + label
      • Air inlet          — blue downward-arrow patch + "Air Inlet" label
      • Side outlet        — green rightward-arrow patch + "Outlet" label
      • Nozzle / injection — white star marker at patient oral-cavity position
    """
    cp  = mesh["cell_pos"]    # (N, 2)
    bc  = mesh["bc_type"]     # (N,)

    def _bbox(bid):
        mask = bc == bid
        if not mask.any():
            return None
        pts = cp[mask]
        return pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max()

    # ── Dentist obstacle ──────────────────────────────────────────────────
    bb = _bbox(_BC_ID["dentist"])
    if bb:
        x0, x1, y0, y1 = bb
        rect = mpatches.FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0,
            boxstyle="round,pad=0.02",
            linewidth=1.2, edgecolor="#aaaaaa",
            facecolor="#555555", alpha=0.55, zorder=3)
        ax.add_patch(rect)
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.08, "Dentist",
                color="white", fontsize=7, ha="center", va="bottom",
                fontweight="bold", zorder=4)

    # ── Patient obstacle ──────────────────────────────────────────────────
    bb = _bbox(_BC_ID["patient"])
    if bb:
        x0, x1, y0, y1 = bb
        rect = mpatches.FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0,
            boxstyle="round,pad=0.02",
            linewidth=1.2, edgecolor="#ddaa77",
            facecolor="#7a5230", alpha=0.55, zorder=3)
        ax.add_patch(rect)
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.08, "Patient",
                color="#ffddaa", fontsize=7, ha="center", va="bottom",
                fontweight="bold", zorder=4)

    # ── Air inlet (ceiling slot) ──────────────────────────────────────────
    bb = _bbox(_BC_ID["inlet"])
    if bb:
        x0, x1, _, y1 = bb
        cx = (x0 + x1) / 2
        # small filled rectangle at the ceiling
        rect = mpatches.FancyBboxPatch(
            (x0 - 0.05, y1 - 0.02), (x1 - x0) + 0.10, 0.06,
            boxstyle="square,pad=0.0",
            linewidth=1.5, edgecolor="#4499ff",
            facecolor="#4499ff", alpha=0.8, zorder=4)
        ax.add_patch(rect)
        # downward arrow
        ax.annotate("", xy=(cx, y1 - 0.20), xytext=(cx, y1 + 0.02),
                    arrowprops=dict(arrowstyle="-|>", color="#4499ff",
                                   lw=1.5), zorder=5)
        ax.text(cx, y1 - 0.28, "Air Inlet",
                color="#4499ff", fontsize=7, ha="center", va="top",
                fontweight="bold", zorder=5)

    # ── Side outlet ───────────────────────────────────────────────────────
    bb = _bbox(_BC_ID["outlet"])
    if bb:
        x1, _, y0, y1 = bb[1], bb[0], bb[2], bb[3]
        cy = (y0 + y1) / 2
        rect = mpatches.FancyBboxPatch(
            (x1 - 0.04, y0 - 0.05), 0.06, (y1 - y0) + 0.10,
            boxstyle="square,pad=0.0",
            linewidth=1.5, edgecolor="#44ff99",
            facecolor="#44ff99", alpha=0.8, zorder=4)
        ax.add_patch(rect)
        ax.annotate("", xy=(x1 + 0.20, cy), xytext=(x1 + 0.02, cy),
                    arrowprops=dict(arrowstyle="-|>", color="#44ff99",
                                   lw=1.5), zorder=5)
        ax.text(x1 + 0.22, cy, "Outlet",
                color="#44ff99", fontsize=7, ha="left", va="center",
                fontweight="bold", zorder=5)

    # ── Nozzle / injection point (patient oral cavity) ────────────────────
    # Injection at (2.40, 0.90) per ConeInjection config
    nozzle_x, nozzle_y = 2.40, 0.90
    ax.plot(nozzle_x, nozzle_y, marker="*", markersize=10,
            color="white", markeredgecolor="#ffaa00",
            markeredgewidth=0.8, zorder=6)
    ax.text(nozzle_x + 0.08, nozzle_y + 0.10, "Nozzle",
            color="white", fontsize=6.5, ha="left", va="bottom",
            bbox=dict(boxstyle="round,pad=0.15",
                      facecolor="black", alpha=0.45), zorder=6)


# ---------------------------------------------------------------------------
#  Data loading
# ---------------------------------------------------------------------------

def _load_npz(path: pathlib.Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {k: np.asarray(data[k]) for k in data.files}


# ---------------------------------------------------------------------------
#  Fluid field helpers
# ---------------------------------------------------------------------------

def _velocity_magnitude(fluid: np.ndarray) -> np.ndarray:
    """(N_cells, 5) → (N_cells,) speed [m/s]."""
    return np.sqrt(fluid[:, 0] ** 2 + fluid[:, 1] ** 2)


def _triangulate(cell_pos: np.ndarray) -> mtri.Triangulation:
    """Delaunay triangulation of unstructured mesh cell centroids."""
    return mtri.Triangulation(cell_pos[:, 0], cell_pos[:, 1])


def _quiver_subsample(cell_pos: np.ndarray, fluid_frame: np.ndarray,
                       n_grid: int = 30) -> tuple:
    """Interpolate velocity onto a regular n_grid×n_grid sub-grid for quiver."""
    try:
        from scipy.interpolate import griddata
    except ImportError:
        return None
    x0, y0 = cell_pos[:, 0].min(), cell_pos[:, 1].min()
    x1, y1 = cell_pos[:, 0].max(), cell_pos[:, 1].max()
    gx = np.linspace(x0, x1, n_grid)
    gy = np.linspace(y0, y1, n_grid)
    GX, GY = np.meshgrid(gx, gy)
    pts = cell_pos
    ux_g = griddata(pts, fluid_frame[:, 0], (GX, GY), method="linear")
    uy_g = griddata(pts, fluid_frame[:, 1], (GX, GY), method="linear")
    return GX, GY, ux_g, uy_g


# ---------------------------------------------------------------------------
#  Main animation builder
# ---------------------------------------------------------------------------

def build_animation(
    rollout:    dict,
    gt:         dict | None,
    mesh:       dict | None = None,
    mode:       str  = "speed",
    fps:        int  = 10,
    particle_size: float = 4.0,
    alpha_fluid: float = 0.75,
    cmap:       str  = "RdYlBu_r",
    quiver_grid: int = 25,
) -> FuncAnimation:
    cell_pos      = rollout["cell_pos"]            # (N_cells, 2)
    fluid_traj    = rollout["fluid_traj"]          # (T, N_cells, 5)
    pred_traj     = rollout["particle_traj"]       # (T, N_part, 2)
    times         = rollout["times"]               # (T,)
    T             = len(times)

    # Fluid field is steady — use frame 0 for the background
    fluid_bg = fluid_traj[0]

    # Choose scalar field for the colour-map
    if mode == "speed":
        scalar = _velocity_magnitude(fluid_bg)
        label  = "|U|  [m/s]"
    elif mode == "ux":
        scalar = fluid_bg[:, 0]
        label  = "U_x  [m/s]"
    elif mode == "uy":
        scalar = fluid_bg[:, 1]
        label  = "U_y  [m/s]"
    else:
        scalar = _velocity_magnitude(fluid_bg)
        label  = "|U|  [m/s]"

    triang = _triangulate(cell_pos)
    vmin, vmax = scalar.min(), scalar.max()
    norm = Normalize(vmin=vmin, vmax=vmax)

    # ── Quiver data (pre-computed once, stays constant) ────────────────────
    quiver_data = None
    if mode == "quiver":
        quiver_data = _quiver_subsample(cell_pos, fluid_bg, n_grid=quiver_grid)

    # ── Streamline grid (pre-computed once) ────────────────────────────────
    stream_data = None
    if mode == "streamline":
        stream_data = _quiver_subsample(cell_pos, fluid_bg, n_grid=50)

    n_panels = 2 if gt is not None else 1
    fig, axes = plt.subplots(1, n_panels,
                             figsize=(6 * n_panels + 0.5, 4.5),
                             facecolor="#0a0a1a")

    if n_panels == 1:
        axes = [axes]

    titles = ["ELGIN (best.pt)"]
    trajs  = [pred_traj]
    if gt is not None:
        titles.append("OpenFOAM ground truth")
        trajs.append(gt["particle_traj"])

    # ── Draw static fluid background on each panel ─────────────────────────
    fluid_artists = []
    for ax, title in zip(axes, titles):
        ax.set_facecolor("#0a0a1a")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_edgecolor("white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.set_title(title, color="white", fontsize=10)
        ax.set_xlabel("x [m]", color="white")
        ax.set_ylabel("y [m]", color="white")
        ax.set_aspect("equal")

        if mode in ("speed", "ux", "uy"):
            tcf = ax.tricontourf(triang, scalar, levels=64,
                                 cmap=cmap, norm=norm, alpha=alpha_fluid)
            # Add colourbar on first panel only
            if ax is axes[0]:
                cbar = fig.colorbar(tcf, ax=axes[-1],
                                    fraction=0.03, pad=0.02)
                cbar.set_label(label, color="white", fontsize=8)
                cbar.ax.yaxis.set_tick_params(color="white")
                plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
            fluid_artists.append(tcf)

        elif mode == "quiver" and quiver_data is not None:
            GX, GY, ux_g, uy_g = quiver_data
            spd_g = np.sqrt(ux_g ** 2 + uy_g ** 2)
            q = ax.quiver(GX, GY, ux_g, uy_g, spd_g,
                          cmap=cmap, norm=norm,
                          alpha=alpha_fluid, scale=30,
                          width=0.003, headwidth=4)
            if ax is axes[0]:
                fig.colorbar(q, ax=axes[-1], fraction=0.03, pad=0.02,
                             label=label)

        elif mode == "streamline" and stream_data is not None:
            GX, GY, ux_g, uy_g = stream_data
            gx = GX[0, :]
            gy = GY[:, 0]
            strm = ax.streamplot(gx, gy, ux_g, uy_g,
                                  color=np.sqrt(ux_g ** 2 + uy_g ** 2),
                                  cmap=cmap, norm=norm,
                                  density=1.2, linewidth=0.8,
                                  arrowsize=0.8)
            if ax is axes[0]:
                fig.colorbar(strm.lines, ax=axes[-1],
                             fraction=0.03, pad=0.02, label=label)

        # Domain boundary box
        dom_x = [cell_pos[:, 0].min(), cell_pos[:, 0].max()]
        dom_y = [cell_pos[:, 1].min(), cell_pos[:, 1].max()]
        ax.set_xlim(dom_x[0] - 0.05, dom_x[1] + 0.30)   # extra right margin for outlet label
        ax.set_ylim(dom_y[0] - 0.05, dom_y[1] + 0.15)   # extra top margin for inlet label

        # Geometry overlay (obstacles + inlet + outlet + nozzle)
        if mesh is not None:
            _draw_geometry_overlay(ax, mesh)

    # ── Particle scatter objects (updated each frame) ──────────────────────
    scatters = []
    for ax, traj in zip(axes, trajs):
        sc = ax.scatter([], [], s=particle_size, c="#39FF14",   # neon lime-green
                        alpha=0.92, linewidths=0.4,
                        edgecolors="#000000", zorder=5)
        scatters.append(sc)

    # ── Time / info text ───────────────────────────────────────────────────
    texts = []
    for ax in axes:
        tx = ax.text(0.02, 0.97, "", transform=ax.transAxes,
                     color="white", fontsize=8, va="top",
                     bbox=dict(boxstyle="round,pad=0.3",
                               facecolor="black", alpha=0.5))
        texts.append(tx)

    fig.tight_layout()

    def _update(frame):
        t = times[frame]
        for sc, traj in zip(scatters, trajs):
            if frame < len(traj):
                pos = traj[frame]             # (N_part, 2)
                alive = np.all(np.isfinite(pos), axis=1)
                sc.set_offsets(pos[alive])
        n_alive = int(np.all(np.isfinite(pred_traj[frame]), axis=1).sum())
        info = f"t = {t:.2f} s\nalive = {n_alive}"
        for tx in texts:
            tx.set_text(info)
        return scatters + texts

    anim = FuncAnimation(fig, _update, frames=T,
                         interval=1000 // fps, blit=True)
    return fig, anim


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rollout", type=pathlib.Path, required=True,
                    help="Path to rollout.npz produced by rollout.py")
    ap.add_argument("--gt", type=pathlib.Path, default=None,
                    help="Optional gt.npz for side-by-side GT panel")
    ap.add_argument("--mesh", type=pathlib.Path, default=None,
                    help="Optional mesh_graph.npz — adds inlet/outlet/dentist/patient overlays")
    ap.add_argument("--output", type=pathlib.Path,
                    default=pathlib.Path("fluid_particles.mp4"),
                    help="Output file (.mp4 or .gif).  Default: fluid_particles.mp4")
    ap.add_argument("--mode", default="speed",
                    choices=["speed", "ux", "uy", "quiver", "streamline"],
                    help="Fluid display mode (default: speed = |U| colourmap)")
    ap.add_argument("--fps",  type=int,   default=10,
                    help="Frames per second (default 10)")
    ap.add_argument("--cmap", default="RdYlBu_r",
                    help="Matplotlib colourmap for fluid field (default RdYlBu_r)")
    ap.add_argument("--alpha", type=float, default=0.70,
                    help="Opacity of fluid background (0–1, default 0.70)")
    ap.add_argument("--particle_size", type=float, default=8.0,
                    help="Scatter dot size for particles (default 8)")
    ap.add_argument("--quiver_grid", type=int, default=25,
                    help="Grid resolution for quiver/streamline mode (default 25)")
    args = ap.parse_args()

    if not args.rollout.exists():
        sys.exit(f"[ERROR] rollout not found: {args.rollout}")

    print(f"  Loading rollout : {args.rollout}")
    rollout = _load_npz(args.rollout)

    gt = None
    if args.gt is not None:
        if not args.gt.exists():
            print(f"  [WARNING] gt file not found: {args.gt} — single panel only")
        else:
            print(f"  Loading GT      : {args.gt}")
            raw_gt = _load_npz(args.gt)
            # Normalise key name: gt.npz uses 'positions', rollout.npz uses 'particle_traj'
            if "positions" in raw_gt and "particle_traj" not in raw_gt:
                raw_gt["particle_traj"] = raw_gt["positions"]
            gt = raw_gt

    # Optional mesh geometry
    mesh = None
    if args.mesh is not None:
        if args.mesh.exists():
            print(f"  Loading mesh    : {args.mesh}")
            mesh = _load_npz(args.mesh)
        else:
            print(f"  [WARNING] mesh not found: {args.mesh} — geometry overlay skipped")

    T = len(rollout["times"])
    print(f"  Frames          : {T}  ({rollout['times'][0]:.1f}–{rollout['times'][-1]:.1f} s)")
    print(f"  Cells           : {rollout['cell_pos'].shape[0]}")
    print(f"  Particles       : {rollout['particle_traj'].shape[1]}")
    print(f"  Fluid mode      : {args.mode}")
    print(f"  Geometry overlay: {'yes' if mesh is not None else 'no'}")

    fig, anim = build_animation(
        rollout, gt,
        mesh=mesh,
        mode=args.mode,
        fps=args.fps,
        particle_size=args.particle_size,
        alpha_fluid=args.alpha,
        cmap=args.cmap,
        quiver_grid=args.quiver_grid,
    )

    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    suffix = out.suffix.lower()

    if suffix == ".gif":
        writer = PillowWriter(fps=args.fps)
        anim.save(str(out), writer=writer)
    else:
        try:
            writer = FFMpegWriter(fps=args.fps,
                                  extra_args=["-vcodec", "libx264",
                                              "-pix_fmt", "yuv420p"])
            anim.save(str(out), writer=writer)
        except Exception as exc:
            print(f"  [WARNING] ffmpeg failed ({exc}) — falling back to GIF")
            gif_out = out.with_suffix(".gif")
            anim.save(str(gif_out), writer=PillowWriter(fps=args.fps))
            out = gif_out

    plt.close(fig)
    print(f"\n  Saved: {out}")
    print(f"  ({T} frames, {args.fps} fps, mode={args.mode})")


if __name__ == "__main__":
    main()
