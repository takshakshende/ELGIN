"""extract_fields.py — Extract Eulerian fluid fields *and* Lagrangian parcel
trajectories (with proper per-parcel ID tracking) from an OpenFOAM case.

Compared with the original revision this version:

  * Reads parcel ``origId`` from the VTK file (or from the ASCII
    ``<time>/lagrangian/<cloud>/origId`` when VTK is unavailable) and
    assembles a **full-timeline union table** indexed by parcel ID.  At each
    frame, ``particle_pos[t, i, :]`` is therefore the position of the *same
    physical parcel* across time — not a permutation-noisy snapshot of the
    VTK array order.
  * Saves ``particle_alive_mask (T, N)`` and ``orig_ids (N,)`` so downstream
    code can skip frames where a parcel has not yet been injected or has
    already evaporated/escaped.
  * Keeps the same ``particle_pos / particle_diam / particle_dens`` keys
    expected by the dataset / rollout / animation code.  Positions for
    non-alive frames are filled with NaN.

Usage
-----
    python cfd_gnn/data/extract_fields.py \\
        --case_dir openfoam/Sweep_Case_03 \\
        --output   datasets/cfd_gnn/case_03.npz \\
        --t_start  2.0 \\
        --t_end    28.0 \\
        --dt_keep  0.1

Output .npz structure:
    fluid_U                (T, N_cells, 2)   velocity [m/s]
    fluid_p                (T, N_cells)      kinematic pressure [m^2/s^2]
    fluid_k                (T, N_cells)      turbulent kinetic energy
    fluid_omega            (T, N_cells)      specific dissipation rate
    cell_pos               (N_cells, 2)      cell centroid positions [m]
    d_wall                 (N_cells,)        distance to nearest wall [m]
    times                  (T,)              physical times [s]
    particle_pos           (T, N_part, 2)    NaN where parcel is not alive
    particle_vel           (T, N_part, 2)    NaN where parcel is not alive
    particle_diam          (T, N_part)       NaN where parcel is not alive
    particle_dens          (N_part,)         constant per parcel
    particle_alive_mask    (T, N_part) bool
    orig_ids               (N_part,) int64
"""

from __future__ import annotations
import argparse
import pathlib
import re
import sys
from typing import Dict, List, Optional, Tuple

# Force UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

import numpy as np

try:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    VTK_AVAILABLE = True
except ImportError:
    VTK_AVAILABLE = False
    print("[WARNING] VTK not available. Use 'pip install vtk'. "
          "Falling back to direct OpenFOAM reader for parcel data.")


# ---------------------------------------------------------------------------
#  VTK helpers (Eulerian)
# ---------------------------------------------------------------------------

def _read_vtk_unstructured(vtk_path: pathlib.Path) -> Optional[dict]:
    """Read a single VTK Unstructured Grid file and return field dict."""
    if not VTK_AVAILABLE:
        return None
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(str(vtk_path))
    reader.Update()
    grid = reader.GetOutput()

    pd = grid.GetPointData()
    cd = grid.GetCellData()

    def _get(data, name):
        arr = data.GetArray(name)
        if arr is None:
            return None
        return vtk_to_numpy(arr)

    ccs = vtk.vtkCellCenters()
    ccs.SetInputData(grid)
    ccs.Update()
    centres = vtk_to_numpy(ccs.GetOutput().GetPoints().GetData())

    return {
        "cell_pos":  centres[:, :2].astype(np.float32),
        "U":   _get(cd, "U")  if _get(cd, "U") is not None else _get(pd, "U"),
        "p":   _get(cd, "p")  if _get(cd, "p") is not None else _get(pd, "p"),
        "k":   _get(cd, "k")  if _get(cd, "k") is not None else _get(pd, "k"),
        "omega": (_get(cd, "omega") if _get(cd, "omega") is not None
                  else _get(pd, "omega")),
    }


# ---------------------------------------------------------------------------
#  VTK helpers (Lagrangian — now reads origId)
# ---------------------------------------------------------------------------

def _read_lagrangian_vtk(vtk_path: pathlib.Path) -> Optional[dict]:
    """Read a Lagrangian parcel VTK file *with* parcel IDs.

    Returns dict with keys ``orig_id, pos, vel, diameter, density`` or None
    if the file cannot be read.  Per-parcel ``orig_id`` is required — without
    it parcels cannot be tracked across frames.
    """
    if not VTK_AVAILABLE:
        return None

    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(vtk_path))
    reader.Update()
    pd_data = reader.GetOutput()
    if pd_data is None:
        return None

    n_pts = pd_data.GetNumberOfPoints()
    if n_pts == 0:
        return {
            "orig_id":  np.empty(0, dtype=np.int64),
            "pos":      np.empty((0, 2), dtype=np.float32),
            "vel":      np.empty((0, 2), dtype=np.float32),
            "diameter": np.empty(0, dtype=np.float32),
            "density":  np.empty(0, dtype=np.float32),
        }

    point_data = pd_data.GetPointData()

    def _g(name):
        arr = point_data.GetArray(name)
        return vtk_to_numpy(arr) if arr is not None else None

    pos = vtk_to_numpy(pd_data.GetPoints().GetData())[:, :2].astype(np.float32)

    orig = None
    for key in ("origId", "Id", "id", "parcelId"):
        a = _g(key)
        if a is not None:
            orig = np.asarray(a, dtype=np.int64).reshape(-1)
            break
    if orig is None:
        # No ID — caller will fall back to direct reader.  We still return
        # positions so the Eulerian flow can be tested in isolation.
        orig = np.full(pos.shape[0], -1, dtype=np.int64)

    _vel  = _g("U")
    _diam = _g("d")
    _dens = _g("rho")

    vel = (_vel[:, :2].astype(np.float32) if _vel is not None
           else np.zeros_like(pos, dtype=np.float32))
    diam = (np.asarray(_diam, dtype=np.float32) if _diam is not None
            else np.full(pos.shape[0], 5e-6, dtype=np.float32))
    dens = (np.asarray(_dens, dtype=np.float32) if _dens is not None
            else np.full(pos.shape[0], 1000.0, dtype=np.float32))

    return {
        "orig_id":  orig,
        "pos":      pos,
        "vel":      vel,
        "diameter": diam,
        "density":  dens,
    }


# ---------------------------------------------------------------------------
#  Direct OpenFOAM reader (fallback when VTK is unavailable or has no IDs)
# ---------------------------------------------------------------------------

_VEC_RE = re.compile(r"\(([^)]+)\)")


def _read_inlet_velocity(case_dir: pathlib.Path,
                         patch_keyword: str = "inlet") -> tuple[float, float]:
    """Read the prescribed airInlet velocity from ``<case>/0/U``.

    Returns ``(Ux, Uy)`` of the first patch whose name contains ``patch_keyword``
    and whose ``value`` keyword is ``uniform``.  Falls back to ``(0.0, 0.0)`` if
    nothing matches (synthetic / non-standard cases).
    """
    u_file = case_dir / "0" / "U"
    if not u_file.exists():
        return (0.0, 0.0)
    text = u_file.read_text(encoding="latin-1", errors="replace")
    bf_idx = text.find("boundaryField")
    if bf_idx < 0:
        return (0.0, 0.0)
    bf = text[bf_idx:]

    # Find the named patch block
    name_re = re.compile(rf"\b\w*{patch_keyword}\w*\b", re.IGNORECASE)
    m = name_re.search(bf)
    if m is None:
        return (0.0, 0.0)
    block_start = m.end()
    # Match braces to capture the patch block
    depth = 0
    block_end = -1
    for i in range(block_start, len(bf)):
        if bf[i] == "{":
            depth += 1
        elif bf[i] == "}":
            depth -= 1
            if depth == 0:
                block_end = i
                break
    if block_end < 0:
        return (0.0, 0.0)
    block = bf[block_start: block_end]

    # Look for "value uniform (Ux Uy Uz)"
    uni = re.search(r"uniform\s*\(([^)]+)\)", block)
    if uni:
        comps = uni.group(1).split()
        if len(comps) >= 2:
            try:
                return (float(comps[0]), float(comps[1]))
            except ValueError:
                pass
    return (0.0, 0.0)


def _parse_foam_positions(path: pathlib.Path) -> np.ndarray:
    content = path.read_text(encoding="latin-1", errors="replace")
    xy_list: List[Tuple[float, float]] = []
    in_block = False
    count: Optional[int] = None
    for line in content.splitlines():
        stripped = line.strip()
        if not in_block:
            if stripped.isdigit():
                count = int(stripped)
            elif stripped == "(" and count is not None:
                in_block = True
        else:
            if stripped.startswith(")"):
                break
            m = _VEC_RE.search(stripped)
            if m:
                coords = m.group(1).split()
                if len(coords) >= 2:
                    xy_list.append((float(coords[0]), float(coords[1])))
    if not xy_list:
        return np.empty((0, 2), dtype=np.float32)
    return np.array(xy_list, dtype=np.float32)


def _parse_foam_label_field(path: pathlib.Path) -> np.ndarray:
    content = path.read_text(encoding="latin-1", errors="replace")
    values: List[int] = []
    in_block = False
    count: Optional[int] = None
    for line in content.splitlines():
        stripped = line.strip()
        if not in_block:
            if stripped.isdigit():
                count = int(stripped)
            elif stripped == "(" and count is not None:
                in_block = True
        else:
            if stripped.startswith(")"):
                break
            try:
                values.append(int(stripped))
            except ValueError:
                pass
    if not values:
        return np.empty(0, dtype=np.int64)
    return np.array(values, dtype=np.int64)


def _parse_foam_scalar_field(path: pathlib.Path) -> np.ndarray:
    """Parse a uniform / non-uniform scalar field (one float per line)."""
    if not path.exists():
        return np.empty(0, dtype=np.float32)
    content = path.read_text(encoding="latin-1", errors="replace")
    vals: List[float] = []
    in_block = False
    count: Optional[int] = None
    for line in content.splitlines():
        stripped = line.strip()
        if not in_block:
            if stripped.isdigit():
                count = int(stripped)
            elif stripped == "(" and count is not None:
                in_block = True
        else:
            if stripped.startswith(")"):
                break
            try:
                vals.append(float(stripped))
            except ValueError:
                pass
    if not vals:
        return np.empty(0, dtype=np.float32)
    return np.array(vals, dtype=np.float32)


def _parse_foam_vector_field(path: pathlib.Path) -> np.ndarray:
    if not path.exists():
        return np.empty((0, 2), dtype=np.float32)
    content = path.read_text(encoding="latin-1", errors="replace")
    out: List[Tuple[float, float]] = []
    in_block = False
    count: Optional[int] = None
    for line in content.splitlines():
        stripped = line.strip()
        if not in_block:
            if stripped.isdigit():
                count = int(stripped)
            elif stripped == "(" and count is not None:
                in_block = True
        else:
            if stripped.startswith(")"):
                break
            m = _VEC_RE.search(stripped)
            if m:
                coords = m.group(1).split()
                if len(coords) >= 2:
                    out.append((float(coords[0]), float(coords[1])))
    if not out:
        return np.empty((0, 2), dtype=np.float32)
    return np.array(out, dtype=np.float32)


def _read_lagrangian_direct(case_dir: pathlib.Path,
                            cloud: str,
                            t: float,
                            tol: float = 1e-4) -> Optional[dict]:
    """Read parcel state from the OpenFOAM time directory ``<t>/lagrangian/<cloud>/``.

    Returns dict like :func:`_read_lagrangian_vtk` or ``None`` if the time
    directory has no Lagrangian data.

    The OpenFOAM time directory name can be formatted in several ways
    (e.g. ``2.1``, ``2.10``, ``2.10000``).  We first try the obvious string
    formattings and then fall back to a tolerance-matched scan over all
    numeric subdirectories.
    """
    candidates = [f"{t:g}", f"{t:.1f}", f"{t:.2f}", f"{t:.3f}", f"{t:.4f}"]
    t_dir: Optional[pathlib.Path] = None
    for c in candidates:
        p = case_dir / c / "lagrangian" / cloud
        if p.is_dir():
            t_dir = p
            break
    if t_dir is None:
        # Tolerance scan
        best_dir, best_err = None, float("inf")
        for entry in case_dir.iterdir():
            try:
                t_entry = float(entry.name)
            except (ValueError, TypeError):
                continue
            err = abs(t_entry - t)
            if err < best_err:
                best_err, best_dir = err, entry
        if best_dir is not None and best_err <= tol:
            cand = best_dir / "lagrangian" / cloud
            if cand.is_dir():
                t_dir = cand
    if t_dir is None:
        return None

    pos_file = t_dir / "positions"
    id_file  = t_dir / "origId"
    if not pos_file.exists() or not id_file.exists():
        return None

    pos  = _parse_foam_positions(pos_file)
    orig = _parse_foam_label_field(id_file)
    n    = min(len(pos), len(orig))
    pos, orig = pos[:n], orig[:n]

    vel  = _parse_foam_vector_field(t_dir / "U")[:n]
    if vel.shape[0] < n:
        vel = np.zeros((n, 2), dtype=np.float32)
    diam = _parse_foam_scalar_field(t_dir / "d")[:n]
    if diam.shape[0] < n:
        diam = np.full(n, 5e-6, dtype=np.float32)
    dens = _parse_foam_scalar_field(t_dir / "rho")[:n]
    if dens.shape[0] < n:
        dens = np.full(n, 1000.0, dtype=np.float32)

    return {
        "orig_id":  orig,
        "pos":      pos,
        "vel":      vel,
        "diameter": diam,
        "density":  dens,
    }


# ---------------------------------------------------------------------------
#  Synthetic data generator (kept for unit tests)
# ---------------------------------------------------------------------------

def _synthetic_case(
    n_cells: int = 800,
    n_part:  int = 300,
    n_time:  int = 26,
    Lx: float = 4.0,
    Ly: float = 3.0,
) -> dict:
    rng  = np.random.default_rng(0)
    xc   = rng.uniform(0, Lx, n_cells).astype(np.float32)
    yc   = rng.uniform(0, Ly, n_cells).astype(np.float32)
    cell_pos = np.stack([xc, yc], axis=-1)
    d_wall   = np.minimum(
        np.minimum(xc, Lx - xc),
        np.minimum(yc, Ly - yc)
    ).astype(np.float32)

    T = n_time
    fluid_U     = rng.normal(0, 0.2, (T, n_cells, 2)).astype(np.float32)
    fluid_p     = rng.normal(0, 0.01,(T, n_cells)).astype(np.float32)
    fluid_k     = np.abs(rng.normal(0.01,0.005,(T,n_cells))).astype(np.float32)
    fluid_omega = np.abs(rng.normal(10.0, 2.0, (T, n_cells))).astype(np.float32)

    part_pos  = rng.uniform([0,0],[Lx,Ly],(T,n_part,2)).astype(np.float32)
    part_vel  = rng.normal(0,0.5,(T,n_part,2)).astype(np.float32)
    part_diam_t = np.full((T, n_part), 5e-6, np.float32)
    part_dens = np.full(n_part, 1000.0, dtype=np.float32)
    times     = np.linspace(2.0, 28.0, T).astype(np.float32)
    alive     = np.ones((T, n_part), dtype=bool)
    orig      = np.arange(n_part, dtype=np.int64)

    domain_bounds = np.array([[0.0, Lx], [0.0, Ly]], dtype=np.float32)
    bc_type       = np.zeros(n_cells, dtype=np.int64)
    wall_normal   = np.zeros((n_cells, 2), dtype=np.float32)
    inlet_velocity = np.array([0.0, -0.1], dtype=np.float32)

    return dict(
        fluid_U=fluid_U, fluid_p=fluid_p,
        fluid_k=fluid_k, fluid_omega=fluid_omega,
        cell_pos=cell_pos, d_wall=d_wall,
        wall_normal=wall_normal, bc_type=bc_type,
        domain_bounds=domain_bounds, inlet_velocity=inlet_velocity,
        times=times,
        particle_pos=part_pos, particle_vel=part_vel,
        particle_diam=part_diam_t, particle_dens=part_dens,
        particle_alive_mask=alive, orig_ids=orig,
    )


# ---------------------------------------------------------------------------
#  Lagrangian frame collection
# ---------------------------------------------------------------------------

def _collect_lag_frames(
    selected_vtks: List[Tuple[pathlib.Path, float]],
    cloud_dir:     Optional[pathlib.Path],
    lag_index_map: dict,
    case_dir:      pathlib.Path,
    cloud:         str,
) -> List[Optional[dict]]:
    """For each selected (vtk_path, t) pair, return a parcel-state dict
    (with ``orig_id``) if available, else ``None``.

    Order of preference per frame:
      1. The matching VTK file under ``VTK/lagrangian/<cloud>/`` (origId
         present).
      2. Direct read from ``<t>/lagrangian/<cloud>/`` (origId always
         present in foam-extend / OpenFOAM output).
    """
    out: List[Optional[dict]] = []
    have_warned_no_id = False

    def _stem_index(p: pathlib.Path) -> int:
        try:
            return int(p.stem.split("_")[-1])
        except ValueError:
            return -1

    for vtk_path, t in selected_vtks:
        frame: Optional[dict] = None

        # Try VTK first
        if cloud_dir is not None:
            vtk_idx = _stem_index(vtk_path)
            lag_vtk = lag_index_map.get(vtk_idx)
            if lag_vtk is None:
                for offset in range(1, 6):
                    lag_vtk = (lag_index_map.get(vtk_idx + offset)
                               or lag_index_map.get(vtk_idx - offset))
                    if lag_vtk:
                        break
            if lag_vtk and lag_vtk.exists():
                frame = _read_lagrangian_vtk(lag_vtk)
                if (frame is not None
                        and frame["orig_id"].size > 0
                        and (frame["orig_id"] < 0).any()):
                    if not have_warned_no_id:
                        print("  [warn] VTK file has no origId — falling "
                              "back to direct OpenFOAM reader for parcel IDs.")
                        have_warned_no_id = True
                    frame = None

        # Fallback: read directly from time directory
        if frame is None:
            frame = _read_lagrangian_direct(case_dir, cloud, t)

        out.append(frame)
    return out


def _build_full_timeline(
    lag_frames: List[Optional[dict]],
    times:      List[float],
    n_target:   Optional[int],
) -> Dict[str, np.ndarray]:
    """Assemble a per-parcel-ID full-timeline trajectory table.

    Returns dict with keys ``particle_pos, particle_vel, particle_diam,
    particle_dens, particle_alive_mask, orig_ids``.  ``particle_pos`` and
    siblings carry NaN at frames where a parcel is not alive.
    """
    valid_frames = [(t, f) for t, f in zip(times, lag_frames)
                    if f is not None and f["orig_id"].size > 0]
    if not valid_frames:
        T = len(times)
        return dict(
            particle_pos        = np.full((T, 0, 2), np.nan, np.float32),
            particle_vel        = np.full((T, 0, 2), np.nan, np.float32),
            particle_diam       = np.full((T, 0),    np.nan, np.float32),
            particle_dens       = np.empty(0, np.float32),
            particle_alive_mask = np.zeros((T, 0), dtype=bool),
            orig_ids            = np.empty(0, np.int64),
        )

    # Union of all parcel IDs ever seen
    all_ids = set()
    for _, f in valid_frames:
        all_ids.update(f["orig_id"].tolist())
    all_ids_sorted = sorted(all_ids)

    # Optionally subsample to a target parcel count.  We pick parcels whose
    # alive lifetime is longest so the dataset is biased toward parcels with
    # many usable training windows.
    if n_target is not None and len(all_ids_sorted) > n_target:
        lifetime = {pid: 0 for pid in all_ids_sorted}
        for _, f in valid_frames:
            for pid in f["orig_id"].tolist():
                lifetime[pid] += 1
        # Top-N by lifetime, ties broken by ID order
        all_ids_sorted = sorted(
            sorted(all_ids_sorted, key=lambda pid: (-lifetime[pid], pid))[:n_target]
        )

    id_to_idx = {pid: i for i, pid in enumerate(all_ids_sorted)}
    T = len(times)
    N = len(all_ids_sorted)

    pos  = np.full((T, N, 2), np.nan, dtype=np.float32)
    vel  = np.full((T, N, 2), np.nan, dtype=np.float32)
    diam = np.full((T, N),    np.nan, dtype=np.float32)
    dens = np.full(N,         1000.0,  dtype=np.float32)
    alive = np.zeros((T, N), dtype=bool)

    valid_iter = iter(valid_frames)
    next_t, next_f = next(valid_iter, (None, None))
    for t_i, t in enumerate(times):
        if next_t is None or abs(next_t - t) > 1e-6:
            continue  # this frame had no Lagrangian data
        f = next_f
        next_t, next_f = next(valid_iter, (None, None))
        for j, pid in enumerate(f["orig_id"].tolist()):
            i = id_to_idx.get(pid)
            if i is None:
                continue
            pos[t_i, i]   = f["pos"][j]
            vel[t_i, i]   = f["vel"][j]
            diam[t_i, i]  = f["diameter"][j]
            dens[i]       = f["density"][j]
            alive[t_i, i] = True

    # Lifetime stats
    if N > 0:
        lt = alive.sum(axis=0)
        print(f"  Parcel-ID tracking: N={N}, lifetime "
              f"min={int(lt.min())}, median={int(np.median(lt))}, "
              f"max={int(lt.max())} of {T} frames; "
              f"avg alive/frame={alive.sum(axis=1).mean():.0f}")

    return dict(
        particle_pos        = pos,
        particle_vel        = vel,
        particle_diam       = diam,
        particle_dens       = dens,
        particle_alive_mask = alive,
        orig_ids            = np.array(all_ids_sorted, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
#  Main extraction routine
# ---------------------------------------------------------------------------

def extract_case(
    case_dir: pathlib.Path,
    output:   pathlib.Path,
    t_start:  float = 2.0,
    t_end:    float = 28.0,
    dt_keep:  float = 0.1,
    n_particles: Optional[int] = None,
    cloud:       str  = "reactingCloud1",
    use_synthetic: bool = False,
) -> None:
    """Extract Eulerian fields and ID-tracked parcel trajectories.

    Args:
        case_dir:    OpenFOAM case directory.
        output:      Output .npz path.
        t_start:     Window start [s].
        t_end:       Window end [s].
        dt_keep:     Sampling interval [s] — must be a multiple of the CFD
                     write interval.
        n_particles: Optional cap on the number of parcels to keep (selected
                     by longest lifetime).  ``None`` keeps every parcel that
                     appears in the window.
        cloud:       Lagrangian cloud name (default ``reactingCloud1``).
        use_synthetic: Generate synthetic data without OpenFOAM (testing).
    """
    output.parent.mkdir(parents=True, exist_ok=True)

    if use_synthetic or not VTK_AVAILABLE:
        print(f"  [synthetic] Generating synthetic test data -> {output.name}")
        data = _synthetic_case()
        np.savez_compressed(output, **data)
        print(f"  Saved: {output}  ({sum(v.nbytes for v in data.values())//1024} KB)")
        return

    # ── Discover Eulerian VTKs and physical-time mapping ───────────────────
    vtk_dir = case_dir / "VTK"
    if not vtk_dir.exists():
        print(f"  [ERROR] No VTK directory in {case_dir}. Run foamToVTK first.")
        sys.exit(1)
    case_name = case_dir.name

    def _stem_index(p: pathlib.Path) -> int:
        try:
            return int(p.stem.split("_")[-1])
        except ValueError:
            return -1

    all_vtk = sorted(
        [p for p in vtk_dir.glob(f"{case_name}_*.vtk") if _stem_index(p) >= 0],
        key=_stem_index,
    )
    if not all_vtk:
        all_vtk = sorted(
            [p for p in vtk_dir.glob("*.vtk") if _stem_index(p) >= 0],
            key=_stem_index,
        )

    def _is_time_dir(p: pathlib.Path) -> bool:
        try:
            float(p.name); return p.is_dir()
        except ValueError:
            return False

    time_dirs = sorted(
        [p for p in case_dir.iterdir() if _is_time_dir(p)],
        key=lambda p: float(p.name),
    )

    if len(all_vtk) == len(time_dirs) and time_dirs:
        vtk_to_time = {p: float(td.name)
                       for p, td in zip(all_vtk, time_dirs)}
    elif all_vtk:
        t_min_dir = float(time_dirs[0].name)  if time_dirs else 0.0
        t_max_dir = float(time_dirs[-1].name) if time_dirs else 30.0
        indices = [_stem_index(p) for p in all_vtk]
        i_min, i_max = indices[0], indices[-1]
        span = max(i_max - i_min, 1)
        vtk_to_time = {
            p: t_min_dir + ((_stem_index(p) - i_min) / span) * (t_max_dir - t_min_dir)
            for p in all_vtk
        }
    else:
        print(f"  [ERROR] No VTK files found in {vtk_dir}.")
        sys.exit(1)

    t_prev = -1e9
    selected: List[Tuple[pathlib.Path, float]] = []
    for p in all_vtk:
        t = vtk_to_time[p]
        if t < t_start or t > t_end:
            continue
        if t - t_prev >= dt_keep - 1e-6:
            selected.append((p, t))
            t_prev = t

    if not selected:
        print(f"  [ERROR] No VTK files found in t=[{t_start},{t_end}].")
        sys.exit(1)
    print(f"  Reading {len(selected)} snapshots from {case_dir.name} ...")

    # ── Read first file to get mesh topology ──────────────────────────────
    first = _read_vtk_unstructured(selected[0][0])
    cell_pos = first["cell_pos"]
    N_cells  = cell_pos.shape[0]
    domain_bounds = np.array([
        [cell_pos[:, 0].min(), cell_pos[:, 0].max()],
        [cell_pos[:, 1].min(), cell_pos[:, 1].max()],
    ], dtype=np.float32)

    # ── Real polyMesh-based d_wall, wall_normal, bc_type ───────────────────
    d_wall      = None
    wall_normal = None
    bc_type     = None
    poly_path   = case_dir / "constant" / "polyMesh"
    if poly_path.exists():
        try:
            from elgin.data.mesh_to_graph import build_mesh_graph
            mg = build_mesh_graph(case_dir)
            # Cell ordering between VTK and polyMesh can differ; only use the
            # mesh-graph fields when the cell counts agree.
            if mg["cell_pos"].shape[0] == N_cells:
                d_wall      = mg["d_wall"]
                wall_normal = mg["wall_normal"]
                bc_type     = mg["bc_type"]
                domain_bounds = mg["domain_bounds"]
            else:
                print(f"  [warn] polyMesh has {mg['cell_pos'].shape[0]} cells "
                      f"but VTK has {N_cells} — falling back to bounding-box "
                      f"d_wall (no obstacle awareness)")
        except Exception as e:
            print(f"  [warn] could not build polyMesh graph: {e}")

    if d_wall is None:
        # Bounding-box fallback (no obstacle awareness, but better than nothing)
        d_wall = np.minimum(
            np.minimum(cell_pos[:, 0] - domain_bounds[0, 0],
                       domain_bounds[0, 1] - cell_pos[:, 0]),
            np.minimum(cell_pos[:, 1] - domain_bounds[1, 0],
                       domain_bounds[1, 1] - cell_pos[:, 1])
        ).astype(np.float32)
        wall_normal = np.zeros((N_cells, 2), dtype=np.float32)
        bc_type     = np.zeros(N_cells, dtype=np.int64)

    # ── Per-case prescribed airInlet velocity (used for conditioning) ──────
    inlet_velocity = np.array(_read_inlet_velocity(case_dir),
                              dtype=np.float32)
    print(f"  Inlet velocity (from 0/U airInlet): "
          f"({inlet_velocity[0]:.4f}, {inlet_velocity[1]:.4f}) m/s  "
          f"|U_in|={np.linalg.norm(inlet_velocity):.4f}")

    # ── Build map of Lagrangian VTK files by step index ───────────────────
    lag_base = vtk_dir / "lagrangian"
    cloud_dir: Optional[pathlib.Path] = None
    if lag_base.exists():
        for candidate in [cloud, "reactingCloud1", "kinematicCloud", "coalCloud"]:
            if (lag_base / candidate).is_dir():
                cloud_dir = lag_base / candidate
                break
        if cloud_dir is None:
            subdirs = [d for d in lag_base.iterdir() if d.is_dir()]
            if subdirs:
                cloud_dir = subdirs[0]
    if cloud_dir is not None:
        print(f"  Lagrangian cloud (VTK): {cloud_dir.name}")

    lag_index_map: dict = {}
    if cloud_dir and cloud_dir.exists():
        for lp in cloud_dir.glob("*.vtk"):
            idx = _stem_index(lp)
            if idx >= 0:
                lag_index_map[idx] = lp

    # ── Read Eulerian fields and per-frame parcel state ───────────────────
    fluid_U, fluid_p, fluid_k, fluid_omega = [], [], [], []
    times: List[float] = []
    for vtk_path, t in selected:
        times.append(t)
        f = _read_vtk_unstructured(vtk_path)
        fluid_U.append(f["U"][:, :2] if f["U"] is not None
                       else np.zeros((N_cells, 2), np.float32))
        fluid_p.append(f["p"] if f["p"] is not None
                       else np.zeros(N_cells, np.float32))
        fluid_k.append(f["k"] if f["k"] is not None
                       else np.full(N_cells, 0.01, np.float32))
        fluid_omega.append(f["omega"] if f["omega"] is not None
                           else np.full(N_cells, 10.0, np.float32))

    lag_frames = _collect_lag_frames(selected, cloud_dir, lag_index_map,
                                     case_dir, cloud)
    n_lag = sum(1 for f in lag_frames if f is not None)
    if n_lag == 0:
        print(f"  [WARNING] No Lagrangian data found for cloud='{cloud}' in "
              f"{case_dir}. Particle fields will be empty.")

    parcel_data = _build_full_timeline(lag_frames, times, n_target=n_particles)

    # ── Assemble Eulerian arrays ──────────────────────────────────────────
    fluid_U     = np.stack(fluid_U,     axis=0).astype(np.float32)
    fluid_p     = np.stack(fluid_p,     axis=0).astype(np.float32)
    fluid_k     = np.stack(fluid_k,     axis=0).astype(np.float32)
    fluid_omega = np.stack(fluid_omega, axis=0).astype(np.float32)
    times_arr   = np.array(times, dtype=np.float32)

    # ── Save ──────────────────────────────────────────────────────────────
    np.savez_compressed(
        output,
        fluid_U        = fluid_U,
        fluid_p        = fluid_p,
        fluid_k        = fluid_k,
        fluid_omega    = fluid_omega,
        cell_pos       = cell_pos,
        d_wall         = d_wall,
        wall_normal    = wall_normal,
        bc_type        = bc_type,
        domain_bounds  = domain_bounds,
        inlet_velocity = inlet_velocity,
        times          = times_arr,
        **parcel_data,
    )
    size_kb = output.stat().st_size // 1024
    N_part = parcel_data["particle_pos"].shape[1]
    print(f"  Saved: {output}  ({size_kb} KB)  "
          f"| cells={N_cells}  parcels={N_part}  timesteps={len(times)}")


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case_dir", type=pathlib.Path, required=True)
    parser.add_argument("--output",   type=pathlib.Path, required=True)
    parser.add_argument("--t_start",  type=float, default=2.0)
    parser.add_argument("--t_end",    type=float, default=28.0)
    parser.add_argument("--dt_keep",  type=float, default=0.1)
    parser.add_argument("--n_particles", type=int, default=None,
                        help="Cap on the number of parcels to retain "
                             "(selected by longest lifetime). Default: keep all.")
    parser.add_argument("--cloud",       type=str, default="reactingCloud1")
    parser.add_argument("--synthetic",   action="store_true",
                        help="Generate synthetic data (no OpenFOAM required).")
    args = parser.parse_args()
    extract_case(
        args.case_dir, args.output,
        t_start=args.t_start, t_end=args.t_end,
        dt_keep=args.dt_keep, n_particles=args.n_particles,
        cloud=args.cloud, use_synthetic=args.synthetic,
    )


if __name__ == "__main__":
    main()
