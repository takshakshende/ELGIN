"""mesh_to_graph.py — Convert OpenFOAM blockMesh to a graph for the EulerianGNN.

This script reads the OpenFOAM mesh topology (constant/polyMesh/) and builds
the graph that the Eulerian GNN operates on:

  Nodes: cell centroids  [N_cells × dim]
  Edges: faces between adjacent cells
  Edge features: face normals, face areas, face-to-face distances

The mesh graph is computed ONCE per geometry and reused for all simulations
with that domain (same mesh, different inlet/spray conditions).

Output .npz structure:
    cell_pos       (N_cells, dim)   cell centroid coordinates [m]
    edge_index     (2, E)           [src, dst] face connectivity
    face_normals   (E, dim)         outward unit normal at each face
    face_areas     (E,)             face area / edge length [m or m²]
    face_dists     (E,)             |x_src - x_dst|  [m]
    cell_volumes   (N_cells,)       cell volume / area [m² or m³]
    bc_type        (N_cells,)       integer BC type per cell
    n_cells        scalar           total number of cells

BC type encoding:
    0 — interior cell
    1 — inlet face  (prescribed velocity)
    2 — outlet face (zero-gradient pressure)
    3 — wall        (no-slip, U=0)
    4 — symmetry    (zero normal gradient)

Usage
-----
    python elgin/data/mesh_to_graph.py \\
        --case_dir  openfoam/dentalRoom2D \\
        --output    experiments/elgin_case03/datasets/mesh_graph.npz
"""

from __future__ import annotations
import argparse
import pathlib
import sys
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
#  OpenFOAM polyMesh reader (pure Python, no VTK needed)
# ---------------------------------------------------------------------------

def _read_foam_list(path: pathlib.Path) -> list:
    """Read an OpenFOAM ASCII list file, return list of entries."""
    lines = path.read_text(encoding="utf-8").splitlines()
    data = []
    in_list = False
    for line in lines:
        line = line.strip()
        if line.startswith("//") or line.startswith("FoamFile") or line == "":
            continue
        if line.endswith("(") and not in_list:
            in_list = True
            continue
        if line == ")" and in_list:
            in_list = False
            continue
        if in_list:
            data.append(line)
    return data


def _parse_points(case_dir: pathlib.Path) -> np.ndarray:
    path = case_dir / "constant" / "polyMesh" / "points"
    if not path.exists():
        return None
    raw = _read_foam_list(path)
    pts = []
    for line in raw:
        line = line.replace("(", "").replace(")", "")
        vals = line.split()
        if len(vals) >= 2:
            pts.append([float(v) for v in vals[:3]])
    return np.array(pts, dtype=np.float32)


def _parse_faces(case_dir: pathlib.Path):
    path = case_dir / "constant" / "polyMesh" / "faces"
    if not path.exists():
        return None
    raw = _read_foam_list(path)
    faces = []
    for line in raw:
        line = line.strip()
        if not line:
            continue
        if "(" in line:
            inner = line[line.index("(")+1:line.index(")")]
            verts = list(map(int, inner.split()))
            faces.append(verts)
    return faces


def _parse_owner_neighbour(case_dir: pathlib.Path):
    owner_path = case_dir / "constant" / "polyMesh" / "owner"
    neigh_path = case_dir / "constant" / "polyMesh" / "neighbour"
    if not owner_path.exists():
        return None, None
    owner  = [int(x) for x in _read_foam_list(owner_path) if x.strip().lstrip('-').isdigit()]
    neigh  = [int(x) for x in _read_foam_list(neigh_path) if x.strip().lstrip('-').isdigit()]
    return owner, neigh


def _parse_boundary(case_dir: pathlib.Path):
    """Parse boundary patches → return list of (name, type, startFace, nFaces)."""
    path = case_dir / "constant" / "polyMesh" / "boundary"
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    patches = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Skip empty lines, comments, FoamFile header, parens, count lines
        # (the patch list is preceded by an integer giving the number of
        # patches), and OpenFOAM keyword lines (anything ending in ';').
        if (line
                and not line.startswith("//")
                and not line.startswith("FoamFile")
                and not line.startswith("{")
                and line != "("
                and line != ")"
                and not line.isdigit()
                and not line.endswith(";")):
            # Potential patch name
            name = line
            # Look for type, startFace, nFaces in the next brace block
            block_text = ""
            j = i + 1
            depth = 0
            while j < len(lines):
                bl = lines[j].strip()
                if "{" in bl: depth += 1
                if "}" in bl:
                    depth -= 1
                    if depth <= 0:
                        break
                block_text += bl + " "
                j += 1
            def _extract(key, text):
                idx = text.find(key)
                if idx < 0: return None
                rest = text[idx + len(key):].strip()
                val = rest.split()[0].rstrip(";")
                return val
            btype     = _extract("type",       block_text)
            start_f   = _extract("startFace",  block_text)
            n_faces   = _extract("nFaces",     block_text)
            if btype and start_f and n_faces:
                patches.append((name, btype, int(start_f), int(n_faces)))
            i = j
        i += 1
    return patches


# BC type mapping
#
#   0 = interior  (generic ``patch`` whose name doesn't match anything below
#                  is treated as interior so the network falls back to its
#                  learned bulk dynamics)
#   1 = airInlet            -- velocity / k / omega prescribed
#   2 = airOutlet           -- zero-gradient outflow
#   3 = generic room wall   -- left/right walls, walls without a known label
#   4 = floor
#   5 = ceiling
#   6 = dentistObstacle     -- dentist body / chest / face
#   7 = patientObstacle     -- patient face / chest
#   8 = symmetryPlane
#
# ``bc_type_count`` in config.py must be at least max(bc_id) + 1 = 9.
#
# We classify by patch *name* first (so the airInlet/airOutlet patches that
# OpenFOAM declares with `type patch` are picked up correctly) and only fall
# back on the OpenFOAM `type` keyword when the name doesn't match.
_NAME_RULES: list[tuple[str, int]] = [
    ("inlet",   1),
    ("outlet",  2),
    ("floor",   4),
    ("ceiling", 5),
    ("dentist", 6),
    ("patient", 7),
]

_TYPE_MAP = {
    "patch":          0,
    "wall":           3,
    "inlet":          1,
    "outlet":         2,
    "symmetryPlane":  8,
    "empty":          0,
    "cyclic":         0,
}


def classify_patch(name: str, ftype: str | None) -> int:
    """Map an OpenFOAM (patchName, patchType) to an ELGIN BC integer."""
    n = name.lower()
    for kw, bc_id in _NAME_RULES:
        if kw in n:
            return bc_id
    if ftype is not None:
        return _TYPE_MAP.get(ftype, 0)
    return 0


# Reverse lookup for diagnostic prints
_BC_NAME = {
    0: "interior", 1: "inlet",   2: "outlet",
    3: "wall",     4: "floor",   5: "ceiling",
    6: "dentist",  7: "patient", 8: "symmetry",
}


def _wall_bc_ids() -> set[int]:
    """BC ids that represent solid no-slip surfaces (used for d_wall)."""
    return {3, 4, 5, 6, 7}   # generic_wall, floor, ceiling, dentist, patient


def build_mesh_graph(case_dir: pathlib.Path) -> dict:
    """Build the Eulerian GNN graph from OpenFOAM polyMesh.

    Returns a dict with:
        cell_pos       (N_cells, 2)
        edge_index     (2, E)         bidirectional internal-face graph
        face_normals   (E, 2)         outward normal at each edge (E = 2 * n_internal)
        face_areas     (E,)
        face_dists     (E,)
        cell_volumes   (N_cells,)
        bc_type        (N_cells,)     0..8, see classify_patch()
        face_type      (E,)           0..4 BC label per *internal* edge (one
                                      side per pair).  All 0 here because
                                      internal faces are not boundary faces;
                                      kept for the model's edge-BC embedding.
        d_wall         (N_cells,)     true Euclidean distance to nearest
                                      solid wall face (incl. dentist & patient)
        wall_normal    (N_cells, 2)   unit vector from cell centroid to its
                                      nearest wall-face midpoint (zero for
                                      cells that have no wall in the case)
        wall_face_pos  (N_walls, 2)   midpoints of every wall face
        wall_face_normal (N_walls, 2) outward unit normal of every wall face
        wall_face_bc   (N_walls,)     BC id of each wall face (3..7)
        domain_bounds  (2, 2)         [[xmin, xmax], [ymin, ymax]]
        patch_info     dict           name -> (bc_id, n_faces) — for diagnostics
        n_cells        scalar
    """
    points  = _parse_points(case_dir)
    faces   = _parse_faces(case_dir)
    owner, neighbour = _parse_owner_neighbour(case_dir)
    patches = _parse_boundary(case_dir)

    if points is None or faces is None or owner is None:
        raise FileNotFoundError(
            f"OpenFOAM polyMesh not found in {case_dir}/constant/polyMesh/. "
            "Ensure you have run blockMesh and that constant/polyMesh/ exists."
        )

    n_cells = max(owner) + 1

    # Cell centroids: average of face-point coordinates
    cell_pts = [[] for _ in range(n_cells)]
    for fi, face_verts in enumerate(faces):
        c = owner[fi]
        fc = points[face_verts].mean(axis=0)
        cell_pts[c].append(fc)
    cell_pos = np.array([np.mean(pts, axis=0) if pts else [0., 0., 0.]
                         for pts in cell_pts], dtype=np.float32)[:, :2]

    # ── Internal-face graph ────────────────────────────────────────────────
    n_internal = len(neighbour)
    src_idx = np.array(owner[:n_internal], dtype=np.int64)
    dst_idx = np.array(neighbour,          dtype=np.int64)
    edge_index = np.stack([
        np.concatenate([src_idx, dst_idx]),
        np.concatenate([dst_idx, src_idx]),
    ], axis=0)

    normals_raw, areas_raw = [], []
    for fi in range(n_internal):
        verts = faces[fi]
        pts_f = points[verts]
        p0, p1 = pts_f[0, :2], pts_f[1 % len(verts), :2]
        edge_v = p1 - p0
        length = np.linalg.norm(edge_v)
        if length < 1e-10:
            normal = np.array([1.0, 0.0])
        else:
            normal = np.array([-edge_v[1], edge_v[0]]) / length
            c_owner, c_neigh = owner[fi], neighbour[fi]
            dir_vec = cell_pos[c_neigh] - cell_pos[c_owner]
            if np.dot(normal, dir_vec) < 0:
                normal = -normal
        normals_raw.append(normal)
        areas_raw.append(length)

    normals = np.stack(normals_raw).astype(np.float32)
    areas   = np.array(areas_raw, dtype=np.float32)
    face_normals = np.concatenate([normals, -normals], axis=0)
    face_areas   = np.concatenate([areas, areas])

    src_pos = cell_pos[edge_index[0]]
    dst_pos = cell_pos[edge_index[1]]
    face_dists = np.linalg.norm(dst_pos - src_pos, axis=-1).astype(np.float32)
    face_dists = np.maximum(face_dists, 1e-8)

    # 2-D approximate cell areas
    area_accum = np.zeros(n_cells, dtype=np.float32)
    for fi, a in enumerate(areas):
        area_accum[owner[fi]] += a
        if fi < n_internal:
            area_accum[neighbour[fi]] += a
    cell_volumes = (area_accum / 4.0).clip(min=1e-8)

    # ── Boundary-face geometry per patch ───────────────────────────────────
    bc_type     = np.zeros(n_cells, dtype=np.int64)
    patch_info: dict[str, tuple[int, int]] = {}

    wall_face_pos:    list[np.ndarray] = []
    wall_face_normal: list[np.ndarray] = []
    wall_face_bc:     list[int]         = []

    wall_ids = _wall_bc_ids()

    for patch_name, patch_type, start_face, n_patch_faces in patches:
        bc_id = classify_patch(patch_name, patch_type)
        patch_info[patch_name] = (bc_id, n_patch_faces)

        # Skip patches classified as interior/empty (e.g. ``frontAndBack``
        # in a 2-D extruded mesh).  Every cell touches them, so writing
        # bc_id=0 there would clobber the real BCs assigned by walls /
        # inlets / outlets.
        if bc_id == 0:
            continue

        for fi in range(start_face, start_face + n_patch_faces):
            if fi >= len(owner):
                continue

            if bc_type[owner[fi]] == 0:
                bc_type[owner[fi]] = bc_id

            # Geometric data for any solid wall (incl. dentist/patient)
            if bc_id in wall_ids and fi < len(faces):
                verts = faces[fi]
                pts_f = points[verts]
                p0, p1 = pts_f[0, :2], pts_f[1 % len(verts), :2]
                edge_v = p1 - p0
                length = np.linalg.norm(edge_v)
                if length < 1e-12:
                    continue
                fmid   = 0.5 * (p0 + p1)
                normal = np.array([-edge_v[1], edge_v[0]], dtype=np.float32) / length
                # Orient outward (away from the owner cell)
                if np.dot(fmid - cell_pos[owner[fi]], normal) < 0:
                    normal = -normal
                wall_face_pos.append(fmid.astype(np.float32))
                wall_face_normal.append(normal.astype(np.float32))
                wall_face_bc.append(bc_id)

    # ── Real d_wall and wall_normal per cell ───────────────────────────────
    if wall_face_pos:
        wfp = np.stack(wall_face_pos)              # (W, 2)
        wfn = np.stack(wall_face_normal)           # (W, 2)
        wfb = np.array(wall_face_bc, dtype=np.int64)
        # Brute-force nearest-wall search (n_cells ≤ ~20k, n_walls ≤ ~1k)
        diff = cell_pos[:, None, :] - wfp[None, :, :]      # (N, W, 2)
        dist = np.linalg.norm(diff, axis=-1)               # (N, W)
        nearest = dist.argmin(axis=1)
        d_wall  = dist[np.arange(n_cells), nearest].astype(np.float32)
        # Unit vector from cell centroid TOWARDS its nearest wall point
        delta   = wfp[nearest] - cell_pos                  # (N, 2)
        norm_d  = np.linalg.norm(delta, axis=-1, keepdims=True).clip(min=1e-12)
        wall_normal = (delta / norm_d).astype(np.float32)
    else:
        wfp = np.zeros((0, 2), dtype=np.float32)
        wfn = np.zeros((0, 2), dtype=np.float32)
        wfb = np.zeros(0, dtype=np.int64)
        d_wall      = np.full(n_cells, 1.0, dtype=np.float32)
        wall_normal = np.zeros((n_cells, 2), dtype=np.float32)

    # face_type (per-edge BC label) — internal faces are 0
    face_type = np.zeros(2 * n_internal, dtype=np.int64)

    domain_bounds = np.array([
        [cell_pos[:, 0].min(), cell_pos[:, 0].max()],
        [cell_pos[:, 1].min(), cell_pos[:, 1].max()],
    ], dtype=np.float32)

    # Diagnostics
    print("  Patch summary:")
    for pname, (bid, nf) in patch_info.items():
        print(f"    {pname:24s}  bc_id={bid} ({_BC_NAME.get(bid,'?'):8s})"
              f"  n_faces={nf}")
    n_wall_faces = wfp.shape[0]
    print(f"  d_wall: range [{d_wall.min():.3f}, {d_wall.max():.3f}] m"
          f" computed from {n_wall_faces} solid-wall faces")

    return dict(
        cell_pos          = cell_pos,
        edge_index        = edge_index,
        face_normals      = face_normals,
        face_areas        = face_areas,
        face_dists        = face_dists,
        face_type         = face_type,
        cell_volumes      = cell_volumes,
        bc_type           = bc_type.astype(np.int64),
        d_wall            = d_wall,
        wall_normal       = wall_normal,
        wall_face_pos     = wfp,
        wall_face_normal  = wfn,
        wall_face_bc      = wfb,
        domain_bounds     = domain_bounds,
        n_cells           = np.array([n_cells]),
    )


def build_synthetic_graph(n_cells: int = 800, Lx: float = 4.0,
                           Ly: float = 3.0) -> dict:
    """Build a structured 2D grid graph for testing (no OpenFOAM needed)."""
    nx = int(np.sqrt(n_cells * Lx / Ly))
    ny = int(n_cells / nx)
    n_cells = nx * ny
    xs = np.linspace(Lx / (2*nx), Lx - Lx/(2*nx), nx, dtype=np.float32)
    ys = np.linspace(Ly / (2*ny), Ly - Ly/(2*ny), ny, dtype=np.float32)
    XX, YY = np.meshgrid(xs, ys)
    cell_pos = np.stack([XX.ravel(), YY.ravel()], axis=-1)

    # Structured neighbour connectivity
    edges_src, edges_dst = [], []
    normals_list, areas_list = [], []
    dx = xs[1] - xs[0]; dy = ys[1] - ys[0]

    def idx(i, j): return j * nx + i

    for j in range(ny):
        for i in range(nx):
            ci = idx(i, j)
            # Right neighbour
            if i + 1 < nx:
                cj = idx(i+1, j)
                edges_src += [ci, cj]; edges_dst += [cj, ci]
                normals_list += [[1.,0.], [-1.,0.]]
                areas_list   += [dy, dy]
            # Top neighbour
            if j + 1 < ny:
                cj = idx(i, j+1)
                edges_src += [ci, cj]; edges_dst += [cj, ci]
                normals_list += [[0.,1.], [0.,-1.]]
                areas_list   += [dx, dx]

    edge_index   = np.array([edges_src, edges_dst], dtype=np.int64)
    face_normals = np.array(normals_list, dtype=np.float32)
    face_areas   = np.array(areas_list,  dtype=np.float32)
    src_p = cell_pos[edge_index[0]]; dst_p = cell_pos[edge_index[1]]
    face_dists   = np.linalg.norm(dst_p - src_p, axis=-1).clip(1e-8).astype(np.float32)
    cell_volumes = np.full(n_cells, dx * dy, dtype=np.float32)

    bc_type = np.zeros(n_cells, dtype=np.int64)
    # Left column = inlet (1), right = outlet (2), bottom = floor (4),
    # top = ceiling (5)
    for j in range(ny):
        bc_type[idx(0, j)]    = 1
        bc_type[idx(nx-1, j)] = 2
    for i in range(nx):
        bc_type[idx(i, 0)]    = 4
        bc_type[idx(i, ny-1)] = 5

    # Synthetic d_wall = vertical distance to nearest top/bottom edge
    d_wall = np.minimum(cell_pos[:, 1], Ly - cell_pos[:, 1]).astype(np.float32)
    wall_normal = np.zeros_like(cell_pos)
    wall_normal[:, 1] = np.where(cell_pos[:, 1] < Ly / 2, -1.0, 1.0)

    domain_bounds = np.array([[0.0, Lx], [0.0, Ly]], dtype=np.float32)

    return dict(
        cell_pos=cell_pos, edge_index=edge_index,
        face_normals=face_normals, face_areas=face_areas,
        face_dists=face_dists, cell_volumes=cell_volumes,
        bc_type=bc_type,
        face_type=np.zeros(edge_index.shape[1], dtype=np.int64),
        d_wall=d_wall,
        wall_normal=wall_normal,
        wall_face_pos=np.zeros((0, 2), dtype=np.float32),
        wall_face_normal=np.zeros((0, 2), dtype=np.float32),
        wall_face_bc=np.zeros(0, dtype=np.int64),
        domain_bounds=domain_bounds,
        n_cells=np.array([n_cells]),
    )


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case_dir", type=pathlib.Path)
    parser.add_argument("--output",   type=pathlib.Path, required=True)
    parser.add_argument("--synthetic", action="store_true",
                        help="Build a structured Cartesian grid (no OpenFOAM).")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.synthetic or args.case_dir is None:
        print("[mesh_to_graph] Building synthetic structured grid ...")
        g = build_synthetic_graph()
    else:
        print(f"[mesh_to_graph] Reading polyMesh from {args.case_dir} ...")
        g = build_mesh_graph(args.case_dir)

    np.savez_compressed(args.output, **g)
    N = int(g["n_cells"][0])
    E = g["edge_index"].shape[1]
    print(f"  Saved: {args.output}  (cells={N}, edges={E})")

if __name__ == "__main__":
    main()
