"""CfdGNNDataset — PyTorch dataset for the full ELGIN framework.
"""

from __future__ import annotations
import pathlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _alive_window_mask(alive: np.ndarray,
                       t: int,
                       history_len: int,
                       future_len: int = 1) -> np.ndarray:
    """Boolean mask of parcels alive in every frame of [t-H, t+future_len].

    ``future_len = 1`` reproduces the original one-step training window
    [t-H, t+1].  ``future_len = K`` requires the parcel to also be alive at
    every step out to t+K, which is what genuine multi-step BPTT
    supervision needs.
    """
    return alive[t - history_len: t + 1 + future_len].all(axis=0)   # (N,)


def _resolve_alive_mask(data: np.ndarray | dict,
                        npz_path: pathlib.Path) -> np.ndarray:
    """Return ``alive_mask (T, N)`` for an .npz file.

    Falls back to a not-NaN check on ``particle_pos`` if the file predates
    the ID-tracking change.  As a last resort treats every parcel as alive
    throughout the window — this preserves backward compatibility with the
    legacy zero-padded extractor but should be considered unsafe for
    training (positions across time do not represent the same physical
    parcel).
    """
    if "particle_alive_mask" in data:
        return np.asarray(data["particle_alive_mask"], dtype=bool)

    pos = data["particle_pos"]                                # (T, N, 2)
    if np.isnan(pos).any():
        return ~np.isnan(pos).any(axis=-1)                    # (T, N)

    print(f"  [warn] {npz_path.name}: no particle_alive_mask and no NaNs in "
          f"particle_pos — assuming every parcel is alive in every frame. "
          f"Re-run extract_fields.py to enable per-ID tracking.")
    return np.ones(pos.shape[:2], dtype=bool)


# ---------------------------------------------------------------------------
#  Dataset
# ---------------------------------------------------------------------------

class CfdGNNDataset(Dataset):
    """Dataset for training the full ELGIN model.

    Args:
        npz_files:    List of .npz files from extract_fields.py.
        history_len:  Number of past frames H (particle history window).
        noise_std:    Gaussian noise added to inputs (0 for val/test).
        n_particles:  Fixed parcel count per sample. For each (case, t) we
                      sample this many parcels from those that are alive
                      throughout the [t-H, t+1] window. ``None`` falls back
                      to the smallest workable count across all cases.
        seed:         Base seed for reproducible parcel sub-sampling.
    """

    def __init__(
        self,
        npz_files:   List[pathlib.Path],
        history_len: int   = 5,
        noise_std:   float = 0.0,
        n_particles: Optional[int] = None,
        seed:        int   = 42,
        future_len:  int   = 1,
    ):
        """Args (additions):
            future_len: number of future ground-truth particle frames to
                        return per sample.  ``1`` keeps the legacy one-step
                        training behaviour; ``K > 1`` enables genuine
                        K-step BPTT supervision in Stage 4.
        """
        super().__init__()
        self.history_len = history_len
        self.future_len  = max(1, int(future_len))
        self.noise_std   = noise_std
        self.seed        = seed

        if not npz_files:
            raise ValueError("npz_files is empty.")

        # ── First pass: probe sizes and decide on a target N ────────────────
        per_case_probe: List[Tuple[pathlib.Path, np.ndarray, int]] = []
        for p in npz_files:
            d = np.load(p, mmap_mode="r")
            alive = _resolve_alive_mask(d, p)
            T = alive.shape[0]
            min_alive_per_window = []
            for t in range(history_len, T - self.future_len):
                m = _alive_window_mask(alive, t, history_len,
                                       self.future_len).sum()
                if m > 0:
                    min_alive_per_window.append(int(m))
            best_floor = (max(min_alive_per_window) if min_alive_per_window
                          else 0)
            per_case_probe.append((p, alive, best_floor))

        if n_particles is None:
            target_N = min(c[2] for c in per_case_probe if c[2] > 0)
            target_N = max(1, target_N)
        else:
            target_N = int(n_particles)
        self.n_particles = target_N

        # ── Second pass: load and build (case, t) -> alive-particle index ──
        self._cases: List[Dict[str, np.ndarray]] = []
        self._index: List[Tuple[int, int, np.ndarray]] = []  # (ci, t, alive_idx)

        skipped = 0
        for ci, (npz_path, alive, _) in enumerate(per_case_probe):
            data = np.load(npz_path)
            pos  = np.asarray(data["particle_pos"], dtype=np.float32)
            # Replace NaN with zeros — caller never sees frames where the
            # parcel is not alive because we mask via _alive_window_mask.
            pos_clean = np.where(np.isnan(pos), 0.0, pos)

            # Diameter and density may be (T, N) or (N,) depending on
            # extractor version.
            d_p_arr = np.asarray(data["particle_diam"], dtype=np.float32)
            if d_p_arr.ndim == 2:
                # Use per-parcel last-known diameter as a constant proxy
                # (good enough for the drag feature when evap is off).
                d_p_const = np.where(np.isnan(d_p_arr), 5e-6, d_p_arr)
                d_p_const = d_p_const.mean(axis=0).astype(np.float32)
            else:
                d_p_const = np.where(np.isnan(d_p_arr), 5e-6, d_p_arr).astype(np.float32)

            rho_p = np.asarray(data["particle_dens"], dtype=np.float32)
            if rho_p.ndim == 2:
                rho_p = rho_p.mean(axis=0).astype(np.float32)
            rho_p = np.where(np.isnan(rho_p) | (rho_p <= 0.0),
                             1000.0, rho_p).astype(np.float32)

            inlet_vel = (np.asarray(data["inlet_velocity"], dtype=np.float32)
                         if "inlet_velocity" in data.files
                         else np.zeros(2, dtype=np.float32))

            case = {
                "fluid_U":      np.asarray(data["fluid_U"],     dtype=np.float32),
                "fluid_p":      np.asarray(data["fluid_p"],     dtype=np.float32),
                "fluid_k":      np.asarray(data["fluid_k"],     dtype=np.float32),
                "fluid_omega":  np.asarray(data["fluid_omega"], dtype=np.float32),
                "particle_pos": pos_clean,
                "particle_diam": d_p_const,
                "particle_dens": rho_p,
                "alive":         alive,
                "inlet_velocity": inlet_vel,
            }
            T = case["fluid_U"].shape[0]
            self._cases.append(case)

            n_added = 0
            for t in range(history_len, T - self.future_len):
                window_mask = _alive_window_mask(alive, t, history_len,
                                                 self.future_len)
                alive_idx = np.flatnonzero(window_mask)
                if alive_idx.size < target_N:
                    skipped += 1
                    continue
                self._index.append((ci, t, alive_idx))
                n_added += 1
            # End per-case loop

        if not self._index:
            raise RuntimeError(
                "No training windows have at least n_particles alive parcels. "
                f"Tried n_particles={target_N}. Reduce n_particles or extend "
                "the simulation window in extract_fields.py."
            )

        print(f"  [CfdGNNDataset] {len(npz_files)} cases | "
              f"N_particles/sample={target_N} | "
              f"{len(self._index)} samples ({skipped} skipped: not enough "
              f"alive parcels in window)")

    def __len__(self) -> int:
        return len(self._index)

    def _sample_indices(self, ci: int, t: int,
                        alive_idx: np.ndarray) -> np.ndarray:
        """Deterministic per-sample parcel selection."""
        rng = np.random.default_rng((self.seed, ci, t))
        sel = rng.choice(alive_idx, size=self.n_particles, replace=False)
        sel.sort()
        return sel

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ci, t, alive_idx = self._index[idx]
        case = self._cases[ci]
        sel  = self._sample_indices(ci, t, alive_idx)

        def _fluid(tt: int) -> np.ndarray:
            U   = case["fluid_U"][tt]
            p   = case["fluid_p"][tt, :, None]
            k   = case["fluid_k"][tt, :, None]
            om  = case["fluid_omega"][tt, :, None]
            return np.concatenate([U, p, k, om], axis=-1)

        fluid_in  = torch.from_numpy(_fluid(t))
        fluid_tgt = torch.from_numpy(_fluid(t + 1))

        # Future fluid trajectory for genuine K-step BPTT with GT fluid.
        # Shape: (K, N_cells, 5).  Element [k] is the fluid field at time
        # t + 1 + k, i.e. the fluid the Lagrangian should see when
        # predicting the particle position at step k+1 of the rollout.
        # When ``future_len = 1`` this collapses to a single frame and is
        # equivalent to the legacy ``fluid_tgt``.
        if self.future_len > 1:
            fluid_traj_future_list = [
                _fluid(t + 1 + k) for k in range(self.future_len)
            ]
            fluid_traj_future = torch.from_numpy(
                np.stack(fluid_traj_future_list, axis=0)        # (K, N_cells, 5)
            ).contiguous()
        else:
            fluid_traj_future = fluid_tgt.unsqueeze(0)          # (1, N_cells, 5)

        p_hist = case["particle_pos"][t - self.history_len: t + 1, sel, :]
        p_hist = torch.from_numpy(np.transpose(p_hist, (1, 0, 2)))   # (N, H+1, 2)
        p_tgt  = torch.from_numpy(case["particle_pos"][t + 1, sel, :])

        # Future trajectory window for genuine K-step BPTT supervision.
        # Shape: (N, K, 2) where K = future_len; element [:, 0, :] equals
        # ``particle_tgt`` (one-step lookahead) for backward compatibility.
        p_traj_future_np = case["particle_pos"][
            t + 1: t + 1 + self.future_len, sel, :
        ]                                                           # (K, N, 2)
        p_traj_future = torch.from_numpy(
            np.transpose(p_traj_future_np, (1, 0, 2))               # (N, K, 2)
        ).contiguous()

        d_p    = torch.from_numpy(case["particle_diam"][sel])
        rho_p  = torch.from_numpy(case["particle_dens"][sel])

        if self.noise_std > 0.0:
            fluid_in = fluid_in + torch.randn_like(fluid_in) * self.noise_std

        return {
            "fluid_in":      fluid_in,
            "fluid_tgt":     fluid_tgt,
            "fluid_traj_future": fluid_traj_future,
            "particle_hist": p_hist,
            "particle_tgt":  p_tgt,
            "particle_traj_future": p_traj_future,
            "d_p":           d_p,
            "rho_p":         rho_p,
            "inlet_velocity": torch.from_numpy(case["inlet_velocity"]),
        }


# ---------------------------------------------------------------------------
#  Normalisation statistics  (now mask-aware)
# ---------------------------------------------------------------------------

def compute_normalisation_stats(
    npz_files: List[pathlib.Path],
    history_len: int = 5,
) -> Dict[str, np.ndarray]:
    """Compute per-variable mean and std from training files.

    Velocities/accelerations are computed only over time-pairs where the
    *same* parcel is alive in both frames (i.e. inside its own
    ``alive_mask`` window).  This stops zero-padded ghost parcels from
    inflating ``acc_std``.
    """
    all_U, all_p, all_k, all_omega = [], [], [], []
    all_vel, all_acc = [], []

    for f in npz_files:
        d = np.load(f)
        all_U.append(np.asarray(d["fluid_U"]).reshape(-1, 2))
        all_p.append(np.asarray(d["fluid_p"]).ravel())
        all_k.append(np.asarray(d["fluid_k"]).ravel())
        all_omega.append(np.asarray(d["fluid_omega"]).ravel())

        pos   = np.asarray(d["particle_pos"], dtype=np.float32)   # (T, N, 2)
        alive = _resolve_alive_mask(d, f)                         # (T, N)
        T = pos.shape[0]

        if T >= 2:
            both = alive[1:] & alive[:-1]                         # (T-1, N)
            vel  = pos[1:] - pos[:-1]                             # (T-1, N, 2)
            sel  = both & ~np.isnan(vel).any(axis=-1)
            if sel.any():
                all_vel.append(vel[sel])
        if T >= 3:
            triple = alive[2:] & alive[1:-1] & alive[:-2]
            acc = pos[2:] - 2 * pos[1:-1] + pos[:-2]
            sel = triple & ~np.isnan(acc).any(axis=-1)
            if sel.any():
                all_acc.append(acc[sel])

    U_arr = np.concatenate(all_U)
    p_arr = np.concatenate(all_p)
    k_arr = np.concatenate(all_k)
    om_arr = np.concatenate(all_omega)

    fluid_mean = np.concatenate([
        U_arr.mean(0), [p_arr.mean()], [k_arr.mean()], [om_arr.mean()]
    ]).astype(np.float32)
    fluid_std = np.concatenate([
        U_arr.std(0).clip(1e-8),
        [p_arr.std().clip(1e-8)],
        [k_arr.std().clip(1e-8)],
        [om_arr.std().clip(1e-8)],
    ]).astype(np.float32)

    vel_arr = (np.concatenate(all_vel) if all_vel
               else np.zeros((1, 2), np.float32))
    acc_arr = (np.concatenate(all_acc) if all_acc
               else np.zeros((1, 2), np.float32))

    return {
        "fluid_mean": fluid_mean,
        "fluid_std":  fluid_std,
        "vel_mean":   vel_arr.mean(0).tolist(),
        "vel_std":    vel_arr.std(0).clip(1e-8).tolist(),
        "acc_mean":   acc_arr.mean(0).tolist(),
        "acc_std":    acc_arr.std(0).clip(1e-8).tolist(),
    }
