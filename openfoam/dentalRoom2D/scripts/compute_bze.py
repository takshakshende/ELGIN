#!/usr/bin/env python3
"""
compute_bze.py — Breathing-Zone Exposure post-processor for Revised_Dental cases.
"""

import sys
import os
import struct
import csv
import numpy as np
from pathlib import Path


# ── Breathing zone bounds ────────────────────────────────────────────────────
BZ_X_MIN = 1.30
BZ_X_MAX = 1.80
BZ_Y_MIN = 1.525
BZ_Y_MAX = 1.675

# Total parcels injected (from kinematicCloudProperties: parcelsPerSecond * duration)
PARCELS_PER_SEC = 5000
DURATION        = 30.0          # [s]
N_TOTAL_INJECTED = int(PARCELS_PER_SEC * DURATION)


def read_positions_ascii(filepath):
    """
    Read a foam-extend 4.1 Lagrangian 'positions' file (ASCII format).
    Returns an (N, 3) float64 array of [x, y, z] parcel positions.
    """
    positions = []
    with open(filepath, 'r') as fh:
        lines = fh.readlines()

    # Find the line that contains just the parcel count (an integer)
    in_data = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('(') or stripped.startswith(')'):
            in_data = stripped.startswith('(')
            continue
        if in_data:
            # Each position line: (x y z)
            stripped = stripped.strip('()')
            parts = stripped.split()
            if len(parts) == 3:
                try:
                    positions.append([float(p) for p in parts])
                except ValueError:
                    pass
    return np.array(positions) if positions else np.zeros((0, 3))


def get_time_dirs(case_dir):
    """
    Return a sorted list of (time_value, path) for all numeric time directories
    that contain a lagrangian/kinematicCloud/positions file.
    """
    time_dirs = []
    for entry in case_dir.iterdir():
        if not entry.is_dir():
            continue
        try:
            t = float(entry.name)
        except ValueError:
            continue
        # Support both old kinematicCloud and new reactingCloud1 (evaporation runs)
        pos_file = entry / 'lagrangian' / 'reactingCloud1' / 'positions'
        if not pos_file.exists():
            pos_file = entry / 'lagrangian' / 'kinematicCloud' / 'positions'
        if pos_file.exists():
            time_dirs.append((t, entry, pos_file))
    return sorted(time_dirs, key=lambda x: x[0])


def compute_bze_timeseries(case_dir):
    """
    Compute BZE fraction at each output time step.
    Returns a list of dicts with keys: time, n_bz, n_total, bze.
    """
    time_dirs = get_time_dirs(case_dir)
    if not time_dirs:
        print(f"  WARNING: no Lagrangian position files found in {case_dir}")
        return []

    records = []
    for t, tdir, pos_file in time_dirs:
        positions = read_positions_ascii(pos_file)
        n_total = len(positions)
        if n_total == 0:
            bze = 0.0
            n_bz = 0
        else:
            in_bz = (
                (positions[:, 0] >= BZ_X_MIN) & (positions[:, 0] <= BZ_X_MAX) &
                (positions[:, 1] >= BZ_Y_MIN) & (positions[:, 1] <= BZ_Y_MAX)
            )
            n_bz = int(in_bz.sum())
            # BZE = parcels currently in BZ / total injected up to this time
            n_injected_so_far = min(int(PARCELS_PER_SEC * t) + 1, N_TOTAL_INJECTED)
            bze = n_bz / max(n_injected_so_far, 1)

        records.append({'time': t, 'n_bz': n_bz, 'n_total': n_total, 'bze': bze})
        print(f"  t={t:6.2f}s  n_parcels={n_total:6d}  n_BZ={n_bz:5d}  BZE={bze:.4f}")

    return records


def write_output(case_dir, records):
    """Write BZE time series to postProcessing/bze.csv and summary file."""
    pp_dir = case_dir / 'postProcessing'
    pp_dir.mkdir(exist_ok=True)

    csv_path = pp_dir / 'bze.csv'
    with open(csv_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=['time', 'n_bz', 'n_total', 'bze'])
        writer.writeheader()
        writer.writerows(records)
    print(f"\n  BZE time series written to {csv_path}")

    if not records:
        return

    bze_vals = np.array([r['bze'] for r in records])
    times    = np.array([r['time'] for r in records])

    peak_bze    = float(bze_vals.max())
    peak_time   = float(times[bze_vals.argmax()])
    mean_bze    = float(bze_vals.mean())
    final_bze   = float(bze_vals[-1]) if len(bze_vals) else 0.0
    # BZE threshold exceeded fraction (paper uses BZE_thr = 0.05 = 5%)
    thr         = 0.05
    frac_above  = float((bze_vals > thr).mean())

    summary_path = pp_dir / 'bze_summary.txt'
    with open(summary_path, 'w') as fh:
        fh.write("===== Breathing Zone Exposure Summary =====\n")
        fh.write(f"Case directory  : {case_dir}\n")
        fh.write(f"BZ definition   : x=[{BZ_X_MIN},{BZ_X_MAX}] m, y=[{BZ_Y_MIN},{BZ_Y_MAX}] m\n")
        fh.write(f"BZE threshold   : {thr*100:.0f}% (BZE_thr = {thr})\n\n")
        fh.write(f"Peak BZE        : {peak_bze:.4f} ({peak_bze*100:.2f}%) at t={peak_time:.2f} s\n")
        fh.write(f"Time-mean BZE   : {mean_bze:.4f} ({mean_bze*100:.2f}%)\n")
        fh.write(f"Final BZE       : {final_bze:.4f} ({final_bze*100:.2f}%)\n")
        fh.write(f"Fraction above threshold: {frac_above*100:.1f}% of time steps\n")
        fh.write("\nINFECTION RISK INTERPRETATION (indicative only):\n")
        if peak_bze > 0.20:
            fh.write("  HIGH risk — >20% of injected aerosol reached breathing zone.\n")
        elif peak_bze > 0.05:
            fh.write("  MODERATE risk — peak BZE exceeded 5% threshold.\n")
        else:
            fh.write("  LOW risk — peak BZE remained below 5% threshold.\n")

    print(f"  Summary written to {summary_path}")
    print(f"  Peak BZE = {peak_bze*100:.2f}%  |  Mean BZE = {mean_bze*100:.2f}%")


def main():
    if len(sys.argv) < 2:
        case_dir = Path('.')
    else:
        case_dir = Path(sys.argv[1])

    if not case_dir.is_dir():
        print(f"ERROR: case directory '{case_dir}' not found.")
        sys.exit(1)

    print(f"\n=== compute_bze.py: Breathing Zone Exposure ===")
    print(f"  Case dir  : {case_dir.resolve()}")
    print(f"  BZ bounds : x=[{BZ_X_MIN},{BZ_X_MAX}] m, y=[{BZ_Y_MIN},{BZ_Y_MAX}] m\n")

    records = compute_bze_timeseries(case_dir)
    write_output(case_dir, records)


if __name__ == '__main__':
    main()
