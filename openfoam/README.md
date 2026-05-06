# foam-extend 4.1 case — 2D dental-room nanoparticle dispersion

Solver: **`uncoupledKinematicParcelFoam`** — the foam-extend 4.1 equivalent of a one-way-coupled dilute Lagrangian aerosol solver. It advances the incompressible RANS carrier flow (PIMPLE) and tracks kinematic parcels alongside, with **no feedback** from the dispersed phase to the fluid (appropriate for the very dilute biofluid loading produced by a dental spray).

> Note on version: this case is written for foam-extend 4.1 (sourceforge.net/projects/foam-extend). It deliberately uses the classic pre-OpenFOAM-4 layout — blockMeshDict in `constant/polyMesh/`, a split `turbulenceProperties` / `RASProperties`, and the `injectionModel` / `Coeffs` single-model pattern for the cloud.

## Physical setup

- **Domain.** 4 m (width) x 3 m (height) x 0.01 m (pseudo-2D slab, one cell thick in z), representing a vertical slice through a dental treatment room.
- **Ventilation.** Air supply inlet on the ceiling (x in [2.0, 3.0] m, velocity 0.5 m/s downward); air extraction outlet on the floor (x in [0.0, 1.0] m). Other boundaries are no-slip walls.
- **Turbulence.** RANS with k-omega SST.
- **Carrier phase.** Dry air (rho = 1.2 kg/m^3, nu = 1.5e-5 m^2/s).
- **Dispersed phase.** Kinematic Lagrangian cloud of water-like droplets (rho = 1000 kg/m^3), injected 20 cm above the patient head at (x = 2.0, y = 1.3, z = 0.005) m, directed straight upward at 10 m/s with a 20 deg half-angle cone. Parcel diameters drawn from a Rosin-Rammler distribution (1 - 50 micron, d = 20 micron, n = 2). `sphereDrag` + `gravity` particle forces, `stochasticDispersionRAS` subgrid dispersion model, `standardWallInteraction` (stick) patch interaction.
- **Duration.** 30 s (~3 air-exchange cycles for the chosen ventilation geometry), dt = 5 ms, CFL-limited.

## Files

| Path | Purpose |
|---|---|
| `dentalRoom2D/0/U` | velocity boundary conditions |
| `dentalRoom2D/0/p` | pressure boundary conditions |
| `dentalRoom2D/0/k`, `omega`, `nut` | turbulence fields |
| `dentalRoom2D/constant/polyMesh/blockMeshDict` | mesh definition (foam-extend expects this under `constant/polyMesh/`) |
| `dentalRoom2D/constant/kinematicCloudProperties` | Lagrangian cloud definition |
| `dentalRoom2D/constant/transportProperties` | air viscosity and density |
| `dentalRoom2D/constant/turbulenceProperties` | turbulence family selector (`simulationType RASModel`) |
| `dentalRoom2D/constant/RASProperties` | RAS model choice (`kOmegaSST`) |
| `dentalRoom2D/constant/g` | gravity vector |
| `dentalRoom2D/system/controlDict` | solver control and I/O |
| `dentalRoom2D/system/fvSchemes` | discretisation schemes |
| `dentalRoom2D/system/fvSolution` | linear-solver + PIMPLE settings |
| `dentalRoom2D/Allrun` | full pipeline (mesh + solve + foamToVTK) |
| `dentalRoom2D/Allclean` | clean the case directory |

Note that compared to the original OpenFOAM 7 layout, the 2D dental-room case intentionally **does not** include:

- `0/alphac` — not needed by `uncoupledKinematicParcelFoam` (no Eulerian-Lagrangian two-fluid coupling).
- The `cloudInfo` function object — that object is an OpenFOAM >= 4 addition and does not exist in foam-extend 4.1.

## How to run

```bash
# Source the foam-extend 4.1 environment first, e.g.:
#   source $FOAM_INST_DIR/foam-extend-4.1/etc/bashrc

cd dentalRoom2D
./Allrun          # blockMesh + uncoupledKinematicParcelFoam + foamToVTK
```

Parcel positions are written to `<time>/lagrangian/kinematicCloud/positions` at each output interval; parcel diameters, velocities, and IDs live in the adjacent field files (`d`, `U`, `origId`). The `Allrun` script additionally calls `foamToVTK` at the end so the downstream Python pipeline (`scripts/extract_trajectories.py`) can read the parcel tracks directly with `pyvista`.

## Parameter sweep (training-data generation)

To produce a training dataset across a design space, loop `Allrun` over:

- air-supply velocity (0.25 - 1.0 m/s),
- injection velocity `UMag` (5 - 20 m/s),
- injection half-angle `thetaOuter` (10 - 45 deg),
- particle size distribution (`d`, `n` in the Rosin-Rammler block).

Each run yields a trajectory file that becomes one training trajectory for the downstream GNS. Thirty runs are a reasonable pilot-scale dataset for the portfolio demonstration; push to a few hundred for a publication-quality study.

## Known limitations

- The geometry here is an idealised rectangular room without patient / dental chair / staff obstacles. Real-clinic CAD can be imported via `snappyHexMesh` (available in foam-extend 4.1) in a follow-up iteration.
- The flow is two-dimensional for training-data-generation tractability. Three-dimensional extension is straightforward but changes the data-pipeline normalisation statistics.
- foam-extend 4.1 uses dev-of-transpose-of-grad (`dev(T(grad(U)))`) in `divSchemes` rather than the `dev2(...)` form used by OpenFOAM.org >= 4. If you port the case to a newer OpenFOAM, update `system/fvSchemes` accordingly.
