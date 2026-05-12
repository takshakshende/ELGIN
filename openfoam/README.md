# OpenFOAM reference case — `dentalRoom2D` / `Sweep_Case_03`

This folder ships the **foam-extend 4.1** Computational Fluid Dynamics
(CFD) case used to generate the ground-truth aerosol trajectories
evaluated in the ELGIN paper.

| Case folder | Description |
|---|---|
| `dentalRoom2D/` (= `Sweep_Case_03`) | Single demonstration case used throughout the paper.  Ceiling supply velocity $V_{\rm in}=\mathbf{0.10}\,\mathrm{m\,s^{-1}}$, nozzle exit speed $U_{\rm mag}=\mathbf{30}\,\mathrm{m\,s^{-1}}$, spray half-angle $\theta=20^{\circ}$.  This is the held-out test case for the single-case ELGIN checkpoint reported in the paper. |

> **Scope of this repository.** A single CFD case is shipped here so that
> any reader can reproduce the ELGIN training pipeline end-to-end with
> a self-contained data source. 

## Solver and physics

| Quantity | Value |
|---|---|
| Solver | `reactingParcelFoam` (foam-extend 4.1) — transient compressible Navier–Stokes, $k$–$\omega$ SST RANS, fully coupled to the reacting Lagrangian parcel cloud |
| Carrier phase | Dry air, $\rho_{\rm air}\!\approx\!1.2\,\mathrm{kg\,m^{-3}}$, $\nu\!\approx\!1.5\!\times\!10^{-5}\,\mathrm{m^{2}\,s^{-1}}$ |
| Dispersed phase | Water-like saliva droplets, $\rho_p\!=\!997\,\mathrm{kg\,m^{-3}}$, Rosin–Rammler PDF $d_p\in[1,50]\,\mu\mathrm{m}$, $\overline{d_p}\!=\!20\,\mu\mathrm{m}$, $n\!=\!2$ |
| Drag | Schiller–Naumann (`SphereDrag`) |
| Dispersion | Discrete Random Walk (`StochasticDispersionRAS`), turbulent Schmidt number $\mathrm{Sc}_t\!=\!0.7$ implicit |
| Phase change | Wells' $d^{2}$-law via `LiquidEvaporation` (Spalding $B_M$) |
| Wall interaction | `StandardWallInteraction` with `escape` (parcel deposited on impact) |

## Geometry (pseudo-2D dental treatment room)

```
       y=3.0  ┌────────────────  ▲ airInlet  ────────────────┐
              │                  (x∈[1.90,2.10], 0.20 m)     │
              │                                              │
              │                                              │
       y=1.60 │                                  ◀───────────┤  airOutlet
              │                                              │  (x=4.0,
       y=1.40 │                                              │   y∈[1.40,1.60])
              │   ┌─┐                       ┌─┐              │
       y=1.30 │   │ │                       │ │ ← Breathing  │
              │   │ │ Dentist               │ │   zone of    │
       y=0.90 │   │ │ x∈[1.40,1.60]         │ │   dentist    │
              │   │ │ y∈[0,1.40]            │ │              │
              │   │ │                       │ │              │
       y=0.80 │   │ │       Patient: x∈[2.40,2.60], y∈[0,0.80]
              │   │ │      ┌─┐                                │
              │   │ │      │ │                                │
              │   │ │      │ │   ● Nozzle (2.40, 0.90, 0.005) │
       y=0.00 └───┴─┴──────┴─┴────────────────────────────────┘
              x=0.0      1.4 1.6 1.9 2.1  2.4 2.6        x=4.0
```

| Patch | Type | Location |
|---|---|---|
| `airInlet` | velocity inlet | ceiling slot, $x\in[1.90,\,2.10]\,\mathrm{m}$, $y=3.0\,\mathrm{m}$ |
| `airOutlet` | pressure outlet | right wall, $y\in[1.40,\,1.60]\,\mathrm{m}$, $x=4.0\,\mathrm{m}$ |
| `dentistObstacle` | no-slip wall | $x\in[1.40,1.60]$, $y\in[0,1.40]\,\mathrm{m}$ |
| `patientObstacle` | no-slip wall | $x\in[2.40,2.60]$, $y\in[0,0.80]\,\mathrm{m}$ |
| `floor` / `ceiling` / `leftWall` / `rightWall` | no-slip walls | room boundaries |
| `frontAndBack` | empty (pseudo-2D) | $z=0$ and $z=0.01\,\mathrm{m}$ |

Particle injection is a `ConeInjection` from the patient's oral cavity
at $(2.40,\,0.90,\,0.005)\,\mathrm{m}$, directed horizontally toward
the dentist ($-\hat{\mathbf{x}}$), with magnitude $U_{\rm mag}=30\,
\mathrm{m\,s^{-1}}$ and half-angle $\theta=20^{\circ}$.

## Mesh

Structured 25-block `blockMesh` carved around the dentist and patient
obstacles.

- **Raw cell count**: 8 000 (paper SI Table~S2 "Mesh (raw)")
- **Active fluid cells** after obstacle carve-out: **7 704** (paper SI Table~S2 "Mesh (active fluid)")
- Cell size ≈ 0.05 m × 0.03 m (uniform); pseudo-2D, single cell of
  thickness $\Delta z = 0.01\,\mathrm{m}$

## How to run

Source the foam-extend 4.1 environment first, then:

```bash
source $FOAM_INST_DIR/foam-extend-4.1/etc/bashrc

cd dentalRoom2D
./Allrun          # blockMesh → reactingParcelFoam (RANS pre-run) →
                  #             reactingParcelFoam (Lagrangian) → foamToVTK
```


Parcel positions, velocities, diameters, temperatures and active flags
are written to `<time>/lagrangian/reactingCloud1/{positions,U,d,T,active,
origId,...}` at every output interval (`writeInterval 0.1`).  These
files are ingested directly by the ELGIN data pipeline
(`elgin/data/extract_fields.py`).


