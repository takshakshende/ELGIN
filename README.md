# ELGIN — Eulerian–Lagrangian Graph Interaction Network

<p align="center">
  <img src="assets/elgin_banner.png" alt="ELGIN Banner" width="800"/>
</p>

<p align="center">
  <a href="#motivation">Motivation</a> •
  <a href="#methodology">Methodology</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#training">Training</a> •
  <a href="#prediction">Prediction</a> •
  <a href="#results">Results</a> •
  <a href="#citation">Citation</a>
</p>

<p align="center">
  <img src="assets/fluid_speed_compare.gif"
       alt="ELGIN vs OpenFOAM — fluid velocity field with aerosol particle cloud (26-second rollout)"
       width="720"/>
  <br/>
  <em>
  <strong>Left:</strong> ELGIN prediction &nbsp;|&nbsp;
  <strong>Right:</strong> OpenFOAM (reactingParcelFoam) ground truth.<br/>
  Background colour = air speed |U| [m/s] &nbsp;·&nbsp;
  Lime dots = aerosol particles &nbsp;·&nbsp;
  Full 26-second rollout predicted in ~64 s on an NVIDIA Quadro P1000 (4 GB VRAM).
  </em>
</p>

---

## Motivation

Dental procedures — high-speed drilling (≥ 300,000 rpm), ultrasonic
scaling, and air-polishing — generate polydisperse bioaerosol
clouds whose sub-50 µm droplet nuclei can remain airborne for tens of
minutes in poorly ventilated clinical rooms.  These particles carry
bacteria (*M. tuberculosis*, oral streptococci), viruses (SARS-CoV-2,
hepatitis B), and fungal spores, creating a quantifiable infection risk
for patients, dental workers, and bystanders.

Classical Euler–Lagrange CFD with foam-extend 4.1
`reactingParcelFoam` resolves this aerosol transport with high
fidelity, but requires ≈ 40 CPU-minutes per case, far too slow
for per-appointment infection-risk screening, Monte Carlo ventilation
sweeps, or personalised treatment planning.

**ELGIN** is a physics-informed Graph Neural Network surrogate that
provides, on the single demonstration case shipped
with this repository:

| Property | Value |
|---|---|
| Rollout wall-clock | **~64 s** per 26-second trajectory |
| Speed-up over CFD | **~37×** |
| Trajectory fidelity (MDE) | **16.2 %** of room width (4 m) — i.e. ~0.65 m mean parcel displacement error |
| Cloud spread fidelity (Rg-err) | **6.6 %** |
| Kinetic-energy preservation (KE-ratio) | **0.66** (∈ [0, 1]) |
| GPU memory | **≈ 1.1 GB** peak at inference |

> **Single-case demonstration.** The public checkpoint shipped here was
> trained on a single OpenFOAM case (`Sweep_Case_03`) and evaluated on
> the same held-out 26-s rollout.

While developed for dental bioaerosol dispersion, the ELGIN
architecture is **domain-agnostic**: it applies equally to industrial
sprays, atmospheric particle transport, sediment dynamics,
pharmaceutical inhaler design, and any other dispersed-phase
particle-in-fluid problem.

---

## Methodology

ELGIN is a **coupled dual-graph neural network** that simultaneously
models the carrier fluid phase (Eulerian) and the dispersed particle
phase (Lagrangian) on the same unstructured computational mesh.

### Architecture overview

```
OpenFOAM polyMesh
        │
        ▼
┌───────────────────────────────────────────────────────┐
│           ELGIN — dual-graph surrogate                │
│                                                       │
│  ┌─────────────────────┐    IDW     ┌───────────────┐ │
│  │  Eulerian sub-net   │ ─────────► │ Lagrangian    │ │
│  │  (Graph Transformer │            │ sub-net       │ │
│  │   K_E = 4 blocks)   │            │ (Interaction  │ │
│  │                     │            │  Network      │ │
│  │  ➜ RANS velocity    │            │  K_L = 4 blk) │ │
│  │  ➜ pressure proj.   │            │               │ │
│  │  ➜ turbulence clos. │            │ ➜ particle    │ │
│  └─────────────────────┘            │   positions   │ │
│                                     └───────────────┘ │
└───────────────────────────────────────────────────────┘
        │
        ▼
  26-second rollout  (260 frames, Δt = 0.1 s)
```

### Key components (paper Sec. III, Table III)

| Component | Description |
|---|---|
| **Eulerian GNN** | K_E = 4 multi-head Graph Transformer blocks (4 attention heads, d_h = 64) on the RANS mesh; predicts residual increments of (U, p, k, ω) |
| **Pressure projection** | Jacobi-preconditioned conjugate-gradient solve (PCG) enforcing a graph-edge approximation of the finite-volume divergence-free constraint; fully differentiable |
| **Turbulence closure** | Learned eddy-viscosity head regularised towards the k–ω SST algebraic formula |
| **Cross-graph coupling** | Inverse-distance-weighted (IDW) interpolation of fluid fields from mesh cells to particle positions (k = 4 nearest neighbours) |
| **Lagrangian GNN** | K_L = 4 Interaction Network blocks on a radius graph (r_c = 0.10 m for ELGIN; 0.30 m for the M0 baseline) |
| **LSTM encoder** | Encodes H = 4 most recent finite-difference particle velocities into a 32-dimensional temporal state |
| **Rotation-invariant edges** | Edge features expressed in local-frame reference coordinates; combined with the Cartesian drag channel the full feature vector is rotation-covariant |
| **Optional stochastic decoder** | VAE-style probabilistic acceleration head (reparametrisation trick) for turbulent-dispersion uncertainty — **off** in the production single-case checkpoint, available via `--full_model` |
| **Symplectic integrator** | Störmer–Verlet kick–drift–kick scheme; preserves the discrete symplectic two-form and reduces energy drift |
| **Embedded physics** | Cunningham- and Schiller–Naumann-corrected Stokes drag, optional McLaughlin-corrected Saffman shear-lift, optional Brownian diffusion, Discrete Random Walk (DRW) turbulent dispersion, optional Wells' D²-law evaporation (Saffman, Brownian and evaporation are **off** in the production checkpoint) |

### Physics features encoded per particle node

```
f_i^L = [ LSTM(velocity history),   # temporal context
           box-SDF distances,         # geometry awareness
           BC-type embedding,         # boundary classification
           Stokes drag acceleration,  # analytic physics
           log(d_p),                  # particle size
           TKE from fluid field,      # turbulent dispersion
           d_wall, wall_normal ]      # obstacle proximity
```

### Four-stage training curriculum (paper Sec. IV)

```
Stage 1 — Eulerian pre-training      (20 %, 60 ep at n=300, lr = 5e-4)
  └─ Train fluid GNN on static RANS snapshots

Stage 2 — Particle supervised        (20 %, 60 ep, lr = 5e-4, Eulerian frozen)
  └─ Teacher-forced one-step acceleration MSE + noise augmentation (3e-4 m)

Stage 3 — PDE-informed joint         (40 %, 120 ep, lr = 1e-4)
  └─ Add continuity, momentum, turbulence, and angular-momentum losses

Stage 4 — BPTT rollout fine-tune     (20 %, 60 ep, lr = 5e-5)
  └─ 5-step autoregressive unroll; 70/30 BPTT/one-step loss blend
     + rollout noise σ_n = 6e-4 m to cure covariate shift
```

---

## Installation

### Prerequisites

- Python ≥ 3.9
- PyTorch ≥ 2.0 (CUDA recommended)
- [ffmpeg](https://ffmpeg.org/) installed system-wide (for MP4 animation output)

```bash
# 1. Clone the repository
git clone https://github.com/TakshakShende/ELGIN.git
cd ELGIN

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install ELGIN as an editable package
pip install -e .
```

---

## Quick Start

### Training on the bundled OpenFOAM case (Windows / PowerShell)

```powershell
# Paper-spec recipe: 300 epochs, d_h = 64, K_E = K_L = 4, BPTT = 5
.\scripts\run_training.ps1
```

The script will:
1. Extract field data from `openfoam/dentalRoom2D` (U, p, k, ω, parcel tracks)
2. Build the mesh graph (`mesh_graph.npz`)
3. Run the four-stage 60 / 60 / 120 / 60 training curriculum
4. Perform a full 26-second autoregressive rollout on the trained model
5. Generate two animations:
   - `fluid_speed_particles.mp4` — fluid velocity colourmap + ELGIN particle cloud
   - `fluid_speed_compare.mp4` — ELGIN prediction vs OpenFOAM ground truth, side-by-side

### Predicting on an unknown case

```powershell
.\scripts\predict_new_case.ps1 `
    -InputPath  "path\to\your\OpenFOAM_case" `
    -OutputDir  "predictions\my_new_case"
```

Or directly in Python:

```python
from elgin import ELGINModel, ELGINConfig, load_checkpoint

model, cfg = load_checkpoint("experiments/elgin_case03/models/best.pt")
model.eval()

new_fluid, new_particles, *_ = model.step(
    fluid_field   = fluid_t,      # (N_cells, 5)  RANS fields
    particle_pos  = positions_t,  # (N_p, 2)      parcel positions
    pos_hist      = history_t,    # (N_p, H+1, 2) position history
    particle_type = ptypes,       # (N_p,)
    d_p           = diameters,    # (N_p,)
    rho_p         = densities,    # (N_p,)
    edge_index    = mesh_edges,   # (2, E)
    cell_pos      = cell_centres, # (N_cells, 2)
)
```

---

## Training

### Prepare your OpenFOAM data

The training pipeline reads **foam-extend 4.1** `reactingParcelFoam`
case data.  Your case directory must contain the standard solver
output:

```
my_case/
├── 0.orig/     ← initial conditions (renamed to 0/ by Allrun)
├── constant/   ← polyMesh, reactingCloud1Properties, thermophysical etc.
├── system/     ← controlDict, fvSchemes, fvSolution
└── 0.1/        ← first time step (and subsequent steps)
    ├── U
    ├── p
    ├── k
    ├── omega
    └── lagrangian/
        └── reactingCloud1/
            ├── positions
            ├── U
            ├── d
            └── origId
```

The single `Sweep_Case_03` shipped in `openfoam/dentalRoom2D` matches
this layout; see `openfoam/README.md` for the full physical setup.

### Run training (PowerShell)

```powershell
.\scripts\run_training.ps1 `
    -Epochs    300 `
    -BatchSize 4 `
    -BpttSteps 5
```

| Parameter | Default | Description |
|---|---|---|
| `-CaseName` | `dentalRoom2D` | Name of the OpenFOAM case under `openfoam/` |
| `-Epochs` | `300` | Total training epochs (paper Table III) |
| `-BatchSize` | `4` | Mini-batch size (reduce if GPU OOM) |
| `-HiddenSize` | `64` | Latent dimension d_h (paper Table III) |
| `-MpSteps` | `4` | K_E and K_L message-passing depth (paper Table III) |
| `-BpttSteps` | `5` | BPTT unroll steps in Stage 4 (paper Table III) |
| `-NParticles` | `1000` | Subsample to this many evaluator parcels |
| `-SkipExtract` | switch | Skip field extraction (if NPZ already exists) |
| `-SkipAnimate` | switch | Skip animation generation |

### Monitor training

Live logs are written to `experiments\elgin_case03\results\logs\train.log`.
Checkpoint `best.pt` is saved whenever validation loss improves.

```
Stage 2 (particle)  epoch  10/60  lr=5.00e-04  mse_part=0.00412  val=0.00389  ← new best
Stage 2 (particle)  epoch  11/60  lr=4.98e-04  mse_part=0.00398  val=0.00401
...
Stage 4 (rollout)   epoch   1/60  lr=5.00e-05  bptt=0.00031  val_bptt=0.00028  ← new best
```

---

## Prediction on an Unknown Situation

Once trained, ELGIN predicts aerosol dispersion for any new case — no
CFD solver required at inference.

```powershell
.\scripts\predict_new_case.ps1 `
    -InputPath  "path\to\new_openfoam_case_or_npz" `
    -OutputDir  "predictions\prediction" `
    -ModelDir   "experiments\elgin_case03\models" `
    -MeshPath   "experiments\elgin_case03\datasets\mesh_graph.npz" `
    -NSteps     255
```

Outputs:
- `rollout.npz` — full particle trajectory array `(T, N_p, 2)`
- `clinical_metrics.json` — MDE, KE-ratio, Rg-err (if ground truth available)
- `fluid_particles.mp4` — animation

---


### Headline parameters of `Sweep_Case_03`

| Parameter | Value |
|---|---|
| Solver | `reactingParcelFoam` (foam-extend 4.1, fully coupled compressible RANS + reacting parcel cloud) |
| Domain | 4 m × 3 m × 0.01 m pseudo-2D dental room |
| Obstacles | Dentist (x∈[1.40,1.60] m, y∈[0,1.40] m); Patient (x∈[2.40,2.60] m, y∈[0,0.80] m) |
| Turbulence | RANS k–ω SST |
| Ceiling supply inlet | x ∈ [1.90, 2.10] m, V_in = 0.10 m/s downward |
| Side pressure outlet | x = 4.0 m, y ∈ [1.40, 1.60] m |
| Aerosol injection | (2.40, 0.90, 0.005) m, horizontal toward dentist, 20° half-cone |
| Jet exit speed | U_mag = 30 m/s |
| Particle size | Rosin–Rammler, 1–50 µm, d̄ = 20 µm, n = 2 |
| Simulation time | 30 s, adaptive Δt ≤ 0.01 s, maxCo = 0.3 |
| Mesh | 8 000 raw cells / 7 704 active fluid cells after obstacle carve-out |
| Wall-clock | ≈ 40 min on a single CPU core |

### How to run

```bash
source $FOAM_INST_DIR/foam-extend-4.1/etc/bashrc

cd openfoam/dentalRoom2D
./Allrun          # blockMesh → reactingParcelFoam → foamToVTK
```

Parcel positions are written to
`<time>/lagrangian/reactingCloud1/positions` at each save interval.
The ELGIN data pipeline (`elgin/data/extract_fields.py`) reads these
files directly to build the training `.npz` datasets.

---

## Results

Evaluated on the held-out 26-second rollout of `Sweep_Case_03`
(proof-of-concept single-case run):

| Model | MDE (% of L_ref) | KE-ratio | Rg-err (%) | Rollout time |
|---|---|---|---|---|
| Baseline GNS (M0)  | 19.56 | 1.057 | 9.85 | ~27 s |
| **ELGIN**          | **16.20** | **0.659** | **6.58** | **~64 s** |

> MDE = Mean Displacement Error normalised by room width (L_ref = 4.0 m);
> e.g. M0 → ~0.78 m, ELGIN → ~0.65 m mean parcel displacement error.<br/>
> KE-ratio = 1 is exact kinetic-energy conservation; > 1 indicates
> spurious energy injection; < 1 indicates numerical dissipation.<br/>
> Rg-err = relative error in the cloud's radius-of-gyration.<br/>
> All models run on an NVIDIA Quadro P1000 (4 GB VRAM).  Peak GPU
> memory at inference: ≈ 1.1 GB.

ELGIN achieves a **17 % relative reduction in trajectory error**
(MDE 16.20 % vs M0's 19.56 %) and a **33 % relative reduction in
cloud-spread error** (Rg-err 6.58 % vs 9.85 %) by exposing each
particle to the full RANS velocity and turbulent-kinetic-energy field
at every rollout step.  The trade-off is a regression in the
KE-ratio (0.659 vs 1.057), reflecting added numerical dissipation
from the physics-informed projection and the deterministic
(VAE-off) production decoder.  The full 16 / 2 / 2 retraining on
the 20-case sweep currently in progress is expected to shrink both
the dissipation and the remaining displacement error.

### Side-by-side comparison

<p align="center">
  <img src="assets/fluid_speed_compare.gif"
       alt="ELGIN vs OpenFOAM ground truth side-by-side" width="720"/>
  <br/>
  <em>Left: ELGIN prediction. &nbsp; Right: OpenFOAM (reactingParcelFoam) ground truth.<br/>
  Case: Sweep_Case_03 — ceiling inlet V<sub>in</sub> = 0.10 m/s,
  nozzle U<sub>mag</sub> = 30 m/s, cone half-angle θ = 20°.</em>
</p>

---

## Citation

If you use ELGIN in your research, please cite:

```bibtex
@article{Shende2026ELGIN,
  title   = {Physics-Informed Graph Neural Network Surrogates for Turbulent
             Nanoparticle Dispersion in Dental Clinical Environments},
  author  = {Shende, Takshak and Popov, Viktor},
  journal = {arXiv},
  year    = {2026},
  note    = {ELGIN: Eulerian--Lagrangian Graph Interaction Network},
  url     = {https://github.com/TakshakShende/ELGIN}
}
```

---

## Author

**Dr Takshak Shende**
Ascend Technologies Ltd,
Southampton, United Kingdom
✉ takshak.shende@gmail.com

---

## Licence

[MIT](LICENSE) © 2026 Dr Takshak Shende
