"""ELGIN — Eulerian–Lagrangian Graph Interaction Network.

A physics-informed Graph Neural Network surrogate for fast simulation
of particle-laden turbulent flows, originally developed for dental
bioaerosol dispersion in indoor clinical environments.  A 26-second
rollout completes in approximately 64 s on a 4 GB GPU — fast enough
for per-appointment infection-risk screening — and is roughly 37x
faster than the reference foam-extend 4.1 reactingParcelFoam solver
on the same single-case demonstration problem.

The public checkpoint accompanying this repository was trained on
a single held-out OpenFOAM case (Sweep_Case_03); a full 16/2/2
train/validation/test retraining on the planned 20-case factorial
sweep is in progress.

Author : Dr Takshak Shende
         Department of Mechanical Engineering, University College London (UCL)
"""

from elgin.model.cfd_gnn import CfdGNN as ELGINModel
from elgin.model.cfd_gnn import save_cfd_gnn_checkpoint as save_checkpoint
from elgin.model.cfd_gnn import load_cfd_gnn_checkpoint as load_checkpoint
from elgin.model.config  import CfdGNNConfig as ELGINConfig

__version__ = "1.0.0"
__author__  = "Dr Takshak Shende"
__all__     = ["ELGINModel", "ELGINConfig", "save_checkpoint", "load_checkpoint"]
