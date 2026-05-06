"""ELGIN — Eulerian–Lagrangian Graph Interaction Network.

A physics-informed Graph Neural Network surrogate for real-time
simulation of particle-laden turbulent flows.

Author : Dr Takshak Shende
         Department of Mechanical Engineering, University College London (UCL)
"""

from elgin.model.elgin  import CfdGNN as ELGINModel
from elgin.model.elgin  import save_cfd_gnn_checkpoint as save_checkpoint
from elgin.model.elgin  import load_cfd_gnn_checkpoint as load_checkpoint
from elgin.model.config import CfdGNNConfig as ELGINConfig

__version__ = "1.0.0"
__author__  = "Dr Takshak Shende"
__all__     = ["ELGINModel", "ELGINConfig", "save_checkpoint", "load_checkpoint"]
