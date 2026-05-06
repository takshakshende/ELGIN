"""EulerianGNN — MeshGraphNet-style fluid field predictor.

Improvements (Round 1 + Round 2)
----------------------------------
#5  Single-head attention gate     (kept for cfg.use_graph_transformer=False)
#6  Face-type embedding in edges

S5  Graph Transformer with multi-head scaled-dot-product attention.
    Replaces the sigmoid gate when cfg.use_graph_transformer=True.

    Multi-head attention:
        Q_i   = W_Q h_i,   K_j   = W_K h_j,   V_ij = W_V e_{ij}
        alpha_{ij}^h = softmax_j( (Q_i^h . K_j^h) / sqrt(d_k) )
        m_i   = W_O concat_h( sum_j alpha_{ij}^h * V_{ij}^h )

    Compared to the single-head sigmoid gate:
      - softmax normalisation prevents attention collapse
      - multiple heads attend to different physical scales simultaneously
        (turbulent fluctuations vs mean convective flow)
      - interpretable attention maps for ablation analysis

    References
    ----------
    - Brody, Alon & Yahav (2022) GAT v2
    - Shi et al. (2021) Masked Label Prediction (Graph Transformer)
    - Pfaff et al. (2021) MeshGraphNets — Eulerian mesh GNN
    - Review paper Section 4.3
"""

from __future__ import annotations
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CfdGNNConfig


# ---------------------------------------------------------------------------
#  MLP helper
# ---------------------------------------------------------------------------

def _mlp(in_dim: int, hidden: int, out_dim: int, layers: int,
         layernorm: bool = True) -> nn.Sequential:
    assert layers >= 1
    mods: list = [nn.Linear(in_dim, hidden), nn.ReLU()]
    for _ in range(layers - 1):
        mods += [nn.Linear(hidden, hidden), nn.ReLU()]
    mods.append(nn.Linear(hidden, out_dim))
    if layernorm:
        mods.append(nn.LayerNorm(out_dim))
    return nn.Sequential(*mods)


# ---------------------------------------------------------------------------
#  S5 — Graph Transformer Interaction Block (multi-head attention)
# ---------------------------------------------------------------------------

class GraphTransformerBlock(nn.Module):
    """Graph Transformer block with multi-head scaled-dot-product attention.

    For each edge (i -> j):
        e_{ij}^new = MLP_e([h_i || h_j || e_{ij}])

    Multi-head attention gate (H heads, d_k = hidden // H):
        Q_{ij} = W_Q^h h_i
        K_{ij} = W_K^h h_j
        V_{ij} = W_V^h e_{ij}^new
        alpha_{ij}^h = exp(Q^h . K^h / sqrt(d_k)) / Z_i^h
        msg_i = W_O [ concat_h sum_j alpha_{ij}^h V_{ij}^h ]

    Node update with residual:
        h_i^new = h_i + MLP_n([h_i || msg_i])

    The softmax is computed over all edges incoming to node i.

    References
    ----------
    - Brody et al. (2022), How Attentive are Graph Attention Networks?
    - Shi et al. (2021), Masked Label Prediction: Unified Message Passing
    - Vaswani et al. (2017), Attention Is All You Need
    """

    def __init__(self, hidden: int, mlp_layers: int, n_heads: int = 4):
        super().__init__()
        assert hidden % n_heads == 0, (
            f"hidden ({hidden}) must be divisible by n_heads ({n_heads})"
        )
        self.n_heads = n_heads
        self.d_k     = hidden // n_heads

        self.edge_mlp  = _mlp(3 * hidden, hidden, hidden, mlp_layers)
        self.node_mlp  = _mlp(2 * hidden, hidden, hidden, mlp_layers)

        # Multi-head projections
        self.W_Q = nn.Linear(hidden, hidden, bias=False)
        self.W_K = nn.Linear(hidden, hidden, bias=False)
        self.W_V = nn.Linear(hidden, hidden, bias=False)
        self.W_O = nn.Linear(hidden, hidden, bias=False)

    def forward(
        self,
        x:          torch.Tensor,   # (N, H)
        edge_index: torch.Tensor,   # (2, E)
        edge_attr:  torch.Tensor,   # (E, H)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index
        N, H     = x.shape
        n_h      = self.n_heads
        d_k      = self.d_k
        E        = src.shape[0]

        # ── Edge update ─────────────────────────────────────────────────────
        e_new = self.edge_mlp(torch.cat([x[dst], x[src], edge_attr], dim=-1))

        if E == 0:
            # No edges — return with zero aggregation
            x_new = x + self.node_mlp(torch.cat([x, x.new_zeros(N, H)], dim=-1))
            return x_new, edge_attr + e_new

        # ── Multi-head attention ─────────────────────────────────────────────
        # Project to Q, K, V
        Q = self.W_Q(x[dst]).reshape(E, n_h, d_k)        # (E, n_h, d_k)
        K = self.W_K(x[src]).reshape(E, n_h, d_k)        # (E, n_h, d_k)
        V = self.W_V(e_new).reshape(E, n_h, d_k)         # (E, n_h, d_k)

        # Scaled dot-product scores
        scores = (Q * K).sum(-1) / math.sqrt(d_k)        # (E, n_h)

        # Per-destination-node softmax via scatter_softmax equivalent
        # Use log-sum-exp for numerical stability
        # scores_max: max score per (dst, head) pair
        scores_max = scores.new_full((N, n_h), float("-inf"))
        idx_nh = dst.unsqueeze(-1).expand(E, n_h)
        scores_max = scores_max.scatter_reduce(
            0, idx_nh, scores, reduce="amax", include_self=True
        )
        scores_shifted = scores - scores_max[dst]         # (E, n_h)
        exp_scores     = scores_shifted.exp()             # (E, n_h)

        # Sum of exp scores per (dst, head)
        sum_exp = exp_scores.new_zeros(N, n_h)
        sum_exp = sum_exp.scatter_add(0, idx_nh, exp_scores)
        alpha   = exp_scores / (sum_exp[dst] + 1e-12)    # (E, n_h)  normalized

        # Weighted message: sum_j alpha_{ij}^h * V_{ij}^h
        alpha_v = alpha.unsqueeze(-1) * V                 # (E, n_h, d_k)
        alpha_v = alpha_v.reshape(E, H)                   # (E, H)

        idx_h = dst.unsqueeze(-1).expand(E, H)
        msg   = alpha_v.new_zeros(N, H).scatter_add(0, idx_h, alpha_v)  # (N, H)

        # Output projection
        msg = self.W_O(msg)                               # (N, H)

        # Node update with residual
        x_new = x + self.node_mlp(torch.cat([x, msg], dim=-1))
        return x_new, edge_attr + e_new


# ---------------------------------------------------------------------------
#  Single-head attention block (Round-1 #5 — kept as fallback)
# ---------------------------------------------------------------------------

class FluidInteractionBlock(nn.Module):
    """Single-head sigmoid-gated interaction block (Round-1 improvement #5)."""

    def __init__(self, hidden: int, mlp_layers: int,
                 use_attention: bool = True):
        super().__init__()
        self.use_attention = use_attention
        self.edge_mlp = _mlp(3 * hidden, hidden, hidden, mlp_layers)
        self.node_mlp = _mlp(2 * hidden, hidden, hidden, mlp_layers)
        if use_attention:
            self.gate = nn.Sequential(
                nn.Linear(3 * hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, 1), nn.Sigmoid(),
            )

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        N = x.shape[0]
        x_i, x_j = x[dst], x[src]
        e_new = self.edge_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))
        msg   = self.gate(torch.cat([x_i, x_j, edge_attr], dim=-1)) * e_new \
                if self.use_attention else e_new
        idx  = dst.unsqueeze(-1).expand_as(msg)
        aggr = torch.zeros(N, msg.shape[-1], dtype=x.dtype,
                           device=x.device).scatter_add(0, idx, msg)
        x_new = x + self.node_mlp(torch.cat([x, aggr], dim=-1))
        return x_new, edge_attr + e_new


# ---------------------------------------------------------------------------
#  EulerianGNN
# ---------------------------------------------------------------------------

class EulerianGNN(nn.Module):
    """GNN for the RANS fluid field with Graph Transformer processor (S5).

    Processor chooses:
        use_graph_transformer=True  : GraphTransformerBlock  (S5, default)
        use_graph_transformer=False : FluidInteractionBlock  (Round-1 #5)
    """

    def __init__(self, cfg: CfdGNNConfig):
        super().__init__()
        self.cfg = cfg
        H = cfg.fluid_hidden

        # Boundary condition embedding
        self.bc_embed = nn.Embedding(cfg.bc_type_count, cfg.bc_embed_dim)

        # S5 / Round-1 #6: face-type embedding
        self.face_type_embed = nn.Embedding(cfg.face_type_count,
                                            cfg.face_type_embed_dim)

        # Node encoder
        # +2*dim for relative-to-bbox wall feature, +bc_embed_dim for BC type
        # embedding, +dim for the per-case inlet conditioning vector.
        self.use_inlet_cond = getattr(cfg, "use_inlet_conditioning", False)
        node_in = (cfg.fluid_node_in
                   + 2 * cfg.dim
                   + cfg.bc_embed_dim
                   + (cfg.dim if self.use_inlet_cond else 0))
        self.node_encoder = _mlp(node_in, H, H, cfg.fluid_mlp_layers)

        # Edge encoder: geometric + face-type
        edge_in = cfg.fluid_edge_in + cfg.face_type_embed_dim
        self.edge_encoder = _mlp(edge_in, H, H, cfg.fluid_mlp_layers)

        # S5: processor — Graph Transformer or single-head gate
        if cfg.use_graph_transformer:
            self.processor = nn.ModuleList(
                GraphTransformerBlock(H, cfg.fluid_mlp_layers,
                                      n_heads=cfg.fluid_attn_heads)
                for _ in range(cfg.fluid_mp_steps)
            )
        else:
            self.processor = nn.ModuleList(
                FluidInteractionBlock(H, cfg.fluid_mlp_layers,
                                      use_attention=cfg.use_fluid_attention)
                for _ in range(cfg.fluid_mp_steps)
            )

        # Decoder
        self.decoder = _mlp(H, H, cfg.fluid_out, cfg.fluid_mlp_layers,
                            layernorm=False)

        self.register_buffer("fluid_mean",
                             torch.tensor(cfg.fluid_mean, dtype=torch.float32))
        self.register_buffer("fluid_std",
                             torch.tensor(cfg.fluid_std,  dtype=torch.float32))

    def _norm(self, q):
        return (q - self.fluid_mean) / (self.fluid_std + 1e-8)

    def _denorm(self, q):
        return q * (self.fluid_std + 1e-8) + self.fluid_mean

    def _node_features(self, field, pos, bc_type, inlet_cond=None):
        cfg    = self.cfg
        bounds = torch.tensor(cfg.domain_bounds,
                              dtype=pos.dtype, device=pos.device)
        room   = torch.stack([bounds[0, 1] - bounds[0, 0],
                               bounds[1, 1] - bounds[1, 0]])
        dist_lo = (pos - bounds[:, 0]) / room
        dist_hi = (bounds[:, 1] - pos) / room
        wall_feat = torch.cat([dist_lo.clamp(-1, 1),
                                dist_hi.clamp(-1, 1)], dim=-1)
        parts = [self._norm(field), wall_feat, self.bc_embed(bc_type)]
        if self.use_inlet_cond:
            if inlet_cond is None:
                inlet_cond = field.new_zeros(cfg.dim)
            parts.append(inlet_cond.view(1, -1).expand(field.shape[0], -1))
        return torch.cat(parts, dim=-1)

    def _edge_features(self, pos, edge_index, face_normals, face_areas,
                       face_type=None):
        src, dst = edge_index
        delta = pos[dst] - pos[src]
        dist  = delta.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        unit  = delta / dist
        geom  = torch.cat([
            face_normals,
            face_areas.unsqueeze(-1) / (self.cfg.L_ref ** 2),
            dist / self.cfg.L_ref,
            unit,
        ], dim=-1)
        E = face_normals.shape[0]
        if face_type is None:
            face_type = torch.zeros(E, dtype=torch.long,
                                    device=face_normals.device)
        ft_feat = self.face_type_embed(face_type.long())
        return torch.cat([geom, ft_feat], dim=-1)

    def forward(self, field, pos, bc_type, edge_index, face_normals,
                face_areas, nu_t=None, face_type=None, inlet_cond=None):
        node_feat = self._node_features(field, pos, bc_type,
                                         inlet_cond=inlet_cond)
        edge_feat = self._edge_features(pos, edge_index, face_normals,
                                        face_areas, face_type)
        x = self.node_encoder(node_feat)
        e = self.edge_encoder(edge_feat)
        for block in self.processor:
            x, e = block(x, edge_index, e)
        return self.decoder(x)

    def next_field(self, field, pos, bc_type, edge_index, face_normals,
                   face_areas, bc_values=None, nu_t=None, face_type=None,
                   inlet_cond=None):
        delta_norm = self.forward(field, pos, bc_type, edge_index,
                                  face_normals, face_areas, nu_t, face_type,
                                  inlet_cond=inlet_cond)
        q_new = field + self._denorm(delta_norm)
        if bc_values is not None:
            is_bc = (bc_type > 0)
            q_new = torch.where(is_bc.unsqueeze(-1).expand_as(q_new),
                                bc_values, q_new)
        q_new = torch.cat([
            q_new[..., :3],
            q_new[..., 3:4].clamp(min=1e-6),
            q_new[..., 4:5].clamp(min=1e-6),
        ], dim=-1)
        return q_new
