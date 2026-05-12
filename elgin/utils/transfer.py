"""transfer.py — Transfer-learning utilities for ELGIN.

Usage
-----
    from elgin.utils.transfer import load_gns_into_lagrangian
    load_gns_into_lagrangian(model.lagrangian_gnn, "gns_checkpoint.pt")

"""

from __future__ import annotations
from typing import Dict, List, Optional
import pathlib
import warnings

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
#  Core transfer function
# ---------------------------------------------------------------------------

def load_gns_into_lagrangian(
    lagrangian_gnn:     nn.Module,
    gns_checkpoint:     str,
    device:             str | torch.device = "cpu",
    strict:             bool = False,
    verbose:            bool = True,
) -> Dict[str, List[str]]:
    """Load a pre-trained GNS checkpoint into a LagrangianGNN.

    Matching strategy (in order):
        1. Exact key match.
        2. Shape-only match: if the shapes align, map regardless of name.
        3. Skip: log skipped keys.

    Args:
        lagrangian_gnn  : the LagrangianGNN module to initialise.
        gns_checkpoint  : path to the .pt checkpoint file.
        device          : device on which to load tensors.
        strict          : if True, raise on any mismatch (default False).
        verbose         : print a summary of loaded/skipped keys.

    Returns:
        dict with keys "loaded", "skipped", "missing"
    """
    dev  = torch.device(device)
    ckpt = torch.load(gns_checkpoint, map_location=dev, weights_only=False)

    # Support checkpoints that wrap the state dict under a 'model' key
    if isinstance(ckpt, dict) and "model" in ckpt:
        gns_state = ckpt["model"]
    elif isinstance(ckpt, dict) and all(
            isinstance(v, torch.Tensor) for v in ckpt.values()):
        gns_state = ckpt
    else:
        gns_state = ckpt

    gnn_state    = lagrangian_gnn.state_dict()
    new_state    = {}
    loaded_keys  = []
    skipped_keys = []
    missing_keys = []

    # ── Pass 1: exact key match ──────────────────────────────────────────────
    for key, param in gnn_state.items():
        if key in gns_state and gns_state[key].shape == param.shape:
            new_state[key] = gns_state[key]
            loaded_keys.append(key)

    # ── Pass 2: shape-only match for remaining keys ──────────────────────────
    # Build a shape -> [list of gns keys] index for unmatched gns tensors
    unmatched_gns = {
        k: v for k, v in gns_state.items()
        if k not in loaded_keys
    }
    shape_to_gns: Dict[tuple, List[str]] = {}
    for k, v in unmatched_gns.items():
        shape_to_gns.setdefault(tuple(v.shape), []).append(k)

    for key, param in gnn_state.items():
        if key in new_state:
            continue
        shape = tuple(param.shape)
        if shape in shape_to_gns and shape_to_gns[shape]:
            gns_key = shape_to_gns[shape].pop(0)
            new_state[key] = gns_state[gns_key]
            loaded_keys.append(f"{key} <- {gns_key} (shape match)")
        else:
            missing_keys.append(key)

    # ── Check for GNS keys that were not used ────────────────────────────────
    used_gns_keys = set()
    for info in loaded_keys:
        # Extract GNS key from "gnn_key <- gns_key (shape match)" if present
        if "<-" in info:
            used_gns_keys.add(info.split("<-")[1].strip().split(" ")[0])
    for k in gns_state:
        if k not in used_gns_keys and k not in new_state:
            skipped_keys.append(k)

    # ── Load ─────────────────────────────────────────────────────────────────
    msg = lagrangian_gnn.load_state_dict(
        {**gnn_state, **{k.split(" ")[0]: v for k, v in
                         {kk: new_state[kk] for kk in loaded_keys
                          if kk.split(" ")[0] in gnn_state}.items()}},
        strict=False
    )

    if verbose:
        print(f"\n[transfer] Loading GNS weights from {gns_checkpoint}")
        print(f"  matched / copied  : {len(loaded_keys)}")
        print(f"  missing (new init): {len(missing_keys)}")
        print(f"  skipped (GNS only): {len(skipped_keys)}")
        if missing_keys:
            print(f"  Missing keys (will use random init):")
            for k in missing_keys[:10]:
                print(f"    - {k}")
            if len(missing_keys) > 10:
                print(f"    ... and {len(missing_keys)-10} more")
        if msg.unexpected_keys:
            print(f"  Unexpected keys: {msg.unexpected_keys[:5]}")

    if strict and (missing_keys or skipped_keys):
        raise RuntimeError(
            f"strict=True: {len(missing_keys)} missing keys, "
            f"{len(skipped_keys)} unused GNS keys."
        )

    return {
        "loaded":  loaded_keys,
        "skipped": skipped_keys,
        "missing": missing_keys,
    }


# ---------------------------------------------------------------------------
#  Layer-wise learning rate (LLRD) for fine-tuning
# ---------------------------------------------------------------------------

def get_layerwise_param_groups(
    model:     nn.Module,
    base_lr:   float,
    decay:     float = 0.85,
    skip_list: Optional[List[str]] = None,
) -> List[Dict]:
    """Assign decayed learning rates to each layer (deepest = highest LR).

    Implements Layerwise Learning Rate Decay (LLRD) from Howard & Ruder
    (2018, ULMFiT) for GNN fine-tuning: the earliest layers (which contain
    the most transferable features) are given the smallest LR to avoid
    catastrophic forgetting.

    Returns a list of param_groups for use with torch.optim.AdamW.

    Example::

        groups = get_layerwise_param_groups(model.lagrangian_gnn, 1e-4)
        optimizer = torch.optim.AdamW(groups, weight_decay=1e-5)
    """
    skip_list = skip_list or []
    # Collect layers in order
    named_layers = [(name, mod) for name, mod in model.named_modules()
                    if isinstance(mod, (nn.Linear, nn.LSTM, nn.Embedding,
                                        nn.LayerNorm))]
    n = len(named_layers)
    groups = []
    for i, (name, layer) in enumerate(named_layers):
        if any(s in name for s in skip_list):
            continue
        lr_i = base_lr * (decay ** (n - 1 - i))
        groups.append({
            "params": list(layer.parameters()),
            "lr": lr_i,
            "name": name,
        })
    return groups


# ---------------------------------------------------------------------------
#  Freeze / unfreeze helpers
# ---------------------------------------------------------------------------

def freeze_encoder(model: nn.Module) -> None:
    """Freeze the node and edge encoders of any GNN (for Stage-1 transfer)."""
    for name, param in model.named_parameters():
        if "encoder" in name:
            param.requires_grad = False


def unfreeze_all(model: nn.Module) -> None:
    """Restore all parameters to trainable."""
    for param in model.parameters():
        param.requires_grad = True


def count_trainable(model: nn.Module) -> int:
    """Return number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
