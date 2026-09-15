"""
sigdist/generators.py:  Reusable generator architectures for distributional training.

All generators use the Full-Input-Cat (FIC) strategy: the conditioning vector
[z, c] is concatenated to the hidden state at every layer.  This exposes the latent and conditioning input at each layer. Its optimization
behaviour is architecture- and task-dependent; it does not guarantee gradient
stability or prevent mode collapse.

Classes:
    FICGenerator   :  no normalisation (recommended default)
    FICGeneratorLN :  LayerNorm after each hidden activation
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["FICGenerator", "FICGeneratorLN", "FICGeneratorBN"]


class FICGenerator(nn.Module):
    """
    Full-Input-Cat generator:  no hidden-layer normalisation.

    Architecture:
        Layer 0 : Linear(noise_dim + cond_dim,          hidden) → ReLU
        Layer i : Linear(hidden + noise_dim + cond_dim, hidden) → ReLU
        Output  : Linear(hidden, out_dim)

    Args:
        out_dim   : Output dimensionality (default 2 for 2D toy tasks).
        noise_dim : Latent noise vector size.
        cond_dim  : Conditioning vector size (e.g. 1 for a binary class flag).
        hidden    : Hidden layer width.
        n_layers  : Number of hidden layers.
    """

    def __init__(
        self,
        out_dim:   int = 2,
        noise_dim: int = 6,
        cond_dim:  int = 1,
        hidden:    int = 128,
        n_layers:  int = 5,
    ) -> None:
        super().__init__()
        cond = noise_dim + cond_dim
        self.layers = nn.ModuleList(
            nn.Linear(cond if i == 0 else hidden + cond, hidden)
            for i in range(n_layers)
        )
        self.fc_out = nn.Linear(hidden, out_dim)

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z : (B, noise_dim) latent noise.
            c : (B, cond_dim)  conditioning vector.

        Returns:
            (B, out_dim) generated samples.
        """
        zc = torch.cat([z, c], dim=-1)
        x  = zc
        for i, fc in enumerate(self.layers):
            x = F.relu(fc(x if i == 0 else torch.cat([x, zc], dim=-1)))
        return self.fc_out(x)


class FICGeneratorLN(nn.Module):
    """
    Full-Input-Cat generator with LayerNorm after each hidden activation.

    LayerNorm normalises per sample rather than across the batch. Its effect
    on distributional training should be evaluated in the intended setting.

    Args: same as FICGenerator.
    """

    def __init__(
        self,
        out_dim:   int = 2,
        noise_dim: int = 6,
        cond_dim:  int = 1,
        hidden:    int = 128,
        n_layers:  int = 5,
    ) -> None:
        super().__init__()
        cond = noise_dim + cond_dim
        self.layers = nn.ModuleList(
            nn.Linear(cond if i == 0 else hidden + cond, hidden)
            for i in range(n_layers)
        )
        self.norms  = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(n_layers))
        self.fc_out = nn.Linear(hidden, out_dim)

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z : (B, noise_dim) latent noise.
            c : (B, cond_dim)  conditioning vector.

        Returns:
            (B, out_dim) generated samples.
        """
        zc = torch.cat([z, c], dim=-1)
        x  = zc
        for i, (fc, ln) in enumerate(zip(self.layers, self.norms)):
            x = F.relu(ln(fc(x if i == 0 else torch.cat([x, zc], dim=-1))))
        return self.fc_out(x)


class FICGeneratorBN(nn.Module):
    """
    Full-Input-Cat generator with BatchNorm after each hidden activation.

    BatchNorm normalises activations across each batch. This variant is
    available for controlled architecture comparisons; no categorical
    incompatibility with distributional losses is claimed.

    Args: same as FICGenerator.
    """

    def __init__(
        self,
        out_dim:   int = 2,
        noise_dim: int = 6,
        cond_dim:  int = 1,
        hidden:    int = 128,
        n_layers:  int = 5,
    ) -> None:
        super().__init__()
        cond = noise_dim + cond_dim
        self.layers = nn.ModuleList(
            nn.Linear(cond if i == 0 else hidden + cond, hidden)
            for i in range(n_layers)
        )
        self.norms  = nn.ModuleList(nn.BatchNorm1d(hidden) for _ in range(n_layers))
        self.fc_out = nn.Linear(hidden, out_dim)

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z : (B, noise_dim) latent noise.
            c : (B, cond_dim)  conditioning vector.

        Returns:
            (B, out_dim) generated samples.
        """
        zc = torch.cat([z, c], dim=-1)
        x  = zc
        for i, (fc, bn) in enumerate(zip(self.layers, self.norms)):
            x = F.relu(bn(fc(x if i == 0 else torch.cat([x, zc], dim=-1))))
        return self.fc_out(x)
