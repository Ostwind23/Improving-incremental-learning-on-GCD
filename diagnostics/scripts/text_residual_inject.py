"""TATRI: Token-Adaptive Text Residual Injection.

Two variants:
  - fixed_gamma:  T' = T + γ · R(T) · mask     (GRMI-symmetric, scalar γ)
  - per_token:    T' = T + Gate(T) · R(T) · mask  (per-token α∈[0,1])

Only new-class text tokens are perturbed; old-class tokens are untouched.
"""
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional


class TextResidualInject(nn.Module):
    """Text-side gated residual, applied to memory_text after encoder.

    Args:
        in_dim: text embedding dim (256 for GroundingDINO-T post text_feat_map).
        hidden_dim: bottleneck width. Default 64.
        gamma_init: initial gate scalar (fixed_gamma mode). Default 0.05.
        mode: 'fixed_gamma' or 'per_token'.
        gate_hidden: hidden dim for per-token gate MLP. Default 32.
    """

    def __init__(self, in_dim: int = 256, hidden_dim: int = 64,
                 gamma_init: float = 0.05, mode: str = 'fixed_gamma',
                 gate_hidden: int = 32):
        super().__init__()
        self.mode = mode
        self.transform = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_dim))

        if mode == 'fixed_gamma':
            self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        elif mode == 'per_token':
            self.gate = nn.Sequential(
                nn.Linear(in_dim, gate_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(gate_hidden, 1),
                nn.Sigmoid())
        else:
            raise ValueError(f"Unknown mode: {mode}")

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[TATRI] mode={mode} hidden={hidden_dim} params={n_params}")

    def forward(self, memory_text: Tensor,
                new_token_mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            memory_text: (bs, T_len, C)
            new_token_mask: (T_len,) float, 1.0 for new-class tokens, 0.0 otherwise
        Returns:
            T': (bs, T_len, C)
        """
        delta = self.transform(memory_text)  # (bs, T_len, C)

        if self.mode == 'fixed_gamma':
            scale = self.gamma
        else:  # per_token
            scale = self.gate(memory_text)  # (bs, T_len, 1)

        if new_token_mask is not None:
            mask = new_token_mask.unsqueeze(0).unsqueeze(-1)  # (1, T_len, 1)
            delta = delta * mask

        return memory_text + scale * delta

    def extra_repr(self) -> str:
        if self.mode == 'fixed_gamma':
            return f'mode=fixed_gamma, gamma={float(self.gamma.detach()):.4f}'
        return f'mode=per_token'
