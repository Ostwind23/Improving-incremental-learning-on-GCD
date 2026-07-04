"""
GRMI: Gated Residual injection on encoder Memory.

Architectures:
  - 'mlp' (default): M' = M + gamma * MLP(M)
  - 'bilinear': M' = M + gamma * LN(Bilinear(M, T_new))

Bilinear variant: R = LN( W_out( W_m(M) * W_t(T_pool) ) )
where T_pool = mean of new-class text embeddings.
LayerNorm on output bounds ||R|| naturally, preventing explosion.
"""
import torch, torch.nn as nn
from torch import Tensor


class ResidualInject(nn.Module):
    """Gated residual for encoder memory. Supports MLP and Bilinear modes.

    Args:
        mode: 'mlp' or 'bilinear'
        in_dim: encoder memory channel (256 for GroundingDINO-T)
        hidden_dim: MLP bottleneck (ignored in bilinear mode)
        bilinear_bottleneck: bilinear interaction dimension (default 64)
        gamma_init: initial gate value
        freeze_gamma: if True, gamma is a fixed buffer
        act_inference: apply residual at inference too
    """

    def __init__(self, in_dim=256, hidden_dim=128, gamma_init=1e-2,
                 dropout=0.0, act_inference=True, freeze_gamma=False,
                 mode='mlp', bilinear_bottleneck=64):
        super().__init__()
        self.mode = mode
        self.in_dim = in_dim

        if mode == 'bilinear':
            self.W_m = nn.Linear(in_dim, bilinear_bottleneck, bias=False)
            self.W_t = nn.Linear(in_dim, bilinear_bottleneck, bias=False)
            self.W_out = nn.Linear(bilinear_bottleneck, in_dim, bias=False)
            self.output_norm = nn.LayerNorm(in_dim, elementwise_affine=False)
            for m in [self.W_m, self.W_t, self.W_out]:
                nn.init.normal_(m.weight, std=0.1)
        else:
            layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True)]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, in_dim))
            self.net = nn.Sequential(*layers)

        if freeze_gamma:
            self.register_buffer('gamma', torch.tensor(float(gamma_init)))
        else:
            self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.act_inference = bool(act_inference)
        self.freeze_gamma = bool(freeze_gamma)

    def forward(self, memory: Tensor, T_new: Tensor = None) -> Tensor:
        """memory: (B, N, C), T_new: (K, C) for bilinear mode."""
        if self.mode == 'bilinear':
            assert T_new is not None, "Bilinear mode requires T_new"
            T_pool = T_new.mean(dim=0, keepdim=True)  # (1, C)
            # Bilinear interaction
            h_m = self.W_m(memory)    # (B*N, bottleneck)
            h_t = self.W_t(T_pool)    # (1, bottleneck)
            h = h_m * h_t.squeeze(0)  # elementwise interaction
            residual = self.W_out(h)
            residual = self.output_norm(residual)  # bounds ||R|| ≈ sqrt(C)
        else:
            residual = self.net(memory)

        # Cache norms for monitoring
        with torch.no_grad():
            self._cached_mem_norm = float(memory.detach().norm(dim=-1).mean())
            self._cached_rm_norm = float(residual.detach().norm(dim=-1).mean())

        # GS-GRMI backward hook
        if self.training and getattr(self, '_gs_ratio', None) is not None:
            r = self._gs_ratio
            residual = residual * 1.0
            residual.register_hook(lambda g, _r=r: g * _r)

        if self.training:
            self._cached_residual = residual

        return memory + self.gamma * residual

    def extra_repr(self) -> str:
        return (f'mode={self.mode}, gamma={float(self.gamma.detach()):.4f}, '
                f'act_inference={self.act_inference}')
