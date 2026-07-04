"""
L1-L3 Orthogonal Residual Inject (lazy init version).
Builds old-class projection on first forward call using text_feat_map.
"""
import torch, torch.nn as nn, torch.nn.functional as F
from torch import Tensor

class ResidualInject(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=128, gamma_init=1e-2,
                 dropout=0.0, act_inference=True, freeze_gamma=False,
                 mode='mlp', bilinear_bottleneck=64, use_old_gate=False,
                 norm_mode='ln', rms_target=2.0, init_std=0.01,
                 ortho_mode='none'):
        super().__init__()
        self.mode = mode; self.in_dim = in_dim
        self.norm_mode = norm_mode; self.rms_target = float(rms_target)
        self.init_std = float(init_std); self.ortho_mode = ortho_mode
        
        if mode == 'bilinear':
            self.W_m = nn.Linear(in_dim, bilinear_bottleneck, bias=False)
            self.W_t = nn.Linear(in_dim, bilinear_bottleneck, bias=False)
            self.W_out = nn.Linear(bilinear_bottleneck, in_dim, bias=False)
            for m in [self.W_m, self.W_t, self.W_out]:
                nn.init.normal_(m.weight, std=self.init_std)
            self.output_norm = nn.LayerNorm(in_dim, elementwise_affine=False)
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, in_dim))

        if freeze_gamma:
            self.register_buffer('gamma', torch.tensor(float(gamma_init)))
        else:
            self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

        # Lazy init: T_old_mean will be set on first forward
        self.register_buffer('T_old_mean', torch.zeros(in_dim))
        self.register_buffer('_ortho_built', torch.tensor(0))

    def _apply_ortho(self, R):
        if self.ortho_mode == 'none' or not self._ortho_built:
            return R
        # L1: subtract projection onto T_old_mean
        v = self.T_old_mean
        vn = v.norm()
        if vn < 1e-4: return R
        v = v / vn
        proj = (R @ v).unsqueeze(-1) * v.unsqueeze(0)
        return R - proj

    def _apply_norm(self, residual):
        if self.norm_mode == 'ln':
            return self.output_norm(residual)
        elif self.norm_mode == 'rms':
            rms = residual.norm(dim=-1).mean()
            return residual * (self.rms_target / (rms + 1e-8))
        else:
            return residual

    def forward(self, memory, T_new=None):
        if self.mode == 'bilinear':
            assert T_new is not None; T_new = T_new.detach()
            T_pool = T_new.mean(dim=0, keepdim=True)
            h = self.W_m(memory) * self.W_t(T_pool).squeeze(0)
            residual = self.W_out(h)
            residual = self._apply_ortho(residual)
            residual = self._apply_norm(residual)
        else:
            residual = self.net(memory)
            residual = self._apply_norm(residual)

        with torch.no_grad():
            self._cached_mem_norm = float(memory.detach().reshape(-1, memory.shape[-1]).norm(dim=-1).mean())
            self._cached_rm_norm = float(residual.detach().reshape(-1, residual.shape[-1]).norm(dim=-1).mean())
        if self.training:
            self._cached_residual = residual
        return memory + self.gamma * residual

    def extra_repr(self):
        return (f"mode={self.mode}, gamma={float(self.gamma.detach()):.4f}, "
                f"norm={self.norm_mode}, ortho={self.ortho_mode}")
