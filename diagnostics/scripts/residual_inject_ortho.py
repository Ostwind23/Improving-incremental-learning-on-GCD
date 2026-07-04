"""
L1-L3 Orthogonal Residual Inject for Bilinear.
Projects R away from old-class feature subspace before injection.
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
        self.use_old_gate = bool(use_old_gate)
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

        # Placeholders for old-class prototypes (set externally)
        self.register_buffer('T_old_mean', torch.zeros(in_dim))       # L1: single vector
        self.register_buffer('T_old_basis', torch.zeros(70, in_dim))  # L2: per-class vectors
        self.register_buffer('V_old_basis', torch.zeros(70, in_dim))  # L3: visual prototypes
        # Pre-built projection matrices (set during detector init)
        self.register_buffer('proj_L1', torch.eye(in_dim))  # I - T @ T^T / |T|^2
        self._ortho_ready = False

    def build_ortho(self, T_old_all):
        """Build projection matrices. Call once after model init with old-class text embeddings.
        T_old_all: (70, L_t, 256) from text_feat_map[0:70]"""
        T_mean = T_old_all.mean(dim=0).mean(dim=0)  # (256,) — average across 70 classes and tokens
        T_mean = T_mean / (T_mean.norm() + 1e-8)
        self.T_old_mean = T_mean
        self.proj_L1 = torch.eye(256, device=T_mean.device) - torch.outer(T_mean, T_mean)
        
        # L2: per-class projection (average across tokens per class, then project)
        # Store basis vectors
        self.T_old_basis = T_old_all.mean(dim=1)  # (70, 256)
        self._ortho_ready = True

    def _apply_ortho(self, R):
        if not self._ortho_ready:
            return R
        if self.ortho_mode == 'L1':
            # Subtract projection onto T_old_mean direction
            return R @ self.proj_L1.T
        elif self.ortho_mode == 'L2':
            # Subtract projection onto span of old-class text vectors
            # Gram-Schmidt: subtract each component
            R_out = R
            for c in range(70):
                v = self.T_old_basis[c]
                v = v / (v.norm() + 1e-8)
                proj = (R_out @ v).unsqueeze(-1) * v.unsqueeze(0)
                R_out = R_out - proj
            return R_out
        elif self.ortho_mode == 'L3':
            # Same as L2 but with visual prototypes (set externally)
            if self.V_old_basis.norm() > 0:
                R_out = R
                for c in range(70):
                    v = self.V_old_basis[c]
                    vn = v.norm()
                    if vn < 1e-4: continue
                    v = v / vn
                    proj = (R_out @ v).unsqueeze(-1) * v.unsqueeze(0)
                    R_out = R_out - proj
                return R_out
            return R
        else:
            return R

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
            residual = self._apply_ortho(residual)  # <-- ortho applied here
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
