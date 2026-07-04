"""
GRMI Bilinear + Global RMS Norm (no per-token LN), 1 epoch test
Variant B2: RMS target = 1.0
"""
import torch.nn as nn
import torch

class ResidualInjectRMS(nn.Module):
    def __init__(self, in_dim=256, gamma_init=0.5, bilinear_bottleneck=64, 
                 rms_target=2.0, freeze_gamma=False):
        super().__init__()
        self.mode = 'bilinear'
        self.in_dim = in_dim
        self.rms_target = float(rms_target)
        
        self.W_m = nn.Linear(in_dim, bilinear_bottleneck, bias=False)
        self.W_t = nn.Linear(in_dim, bilinear_bottleneck, bias=False)
        self.W_out = nn.Linear(bilinear_bottleneck, in_dim, bias=False)
        for m in [self.W_m, self.W_t, self.W_out]:
            nn.init.normal_(m.weight, std=0.1)
        
        if freeze_gamma:
            self.register_buffer('gamma', torch.tensor(float(gamma_init)))
        else:
            self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.freeze_gamma = bool(freeze_gamma)
    
    def forward(self, memory, T_new=None):
        assert T_new is not None
        T_new = T_new.detach()
        T_pool = T_new.mean(dim=0, keepdim=True)
        h = self.W_m(memory) * self.W_t(T_pool).squeeze(0)
        residual = self.W_out(h)
        
        # Global RMS scaling (replaces per-token LN)
        rms = residual.norm(dim=-1).mean()
        residual = residual * (self.rms_target / (rms + 1e-8))
        
        with torch.no_grad():
            self._cached_mem_norm = float(memory.detach().reshape(-1, memory.shape[-1]).norm(dim=-1).mean())
            self._cached_rm_norm = float(residual.detach().reshape(-1, residual.shape[-1]).norm(dim=-1).mean())
        
        if self.training:
            self._cached_residual = residual
        
        return memory + self.gamma * residual
    
    def extra_repr(self):
        return f"mode={self.mode}, gamma={float(self.gamma.detach()):.4f}, rms_target={self.rms_target}"
