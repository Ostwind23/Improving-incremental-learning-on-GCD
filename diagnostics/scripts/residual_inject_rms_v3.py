import torch, torch.nn as nn, torch.nn.functional as F
from torch import Tensor

class ResidualInject(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=128, gamma_init=1e-2,
                 dropout=0.0, act_inference=True, freeze_gamma=False,
                 mode='mlp', bilinear_bottleneck=64, use_old_gate=False,
                 norm_mode='ln', rms_target=2.0, init_std=0.01):
        super().__init__()
        self.mode = mode
        self.in_dim = in_dim
        self.norm_mode = norm_mode
        self.rms_target = float(rms_target)
        self.use_old_gate = bool(use_old_gate)
        self.init_std = float(init_std)
        
        if mode == 'bilinear':
            self.W_m = nn.Linear(in_dim, bilinear_bottleneck, bias=False)
            self.W_t = nn.Linear(in_dim, bilinear_bottleneck, bias=False)
            self.W_out = nn.Linear(bilinear_bottleneck, in_dim, bias=False)
            self.output_norm = nn.LayerNorm(in_dim, elementwise_affine=False)
            for m in [self.W_m, self.W_t, self.W_out]:
                nn.init.normal_(m.weight, std=self.init_std)
            if self.use_old_gate:
                self.gate_net = nn.Sequential(
                    nn.Linear(in_dim, 64), nn.ReLU(inplace=True),
                    nn.Linear(64, 1), nn.Sigmoid())
                nn.init.constant_(self.gate_net[-2].bias, 2.0)
        elif mode == 'tc_mlp':
            self.net = nn.Sequential(
                nn.Linear(in_dim * 2, hidden_dim), nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, in_dim))
            self.output_norm = nn.LayerNorm(in_dim, elementwise_affine=False)
        elif mode == 'crossattn':
            self.W_q = nn.Linear(in_dim, in_dim, bias=False)
            self.W_k = nn.Linear(in_dim, in_dim, bias=False)
            self.q_norm = nn.LayerNorm(in_dim, elementwise_affine=False)
            self.output_norm = nn.LayerNorm(in_dim, elementwise_affine=False)
            self.tau = in_dim ** 0.5
            nn.init.normal_(self.W_q.weight, std=0.05)
            nn.init.normal_(self.W_k.weight, std=0.05)
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, in_dim))
            self.output_norm = nn.LayerNorm(in_dim, elementwise_affine=False)

        if freeze_gamma:
            self.register_buffer('gamma', torch.tensor(float(gamma_init)))
        else:
            self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.act_inference = bool(act_inference)
        self.freeze_gamma = bool(freeze_gamma)

    def _apply_norm(self, residual):
        if self.norm_mode == 'ln':
            return self.output_norm(residual)
        elif self.norm_mode == 'rms':
            # Global RMS: scale so mean per-token norm = target
            # No detach — gradients flow through, magnitude determined by target
            rms = residual.norm(dim=-1).mean()
            return residual * (self.rms_target / (rms + 1e-8))
        else:  # 'none'
            return residual

    def forward(self, memory, T_new=None):
        if self.mode in ('bilinear', 'tc_mlp', 'crossattn'):
            assert T_new is not None
            T_new = T_new.detach()
        
        if self.mode == 'bilinear':
            T_pool = T_new.mean(dim=0, keepdim=True)
            h = self.W_m(memory) * self.W_t(T_pool).squeeze(0)
            residual = self.W_out(h)
            residual = self._apply_norm(residual)
            if self.use_old_gate:
                gate = self.gate_net(memory.detach())
                residual = residual * gate
        elif self.mode == 'tc_mlp':
            T_pool = T_new.mean(dim=0, keepdim=True)
            T_bcast = T_pool.unsqueeze(0).expand(memory.shape[0], memory.shape[1], -1)
            M_aug = torch.cat([memory, T_bcast], dim=-1)
            residual = self.net(M_aug)
            residual = self._apply_norm(residual)
        elif self.mode == 'crossattn':
            Q = self.q_norm(self.W_q(T_new))
            memory_flat = memory.reshape(-1, memory.shape[-1]) if memory.dim() == 3 else memory
            K = self.W_k(memory_flat)
            A = F.softmax(Q @ K.T / self.tau, dim=-1)
            residual = A.T @ Q
            residual = self._apply_norm(residual)
        else:
            residual = self.net(memory)
            residual = self._apply_norm(residual)

        with torch.no_grad():
            self._cached_mem_norm = float(memory.detach().reshape(-1, memory.shape[-1]).norm(dim=-1).mean())
            self._cached_rm_norm = float(residual.detach().reshape(-1, residual.shape[-1]).norm(dim=-1).mean())
            if self.use_old_gate:
                self._cached_gate_mean = float(gate.detach().mean())
        if self.training:
            self._cached_residual = residual
        return memory + self.gamma * residual

    def extra_repr(self):
        return (f"mode={self.mode}, gamma={float(self.gamma.detach()):.4f}, "
                f"norm={self.norm_mode}, old_gate={self.use_old_gate}, init_std={self.init_std}")
