"""
GRMI: Gated Residual injection on encoder Memory.
Supports: mlp, bilinear, tc_mlp, crossattn.

Bilinear with old-class gate (use_old_gate=True):
  R = LN(W_out(W_m(M) * W_t(T_pool))) * sigmoid(MLP_gate(M))
  The gate MLP learns to suppress R at positions where it hurts detection
  (old-class locations) — fully self-supervised, 16K params, no extra loss.
"""
import torch, torch.nn as nn, torch.nn.functional as F
from torch import Tensor


class ResidualInject(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=128, gamma_init=1e-2,
                 dropout=0.0, act_inference=True, freeze_gamma=False,
                 mode='mlp', bilinear_bottleneck=64, use_old_gate=False):
        super().__init__()
        self.mode = mode
        self.in_dim = in_dim
        self.use_old_gate = bool(use_old_gate)
        if self.use_old_gate:
            assert mode == 'bilinear', f"old_gate only supports bilinear mode, got {mode}"

        if mode == 'bilinear':
            self.W_m = nn.Linear(in_dim, bilinear_bottleneck, bias=False)
            self.W_t = nn.Linear(in_dim, bilinear_bottleneck, bias=False)
            self.W_out = nn.Linear(bilinear_bottleneck, in_dim, bias=False)
            self.output_norm = nn.LayerNorm(in_dim, elementwise_affine=False)
            for m in [self.W_m, self.W_t, self.W_out]:
                nn.init.normal_(m.weight, std=0.1)
            if self.use_old_gate:
                self.gate_net = nn.Sequential(
                    nn.Linear(in_dim, 64), nn.ReLU(inplace=True),
                    nn.Linear(64, 1), nn.Sigmoid())
                # Start with gate ≈ 1 everywhere (don't suppress initially)
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

        else:  # mlp
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

    def forward(self, memory: Tensor, T_new: Tensor = None) -> Tensor:
        if self.mode in ('bilinear', 'tc_mlp', 'crossattn'):
            assert T_new is not None, f"{self.mode} requires T_new"
            T_new = T_new.detach()

        if self.mode == 'bilinear':
            T_pool = T_new.mean(dim=0, keepdim=True)
            h = self.W_m(memory) * self.W_t(T_pool).squeeze(0)
            residual = self.W_out(h)
            residual = self.output_norm(residual)
            if self.use_old_gate:
                # Learned spatial gate: suppresses R at positions where
                # the network learns that perturbation hurts detection.
                # Gate AFTER LN: LN first constrains residual to stable scale,
                # then gate selectively attenuates at old-class positions.
                gate = self.gate_net(memory)  # (..., 1)
                residual = residual * gate

        elif self.mode == 'tc_mlp':
            T_pool = T_new.mean(dim=0, keepdim=True)
            T_bcast = T_pool.unsqueeze(0).expand(memory.shape[0], memory.shape[1], -1)
            M_aug = torch.cat([memory, T_bcast], dim=-1)
            residual = self.net(M_aug)
            residual = self.output_norm(residual)

        elif self.mode == 'crossattn':
            Q = self.q_norm(self.W_q(T_new))
            if memory.dim() == 3:
                memory_flat = memory.reshape(-1, memory.shape[-1])
            else:
                memory_flat = memory
            K = self.W_k(memory_flat)
            A = F.softmax(Q @ K.T / self.tau, dim=-1)
            residual = A.T @ Q
            residual = self.output_norm(residual)

        else:  # mlp
            residual = self.net(memory)
            residual = self.output_norm(residual)

        with torch.no_grad():
            self._cached_mem_norm = float(memory.detach().reshape(-1, memory.shape[-1]).norm(dim=-1).mean())
            self._cached_rm_norm = float(residual.detach().reshape(-1, residual.shape[-1]).norm(dim=-1).mean())
            if self.use_old_gate:
                self._cached_gate_mean = float(gate.detach().mean())

        if self.training:
            self._cached_residual = residual

        return memory + self.gamma * residual

    def extra_repr(self) -> str:
        return (f"mode={self.mode}, gamma={float(self.gamma.detach()):.4f}, "
                f"act_inference={self.act_inference}, old_gate={self.use_old_gate}")
