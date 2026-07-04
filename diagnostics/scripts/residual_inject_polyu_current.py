"""
GRMI: Gated Residual injection on encoder Memory.
Supports three modes:
  - 'mlp':      M' = M + gamma * LN(MLP(M))
  - 'bilinear': M' = M + gamma * LN(Bilinear(M, T_pool))
  - 'tc_mlp':   M' = M + gamma * LN(MLP(concat(M, T_pool)))
  - 'crossattn': M' = M + gamma * CrossAttn(M, T_new)

All modes use LayerNorm on output for natural ||R|| bounding (~sqrt(d)=16).
T_new is always detached to prevent gradient flow to text_feat_map/BERT.
"""
import torch, torch.nn as nn, torch.nn.functional as F
from torch import Tensor


class ResidualInject(nn.Module):
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
            T_new = T_new.detach()  # prevent gradient to text_feat_map/BERT

        if self.mode == 'bilinear':
            T_pool = T_new.mean(dim=0, keepdim=True)
            h = self.W_m(memory) * self.W_t(T_pool).squeeze(0)
            residual = self.W_out(h)
            residual = self.output_norm(residual)

        elif self.mode == 'tc_mlp':
            T_pool = T_new.mean(dim=0, keepdim=True)
            T_bcast = T_pool.unsqueeze(0).expand(memory.shape[0], memory.shape[1], -1)
            M_aug = torch.cat([memory, T_bcast], dim=-1)
            residual = self.net(M_aug)
            residual = self.output_norm(residual)

        elif self.mode == 'crossattn':
            Q = self.q_norm(self.W_q(T_new))
            if memory.dim() == 3:
                memory = memory.reshape(-1, memory.shape[-1])
            K = self.W_k(memory)
            A = F.softmax(Q @ K.T / self.tau, dim=-1)
            residual = A.T @ Q
            residual = self.output_norm(residual)

        else:  # mlp
            residual = self.net(memory)
            residual = self.output_norm(residual)

        with torch.no_grad():
            self._cached_mem_norm = float(memory.detach().norm(dim=-1).mean())
            self._cached_rm_norm = float(residual.detach().norm(dim=-1).mean())

        if self.training:
            self._cached_residual = residual

        return memory + self.gamma * residual

    def extra_repr(self) -> str:
        return (f"mode={self.mode}, gamma={float(self.gamma.detach()):.4f}, "
                f"act_inference={self.act_inference}")
